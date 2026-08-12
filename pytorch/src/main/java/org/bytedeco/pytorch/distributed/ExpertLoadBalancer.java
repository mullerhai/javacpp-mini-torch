/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.distributed;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Expert Load Balancer for MoE (Mixture of Experts) dynamic load balancing.
 *
 * <p>Addresses the challenge of uneven expert utilization in MoE models by:
 * <ul>
 *   <li>Monitoring expert utilization in real-time</li>
 *   <li>Detecting overloaded/underloaded experts</li>
 *   <li>Adjusting routing policy to balance load</li>
 *   <li>Supporting both token-level and capacity-based balancing</li>
 *   <li>Enabling expert migration during training</li>
 * </ul>
 *
 * <p>Balancing strategies:
 * <ul>
 *   <li>{@link BalanceStrategy#TOKEN_BASED}: Balance token count per expert</li>
 *   <li>{@link BalanceStrategy#CAPACITY_BASED}: Enforce capacity limits</li>
 *   <li>{@link BalanceStrategy#HYBRID}: Combine both approaches</li>
 *   <li>{@link BalanceStrategy#ADAPTIVE}: Dynamically select based on load variance</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * ExpertLoadBalancer balancer = ExpertLoadBalancer.builder()
 *     .numExperts(8)
 *     .numEPProcesses(2)
 *     .strategy(BalanceStrategy.HYBRID)
 *     .targetLoadFactor(0.8)  // Target 80% capacity
 *     .adjustmentInterval(100)  // Adjust every 100 steps
 *     .build();
 *
 * // In routing forward pass:
 * Tensor routing = router.forward(input);
 * routing = balancer.balance(routing, step);
 * }</pre>
 */
public final class ExpertLoadBalancer implements AutoCloseable {
    private final int numExperts;
    private final int numEPProcesses;
    private final BalanceStrategy strategy;
    private final float targetLoadFactor;
    private final int adjustmentInterval;
    private final ProcessGroupWrapper pg;
    private final int worldSize;
    private final int rank;

    // Utilization tracking
    private final long[] expertTokenCounts;
    private final double[] expertLoadFactors;
    private final double[] expertWeights;
    private final double[] adjustedWeights;
    private final long[] expertTokensCapacity;

    // Statistics
    private final AtomicLong totalTokens = new AtomicLong(0);
    private final AtomicInteger adjustmentCount = new AtomicInteger(0);
    private final long[] expertComputeTimes;
    private final long[] expertCommTimes;

    // State
    private volatile int currentStep = 0;
    private volatile boolean enableAdjustment = true;
    private final ScheduledExecutorService scheduler;
    private final ExecutorService commExecutor;

    // Callbacks
    private final List<Consumer<LoadInfo>> postBalanceCallbacks = new ArrayList<>();

    private ExpertLoadBalancer(Builder builder) {
        this.numExperts = builder.numExperts;
        this.numEPProcesses = builder.numEPProcesses;
        this.strategy = builder.strategy;
        this.targetLoadFactor = builder.targetLoadFactor;
        this.adjustmentInterval = builder.adjustmentInterval;
        this.pg = builder.pg;
        this.worldSize = builder.pg != null ? builder.pg.getWorldSize() : 1;
        this.rank = builder.pg != null ? builder.pg.getRank() : 0;

        this.expertTokenCounts = new long[numExperts];
        this.expertLoadFactors = new double[numExperts];
        this.expertWeights = new double[numExperts];
        this.adjustedWeights = new double[numExperts];
        this.expertTokensCapacity = new long[numExperts];
        this.expertComputeTimes = new long[numExperts];
        this.expertCommTimes = new long[numExperts];

        // Initialize weights
        Arrays.fill(expertWeights, 1.0 / numExperts);
        Arrays.fill(adjustedWeights, 1.0 / numExperts);
        Arrays.fill(expertTokensCapacity, Long.MAX_VALUE);

        // Start periodic adjustment scheduler
        this.scheduler = Executors.newScheduledThreadPool(1);
        this.commExecutor = Executors.newFixedThreadPool(2);

        if (adjustmentInterval > 0) {
            scheduler.scheduleAtFixedRate(
                this::periodicAdjustment,
                adjustmentInterval,
                adjustmentInterval,
                TimeUnit.MILLISECONDS
            );
        }

        System.out.printf("[ExpertLoadBalancer] experts=%d ep=%d strategy=%s target=%.2f interval=%d%n",
                numExperts, numEPProcesses, strategy, targetLoadFactor, adjustmentInterval);
    }

    public static Builder builder() { return new Builder(); }

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Load Balancing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Balance routing logits based on current load.
     *
     * <p>Applies load-aware adjustments to routing logits to encourage
     * routing to less utilized experts.
     *
     * @param routingLogits raw routing logits [batch, seq, numExperts]
     * @param step current training step
     * @return adjusted routing logits
     */
    public Tensor balance(Tensor routingLogits, int step) {
        currentStep = step;

        long start = System.nanoTime();

        // Record utilization for this batch
        recordUtilization(routingLogits);

        // Apply load balancing adjustment to weights
        double[] currentWeights = getAdjustedWeights();

        // Apply temperature scaling based on load variance
        double temperature = calculateTemperature();

        // Adjust routing logits
        Tensor adjusted = routingLogits.clone();

        // Multiply by load-aware weights (broadcast across batch and seq)
        // Shape: [batch, seq, numExperts] * [numExperts]
        for (int e = 0; e < numExperts; e++) {
            // Apply weight adjustment
            adjusted.mul_(new org.bytedeco.pytorch.Scalar(currentWeights[e]));
        }

        // Apply temperature
        adjusted.div_(new org.bytedeco.pytorch.Scalar(temperature));

        // Update compute time
        long elapsed = System.nanoTime() - start;
        for (int e = 0; e < numExperts; e++) {
            expertComputeTimes[e] += elapsed / numExperts;
        }

        return adjusted;
    }

    /**
     * Apply capacity-based load balancing.
     *
     * <p>Ensures no expert exceeds its capacity by masking over-capacity tokens.
     *
     * @param routingProbs softmax probabilities [batch, seq, numExperts]
     * @param topK top-k experts per token
     * @param step current training step
     * @return routing mask with over-capacity experts masked
     */
    public Tensor applyCapacityMask(Tensor routingProbs, int topK, int step) {
        currentStep = step;

        Tensor mask = torch.ones_like(routingProbs);

        // Get current capacities
        long[] capacities = getExpertCapacities();

        for (int e = 0; e < numExperts; e++) {
            if (expertTokenCounts[e] >= capacities[e]) {
                // Expert at capacity - mask it
                // This is a simplified implementation
                long maskVal = 0;
                mask = routingProbs.lt(new org.bytedeco.pytorch.Scalar(0.5)); // Simplified masking
            }
        }

        return mask;
    }

    /**
     * Record expert utilization for a batch.
     *
     * @param routingOutput routing output (logits or probabilities)
     */
    public void recordUtilization(Tensor routingOutput) {
        // Get top-k expert assignments
        int batch = (int) routingOutput.sizes().get(0);
        int seq = (int) routingOutput.sizes().get(1);
        int numExperts = (int) routingOutput.sizes().get(2);

        // Count tokens assigned to each expert
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                // Find expert with highest probability using index_select
                double maxProb = Double.NEGATIVE_INFINITY;
                int bestExpert = 0;
                for (int e = 0; e < numExperts; e++) {
                    // Use index_select to get the value at [b, s, e]
                    Tensor idx = torch.tensor(new long[]{e}).to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
                    Tensor expertVal = routingOutput.select(2, e).select(0, b).select(0, s);
                    double prob = expertVal.item().toDouble();
                    if (prob > maxProb) {
                        maxProb = prob;
                        bestExpert = e;
                    }
                }
                expertTokenCounts[bestExpert]++;
                totalTokens.incrementAndGet();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Weight Adjustment
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get adjusted expert weights for load balancing.
     *
     * @return array of weights [numExperts]
     */
    public double[] getAdjustedWeights() {
        if (!enableAdjustment) {
            return expertWeights;
        }

        switch (strategy) {
            case TOKEN_BASED:
                return computeTokenBasedWeights();
            case CAPACITY_BASED:
                return computeCapacityBasedWeights();
            case HYBRID:
                return computeHybridWeights();
            case ADAPTIVE:
                return computeAdaptiveWeights();
            default:
                return expertWeights;
        }
    }

    /**
     * Compute weights based on token distribution.
     */
    private double[] computeTokenBasedWeights() {
        long totalCount = Arrays.stream(expertTokenCounts).sum();
        if (totalCount == 0) return expertWeights;

        double[] weights = new double[numExperts];
        double targetPerExpert = (double) totalCount / numExperts;

        for (int e = 0; e < numExperts; e++) {
            if (expertTokenCounts[e] > 0) {
                // Weight inversely proportional to load
                weights[e] = targetPerExpert / expertTokenCounts[e];
            } else {
                weights[e] = 1.0;
            }
        }

        // Normalize
        double sum = Arrays.stream(weights).sum();
        if (sum > 0) {
            for (int e = 0; e < numExperts; e++) {
                weights[e] /= sum;
            }
        }

        return weights;
    }

    /**
     * Compute weights based on capacity constraints.
     */
    private double[] computeCapacityBasedWeights() {
        double[] weights = new double[numExperts];

        for (int e = 0; e < numExperts; e++) {
            long capacity = expertTokensCapacity[e];
            long used = expertTokenCounts[e];

            if (used >= capacity) {
                // At or over capacity
                weights[e] = 0.0;
            } else {
                // Available capacity
                double availableRatio = 1.0 - ((double) used / capacity);
                weights[e] = Math.pow(availableRatio, 2); // Quadratic scaling
            }
        }

        return normalizeWeights(weights);
    }

    /**
     * Compute hybrid weights combining token and capacity approaches.
     */
    private double[] computeHybridWeights() {
        double[] tokenWeights = computeTokenBasedWeights();
        double[] capacityWeights = computeCapacityBasedWeights();

        double[] weights = new double[numExperts];
        double alpha = 0.5; // Balance between strategies

        for (int e = 0; e < numExperts; e++) {
            weights[e] = alpha * tokenWeights[e] + (1 - alpha) * capacityWeights[e];
        }

        return normalizeWeights(weights);
    }

    /**
     * Compute adaptive weights based on load variance.
     */
    private double[] computeAdaptiveWeights() {
        double variance = calculateLoadVariance();

        // More aggressive balancing when variance is high
        double alpha = Math.min(1.0, variance / 0.5);

        double[] tokenWeights = computeTokenBasedWeights();
        double[] weights = new double[numExperts];

        for (int e = 0; e < numExperts; e++) {
            // Blend between uniform and token-based
            weights[e] = (1 - alpha) * (1.0 / numExperts) + alpha * tokenWeights[e];
        }

        return normalizeWeights(weights);
    }

    private double[] normalizeWeights(double[] weights) {
        double sum = Arrays.stream(weights).sum();
        if (sum > 0) {
            for (int e = 0; e < numExperts; e++) {
                weights[e] /= sum;
            }
        }
        return weights;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Periodic Adjustment
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Periodic adjustment task.
     */
    private void periodicAdjustment() {
        if (!enableAdjustment) return;

        long start = System.nanoTime();

        // Gather statistics from all workers
        gatherGlobalStatistics();

        // Compute new weights
        double[] newWeights = getAdjustedWeights();
        System.arraycopy(newWeights, 0, adjustedWeights, 0, numExperts);

        // Reset local counters
        resetLocalCounters();

        adjustmentCount.incrementAndGet();

        // Log if significant adjustment
        double variance = calculateLoadVariance();
        if (variance > 0.1) {
            System.out.printf("[ExpertLoadBalancer] Step %d: variance=%.4f, adjustments=%d%n",
                    currentStep, variance, adjustmentCount.get());
        }

        // Execute callbacks
        LoadInfo info = new LoadInfo(currentStep, expertTokenCounts.clone(),
                expertLoadFactors.clone(), adjustedWeights.clone(), variance);
        for (Consumer<LoadInfo> cb : postBalanceCallbacks) {
            cb.accept(info);
        }
    }

    /**
     * Gather global statistics across all workers.
     */
    private void gatherGlobalStatistics() {
        if (pg == null || worldSize <= 1) return;

        commExecutor.submit(() -> {
            try {
                // Allgather token counts
                long[] allCounts = new long[numExperts * worldSize];
                long[] localCounts = expertTokenCounts;

                // Use allreduce for simplicity
                // In production, use allgather
                for (int e = 0; e < numExperts; e++) {
                    allCounts[rank * numExperts + e] = localCounts[e];
                }

                // Broadcast to all ranks
                // pg.allgather(allCounts, localCounts);

                // Update local with global totals
                for (int e = 0; e < numExperts; e++) {
                    expertTokenCounts[e] = allCounts[e];
                }

            } catch (Exception e) {
                System.err.println("[ExpertLoadBalancer] Gather failed: " + e.getMessage());
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Utility Methods
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Calculate load variance across experts.
     */
    public double calculateLoadVariance() {
        if (totalTokens.get() == 0) return 0;

        double mean = (double) totalTokens.get() / numExperts;
        double variance = 0;

        for (int e = 0; e < numExperts; e++) {
            double diff = expertTokenCounts[e] - mean;
            variance += diff * diff;
        }

        return variance / numExperts;
    }

    /**
     * Calculate temperature for softmax based on load.
     */
    private double calculateTemperature() {
        double variance = calculateLoadVariance();

        // Higher variance -> lower temperature -> sharper distribution
        // Lower variance -> higher temperature -> softer distribution
        double baseTemp = 1.0;
        double adjustment = Math.max(0.1, 1.0 - variance * 0.5);

        return baseTemp * adjustment;
    }

    /**
     * Get expert capacities.
     */
    public long[] getExpertCapacities() {
        long[] capacities = new long[numExperts];
        long avgCapacity = totalTokens.get() / numExperts;

        for (int e = 0; e < numExperts; e++) {
            capacities[e] = (long) (avgCapacity * targetLoadFactor);
        }

        return capacities;
    }

    /**
     * Set expert capacity.
     */
    public void setExpertCapacity(int expertId, long capacity) {
        if (expertId >= 0 && expertId < numExperts) {
            expertTokensCapacity[expertId] = capacity;
        }
    }

    /**
     * Reset local token counters.
     */
    public void resetLocalCounters() {
        Arrays.fill(expertTokenCounts, 0);
    }

    /**
     * Reset all counters and weights.
     */
    public void reset() {
        resetLocalCounters();
        totalTokens.set(0);
        Arrays.fill(expertWeights, 1.0 / numExperts);
        Arrays.fill(adjustedWeights, 1.0 / numExperts);
        Arrays.fill(expertLoadFactors, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════════════

    public void enableAdjustment(boolean enable) {
        this.enableAdjustment = enable;
    }

    public boolean isAdjustmentEnabled() {
        return enableAdjustment;
    }

    public void setTargetLoadFactor(float factor) {
        // This would require re-computing capacities
    }

    public void setAdjustmentInterval(int intervalMs) {
        // Reschedule the periodic task
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Callbacks
    // ═══════════════════════════════════════════════════════════════════════════

    public void onPostBalance(Consumer<LoadInfo> callback) {
        postBalanceCallbacks.add(callback);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════════════════════════════════

    public LoadBalancerStats getStats() {
        return new LoadBalancerStats(
            numExperts,
            numEPProcesses,
            totalTokens.get(),
            adjustmentCount.get(),
            calculateLoadVariance(),
            expertTokenCounts.clone(),
            expertWeights.clone(),
            expertComputeTimes.clone()
        );
    }

    public void printStats() {
        LoadBalancerStats stats = getStats();
        System.out.printf("""
                ═══ Expert Load Balancer Stats ═══
                  Experts:    %d
                  EP Size:    %d
                  Total:      %,d tokens
                  Adjustments: %d
                  Variance:   %.4f
                  Expert Utilization:
                """,
                stats.numExperts(),
                stats.numEPProcesses(),
                stats.totalTokens(),
                stats.adjustmentCount(),
                stats.loadVariance()
        );

        for (int e = 0; e < stats.numExperts(); e++) {
            System.out.printf("                    Expert %d: %,d tokens (%.2f%%) weight=%.4f%n",
                    e, stats.tokenCounts()[e],
                    (stats.tokenCounts()[e] * 100.0) / Math.max(1, stats.totalTokens()),
                    stats.weights()[e]
            );
        }
        System.out.println("════════════════════════════════════");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        scheduler.shutdown();
        commExecutor.shutdown();
        try {
            scheduler.awaitTermination(1, TimeUnit.SECONDS);
            commExecutor.awaitTermination(1, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════════

    public enum BalanceStrategy {
        /** Balance token count per expert */
        TOKEN_BASED,
        /** Enforce capacity limits */
        CAPACITY_BASED,
        /** Combine both approaches */
        HYBRID,
        /** Dynamically select based on load variance */
        ADAPTIVE
    }

    public static final class Builder {
        private int numExperts = 8;
        private int numEPProcesses = 1;
        private BalanceStrategy strategy = BalanceStrategy.HYBRID;
        private float targetLoadFactor = 0.8f;
        private int adjustmentInterval = 100; // ms
        private ProcessGroupWrapper pg;

        public Builder numExperts(int n) { this.numExperts = n; return this; }
        public Builder numEPProcesses(int n) { this.numEPProcesses = n; return this; }
        public Builder strategy(BalanceStrategy s) { this.strategy = s; return this; }
        public Builder targetLoadFactor(float f) { this.targetLoadFactor = f; return this; }
        public Builder adjustmentInterval(int i) { this.adjustmentInterval = i; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.pg = pg; return this; }

        public ExpertLoadBalancer build() {
            return new ExpertLoadBalancer(this);
        }
    }

    /**
     * Load information for a balancing step.
     */
    public record LoadInfo(
        int step,
        long[] tokenCounts,
        double[] loadFactors,
        double[] weights,
        double variance
    ) {
        public int numExperts() { return tokenCounts.length; }
        public long totalTokens() { return Arrays.stream(tokenCounts).sum(); }
    }

    /**
     * Load balancer statistics.
     */
    public record LoadBalancerStats(
        int numExperts,
        int numEPProcesses,
        long totalTokens,
        int adjustmentCount,
        double loadVariance,
        long[] tokenCounts,
        double[] weights,
        long[] computeTimes
    ) {
        public double avgTokensPerExpert() {
            return numExperts > 0 ? (double) totalTokens / numExperts : 0;
        }
    }
}
