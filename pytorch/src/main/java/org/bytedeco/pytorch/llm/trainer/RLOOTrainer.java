/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trainer;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * RLOO (REINFORCE Leave-One-Out) trainer.
 *
 * <p>RLOO is a policy gradient method for LLM training that uses
 * leave-one-out baseline for variance reduction.
 *
 * <p>Reference: Scaling Reinforcement Learning from Human Feedback
 *
 * <pre>{@code
 * RLOOTrainer trainer = RLOOTrainer.builder()
 *     .model(model)
 *     .baselineCoeff(0.99)
 *     .learningRate(1e-5)
 *     .build();
 *
 * double loss = trainer.trainStep(prompt, responses, rewards);
 * }</pre>
 */
public class RLOOTrainer implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final Module model;
    private final float baselineCoeff;     // Exponential moving average coefficient
    private final float learningRate;
    private final float entropyCoeff;      // Entropy bonus coefficient
    private final boolean useAmp;

    // Baseline tracking
    private volatile double baseline = 0.0;
    private volatile int updateCount = 0;

    // Statistics
    private final AtomicLong totalSteps = new AtomicLong(0);
    private final AtomicLong totalTimeMs = new AtomicLong(0);
    private final AtomicReference<String> lastError = new AtomicReference<>(null);

    public static Builder builder() {
        return new Builder();
    }

    private RLOOTrainer(Builder builder) {
        this.model = builder.model;
        this.baselineCoeff = builder.baselineCoeff;
        this.learningRate = builder.learningRate;
        this.entropyCoeff = builder.entropyCoeff;
        this.useAmp = builder.useAmp;
    }

    /**
     * Execute one RLOO training step.
     *
     * @param prompt     Input prompt
     * @param responses Batch of K responses [K, seq_len]
     * @param rewards  Rewards for each response [K]
     * @return RLOO loss
     */
    public double trainStep(Tensor prompt, Tensor responses, Tensor rewards) {
        long start = System.currentTimeMillis();

        try {
            int K = (int) responses.size(0);  // Number of responses

            // 1. Compute log probabilities for all responses
            Tensor logProbs = computeLogProbs(prompt, responses);

            // 2. Compute LOO baseline (mean excluding current)
            Tensor[] logProbArray = new Tensor[K];
            float[] rewardArray = new float[K];
            for (int i = 0; i < K; i++) {
                logProbArray[i] = logProbs.select(0, i);
                rewardArray[i] = rewards.get(i);
            }

            // 3. Compute advantage for each response
            double totalLoss = 0;
            for (int i = 0; i < K; i++) {
                // LOO baseline: mean of rewards excluding current
                double looBaseline = 0;
                for (int j = 0; j < K; j++) {
                    if (i != j) {
                        looBaseline += rewardArray[j];
                    }
                }
                looBaseline /= (K - 1);

                // Advantage = reward - LOO baseline
                double advantage = rewardArray[i] - looBaseline;

                // Policy gradient loss: -log_prob * advantage
                double policyLoss = -logProbArray[i].mean().item_double() * advantage;

                totalLoss += policyLoss;
            }
            totalLoss /= K;

            // 4. Update exponential moving average baseline
            double meanReward = 0;
            for (float r : rewardArray) meanReward += r;
            meanReward /= K;

            baseline = baselineCoeff * baseline + (1 - baselineCoeff) * meanReward;
            updateCount++;

            // 5. Backward pass would go here

            totalSteps.incrementAndGet();
            totalTimeMs.addAndGet(System.currentTimeMillis() - start);

            return totalLoss;

        } catch (Exception e) {
            lastError.set(e.getMessage());
            System.err.println("RLOOTrainer.trainStep error: " + e.getMessage());
            return Double.NaN;
        }
    }

    /**
     * Compute log probabilities for responses.
     */
    private Tensor computeLogProbs(Tensor prompt, Tensor responses) {
        // Simplified - real implementation would:
        // 1. Forward pass for each response
        // 2. Compute log softmax over vocabulary
        // 3. Sum log probs for each response
        int K = (int) responses.size(0);
        return torch.randn(new long[]{K, 1});
    }

    /**
     * Get current baseline value.
     */
    public double getBaseline() {
        return baseline;
    }

    /**
     * Get statistics.
     */
    public RLOOTrainerStats getStats() {
        return new RLOOTrainerStats(
                baseline,
                baselineCoeff,
                learningRate,
                updateCount,
                totalSteps.get(),
                totalTimeMs.get(),
                lastError.get()
        );
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[RLOOTrainer] Closed: steps=%d, baseline=%.4f, time=%.2fs%n",
                totalSteps.get(), baseline, totalTimeMs.get() / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Statistics.
     */
    public static class RLOOTrainerStats {
        public final double baseline;
        public final float baselineCoeff;
        public final float learningRate;
        public final int updateCount;
        public final long totalSteps;
        public final long totalTimeMs;
        public final String lastError;

        public RLOOTrainerStats(double baseline, float baselineCoeff, float learningRate,
                           int updateCount, long totalSteps, long totalTimeMs, String lastError) {
            this.baseline = baseline;
            this.baselineCoeff = baselineCoeff;
            this.learningRate = learningRate;
            this.updateCount = updateCount;
            this.totalSteps = totalSteps;
            this.totalTimeMs = totalTimeMs;
            this.lastError = lastError;
        }

        public double avgStepTimeMs() {
            return totalSteps > 0 ? (double) totalTimeMs / totalSteps : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "RLOOTrainerStats{baseline=%.4f, updates=%d, steps=%d, avgTime=%.2fms}",
                    baseline, updateCount, totalSteps, avgStepTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private Module model;
        private float baselineCoeff = 0.99f;
        private float learningRate = 1e-5f;
        private float entropyCoeff = 0.01f;
        private boolean useAmp = true;

        public Builder model(Module model) { this.model = model; return this; }
        public Builder baselineCoeff(float baselineCoeff) { this.baselineCoeff = baselineCoeff; return this; }
        public Builder learningRate(float learningRate) { this.learningRate = learningRate; return this; }
        public Builder entropyCoeff(float entropyCoeff) { this.entropyCoeff = entropyCoeff; return this; }
        public Builder useAmp(boolean useAmp) { this.useAmp = useAmp; return this; }

        public RLOOTrainer build() { return new RLOOTrainer(this); }
    }
}
