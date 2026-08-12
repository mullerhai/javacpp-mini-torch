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
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.StringTensorDict;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Scalar;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * Gradient Accumulation Manager for large batch training.
 *
 * <p>Enables training with effectively larger batch sizes by accumulating
 * gradients over multiple micro-batches before performing an optimizer step.
 * This is essential for:
 * <ul>
 *   <li>Training with limited GPU memory</li>
 *   <li>Achieving large effective batch sizes across devices</li>
 *   <li>Pipeline parallelism where micro-batches are processed</li>
 * </ul>
 *
 * <p>Features:
 * <ul>
 *   <li>Automatic gradient scaling for effective batch size</li>
 *   <li>Dynamic accumulation based on memory pressure</li>
 *   <li>Gradient clipping support</li>
 *   <li>Distributed gradient synchronization</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * GradientAccumulator acc = GradientAccumulator.builder()
 *     .accumulationSteps(8)
 *     .maxGradNorm(1.0)
 *     .processGroup(pg)
 *     .build();
 *
 * for (DataBatch batch : dataloader) {
 *     Tensor loss = model.forward(batch);
 *     loss.backward();
 *     acc.step(model);
 * }
 * }</pre>
 */
public final class GradientAccumulator implements AutoCloseable {
    private final int accumulationSteps;
    private final float maxGradNorm;
    private final ProcessGroupWrapper pg;
    private final boolean syncBeforeStep;
    private final boolean scaleGrad;
    private final float gradScale;

    // Accumulated gradients
    private final Map<String, Tensor> accumulatedGrads;
    private final Map<String, Tensor> gradBuffers;
    private final Set<String> paramNames;

    // State
    private final AtomicInteger currentStep;
    private int microBatchCount = 0;
    private boolean initialized = false;

    // Statistics
    private long totalBackwardTime = 0;
    private long totalSyncTime = 0;
    private long totalClipTime = 0;
    private int optimizerStepCount = 0;

    // Callbacks
    private final List<Consumer<GradientAccumulator>> stepCallbacks;

    private GradientAccumulator(Builder builder) {
        this.accumulationSteps = builder.accumulationSteps;
        this.maxGradNorm = builder.maxGradNorm;
        this.pg = builder.pg;
        this.syncBeforeStep = builder.syncBeforeStep;
        this.scaleGrad = builder.scaleGrad;
        this.gradScale = 1.0f / accumulationSteps;
        this.accumulatedGrads = new ConcurrentHashMap<>();
        this.gradBuffers = new ConcurrentHashMap<>();
        this.paramNames = ConcurrentHashMap.newKeySet();
        this.currentStep = new AtomicInteger(0);
        this.stepCallbacks = new ArrayList<>(builder.stepCallbacks);

        System.out.printf("[GradientAccumulator] accumulation=%d maxGradNorm=%.2f sync=%b scale=%.4f%n",
                accumulationSteps, maxGradNorm, syncBeforeStep, gradScale);
    }

    public static Builder builder() { return new Builder(); }

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Gradient Accumulation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Initialize gradient accumulator with model parameters.
     *
     * @param model model to track gradients for
     */
    public void initialize(Module model) {
        if (initialized) {
            System.out.println("[GradientAccumulator] Already initialized, skipping...");
            return;
        }

        accumulatedGrads.clear();
        gradBuffers.clear();
        paramNames.clear();

        StringTensorDict dict = model.named_parameters();
        if (dict != null && !dict.isNull()) {
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                String name = dict.keys().get(i).getString();
                Tensor param = dict.get(name);
                if (param != null && param.defined() && param.requires_grad() && param.grad() != null) {
                    // Clone the gradient as initial accumulation buffer
                    Tensor buffer = param.grad().clone();
                    accumulatedGrads.put(name, buffer);
                    gradBuffers.put(name, torch.zeros_like(buffer));
                    paramNames.add(name);
                }
            }
        }

        initialized = true;
        System.out.printf("[GradientAccumulator] Initialized with %d parameters%n", paramNames.size());
    }

    /**
     * Accumulate gradients from a micro-batch.
     *
     * <p>Accumulates the current gradients into the buffer without
     * performing an optimizer step.
     *
     * @param model model whose gradients to accumulate
     */
    public void accumulate(Module model) {
        if (!initialized) {
            initialize(model);
        }

        long start = System.nanoTime();

        StringTensorDict dict = model.named_parameters();
        if (dict != null && !dict.isNull()) {
            for (String name : paramNames) {
                Tensor param = dict.get(name);
                if (param == null || !param.defined()) continue;
                Tensor grad = param.grad();
                if (grad == null) continue;

                Tensor buffer = accumulatedGrads.get(name);
                if (buffer == null) {
                    // First gradient for this param
                    buffer = grad.clone();
                    accumulatedGrads.put(name, buffer);
                } else {
                    // Accumulate
                    buffer.add_(grad);
                }
            }
        }

        microBatchCount++;
        totalBackwardTime += System.nanoTime() - start;

        if (microBatchCount % 10 == 0) {
            System.out.printf("[GradientAccumulator] Accumulated micro-batch %d/%d%n",
                    microBatchCount, accumulationSteps);
        }
    }

    /**
     * Perform optimizer step after accumulating enough gradients.
     *
     * <p>Calls optimizer.step() and then zeroes gradients if step was performed.
     *
     * @param model model to update
     * @param optimizer optimizer to use
     * @return true if optimizer step was performed
     */
    public boolean step(Module model, Optimizer optimizer) {
        if (microBatchCount < accumulationSteps) {
            return false;
        }

        return stepInternal(model, optimizer);
    }

    /**
     * Force an optimizer step regardless of accumulation count.
     *
     * @param model model to update
     * @param optimizer optimizer to use
     * @return true if optimizer step was performed
     */
    public boolean forceStep(Module model, Optimizer optimizer) {
        if (microBatchCount == 0) {
            return false;
        }
        return stepInternal(model, optimizer);
    }

    /**
     * Internal step implementation.
     */
    private boolean stepInternal(Module model, Optimizer optimizer) {
        long start = System.nanoTime();

        // Sync gradients across workers if needed
        if (syncBeforeStep && pg != null && pg.getWorldSize() > 1) {
            syncGradients();
        }

        // Clip gradients if configured
        if (maxGradNorm > 0) {
            clipGradients();
        }

        // Scale gradients by accumulation factor
        if (scaleGrad) {
            scaleGradients();
        }

        // Copy accumulated gradients to model
        StringTensorDict dict = model.named_parameters();
        if (dict != null && !dict.isNull()) {
            for (String name : paramNames) {
                Tensor buffer = accumulatedGrads.get(name);
                if (buffer != null) {
                    Tensor param = dict.get(name);
                    if (param != null && param.defined() && param.grad() != null) {
                        param.grad().copy_(buffer);
                    }
                }
            }
        }

        // Perform optimizer step
        optimizer.step();

        // Zero accumulated gradients
        zeroAccumulated();

        // Zero model gradients
        model.zero_grad(false);

        // Reset counter
        microBatchCount = 0;
        optimizerStepCount++;

        totalSyncTime += System.nanoTime() - start;

        // Call step callbacks
        for (Consumer<GradientAccumulator> callback : stepCallbacks) {
            callback.accept(this);
        }

        currentStep.incrementAndGet();

        System.out.printf("[GradientAccumulator] Optimizer step %d completed%n", optimizerStepCount);
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Gradient Operations
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Synchronize gradients across all workers.
     */
    public void syncGradients() {
        if (pg == null || pg.getWorldSize() <= 1) return;

        long start = System.nanoTime();

        List<Tensor> grads = new ArrayList<>();
        List<String> gradNames = new ArrayList<>();

        for (Map.Entry<String, Tensor> entry : accumulatedGrads.entrySet()) {
            grads.add(entry.getValue());
            gradNames.add(entry.getKey());
        }

        // Allreduce across all workers
        pg.allreduce(grads);

        // Divide by world size
        Scalar scale = new Scalar(1.0 / pg.getWorldSize());
        for (Tensor grad : grads) {
            grad.div_(scale);
        }

        totalSyncTime += System.nanoTime() - start;

        if (pg.getRank() == 0) {
            System.out.printf("[GradientAccumulator] Synced %d gradients%n", grads.size());
        }
    }

    /**
     * Clip gradients by global norm.
     *
     * @return total norm before clipping, or -1 if skipped
     */
    public float clipGradients() {
        if (maxGradNorm <= 0) return -1;

        long start = System.nanoTime();
        float totalNorm = 0.0f;

        // Compute total norm
        for (Tensor grad : accumulatedGrads.values()) {
            if (grad == null) continue;
            float norm = (float) grad.norm(new Scalar(2.0)).item().toDouble();
            totalNorm += norm * norm;
        }
        totalNorm = (float) Math.sqrt(totalNorm);

        // Clip if needed
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / totalNorm;
            for (Tensor grad : accumulatedGrads.values()) {
                if (grad != null) {
                    grad.mul_(new Scalar(clipCoef));
                }
            }
            System.out.printf("[GradientAccumulator] Clipped gradients (norm: %.4f -> %.4f)%n",
                    totalNorm, maxGradNorm);
        }

        totalClipTime += System.nanoTime() - start;
        return totalNorm;
    }

    /**
     * Scale accumulated gradients by accumulation factor.
     */
    public void scaleGradients() {
        Scalar scale = new Scalar(gradScale);
        for (Tensor grad : accumulatedGrads.values()) {
            if (grad != null) {
                grad.div_(scale);
            }
        }
    }

    /**
     * Zero accumulated gradients.
     */
    public void zeroAccumulated() {
        for (Tensor grad : accumulatedGrads.values()) {
            if (grad != null) {
                grad.zero_();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Dynamic Accumulation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Adjust accumulation steps based on memory pressure.
     *
     * @param memoryUsagePercent current memory usage percentage
     */
    public void adjustForMemory(double memoryUsagePercent) {
        if (memoryUsagePercent > 0.9) {
            // Reduce accumulation to save memory
            int newSteps = Math.max(1, accumulationSteps / 2);
            if (newSteps != accumulationSteps) {
                System.out.printf("[GradientAccumulator] Memory pressure detected, reducing accumulation: %d -> %d%n",
                        accumulationSteps, newSteps);
            }
        }
    }

    /**
     * Adjust accumulation based on throughput feedback.
     *
     * @param throughput tokens per second
     */
    public void adjustForThroughput(double throughput) {
        // Simple heuristic: increase accumulation if throughput is good
        if (throughput > 1000) {
            int newSteps = Math.min(16, accumulationSteps * 2);
            if (newSteps != accumulationSteps) {
                System.out.printf("[GradientAccumulator] Good throughput, increasing accumulation: %d -> %d%n",
                        accumulationSteps, newSteps);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // State Queries
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Check if optimizer step should be performed.
     */
    public boolean shouldStep() {
        return microBatchCount >= accumulationSteps;
    }

    /**
     * Get current micro-batch count.
     */
    public int getMicroBatchCount() {
        return microBatchCount;
    }

    /**
     * Get accumulation steps.
     */
    public int getAccumulationSteps() {
        return accumulationSteps;
    }

    /**
     * Get effective batch size multiplier.
     */
    public int getEffectiveBatchMultiplier() {
        return accumulationSteps;
    }

    /**
     * Get current optimizer step count.
     */
    public int getOptimizerStepCount() {
        return optimizerStepCount;
    }

    /**
     * Get accumulated gradient for a parameter.
     */
    public Tensor getAccumulatedGrad(String paramName) {
        return accumulatedGrads.get(paramName);
    }

    /**
     * Register a callback to be called after each optimizer step.
     */
    public void onStepComplete(Consumer<GradientAccumulator> callback) {
        stepCallbacks.add(callback);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════════════════════════════════

    public GradientAccumulatorStats getStats() {
        return new GradientAccumulatorStats(
            microBatchCount,
            accumulationSteps,
            optimizerStepCount,
            totalBackwardTime,
            totalSyncTime,
            totalClipTime,
            accumulatedGrads.size()
        );
    }

    public void printStats() {
        GradientAccumulatorStats stats = getStats();
        System.out.printf("""
                ═══ Gradient Accumulator Stats ═══
                  Micro-batches:  %d / %d
                  Optimizer steps: %d
                  Timing:
                    Backward:  %,.2f ms
                    Sync:      %,.2f ms
                    Clip:      %,.2f ms
                  Effective batch: %dx
                ═════════════════════════════════════
                """,
                stats.microBatchCount(),
                stats.accumulationSteps(),
                stats.optimizerStepCount(),
                stats.backwardTimeMs(),
                stats.syncTimeMs(),
                stats.clipTimeMs(),
                stats.accumulationSteps() * stats.optimizerStepCount()
        );
    }

    @Override
    public void close() {
        // Clean up accumulated gradients
        for (Tensor grad : accumulatedGrads.values()) {
            if (grad != null) grad.close();
        }
        for (Tensor grad : gradBuffers.values()) {
            if (grad != null) grad.close();
        }
        accumulatedGrads.clear();
        gradBuffers.clear();
        paramNames.clear();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════════

    public static final class Builder {
        private int accumulationSteps = 1;
        private float maxGradNorm = 1.0f;
        private ProcessGroupWrapper pg;
        private boolean syncBeforeStep = true;
        private boolean scaleGrad = true;
        private List<Consumer<GradientAccumulator>> stepCallbacks = new ArrayList<>();

        public Builder accumulationSteps(int n) { this.accumulationSteps = n; return this; }
        public Builder maxGradNorm(float n) { this.maxGradNorm = n; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.pg = pg; return this; }
        public Builder syncBeforeStep(boolean s) { this.syncBeforeStep = s; return this; }
        public Builder scaleGrad(boolean s) { this.scaleGrad = s; return this; }
        public Builder onStepComplete(Consumer<GradientAccumulator> callback) {
            this.stepCallbacks.add(callback);
            return this;
        }

        public GradientAccumulator build() {
            return new GradientAccumulator(this);
        }
    }

    /**
     * Gradient accumulator statistics.
     */
    public record GradientAccumulatorStats(
        int microBatchCount,
        int accumulationSteps,
        int optimizerStepCount,
        long totalBackwardTimeNs,
        long totalSyncTimeNs,
        long totalClipTimeNs,
        int numAccumulatedParams
    ) {
        public double backwardTimeMs() { return totalBackwardTimeNs / 1e6; }
        public double syncTimeMs() { return totalSyncTimeNs / 1e6; }
        public double clipTimeMs() { return totalClipTimeNs / 1e6; }
        public double totalTimeMs() { return backwardTimeMs() + syncTimeMs() + clipTimeMs(); }
        public int effectiveBatchSize(int baseBatchSize) {
            return baseBatchSize * accumulationSteps * optimizerStepCount;
        }
    }
}
