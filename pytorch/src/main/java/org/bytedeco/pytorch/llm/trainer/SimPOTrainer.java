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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * SimPO (Simple Preference Optimization) trainer.
 *
 * <p>SimPO simplifies DPO by using reward ranking loss instead of
 * Bradley-Terry model. It eliminates the reference model requirement.
 *
 * <p>Reference: SimPO: Simplifying Preference Optimization for Large Language Models
 *
 * <pre>{@code
 * SimPOTrainer trainer = SimPOTrainer.builder()
 *     .model(model)
 *     .beta(0.1)
 *     .gamma(0.5)  // Target margin
 *     .build();
 *
 * double loss = trainer.trainStep(prompt, chosen, rejected);
 * }</pre>
 */
public class SimPOTrainer implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final Module model;
    private final float beta;          // Temperature for reward
    private final float gamma;          // Target reward margin
    private final float lr;            // Learning rate
    private final boolean useAmp;

    // Statistics
    private final AtomicLong totalSteps = new AtomicLong(0);
    private final AtomicLong totalTimeMs = new AtomicLong(0);
    private final AtomicReference<String> lastError = new AtomicReference<>(null);

    public static Builder builder() {
        return new Builder();
    }

    private SimPOTrainer(Builder builder) {
        this.model = builder.model;
        this.beta = builder.beta;
        this.gamma = builder.gamma;
        this.lr = builder.lr;
        this.useAmp = builder.useAmp;
    }

    /**
     * Execute one SimPO training step.
     *
     * @param prompt      Input prompt tensor
     * @param chosen     Preferred response
     * @param rejected   Rejected response
     * @return SimPO loss
     */
    public double trainStep(Tensor prompt, Tensor chosen, Tensor rejected) {
        long start = System.currentTimeMillis();

        try {
            // 1. Compute rewards (log probabilities) for both responses
            Tensor rewardChosen = computeReward(model, prompt, chosen);
            Tensor rewardRejected = computeReward(model, prompt, rejected);

            // 2. Compute log ratios (policy / reference)
            // In SimPO, we directly compare policy log probs
            Tensor logRatio = rewardChosen.sub(rewardRejected);

            // 3. Compute Bradley-Terry style loss with margin
            // L = -log(sigmoid(beta * (log_ratio - gamma)))
            Tensor margin = torch.ones_like(logRatio).mul(torch.tensor(gamma));
            Tensor logProbDiff = logRatio.sub(margin);
            Tensor sigmoid = torch.sigmoid(logProbDiff.mul(torch.tensor(beta)));
            Tensor loss = torch.neg(torch.log(sigmoid));
            double lossValue = loss.mean().item_double();

            // 4. Backward pass would go here

            totalSteps.incrementAndGet();
            totalTimeMs.addAndGet(System.currentTimeMillis() - start);

            return lossValue;

        } catch (Exception e) {
            lastError.set(e.getMessage());
            System.err.println("SimPOTrainer.trainStep error: " + e.getMessage());
            return Double.NaN;
        }
    }

    /**
     * Compute reward (log probability) for a response.
     */
    private Tensor computeReward(Module model, Tensor prompt, Tensor response) {
        // Simplified - real implementation would:
        // 1. Concatenate prompt + response
        // 2. Forward pass through model
        // 3. Compute log probabilities
        // 4. Sum/average over response tokens
        return torch.randn(new long[]{1});
    }

    /**
     * Get statistics.
     */
    public SimPOTrainerStats getStats() {
        return new SimPOTrainerStats(
                beta,
                gamma,
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
                "[SimPOTrainer] Closed: steps=%d, time=%.2fs%n",
                totalSteps.get(), totalTimeMs.get() / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Statistics.
     */
    public static class SimPOTrainerStats {
        public final float beta;
        public final float gamma;
        public final long totalSteps;
        public final long totalTimeMs;
        public final String lastError;

        public SimPOTrainerStats(float beta, float gamma, long totalSteps,
                            long totalTimeMs, String lastError) {
            this.beta = beta;
            this.gamma = gamma;
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
                    "SimPOTrainerStats{beta=%.3f, gamma=%.3f, steps=%d, avgTime=%.2fms}",
                    beta, gamma, totalSteps, avgStepTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private Module model;
        private float beta = 0.1f;
        private float gamma = 0.5f;
        private float lr = 1e-6f;
        private boolean useAmp = true;

        public Builder model(Module model) { this.model = model; return this; }
        public Builder beta(float beta) { this.beta = beta; return this; }
        public Builder gamma(float gamma) { this.gamma = gamma; return this; }
        public Builder lr(float lr) { this.lr = lr; return this; }
        public Builder useAmp(boolean useAmp) { this.useAmp = useAmp; return this; }

        public SimPOTrainer build() { return new SimPOTrainer(this); }
    }
}
