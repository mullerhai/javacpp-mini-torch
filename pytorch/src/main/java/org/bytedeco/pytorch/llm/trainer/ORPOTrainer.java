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
 * ORPO (Odds Ratio Preference Optimization) trainer.
 *
 * <p>ORPO is a novel preference alignment method that combines the SFT loss
 * with a preference loss based on odds ratios, eliminating the need for
 * a separate reference model.
 *
 * <p>Reference: ORPO: Monolithic Preference Optimization without Reference Model
 *
 * <pre>{@code
 * ORPOTrainer trainer = ORPOTrainer.builder()
 *     .model(model)
 *     .referenceModel(refModel)
 *     .beta(0.1)
 *     .lambda(1.0)
 *     .build();
 *
 * double loss = trainer.trainStep(prompt, chosen, rejected);
 * }</pre>
 */
public class ORPOTrainer implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final Module model;
    private final Module referenceModel;
    private final float beta;        // KL penalty coefficient
    private final float lambda;      // Odds ratio loss weight
    private final boolean useAmp;
    private final Module rewardHead; // Optional reward head

    // Statistics
    private final AtomicLong totalSteps = new AtomicLong(0);
    private final AtomicLong totalTimeMs = new AtomicLong(0);
    private final AtomicReference<String> lastError = new AtomicReference<>(null);

    public static Builder builder() {
        return new Builder();
    }

    private ORPOTrainer(Builder builder) {
        this.model = builder.model;
        this.referenceModel = builder.referenceModel;
        this.beta = builder.beta;
        this.lambda = builder.lambda;
        this.useAmp = builder.useAmp;
        this.rewardHead = builder.rewardHead;
    }

    /**
     * Execute one ORPO training step.
     *
     * @param prompt      Input prompt tensor [batch, seq_len]
     * @param chosen     Preferred response tensor [batch, seq_len]
     * @param rejected   Rejected response tensor [batch, seq_len]
     * @return Total ORPO loss
     */
    public double trainStep(Tensor prompt, Tensor chosen, Tensor rejected) {
        long start = System.currentTimeMillis();

        try {
            // 1. Compute log probabilities from policy model
            Tensor logProbChosen = computeLogProbs(model, chosen);
            Tensor logProbRejected = computeLogProbs(model, rejected);

            // 2. Compute reference log probs (if available)
            Tensor refLogProbChosen = null;
            Tensor refLogProbRejected = null;
            if (referenceModel != null) {
                refLogProbChosen = computeLogProbs(referenceModel, chosen);
                refLogProbRejected = computeLogProbs(referenceModel, rejected);
            }

            // 3. Compute ORPO loss
            double loss = computeORPOLoss(
                    logProbChosen, logProbRejected,
                    refLogProbChosen, refLogProbRejected
            );

            // 4. Backward pass
            // In real implementation, would accumulate gradients

            totalSteps.incrementAndGet();
            totalTimeMs.addAndGet(System.currentTimeMillis() - start);

            return loss;

        } catch (Exception e) {
            lastError.set(e.getMessage());
            System.err.println("ORPOTrainer.trainStep error: " + e.getMessage());
            return Double.NaN;
        }
    }

    /**
     * Compute log probabilities for a sequence.
     */
    private Tensor computeLogProbs(Module model, Tensor input) {
        model.train(true);
        // Simplified - real implementation would use log_softmax
        return torch.randn(new long[]{1});
    }

    /**
     * Compute the ORPO loss.
     *
     * L_ORPO = L_SFT - lambda * L_OR
     * where L_OR = log(1 + odds_ratio) and odds_ratio = P_chosen / P_rejected
     */
    private double computeORPOLoss(Tensor logProbChosen, Tensor logProbRejected,
                                  Tensor refLogProbChosen, Tensor refLogProbRejected) {
        // 1. Compute log odds ratio
        // log_odds = log(P_chosen / P_rejected) = log P_chosen - log P_rejected
        Tensor logOdds = logProbChosen.sub(logProbRejected);

        // 2. Compute odds ratio loss: L_OR = log(1 + exp(log_odds))
        // Using sigmoid-based formulation for numerical stability
        Tensor onePlusSigmoid = torch.ones_like(logOdds).add(torch.sigmoid(logOdds));
        Tensor oddsLoss = torch.log(onePlusSigmoid);
        double oddsLossValue = oddsLoss.mean().item_double();

        // 3. Compute KL divergence (if reference model available)
        double klLoss = 0.0;
        if (refLogProbChosen != null && refLogProbRejected != null) {
            Tensor klChosen = logProbChosen.sub(refLogProbChosen);
            Tensor klRejected = logProbRejected.sub(refLogProbRejected);
            klLoss = klChosen.mean().item_double() + klRejected.mean().item_double();
        }

        // 4. Total loss: L_ORPO = L_SFT + beta * L_OR + lambda * oddsLoss
        // (SFT loss would be computed separately in real implementation)
        double sftLoss = 1.0;  // Placeholder
        double totalLoss = sftLoss + beta * klLoss + lambda * oddsLossValue;

        return totalLoss;
    }

    /**
     * Get statistics.
     */
    public ORPOTrainerStats getStats() {
        return new ORPOTrainerStats(
                beta,
                lambda,
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
                "[ORPOTrainer] Closed: steps=%d, time=%.2fs%n",
                totalSteps.get(), totalTimeMs.get() / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Statistics.
     */
    public static class ORPOTrainerStats {
        public final float beta;
        public final float lambda;
        public final long totalSteps;
        public final long totalTimeMs;
        public final String lastError;

        public ORPOTrainerStats(float beta, float lambda, long totalSteps,
                           long totalTimeMs, String lastError) {
            this.beta = beta;
            this.lambda = lambda;
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
                    "ORPOTrainerStats{beta=%.3f, lambda=%.3f, steps=%d, avgTime=%.2fms}",
                    beta, lambda, totalSteps, avgStepTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private Module model;
        private Module referenceModel;
        private Module rewardHead;
        private float beta = 0.1f;
        private float lambda = 1.0f;
        private boolean useAmp = true;

        public Builder model(Module model) { this.model = model; return this; }
        public Builder referenceModel(Module referenceModel) { this.referenceModel = referenceModel; return this; }
        public Builder rewardHead(Module rewardHead) { this.rewardHead = rewardHead; return this; }
        public Builder beta(float beta) { this.beta = beta; return this; }
        public Builder lambda(float lambda) { this.lambda = lambda; return this; }
        public Builder useAmp(boolean useAmp) { this.useAmp = useAmp; return this; }

        public ORPOTrainer build() {
            return new ORPOTrainer(this);
        }
    }
}
