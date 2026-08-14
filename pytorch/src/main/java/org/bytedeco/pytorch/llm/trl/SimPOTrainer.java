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
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.trl.config.SimPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.SimPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.TensorVector;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * SimPO (Simple Preference Optimization) trainer.
 *
 * <p>SimPO removes the reference model term from DPO and uses a target margin
 * on log probabilities instead. This provides:
 * <ul>
 *   <li>Reference-free training (reduced memory and compute)</li>
 *   <li>More robust to hyperparameters</li>
 *   <li>Better alignment with generation metrics (length bias correction)</li>
 *   <li>Simplified implementation and deployment</li>
 * </ul>
 *
 * <p>The key innovation is the reward margin term:
 * <pre>
 *   reward_margin = β * (π(y_w) / |y_w| - π(y_l) / |y_l|) - γ
 * </pre>
 * where γ is the target margin and lengths are normalized.
 *
 * <p>Reference: "SimPO: Simple Preference Optimization" (Meng et al., 2024)
 * <a href="https://arxiv.org/abs/2405.14734">arXiv:2405.14734</a>
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids}</li>
 *   <li>optional {@code chosen_attention_mask}, {@code rejected_attention_mask}</li>
 *   <li>or precomputed {@code policy_chosen_logps} / {@code policy_rejected_logps}</li>
 * </ul>
 *
 * <pre>{@code
 * SimPOConfig config = SimPOConfig.builder()
 *     .beta(2.0)
 *     .targetMargin(1.0)
 *     .lengthNormalize(true)
 *     .learningRate(1e-6)
 *     .build();
 *
 * try (SimPOTrainer trainer = new SimPOTrainer(policyModel, policyForward, optimizer, config)) {
 *     for (Map<String, Tensor> batch : dataloader) {
 *         trainer.trainingStep(batch);
 *     }
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SimPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";
    public static final String ALGORITHM_ID = "simpo";

    private volatile boolean closed;

    private final Module policy;
    private final LlmForward policyForward;
    private final SimPOConfig simpoConfig;
    private final TensorVector params;

    // Metrics tracking
    private double totalRewardMargin;
    private double rewardMarginCount;

    /**
     * Create SimPO trainer.
     *
     * @param policy Policy model to train
     * @param policyForward Forward function for policy
     * @param optimizer Optimizer
     * @param config SimPO configuration
     */
    public SimPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            SimPOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.simpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();

        System.out.printf(
                "[SimPOTrainer v%s] beta=%.2f, targetMargin=%.2f, lengthNorm=%s%n",
                VERSION, simpoConfig.beta(), simpoConfig.targetMargin(),
                simpoConfig.lengthNormalize());
    }

    // ==================== BaseTrainer Overrides ====================

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        policy.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        // Fast path: precomputed log-probs
        if (batch.containsKey("policy_chosen_logps")
                && batch.get("policy_chosen_logps") != null
                && batch.get("policy_chosen_logps").defined()) {
            return computeFromLogps(batch);
        }

        // Compute log-probs from input
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");

        // Forward pass
        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        // Get labels
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

        // Compute sequence log-probs
        Tensor chosenLp = computeLogProbs(chosenLogits, chosenLabels, chosenMask);
        Tensor rejectedLp = computeLogProbs(rejectedLogits, rejectedLabels, rejectedMask);

        // Store for loss computation
        batch.put("policy_chosen_logps", chosenLp);
        batch.put("policy_rejected_logps", rejectedLp);

        return computeFromLogps(batch);
    }

    /**
     * Compute SimPO loss from precomputed log-probs.
     */
    private Tensor computeFromLogps(Map<String, Tensor> batch) {
        Tensor chosenLogps = require(batch, "policy_chosen_logps");
        Tensor rejectedLogps = require(batch, "policy_rejected_logps");

        // Track reward margin for monitoring
        Tensor rewardMargin = chosenLogps.sub(rejectedLogps);
        totalRewardMargin += rewardMargin.mean().item_double();
        rewardMarginCount++;

        return SimPOLoss.compute(
                chosenLogps,
                rejectedLogps,
                simpoConfig.beta(),
                simpoConfig.targetMargin(),
                simpoConfig.lengthNormalize());
    }

    /**
     * Compute sequence log-probs with optional length normalization.
     */
    private Tensor computeLogProbs(Tensor logits, Tensor labels, Tensor mask) {
        if (simpoConfig.lengthNormalize()) {
            return LogProbUtils.sequenceMeanLogProbs(logits, labels, mask);
        } else {
            return LogProbUtils.sequenceLogProbs(logits, labels, mask);
        }
    }

    // ==================== Utility Methods ====================

    private static Tensor orElse(Tensor a, Tensor b) {
        return a != null && a.defined() ? a : b;
    }

    private static Tensor require(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalArgumentException("batch missing required key: " + key);
        }
        return t;
    }

    // ==================== Getters ====================

    public Module policy() { return policy; }
    public SimPOConfig config() { return simpoConfig; }

    /**
     * Get algorithm identifier.
     */
    public String algorithm() {
        return ALGORITHM_ID;
    }

    public String algorithmName() {
        return "SimPO (Simple Preference Optimization)";
    }

    // ==================== Metrics ====================

    public double getAverageRewardMargin() {
        return rewardMarginCount > 0 ? totalRewardMargin / rewardMarginCount : 0.0;
    }

    public void resetMetrics() {
        totalRewardMargin = 0.0;
        rewardMarginCount = 0.0;
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        super.close();
        System.out.printf(
                "[SimPOTrainer v%s] Closed: avgRewardMargin=%.4f%n",
                VERSION, getAverageRewardMargin());
    }

    public boolean isClosed() { return closed; }
}
