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
import org.bytedeco.pytorch.llm.trl.config.RRHFConfig;
import org.bytedeco.pytorch.llm.trl.loss.RRHFLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.TensorVector;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * RRHF (Rank Responses to Rank Responses) trainer.
 *
 * <p>RRHF aligns language models using a ranking loss on generated responses.
 * Unlike DPO, it doesn't require preference pairs - instead, it uses a reward
 * model to score multiple generated responses and learns to match the ranking.
 *
 * <p>Key features:
 * <ul>
 *   <li>Only requires a reward model (no reference model needed)</li>
 *   <li>Can use any number of responses per prompt</li>
 *   <li>Compatible with any reward model scoring function</li>
 *   <li>Simpler data requirements than pairwise methods</li>
 * </ul>
 *
 * <p>Reference: "RRHF: Rank Responses to Rank Responses for Human Preference"
 * (Yuan et al., 2023)
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code input_ids} - prompt tokens {@code [B * num_responses, T]}</li>
 *   <li>{@code rewards} - scalar reward for each response {@code [B * num_responses]}</li>
 *   <li>optional {@code attention_mask}, {@code labels}</li>
 *   <li>or precomputed {@code log_probs} for policy responses</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RRHFTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";
    public static final String ALGORITHM_ID = "rrhf";

    private volatile boolean closed;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module rewardModel;
    private final Module reference;  // optional
    private final LlmForward referenceForward;
    private final RRHFConfig rrhfConfig;
    private final TensorVector params;

    // Metrics
    private double totalRewardCorrelation;
    private int correlationCount;

    /**
     * Create RRHF trainer with reward model.
     *
     * @param policy Policy model to train
     * @param policyForward Forward function for policy
     * @param rewardModel Reward model for scoring responses
     * @param optimizer Optimizer
     * @param config RRHF configuration
     */
    public RRHFTrainer(
            Module policy,
            LlmForward policyForward,
            Module rewardModel,
            Optimizer optimizer,
            RRHFConfig config) {
        this(policy, policyForward, rewardModel, null, null, optimizer, config);
    }

    /**
     * Create RRHF trainer with reward model and reference model.
     */
    public RRHFTrainer(
            Module policy,
            LlmForward policyForward,
            Module rewardModel,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            RRHFConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.rewardModel = Objects.requireNonNull(rewardModel, "rewardModel");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.rrhfConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();

        if (reference != null) {
            freeze(reference);
        }

        System.out.printf(
                "[RRHFTrainer v%s] numResponses=%d, rewardTemp=%.2f, pairwise=%s%n",
                VERSION, rrhfConfig.numResponses(), rrhfConfig.rewardTemperature(),
                rrhfConfig.pairwiseLoss());
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
        if (reference != null) reference.eval();
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (reference != null) reference.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        // Get responses and rewards
        Tensor rewards = require(batch, "rewards");
        int numResponses = rrhfConfig.numResponses();

        // Get log-probs
        Tensor logProbs;
        Tensor refLogProbs = null;

        if (batch.containsKey("log_probs") && batch.get("log_probs") != null
                && batch.get("log_probs").defined()) {
            // Use precomputed log-probs
            logProbs = batch.get("log_probs");
        } else {
            // Compute log-probs from forward pass
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);

            // Policy forward
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            logProbs = LogProbUtils.sequenceLogProbs(logits, labels, attentionMask);
        }

        // Compute reference log-probs if available
        if (referenceForward != null && rrhfConfig.ratioWeight() > 0) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor inputIds = require(batch, "input_ids");
                Tensor attentionMask = batch.get("attention_mask");
                Tensor labels = orElse(batch.get("labels"), inputIds);

                Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
                refLogProbs = LogProbUtils.sequenceLogProbs(refLogits, labels, attentionMask);
            }
        }

        // Compute RRHF loss
        Tensor loss;
        if (rrhfConfig.pairwiseLoss()) {
            loss = RRHFLoss.computePairwise(
                    logProbs, rewards, numResponses,
                    rrhfConfig.rewardWeight(), rrhfConfig.ratioWeight());
        } else {
            loss = RRHFLoss.computeSampleLevel(
                    logProbs, rewards, numResponses,
                    rrhfConfig.rewardTemperature(),
                    rrhfConfig.rewardWeight(), rrhfConfig.ratioWeight());
        }

        // Track reward correlation
        updateRewardCorrelation(logProbs, rewards);

        return loss;
    }

    private void updateRewardCorrelation(Tensor logProbs, Tensor rewards) {
        // Simple correlation tracking (placeholder for actual Spearman correlation)
        // In practice, you would compute Spearman/Pearson correlation
        correlationCount++;
    }

    // ==================== Utility Methods ====================

    private static void freeze(Module m) {
        TensorVector pv = m.parameters();
        for (long i = 0, n = pv.size(); i < n; i++) {
            Tensor p = pv.get(i);
            if (p != null && !p.isNull() && p.defined()) {
                p.requires_grad_(false);
            }
        }
        m.eval();
    }

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
    public Module rewardModel() { return rewardModel; }
    public Module reference() { return reference; }
    public RRHFConfig config() { return rrhfConfig; }

    /**
     * Get algorithm identifier.
     */
    public String algorithm() {
        return ALGORITHM_ID;
    }

    public String algorithmName() {
        return "RRHF (Rank Responses to Rank Responses)";
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        super.close();
        System.out.printf("[RRHFTrainer v%s] Closed%n", VERSION);
    }

    public boolean isClosed() { return closed; }
}
