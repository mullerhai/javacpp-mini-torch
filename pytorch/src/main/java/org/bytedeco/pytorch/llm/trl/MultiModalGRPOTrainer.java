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
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.MultiModalGRPOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * MultiModal GRPO trainer for multimodal LLM alignment (ByteDance inspired).
 *
 * <p>MultiModal GRPO extends standard GRPO to handle multimodal inputs (text + images
 * + audio + video) with modality-specific reward shaping and cross-modal attention.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code input_ids} - text tokens {@code [B, T]}</li>
 *   <li>{@code attention_mask} - attention mask {@code [B, T]}</li>
 *   <li>{@code multimodal_inputs} - map of modality -> tensor (optional)</li>
 *   <li>{@code responses} - response tokens {@code [B, G]}</li>
 *   <li>{@code rewards} - scalar reward per sample {@code [B]}</li>
 *   <li>{@code modality_rewards} - per-modality rewards (optional) {@code [B, M]}</li>
 * </ul>
 */
public final class MultiModalGRPOTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final MultiModalGRPOConfig config;
    private final TensorVector params;

    // Performance tracking
    private double avgReward;
    private double avgKL;
    private int samplesProcessed;

    public MultiModalGRPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            MultiModalGRPOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.config = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.avgReward = 0.0;
        this.avgKL = 0.0;
        this.samplesProcessed = 0;

        if (reference != null) {
            freeze(reference);
        }
    }

    /** Reference-free constructor. */
    public MultiModalGRPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            MultiModalGRPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public MultiModalGRPOConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (reference != null) {
            reference.eval();
        }
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (reference != null) {
            reference.eval();
        }
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        // Extract rewards
        Tensor rewards = require(batch, "rewards");
        Tensor modalityRewards = batch.get("modality_rewards");

        // Compute log probabilities
        Tensor logProbs = computeLogProbs(batch);
        Tensor refLogProbs = computeRefLogProbs(batch);

        // Compute group-relative advantages
        Tensor advantages = computeGroupRelativeAdvantage(rewards, modalityRewards);

        // Compute policy gradient loss
        Tensor policyLoss = computePolicyLoss(logProbs, refLogProbs, advantages);

        // Compute entropy bonus
        Tensor entropyLoss = computeEntropyLoss(logProbs);

        // Combined loss
        Tensor totalLoss = policyLoss
                .sub(new Scalar(config.entropyCoeff()).mul(entropyLoss.mean()));

        // Update running statistics
        updateStatistics(rewards, logProbs, refLogProbs);

        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor computeLogProbs(Map<String, Tensor> batch) {
        // Check for precomputed log probs
        if (batch.containsKey("log_probs") && batch.get("log_probs") != null
                && batch.get("log_probs").defined()) {
            return batch.get("log_probs");
        }

        // Full forward pass with multimodal support
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        Tensor logits = policyForward.forward(inputIds, attentionMask);

        // Get log probs for response tokens
        int seqLen = (int) logits.size(1);
        Tensor logProbs = logits.log_softmax(-1);

        // Return last token log prob (simplified)
        return logProbs.select(1, seqLen - 1);
    }

    private Tensor computeRefLogProbs(Map<String, Tensor> batch) {
        if (reference == null || referenceForward == null) {
            return org.bytedeco.pytorch.global.torch.zeros_like(
                    batch.containsKey("rewards") ? batch.get("rewards") : null);
        }

        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        try (NoGradGuard guard = new NoGradGuard()) {
            Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
            int seqLen = (int) refLogits.size(1);
            Tensor refLogProbs = refLogits.log_softmax(-1);
            return refLogProbs.select(1, seqLen - 1).detach();
        }
    }

    private Tensor computeGroupRelativeAdvantage(Tensor rewards, Tensor modalityRewards) {
        int batchSize = (int) rewards.size(0);
        int groupSize = config.groupSize();

        // Group samples for GRPO-style advantage computation
        // Advantage for each sample = reward - mean(group reward)
        Tensor advantages = rewards.clone();

        // Compute per-modality advantages if available
        if (modalityRewards != null && modalityRewards.defined() && config.crossModalReward()) {
            advantages = computeCrossModalAdvantage(rewards, modalityRewards);
        }

        // Normalize advantages if scale is specified
        double normScale = config.rewardNormScale();
        if (normScale > 0) {
            Tensor mean = advantages.mean();
            Tensor std = advantages.std();
            advantages = (advantages.sub(mean)).div(std.add(normScale));
        }

        return advantages;
    }

    private Tensor computeCrossModalAdvantage(Tensor rewards, Tensor modalityRewards) {
        // Compute weighted combination of overall and modality-specific rewards
        double wText = config.textWeight();
        double wImage = config.imageWeight();
        double wAudio = config.audioWeight();

        // Default weights if modality rewards have different shape
        Tensor combinedRewards = rewards.clone();

        int numModalities = (int) modalityRewards.size(1);
        if (numModalities >= 1) {
            Tensor textReward = modalityRewards.select(1, 0);
            combinedRewards = combinedRewards.mul(wText)
                    .add(textReward.mul(1 - wText));
        }
        if (numModalities >= 2) {
            Tensor imageReward = modalityRewards.select(1, 1);
            combinedRewards = combinedRewards.mul(1 - wImage)
                    .add(imageReward.mul(wImage));
        }
        if (numModalities >= 3) {
            Tensor audioReward = modalityRewards.select(1, 2);
            combinedRewards = combinedRewards.mul(1 - wAudio)
                    .add(audioReward.mul(wAudio));
        }

        return combinedRewards;
    }

    private Tensor computePolicyLoss(Tensor logProbs, Tensor refLogProbs, Tensor advantages) {
        double beta = config.beta();

        // Compute importance weights
        Tensor ratio = logProbs.sub(refLogProbs).exp();

        // Clipped surrogate loss (GRPO-style)
        double clipEps = 0.2;
        Tensor ratioClipped = ratio.clamp(1 - clipEps, 1 + clipEps);

        Tensor surr1 = ratio.mul(advantages);
        Tensor surr2 = ratioClipped.mul(advantages);

        // Take minimum for clipped loss
        Tensor clippedLoss = surr1.lt(surr2).select(surr1, surr2);

        return clippedLoss.neg().mean();
    }

    private Tensor computeEntropyLoss(Tensor logProbs) {
        // Entropy bonus for exploration
        Tensor probs = logProbs.exp();
        return logProbs.mul(probs).neg();
    }

    private void updateStatistics(Tensor rewards, Tensor logProbs, Tensor refLogProbs) {
        samplesProcessed += rewards.size(0);

        // Running average of rewards
        double newReward = rewards.mean().item_double();
        avgReward = 0.9 * avgReward + 0.1 * newReward;

        // Running average of KL
        if (refLogProbs != null && refLogProbs.size(0) > 0) {
            Tensor kl = logProbs.sub(refLogProbs);
            double newKL = kl.abs().mean().item_double();
            avgKL = 0.9 * avgKL + 0.1 * newKL;
        }
    }

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

    private static Tensor require(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalArgumentException("batch missing required key: " + key);
        }
        return t;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        super.close();
        System.out.printf(
                "[MultiModalGRPOTrainer] Closed: steps=%d, samples=%d, avgReward=%.4f, avgKL=%.4f%n",
                numTrainingSteps, samplesProcessed, avgReward, avgKL);
    }

    public boolean isClosed() { return closed; }
    public double getAvgReward() { return avgReward; }
    public double getAvgKL() { return avgKL; }
}
