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
import org.bytedeco.pytorch.llm.trl.config.RLOOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.mean;

/**
 * RLOO (REINFORCE Leave-One-Out) trainer for LLM alignment (Meta AI inspired).
 *
 * <p>RLOO is a simplified policy gradient algorithm that estimates advantage by
 * comparing each sample's reward to the mean of other samples in the same batch.
 * Unlike PPO, RLOO doesn't require a separate value function.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code input_ids} - prompt tokens {@code [B, T]}</li>
 *   <li>{@code responses} - response tokens {@code [B, G]} (or use generation)</li>
 *   <li>{@code rewards} - scalar reward for each sample {@code [B]}</li>
 *   <li>{@code log_probs} - log probabilities of responses (optional)</li>
 * </ul>
 *
 * <p>Reference: "A Minimalist Approach to LLM Reinforcement Learning" (Meta AI, 2024)
 */
public final class RLOOTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final Module reference;
    private final LlmForward policyForward;
    private final LlmForward referenceForward;
    private final RLOOConfig config;
    private final TensorVector params;

    // Adaptive KL state
    private double currentBeta;
    private double runningKl;

    public RLOOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            RLOOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.config = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.currentBeta = config.beta();
        this.runningKl = 0.0;

        if (reference != null) {
            freeze(reference);
        }
    }

    /** Policy-only constructor (reference-free). */
    public RLOOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, RLOOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public RLOOConfig config() { return config; }

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

        // Compute log probabilities
        Tensor logProbs = computeLogProbs(batch);

        // Compute LOO advantage: each sample's reward minus mean of others
        Tensor advantages = computeLooAdvantage(rewards);

        // Compute KL penalty
        Tensor klPenalty = computeKlPenalty(batch);

        // Policy gradient loss with KL regularization
        Tensor pgLoss = computePolicyGradientLoss(logProbs, advantages);

        // Combined loss
        Tensor totalLoss = pgLoss.add(new Scalar(currentBeta).mul(klPenalty.mean()));

        // Update adaptive KL
        if (config.useAdaptiveKL()) {
            updateAdaptiveKL(klPenalty.mean().item_double());
        }

        numTrainingSteps++;
        return totalLoss;
    }

    /**
     * Compute LOO (Leave-One-Out) advantage.
     *
     * For each sample i: advantage_i = reward_i - mean(reward_j for j != i)
     *
     * This is a baseline that doesn't require a learned value function.
     */
    private Tensor computeLooAdvantage(Tensor rewards) {
        int batchSize = (int) rewards.size(0);

        // Compute mean reward across batch
        Tensor meanReward = rewards.mean();

        // Expand mean to match batch size
        Tensor expandedMean = meanReward.expand(new long[]{batchSize});

        // LOO advantage: reward_i - mean(other rewards)
        // = reward_i - (sum - reward_i) / (n - 1)
        // = (n * reward_i - sum) / (n - 1)
        double n = batchSize;
        Tensor sumRewards = rewards.sum();

        Tensor looAdvantage = rewards.mul(n).sub(sumRewards).div(n - 1);

        // Optionally apply baseline coefficient
        double alpha = config.baselineCoeff();
        if (alpha < 1.0) {
            // Interpolate between LOO and simple reward baseline
            Tensor simpleAdvantage = rewards.sub(expandedMean);
            looAdvantage = looAdvantage.mul(alpha).add(simpleAdvantage.mul(1 - alpha));
        }

        return looAdvantage;
    }

    private Tensor computeLogProbs(Map<String, Tensor> batch) {
        // Check for precomputed log probs
        if (batch.containsKey("log_probs") && batch.get("log_probs") != null
                && batch.get("log_probs").defined()) {
            return batch.get("log_probs");
        }

        // Full forward pass
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        Tensor logits = policyForward.forward(inputIds, attentionMask);

        // Return per-token log probs for policy gradient
        // This is a simplified version; full implementation would need sequence-level
        int seqLen = (int) logits.size(1);
        Tensor logProbs = logits.log_softmax(-1);

        // For simplicity, return last token log prob per sample
        Tensor lastLogProbs = logProbs.select(1, seqLen - 1);

        return lastLogProbs;
    }

    private Tensor computeKlPenalty(Map<String, Tensor> batch) {
        if (reference == null || referenceForward == null) {
            return org.bytedeco.pytorch.global.torch.zeros_like(
                    batch.containsKey("rewards") ? batch.get("rewards") : null);
        }

        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        // Policy logits
        Tensor policyLogits = policyForward.forward(inputIds, attentionMask);

        // Reference logits (no grad)
        Tensor refLogits;
        try (NoGradGuard guard = new NoGradGuard()) {
            refLogits = referenceForward.forward(inputIds, attentionMask);
        }

        // KL divergence between policy and reference
        Tensor policyLogProbs = policyLogits.log_softmax(-1);
        Tensor refLogProbs = refLogits.log_softmax(-1);

        // KL = sum(p * (log(p) - log(q)))
        Tensor klDiv = policyLogProbs.sub(refLogProbs).mul(policyLogProbs.exp());

        return klDiv.mean();
    }

    private Tensor computePolicyGradientLoss(Tensor logProbs, Tensor advantages) {
        // Policy gradient: -E[advantage * log_prob]
        // For LOO, advantage is already computed per sample
        return logProbs.mul(advantages).neg().mean();
    }

    private void updateAdaptiveKL(double kl) {
        runningKl = 0.9 * runningKl + 0.1 * kl;

        double target = config.klTarget();
        double eps = config.klEpsilon();

        if (runningKl > target * (1 + eps)) {
            // KL too high, increase penalty
            currentBeta *= 1.5;
        } else if (runningKl < target * (1 - eps)) {
            // KL too low, decrease penalty
            currentBeta *= 0.5;
        }

        // Clamp beta to reasonable range
        currentBeta = Math.max(0.001, Math.min(10.0, currentBeta));
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
                "[RLOOTrainer] Closed: trainingSteps=%d, finalBeta=%.4f, finalKl=%.4f%n",
                numTrainingSteps, currentBeta, runningKl);
    }

    public boolean isClosed() { return closed; }

    public double getCurrentBeta() { return currentBeta; }
    public double getRunningKl() { return runningKl; }
}
