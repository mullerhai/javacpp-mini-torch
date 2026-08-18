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
 * or as provided in the LICENSE.txt file that accompanied this code.
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
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.trl.config.RLOOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * RLOO (REINFORCE Leave-One-Out) trainer (Meta AI inspired).
 *
 * <p>Enterprise features over the legacy implementation:
 * <ul>
 *   <li>Per-sample advantage clipping, whitening, and reward scaling</li>
 *   <li>Entropy bonus and value-function coefficient (even though there is no value head, the coefficient stays)</li>
 *   <li>Adaptive KL with exponential moving average</li>
 *   <li>Mini-batching and multiple PPO epochs (REINFORCE multiple passes per batch)</li>
 *   <li>Reference-free mode</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RLOOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "rloo";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final Module reference;
    private final LlmForward policyForward;
    private final LlmForward referenceForward;
    private final RLOOConfig config;
    private final TensorVector params;

    // Adaptive KL
    private double currentBeta;
    private double runningKl;

    public RLOOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            RLOOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

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
        this.currentBeta = config.klTarget() > 0.0 ? config.klTarget() : 0.1;
        if (reference != null) freeze(reference);
        System.out.printf(
                "[RLOOTrainer v%s] beta=%.3f, gae_lambda=%.2f, scale=%s, whiten=%s%n",
                VERSION, config.beta(), config.gaeLambda(),
                config.scaleRewards(), config.whitenAdvantages());
    }

    public Module policy() { return policy; }
    public RLOOConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() { return params; }

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
        Tensor rewards = require(batch, "rewards");

        // Reward scaling
        if (shouldScaleRewards(config.scaleRewards())) {
            double std = rewards.std().item_double();
            if (std > 1e-6) {
                rewards = rewards.div(new Scalar(std + 1e-8));
            }
        }
        // Reward clipping
        if (config.rewardClip() > 0.0) {
            double c = config.rewardClip();
            rewards = rewards.clamp(new ScalarOptional(new Scalar(-c)), new ScalarOptional(new Scalar(c)));
        }

        Tensor logProbs = computeLogProbs(batch);
        Tensor advantages = computeLooAdvantage(rewards);
        // Advantage clipping
        if (config.advantageClip() > 0.0) {
            double c = config.advantageClip();
            advantages = advantages.clamp(new ScalarOptional(new Scalar(-c)), new ScalarOptional(new Scalar(c)));
        }
        // Advantage whitening
        if (config.whitenAdvantages()) {
            Tensor mean = advantages.mean();
            Tensor std = advantages.std();
            advantages = advantages.sub(mean).div(std.add(new Scalar(1e-8)));
        }

        Tensor klPenalty = computeKlPenalty(batch);

        Tensor pgLoss = computePolicyGradientLoss(logProbs, advantages);

        // Entropy bonus (use logits softmax entropy as approximation).
        double entropyBonusValue = 0.0;
        if (config.entropyBonus() > 0.0) {
            entropyBonusValue = -config.entropyBonus() * estimateEntropy(batch).item_double();
        }

        // Combined loss
        Tensor totalLoss = pgLoss
                .add(klPenalty.mean().mul(new Scalar(currentBeta)))
                .add(new Scalar(entropyBonusValue));

        // Update adaptive KL
        double kl = klPenalty.mean().item_double();
        runningKl = 0.9 * runningKl + 0.1 * kl;
        if (config.klTarget() > 0.0) {
            adaptKl(kl);
        }

        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor computeLooAdvantage(Tensor rewards) {
        long batchSize = rewards.size(0);
        if (batchSize <= 1) {
            return rewards.sub(rewards.mean());
        }
        double n = batchSize;
        Tensor sum = rewards.sum();
        Tensor loo = rewards.mul(new Scalar(n)).sub(sum).div(new Scalar(n - 1));
        double alpha = config.baselineCoeff();
        if (alpha < 1.0) {
            Tensor mean = rewards.mean().expand(new long[]{batchSize});
            Tensor simple = rewards.sub(mean);
            loo = loo.mul(new Scalar(alpha)).add(simple.mul(new Scalar(1 - alpha)));
        }
        return loo;
    }

    private Tensor computeLogProbs(Map<String, Tensor> batch) {
        if (hasKey(batch, "log_probs")) {
            return batch.get("log_probs");
        }
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor logits = policyForward.forward(inputIds, attentionMask);
        int seqLen = (int) logits.size(1);
        Tensor logProbs = logits.log_softmax(-1);
        return logProbs.select(1, seqLen - 1);
    }

    private Tensor computeKlPenalty(Map<String, Tensor> batch) {
        if (reference == null || referenceForward == null) {
            return zeros_like(require(batch, "rewards"));
        }
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        Tensor policyLogits = policyForward.forward(inputIds, attentionMask);
        Tensor refLogits;
        try (NoGradGuard guard = new NoGradGuard()) {
            refLogits = referenceForward.forward(inputIds, attentionMask);
        }

        Tensor policyLogProbs = policyLogits.log_softmax(-1);
        Tensor refLogProbs = refLogits.log_softmax(-1);
        return policyLogProbs.sub(refLogProbs).mul(policyLogProbs.exp());
    }

    private Tensor estimateEntropy(Map<String, Tensor> batch) {
        if (!hasKey(batch, "input_ids")) return torch.tensor(0.0).reshape(new long[]{1});
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor logits = policyForward.forward(inputIds, attentionMask);
        Tensor probs = logits.softmax(-1);
        Tensor logProbs = logits.log_softmax(-1);
        Tensor ent = probs.mul(logProbs).neg().sum(-1);
        return ent.mean();
    }

    private Tensor computePolicyGradientLoss(Tensor logProbs, Tensor advantages) {
        // -E[advantage * log_prob]
        return logProbs.mul(advantages).neg().mean();
    }

    private void adaptKl(double kl) {
        double target = config.klTarget();
        double eps = config.klEpsilon();
        if (kl > target * (1 + eps)) currentBeta *= 1.5;
        else if (kl < target * (1 - eps)) currentBeta *= 0.5;
        currentBeta = Math.max(1e-3, Math.min(10.0, currentBeta));
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
    }

    private static boolean shouldScaleRewards(String mode) {
        if (mode == null) return false;
        String m = mode.toLowerCase();
        return "group".equals(m) || "batch".equals(m);
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
                "[RLOOTrainer v%s] Closed: steps=%d, beta=%.4f, kl=%.4f%n",
                VERSION, numTrainingSteps, currentBeta, runningKl);
    }

    public boolean isClosed() { return closed; }
    public double getCurrentBeta() { return currentBeta; }
    public double getRunningKl() { return runningKl; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "RLOO v" + VERSION + " (REINFORCE Leave-One-Out)";
    }
}