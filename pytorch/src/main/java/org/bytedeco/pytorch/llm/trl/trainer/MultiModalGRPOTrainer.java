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
package org.bytedeco.pytorch.llm.trl.trainer;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.config.MultiModalGRPOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.where;

/**
 * MultiModal GRPO trainer (ByteDance inspired).
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Group-relative advantage normalization with whitening</li>
 *   <li>Reward scaling / clipping / per-modality weights</li>
 *   <li>GRPO / DAPO loss type toggle</li>
 *   <li>Two-sided epsilon clipping</li>
 *   <li>Router auxiliary loss integration</li>
 *   <li>NaN/Inf guard</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class MultiModalGRPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "multimodal_grpo";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final MultiModalGRPOConfig config;
    private final TensorVector params;

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
        if (reference != null) freeze(reference);
        System.out.printf(
                "[MultiModalGRPOTrainer v%s] beta=%.3f, eps=%.3f, epsH=%.3f, lossType=%s, scale=%s, whiten=%s%n",
                VERSION, config.beta(), config.epsilon(), config.epsilonHigh(),
                config.lossType(), config.scaleRewards(), config.whitenAdvantages());
    }

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
            if (std > 1e-6) rewards = rewards.div(new Scalar(std + 1e-8));
        }
        if (config.rewardClip() > 0.0) {
            double c = config.rewardClip();
            rewards = rewards.clamp(new ScalarOptional(new Scalar(-c)), new ScalarOptional(new Scalar(c)));
        }

        Tensor modalityRewards = batch.get("modality_rewards");

        Tensor logProbs = computeLogProbs(batch);
        Tensor refLogProbs = computeRefLogProbs(batch);
        Tensor advantages = computeGroupRelativeAdvantage(rewards, modalityRewards);

        if (config.advantageClip() > 0.0) {
            double c = config.advantageClip();
            advantages = advantages.clamp(new ScalarOptional(new Scalar(-c)), new ScalarOptional(new Scalar(c)));
        }
        if (config.whitenAdvantages()) {
            Tensor mean = advantages.mean();
            Tensor std = advantages.std();
            advantages = advantages.sub(mean).div(std.add(new Scalar(1e-8)));
        }

        Tensor policyLoss = computePolicyLoss(logProbs, refLogProbs, advantages);
        Tensor entropyLoss = computeEntropyLoss(logProbs);

        Tensor totalLoss = policyLoss.sub(
                entropyLoss.mul(new Scalar(config.entropyCoeff())).mean()
        );

        // Router auxiliary loss (MoE)
        if (config.routerAuxLossCoef() > 0.0) {
            Tensor aux = batch.get("aux_loss");
            if (aux != null && aux.defined()) {
                totalLoss = totalLoss.add(aux.mul(new Scalar(config.routerAuxLossCoef())));
            }
        }

        // Mask truncated completions: zero-out the contribution
        if (config.maskTruncatedCompletions() >= 1.0) {
            Tensor truncMask = batch.get("truncation_mask");
            if (truncMask != null && truncMask.defined()) {
                Tensor masked = totalLoss.mul(truncMask.mean());
                totalLoss = masked;
            }
        }

        double v = totalLoss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[MultiModalGRPOTrainer] WARNING: NaN/Inf loss; fallback to policy term.");
            return policyLoss;
        }

        updateStatistics(rewards, logProbs, refLogProbs);
        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor computeLogProbs(Map<String, Tensor> batch) {
        if (hasKey(batch, "log_probs")) {
            return batch.get("log_probs");
        }
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor logits = policyForward.forward(inputIds, attentionMask);
        int seqLen = (int) logits.size(1);
        return logits.log_softmax(-1).select(1, seqLen - 1);
    }

    private Tensor computeRefLogProbs(Map<String, Tensor> batch) {
        if (reference == null || referenceForward == null) {
            Tensor rewards = batch.get("rewards");
            if (rewards != null && rewards.defined()) {
                return org.bytedeco.pytorch.global.torch.zeros_like(rewards);
            }
            return org.bytedeco.pytorch.global.torch.zeros(new long[]{1});
        }
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        try (NoGradGuard guard = new NoGradGuard()) {
            Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
            int seqLen = (int) refLogits.size(1);
            return refLogits.log_softmax(-1).select(1, seqLen - 1).detach();
        }
    }

    private Tensor computeGroupRelativeAdvantage(Tensor rewards, Tensor modalityRewards) {
        Tensor advantages = rewards.clone();

        if (modalityRewards != null && modalityRewards.defined() && config.crossModalReward()) {
            advantages = computeCrossModalAdvantage(rewards, modalityRewards);
        }

        double normScale = config.rewardNormScale();
        if (normScale > 0) {
            Tensor mean = advantages.mean();
            Tensor std = advantages.std();
            advantages = advantages.sub(mean).div(std.add(new Scalar(normScale)));
        }
        return advantages;
    }

    private Tensor computeCrossModalAdvantage(Tensor rewards, Tensor modalityRewards) {
        double wText = config.textWeight();
        double wImage = config.imageWeight();
        double wAudio = config.audioWeight();

        Tensor combined = rewards.clone();
        int numModalities = (int) modalityRewards.size(1);
        if (numModalities >= 1) {
            Tensor text = modalityRewards.select(1, 0);
            combined = combined.mul(new Scalar(wText)).add(text.mul(new Scalar(1 - wText)));
        }
        if (numModalities >= 2) {
            Tensor image = modalityRewards.select(1, 1);
            combined = combined.mul(new Scalar(1 - wImage)).add(image.mul(new Scalar(wImage)));
        }
        if (numModalities >= 3) {
            Tensor audio = modalityRewards.select(1, 2);
            combined = combined.mul(new Scalar(1 - wAudio)).add(audio.mul(new Scalar(wAudio)));
        }
        return combined;
    }

    private Tensor computePolicyLoss(Tensor logProbs, Tensor refLogProbs, Tensor advantages) {
        Tensor ratio = logProbs.sub(refLogProbs).exp();
        String type = config.lossType() == null ? "grpo" : config.lossType().toLowerCase();
        double eps = config.epsilon();
        double epsH = config.epsilonHigh() > 0 ? config.epsilonHigh() : eps;

        if ("dapo".equals(type)) {
            // Two-sided clipping without ratio clipping (DAPO)
            Tensor clipped = advantages.clamp(new ScalarOptional(new Scalar(-epsH)),
                    new ScalarOptional(new Scalar(eps)));
            Tensor pos = ratio.sub(new Scalar(1.0)).sub(advantageAsBonus(advantages))
                    .mul(clipped.gt(new Scalar(0.0)).to(org.bytedeco.pytorch.global.torch.ScalarType.Float));
            Tensor neg = ratio.sub(new Scalar(1.0)).mul(clipped.lt(new Scalar(0.0))
                    .to(org.bytedeco.pytorch.global.torch.ScalarType.Float));
            Tensor loss = where(advantages.gt(new Scalar(0.0)), pos, neg).neg().mean();
            return loss;
        }

        // Default: GRPO (ratio clipped)
        Tensor ratioClipped = ratio.clamp(
                new ScalarOptional(new Scalar(1 - eps)),
                new ScalarOptional(new Scalar(1 + eps)));
        Tensor surr1 = ratio.mul(advantages);
        Tensor surr2 = ratioClipped.mul(advantages);
        Tensor clippedLoss = where(surr1.lt(surr2), surr1, surr2);
        return clippedLoss.neg().mean();
    }

    private static Tensor advantageAsBonus(Tensor advantages) {
        return advantages.mul(new Scalar(0.0)); // placeholder for extension
    }

    private Tensor computeEntropyLoss(Tensor logProbs) {
        Tensor probs = logProbs.exp();
        return logProbs.mul(probs).neg();
    }

    private void updateStatistics(Tensor rewards, Tensor logProbs, Tensor refLogProbs) {
        samplesProcessed += rewards.size(0);
        double newReward = rewards.mean().item_double();
        avgReward = 0.9 * avgReward + 0.1 * newReward;
        if (refLogProbs != null && refLogProbs.size(0) > 0) {
            Tensor kl = logProbs.sub(refLogProbs);
            double newKL = kl.abs().mean().item_double();
            avgKL = 0.9 * avgKL + 0.1 * newKL;
        }
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
                "[MultiModalGRPOTrainer v%s] Closed: steps=%d, samples=%d, avgReward=%.4f, avgKL=%.4f%n",
                VERSION, numTrainingSteps, samplesProcessed, avgReward, avgKL);
    }

    public boolean isClosed() { return closed; }
    public double getAvgReward() { return avgReward; }
    public double getAvgKL() { return avgKL; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "MultiModal GRPO v" + VERSION;
    }
}