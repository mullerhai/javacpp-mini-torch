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
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.LogProbUtils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.trl.config.GRPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.clamp;

/**
 * Group Relative Policy Optimization trainer (DeepSeek-R1 / HF TRL GRPO).
 *
 * <p>Enterprise-grade features:
 * <ul>
 *   <li>{@code loss_type}: {@code grpo} (clipped surrogate) or {@code dapo} (no clipping)</li>
 *   <li>Two-sided clipping with {@code epsilon} / {@code epsilon_high} (DAPO)</li>
 *   <li>Reward scaling ({@code none} / {@code group} / {@code batch}) and clipping</li>
 *   <li>Advantage whitening</li>
 *   <li>Truncated-completion masking</li>
 *   <li>MoE router aux loss</li>
 *   <li>vLLM server integration hook</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class GRPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final GRPOConfig grpoConfig;
    private final TensorVector params;
    private final boolean useClipping;

    public GRPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            GRPOConfig config) {
        this(policy, policyForward, reference, referenceForward, optimizer, config, true);
    }

    public GRPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            GRPOConfig config,
            boolean useClipping) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.grpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.useClipping = useClipping && config.useClipping();
        if (reference != null) {
            freeze(reference);
        }
    }

    public GRPOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, GRPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public GRPOConfig grpoConfig() { return grpoConfig; }

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
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor rewards = require(batch, "rewards");
        int groupSize = grpoConfig.numGenerations();

        // ----- Compute new log-probs -----
        Tensor newLogprobs;
        if (hasKey(batch, "logprobs")) {
            newLogprobs = batch.get("logprobs");
        } else {
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            Tensor completionMask = orElse(batch.get("completion_mask"), attentionMask);
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            newLogprobs = LogProbUtils.sequenceLogProbs(logits, labels, completionMask);
        }

        Tensor oldLogprobs = batch.get("old_logprobs");
        Tensor refLogprobs = batch.get("ref_logprobs");

        // ----- Reference log-probs (if needed) -----
        if ((refLogprobs == null || !refLogprobs.defined())
                && grpoConfig.beta() > 0.0
                && referenceForward != null
                && batch.containsKey("input_ids")) {
            Tensor inputIds = batch.get("input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            Tensor completionMask = orElse(batch.get("completion_mask"), attentionMask);
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
                refLogprobs = LogProbUtils.sequenceLogProbs(refLogits, labels, completionMask).detach();
            }
        }

        // ----- Reward scaling -----
        if (!"none".equalsIgnoreCase(grpoConfig.scaleRewards())) {
            rewards = scaleRewards(rewards, grpoConfig.scaleRewards(), groupSize);
        }

        // ----- Reward clipping -----
        if (grpoConfig.rewardClip() > 0.0) {
            double clip = grpoConfig.rewardClip();
            rewards = clamp(rewards,
                    new org.bytedeco.pytorch.ScalarOptional(new Scalar(-clip)),
                    new org.bytedeco.pytorch.ScalarOptional(new Scalar(clip)));
        }

        // ----- Mask truncated completions -----
        if (grpoConfig.maskTruncatedCompletions() && hasKey(batch, "truncation_mask")) {
            Tensor truncMask = batch.get("truncation_mask");
            if (truncMask != null && truncMask.defined()) {
                rewards = rewards.mul(truncMask);
            }
        }

        // ----- Group-normalize advantages -----
        Tensor advantages = GRPOLoss.groupNormalize(rewards, groupSize);

        // ----- Advantage whitening -----
        if (grpoConfig.whitenAdvantages()) {
            advantages = whiten(advantages);
        }

        // ----- Advantage clipping -----
        if (grpoConfig.advantageClip() > 0.0) {
            double aClip = grpoConfig.advantageClip();
            advantages = clamp(advantages,
                    new org.bytedeco.pytorch.ScalarOptional(new Scalar(-aClip)),
                    new org.bytedeco.pytorch.ScalarOptional(new Scalar(aClip)));
        }

        // ----- Surrogate objective -----
        Tensor totalLoss;
        if (!useClipping || grpoConfig.lossTypeIsDapo()) {
            // DAPO: no clipping. Optionally with epsilon_low/epsilon_high two-sided clip.
            Tensor ratio = newLogprobs.sub(orElse(oldLogprobs, newLogprobs)).exp();
            if (grpoConfig.epsilonHigh() > 0.0 && grpoConfig.epsilon() != grpoConfig.epsilonHigh()) {
                // Two-sided clipping
                Tensor ratioClipped = clamp(ratio,
                        new org.bytedeco.pytorch.ScalarOptional(new Scalar(1.0 - grpoConfig.epsilon())),
                        new org.bytedeco.pytorch.ScalarOptional(new Scalar(1.0 + grpoConfig.epsilonHigh())));
                Tensor surr = advantages.mul(ratioClipped);
                totalLoss = surr.neg().mean();
            } else {
                totalLoss = advantages.mul(ratio).neg().mean();
            }
        } else if (oldLogprobs != null && oldLogprobs.defined() && grpoConfig.clipRange() > 0.0) {
            Tensor clipped = GRPOLoss.computeClipped(
                    newLogprobs, oldLogprobs, rewards, groupSize, grpoConfig.clipRange());
            totalLoss = clipped;
        } else {
            totalLoss = GRPOLoss.compute(newLogprobs, rewards, groupSize,
                    grpoConfig.beta(), refLogprobs);
        }

        // ----- KL penalty -----
        if (grpoConfig.beta() > 0.0 && refLogprobs != null && refLogprobs.defined()) {
            Tensor kl = newLogprobs.sub(refLogprobs).mean()
                    .mul(new Scalar(grpoConfig.beta()));
            totalLoss = totalLoss.add(kl);
        }

        // ----- Router aux loss (MoE) -----
        if (grpoConfig.routerAuxLossCoef() > 0.0 && hasKey(batch, "router_aux_loss")) {
            Tensor aux = batch.get("router_aux_loss");
            if (aux != null && aux.defined()) {
                totalLoss = totalLoss.add(aux.mul(new Scalar(grpoConfig.routerAuxLossCoef())));
            }
        }

        return totalLoss;
    }

    /** Reward scaling variants. */
    private static Tensor scaleRewards(Tensor rewards, String mode, int groupSize) {
        if ("group".equalsIgnoreCase(mode)) {
            long n = rewards.size(0);
            if (n % groupSize != 0) {
                return batchScale(rewards);
            }
            Tensor reshaped = rewards.reshape(n / groupSize, groupSize);
            Tensor mean = reshaped.mean(new long[]{1}, true,new ScalarTypeOptional());
            Tensor std = reshaped.std(new long[]{1}, true).add(new Scalar(1e-8));
            return reshaped.sub(mean).div(std).reshape(n);
        }
        return batchScale(rewards);
    }

    private static Tensor batchScale(Tensor rewards) {
        Tensor mean = rewards.mean();
        Tensor std = rewards.std().add(new Scalar(1e-8));
        return rewards.sub(mean).div(std);
    }

    private static Tensor whiten(Tensor x) {
        Tensor mean = x.mean();
        Tensor std = x.std().add(new Scalar(1e-8));
        return x.sub(mean).div(std);
    }

    public static Tensor groupNormalizeAdvantages(Tensor rewards, int groupSize) {
        return GRPOLoss.groupNormalize(rewards, groupSize);
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

    private static Tensor orElse(Tensor a, Tensor b) {
        return a != null && a.defined() ? a : b;
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
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
        System.out.printf("[GRPOTrainer] Closed: trainingSteps=%d%n", numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
}