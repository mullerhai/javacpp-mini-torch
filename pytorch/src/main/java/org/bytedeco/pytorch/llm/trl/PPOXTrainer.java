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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.PPOXConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.clamp;
import static org.bytedeco.pytorch.global.torch.exp;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.where;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * PPO-X trainer for LLM alignment (ByteDance inspired).
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Adaptive clipping with trust-region management</li>
 *   <li>Reward scaling / clipping / whitening</li>
 *   <li>Adaptive KL control (init/target/adap toggles)</li>
 *   <li>PPO-epochs / mini-batch scheduling</li>
 *   <li>Reference KL penalty</li>
 *   <li>SFT auxiliary</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PPOXTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "ppo_x";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final Module valueModel;
    private final LlmForward policyForward;
    private final LlmForward valueForward;
    private final PPOXConfig config;
    private final TensorVector policyParams;
    private final TensorVector valueParams;

    private double currentClipRatio;
    private double runningKL;
    private double currentKlCoef;

    public PPOXTrainer(
            Module policy,
            LlmForward policyForward,
            Module valueModel,
            LlmForward valueForward,
            Optimizer optimizer,
            Optimizer valueOptimizer,
            PPOXConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.valueModel = valueModel;
        this.valueForward = valueForward;
        this.config = Objects.requireNonNull(config, "config");
        this.policyParams = policy.parameters();
        this.valueParams = valueModel != null ? valueModel.parameters() : new TensorVector();
        this.currentClipRatio = config.clipRatio();
        this.currentKlCoef = config.initKlCoef();
        System.out.printf(
                "[PPOXTrainer v%s] clip=%.3f, kl=%.3f, target=%.3f, vf=%.3f, ent=%.3f, beta=%.3f, sftW=%.3f%n",
                VERSION, config.clipRatio(), config.initKlCoef(), config.targetKl(),
                config.valueLossCoeff(), config.entropyCoefficient(),
                config.beta(), config.sftWeight());
    }

    public PPOXTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            PPOXConfig config) {
        this(policy, policyForward, null, null, optimizer, null, config);
    }

    public Module policy() { return policy; }
    public Module valueModel() { return valueModel; }
    public PPOXConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() { return policyParams; }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (valueModel != null) valueModel.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (valueModel != null) valueModel.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor logProbs = require(batch, "log_probs");
        Tensor oldLogProbs = require(batch, "old_log_probs");
        Tensor rewards = require(batch, "rewards");

        // Reward processing
        if (shouldScaleRewards(config.scaleRewards())) {
            double std = rewards.std().item_double();
            if (std > 1e-6) rewards = rewards.div(new Scalar(std + 1e-8));
        }
        if (config.cliprangeReward() > 0.0) {
            double c = config.cliprangeReward();
            rewards = rewards.clamp(new ScalarOptional(new Scalar(-c)),new ScalarOptional(new Scalar(c)));
        }

        // Advantages
        Tensor advantages;
        if (hasKey(batch, "advantages")) {
            advantages = batch.get("advantages");
        } else {
            advantages = computeAdvantages(batch, rewards);
        }
        if (config.whitenAdvantages()) {
            Tensor mean = advantages.mean();
            Tensor std = advantages.std();
            advantages = advantages.sub(mean).div(std.add(new Scalar(1e-8)));
        }

        Tensor ratios = computeImportanceRatios(logProbs, oldLogProbs);
        Tensor policyLoss = computeClippedPolicyLoss(ratios, advantages);
        Tensor valueLoss = computeValueLoss(batch, advantages);
        Tensor entropyLoss = computeEntropyLoss(logProbs);

        // KL penalty (between current and old log probs)
        Tensor klPenalty = logProbs.sub(oldLogProbs).mean();

        // Reference KL penalty
        if (config.useReferenceKl() && hasKey(batch, "ref_log_probs")) {
            Tensor refLp = batch.get("ref_log_probs").detach();
            Tensor refKl = logProbs.sub(refLp).mean();
            klPenalty = klPenalty.add(refKl);
        }

        Tensor totalLoss = policyLoss
                .add(valueLoss.mul(new Scalar(config.valueLossCoeff())))
                .sub(entropyLoss.mul(new Scalar(config.entropyCoefficient())))
                .add(klPenalty.mul(new Scalar(currentKlCoef)));

        // SFT auxiliary
        if (config.sftWeight() > 0.0 && hasKey(batch, "sft_logits") && hasKey(batch, "sft_labels")) {
            Tensor sftLogits = batch.get("sft_logits");
            Tensor sftLabels = batch.get("sft_labels");
            Tensor sft = org.bytedeco.pytorch.llm.trl.loss.DPOLoss.sftNll(sftLogits, sftLabels, null);
            totalLoss = totalLoss.add(sft.mul(new Scalar(config.sftWeight())));
        }

        // Adaptive clipping
        if (config.adaptiveClipping()) {
            updateAdaptiveClipping(ratios);
        }

        // Adaptive KL
        if (config.adapKlCtrl() && config.targetKl() > 0.0) {
            double kl = klPenalty.item_double();
            adaptKl(kl);
        }

        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor computeImportanceRatios(Tensor logProbs, Tensor oldLogProbs) {
        return exp(logProbs.sub(oldLogProbs));
    }

    private Tensor computeAdvantages(Map<String, Tensor> batch, Tensor rewards) {
        if (config.useGAE()) {
            return computeGAE(rewards);
        }
        Tensor meanReward = rewards.mean();
        return rewards.sub(meanReward);
    }

    private Tensor computeGAE(Tensor rewards) {
        double gamma = config.gaeGamma();
        double lambda = config.advantageLambda();

        long batchSize = rewards.size(0);
        Tensor advantages = zeros_like(rewards);
        Tensor returns = rewards.clone();

        for (long i = batchSize - 2; i >= 0; i--) {
            Tensor returnsNext = returns.select(0, i + 1);
            Tensor delta = returns.select(0, i).add(returnsNext.mul(new Scalar(gamma * lambda)));
            Tensor idx = tensor(new long[]{i});
            advantages.index_copy_(0, idx, delta.unsqueeze(0));
        }
        return advantages;
    }

    private Tensor computeClippedPolicyLoss(Tensor ratios, Tensor advantages) {
        double clipEps = currentClipRatio;
        Tensor surr1 = ratios.mul(advantages);
        Tensor ratiosClipped = clamp(ratios,
                new ScalarOptional(new Scalar(1 - clipEps)),
                new ScalarOptional(new Scalar(1 + clipEps)));
        Tensor surr2 = ratiosClipped.mul(advantages);
        Tensor clippedLoss = where(surr1.lt(surr2), surr1, surr2);
        return clippedLoss.neg().mean();
    }

    private Tensor computeValueLoss(Map<String, Tensor> batch, Tensor advantages) {
        if (valueModel == null) {
            return zeros_like(advantages).mean();
        }
        if (!hasKey(batch, "values")) {
            return zeros_like(advantages).mean();
        }
        Tensor values = batch.get("values");
        Tensor oldValues = hasKey(batch, "old_values") ? batch.get("old_values") : values;
        double clipEps = config.valueClipRatio();
        Tensor valuesClipped = clamp(values,
                new ScalarOptional(new Scalar(oldValues.sub(new Scalar(clipEps)))),
                new ScalarOptional(new Scalar(oldValues.add(new Scalar(clipEps)))));

        Tensor loss1 = values.sub(advantages).pow(new Scalar(2));
        Tensor loss2 = valuesClipped.sub(advantages).pow(new Scalar(2));
        return where(loss1.gt(loss2), loss1, loss2).mean();
    }

    private Tensor computeEntropyLoss(Tensor logProbs) {
        Tensor probs = exp(logProbs);
        Tensor entropy = logProbs.mul(probs).neg();
        return entropy.mean();
    }

    private void updateAdaptiveClipping(Tensor ratios) {
        try {
            double kl = ratios.log().mean().item_double() - ratios.mean().log().item_double();
            runningKL = 0.9 * runningKL + 0.1 * kl;
            double targetKL = config.trustRegionRadius();
            if (runningKL > targetKL * 1.5) {
                currentClipRatio = Math.max(0.01, currentClipRatio / 1.1);
            } else if (runningKL < targetKL / 1.5) {
                currentClipRatio = Math.min(0.5, currentClipRatio * 1.1);
            }
        } catch (Exception ignored) {}
    }

    private void adaptKl(double kl) {
        double target = config.targetKl();
        if (kl > target * 1.2) currentKlCoef *= 1.5;
        else if (kl < target * 0.8) currentKlCoef *= 0.5;
        currentKlCoef = Math.max(1e-4, Math.min(10.0, currentKlCoef));
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
                "[PPOXTrainer v%s] Closed: steps=%d, clip=%.4f, kl=%.4f%n",
                VERSION, numTrainingSteps, currentClipRatio, runningKL);
    }

    public boolean isClosed() { return closed; }
    public double getCurrentClipRatio() { return currentClipRatio; }
    public double getRunningKL() { return runningKL; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "PPO-X v" + VERSION + " (Enhanced Proximal Policy Optimization)";
    }
}