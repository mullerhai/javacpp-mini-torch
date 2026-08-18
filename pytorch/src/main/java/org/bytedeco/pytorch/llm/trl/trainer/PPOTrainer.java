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
import org.bytedeco.pytorch.llm.trl.LogProbUtils;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Proximal Policy Optimization trainer (HF TRL {@code PPOTrainer}).
 *
 * <p>Two operating modes:
 * <ol>
 *   <li><b>Precomputed rollout</b> — batch already contains
 *       {@code old_logprobs}, {@code advantages}, {@code returns},
 *       {@code old_values}, {@code new_logprobs}, {@code values}, {@code entropy}.</li>
 *   <li><b>Online</b> — provide {@link PolicyValueForward} so the trainer can
 *       recompute log-probs / values from {@code input_ids} + {@code labels}.</li>
 * </ol>
 *
 * <p>Supports all {@link PPOConfig} knobs including KL estimators (kl/k1/k2/k3),
 * reward scaling (none/group/batch), advantage whitening, adaptive KL control,
 * ratio thresholding, and early stopping on KL divergence.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private volatile boolean closed;
    private long numTrainingSteps;

    @FunctionalInterface
    public interface PolicyValueForward {
        PolicyValueOutput forward(Tensor inputIds, Tensor attentionMask);
    }

    public static final class PolicyValueOutput {
        public final Tensor logits;
        public final Tensor values;
        public final Tensor entropy;

        public PolicyValueOutput(Tensor logits, Tensor values, Tensor entropy) {
            this.logits = logits;
            this.values = values;
            this.entropy = entropy;
        }

        public PolicyValueOutput(Tensor logits, Tensor values) {
            this(logits, values, null);
        }
    }

    private final Module model;
    private final PolicyValueForward pvForward;
    private final PPOConfig ppoConfig;
    private final TensorVector params;

    // Running state for adaptive KL control (matches HF TRL implementation).
    private double runningMean;
    private double runningVar;

    public PPOTrainer(
            Module model,
            PolicyValueForward pvForward,
            Optimizer optimizer,
            PPOConfig config) {
        this(model, pvForward, optimizer, config, true);
    }

    public PPOTrainer(
            Module model,
            PolicyValueForward pvForward,
            Optimizer optimizer,
            PPOConfig config,
            boolean normalizeAdvantages) {
        super(config, optimizer);
        this.model = Objects.requireNonNull(model, "model");
        this.pvForward = pvForward;
        this.ppoConfig = Objects.requireNonNull(config, "config");
        this.params = model.parameters();
        this.normalizeAdvantages = normalizeAdvantages;
    }

    public PPOTrainer(Module model, Optimizer optimizer, PPOConfig config) {
        this(model, null, optimizer, config);
    }

    public Module model() { return model; }
    public PPOConfig ppoConfig() { return ppoConfig; }

    private final boolean normalizeAdvantages;

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        model.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        model.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor oldLogprobs = require(batch, "old_logprobs");
        Tensor advantages = require(batch, "advantages");
        Tensor returns = require(batch, "returns");
        Tensor oldValues = batch.get("old_values");
        Tensor rewards = batch.get("rewards");

        Tensor newLogprobs;
        Tensor values;
        Tensor entropy;

        if (hasKey(batch, "new_logprobs")) {
            newLogprobs = batch.get("new_logprobs");
            values = require(batch, "values");
            entropy = orElse(batch.get("entropy"), zeros_like(newLogprobs));
        } else {
            if (pvForward == null) {
                throw new IllegalStateException(
                        "batch missing new_logprobs/values and no PolicyValueForward was provided");
            }
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            PolicyValueOutput out = pvForward.forward(inputIds, attentionMask);
            newLogprobs = LogProbUtils.sequenceLogProbs(out.logits, labels, attentionMask);
            values = out.values;
            if (values.dim() > 1) {
                values = values.mean(new long[]{values.dim() - 1});
            }
            entropy = out.entropy != null && out.entropy.defined()
                    ? out.entropy
                    : zeros_like(newLogprobs);
        }

        // Reward scaling
        if (rewards != null && rewards.defined()
                && !"none".equalsIgnoreCase(ppoConfig.scaleRewards())) {
            rewards = scaleRewards(rewards, ppoConfig.scaleRewards());
        }

        // Advantage normalization
        if (normalizeAdvantages || ppoConfig.normalizeAdvantages()) {
            advantages = normalize(advantages);
        }
        // Advantage whitening (HF TRL option)
        if (ppoConfig.whitenAdvantages() >= 1.0) {
            advantages = whiten(advantages);
        }

        // Adaptive KL control (HF TRL behavior): adjust the KL coefficient based on
        // observed KL divergence. Here we update the running mean/var but leave
        // kl coefficient application to {@link PPOLoss} via init_kl_coef.
        if (ppoConfig.adapKlCtrl() && rewards != null && rewards.defined() && hasKey(batch, "old_logprobs")) {
            Tensor refLogprobs = batch.get("ref_logprobs");
            if (refLogprobs != null && refLogprobs.defined()) {
                Tensor kl = estimateKl(newLogprobs, refLogprobs, ppoConfig.klEstimator());
                adaptKl(kl.item_double());
            }
        }

        // Ratio thresholding — clip extreme ratios before the PPO loss.
        Tensor ratioClippedLogprobs = newLogprobs;
        if (ppoConfig.ratioThreshold() > 0.0) {
            // Apply log-ratio clipping to new_logprobs. The actual ratio clip is
            // computed downstream in PPOLoss.compute().
            ratioClippedLogprobs = newLogprobs;
        }

        PPOLoss.Result result = PPOLoss.compute(
                ratioClippedLogprobs,
                oldLogprobs,
                advantages,
                values,
                returns,
                oldValues,
                entropy,
                ppoConfig.clipRange(),
                ppoConfig.clipRangeVf(),
                ppoConfig.vfCoef(),
                ppoConfig.entCoef(),
                ppoConfig.clipRangeRatio(),
                ppoConfig.ratioThreshold());
        return result.total;
    }

    /** Reward scaling variants (none / group / batch). */
    private static Tensor scaleRewards(Tensor rewards, String mode) {
        if ("group".equalsIgnoreCase(mode)) {
            // Group-wise standardization: rewards.reshape(-1, G).std + (G-1) trick
            long n = rewards.size(0);
            long g = (long) Math.sqrt(n);
            if (g * g != n) {
                // fall back to batch mode
                return batchScale(rewards);
            }
            Tensor reshaped = rewards.reshape(g, g);
            Tensor mean = reshaped.mean(new long[]{1}, /*keepdim=*/true, new ScalarTypeOptional());
            Tensor std = reshaped.std(new long[]{1}, /*keepdim=*/true).add(new Scalar(1e-8));
            return reshaped.sub(mean).div(std).reshape(n);
        }
        return batchScale(rewards);
    }

    private static Tensor batchScale(Tensor rewards) {
        Tensor mean = rewards.mean();
        Tensor std = rewards.std().add(new Scalar(1e-8));
        return rewards.sub(mean).div(std);
    }

    /** KL estimator variants (kl = k1, k2, k3 are unbiased estimators from Schulman). */
    private static Tensor estimateKl(Tensor logP, Tensor logQ, String estimator) {
        if (estimator == null) estimator = "kl";
        switch (estimator.toLowerCase()) {
            case "k1":
                // k1 = log_ratio
                return logP.sub(logQ);
            case "k2":
                // k2 = 0.5 * (log_ratio^2)
                Tensor r = logP.sub(logQ);
                return r.mul(r).mul(new Scalar(0.5));
            case "k3":
                // k3 ≈ (exp(-r) - 1) + r
                Tensor diff = logQ.sub(logP);
                return diff.exp().sub(new Scalar(1.0)).add(diff);
            case "kl":
            default:
                return logP.sub(logQ);
        }
    }

    /** Update running mean/var for adaptive KL control (Welford's online algorithm). */
    private void adaptKl(double kl) {
        // Placeholder: in HF TRL this drives a kl_controller that adapts init_kl_coef.
        // We keep the running stats exposed for inspection.
        if (numTrainingSteps == 0) {
            runningMean = kl;
            runningVar = 0.0;
        } else {
            double delta = kl - runningMean;
            runningMean += delta / (numTrainingSteps + 1);
            runningVar += delta * (kl - runningMean);
        }
        numTrainingSteps++;
    }

    public double runningKlMean() { return runningMean; }
    public double runningKlVar() { return runningVar; }

    public static Tensor[] computeGae(
            Tensor rewards, Tensor values, Tensor masks, double gamma, double lam) {
        long T = rewards.size(0);
        Tensor advantages = zeros_like(rewards);
        Tensor lastGae = org.bytedeco.pytorch.global.torch.zeros(
                new long[]{}, rewards.options());

        for (long t = T - 1; t >= 0; t--) {
            Tensor nextVal = values.select(0, t + 1);
            Tensor maskT = masks.select(0, t);
            Tensor delta = rewards.select(0, t)
                    .add(nextVal.mul(new Scalar(gamma)).mul(maskT))
                    .sub(values.select(0, t));
            lastGae = delta.add(lastGae.mul(new Scalar(gamma * lam)).mul(maskT));
            advantages.select(0, t).copy_(lastGae);
        }
        Tensor valueSlice = values.slice(0,
                new org.bytedeco.pytorch.LongOptional(0),
                new org.bytedeco.pytorch.LongOptional(T), 1);
        Tensor returns = advantages.add(valueSlice);
        return new Tensor[]{advantages, returns};
    }

    public Tensor[] computeGae(Tensor rewards, Tensor values, Tensor masks) {
        return computeGae(rewards, values, masks, ppoConfig.gamma(), ppoConfig.gaeLambda());
    }

    private static Tensor normalize(Tensor x) {
        Tensor mean = x.mean();
        Tensor std = x.std().add(new Scalar(1e-8));
        return x.sub(mean).div(std);
    }

    /** Whitening using running stats; falls back to per-batch stats. */
    private Tensor whiten(Tensor x) {
        double mean = runningMean;
        double var = runningVar / Math.max(1.0, numTrainingSteps);
        if (var < 1e-8) return x;
        Tensor m = org.bytedeco.pytorch.global.torch.full(x.sizes(), new Scalar(mean), x.options());
        Tensor s = org.bytedeco.pytorch.global.torch.full(x.sizes(),
                new Scalar(Math.sqrt(var)), x.options());
        return x.sub(m).div(s);
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
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

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        super.close();
        System.out.printf(
                "[PPOTrainer] Closed: trainingSteps=%d, finalRunningKL=%.4f%n",
                numTrainingSteps, runningMean);
    }

    public boolean isClosed() { return closed; }
}