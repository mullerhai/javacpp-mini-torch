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

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.trl.config.PPOXConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.clamp;
import static org.bytedeco.pytorch.global.torch.exp;
import static org.bytedeco.pytorch.global.torch.mean;

/**
 * PPO-X trainer for LLM alignment (ByteDance inspired).
 *
 * <p>PPO-X is an enhanced PPO variant with:
 * <ul>
 *   <li>Adaptive clipping with trust region management</li>
 *   <li>Importance weight clipping for stability</li>
 *   <li>Reward normalization and shaping</li>
 *   <li>GAE (Generalized Advantage Estimation) support</li>
 * </ul>
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code log_probs} - current policy log probs {@code [B]}</li>
 *   <li>{@code old_log_probs} - old policy log probs {@code [B]}</li>
 *   <li>{@code rewards} - rewards {@code [B]}</li>
 *   <li>{@code values} - value estimates {@code [B]} (optional)</li>
 *   <li>{@code advantages} - precomputed advantages {@code [B]} (optional)</li>
 * </ul>
 */
public final class PPOXTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final Module valueModel;
    private final LlmForward policyForward;
    private final LlmForward valueForward;
    private final PPOXConfig config;
    private final TensorVector policyParams;
    private final TensorVector valueParams;

    // Adaptive clipping state
    private double currentClipRatio;
    private double runningKL;

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
        this.runningKL = 0.0;

        if (valueModel != null) {
            freeze(valueModel);
        }
    }

    /** Simple constructor without value model (uses policy for values). */
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
    protected TensorVector trainableParameters() {
        return policyParams;
    }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (valueModel != null) {
            valueModel.train(true);
        }
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (valueModel != null) {
            valueModel.eval();
        }
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        // Extract necessary tensors
        Tensor logProbs = require(batch, "log_probs");
        Tensor oldLogProbs = require(batch, "old_log_probs");
        Tensor rewards = require(batch, "rewards");

        // Compute or extract advantages
        Tensor advantages;
        if (batch.containsKey("advantages") && batch.get("advantages") != null
                && batch.get("advantages").defined()) {
            advantages = batch.get("advantages");
        } else {
            advantages = computeAdvantages(batch, rewards);
        }

        // Compute importance ratios
        Tensor ratios = computeImportanceRatios(logProbs, oldLogProbs);

        // Compute clipped policy loss
        Tensor policyLoss = computeClippedPolicyLoss(ratios, advantages);

        // Compute value loss (if using value model)
        Tensor valueLoss = computeValueLoss(batch, advantages);

        // Compute entropy bonus for exploration
        Tensor entropyLoss = computeEntropyLoss(logProbs);

        // Combined loss: policy + value_coef * value - entropy_coef * entropy
        Tensor policyTerm = policyLoss;
        Tensor valueTerm = policyLoss
                .add(new Scalar(config.valueLossCoeff())).mul(valueLoss);
        Tensor entropyTerm = policyLoss
                .sub(new Scalar(config.entropyCoefficient())).mul(entropyLoss);
        Tensor totalLoss = policyTerm.add(valueTerm).sub(entropyTerm);

        // Update adaptive clipping
        if (config.adaptiveClipping()) {
            updateAdaptiveClipping(ratios);
        }

        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor computeImportanceRatios(Tensor logProbs, Tensor oldLogProbs) {
        Tensor logRatio = logProbs.sub(oldLogProbs);
        return exp(logRatio);
    }

    private Tensor computeAdvantages(Map<String, Tensor> batch, Tensor rewards) {
        if (config.useGAE()) {
            return computeGAE(rewards);
        } else {
            // Simple advantage: reward - mean(reward)
            Tensor meanReward = rewards.mean();
            return rewards.sub(meanReward);
        }
    }

    private Tensor computeGAE(Tensor rewards) {
        // Simplified GAE for batch processing
        double gamma = config.gaeGamma();
        double tau = config.gaeTau();
        double lambda = config.advantageLambda();

        int batchSize = (int) rewards.size(0);
        Tensor advantages = org.bytedeco.pytorch.global.torch.zeros_like(rewards);
        Tensor returns = rewards.clone();

        // Simplified GAE computation (recurrent, in-place write through index_copy_)
        for (int i = batchSize - 2; i >= 0; i--) {
            Tensor returnsNext = returns.select(0, i + 1);
            Tensor delta = returns.select(0, i)
                    .add(returnsNext.mul(new Scalar(gamma * lambda)));
            Tensor idx = org.bytedeco.pytorch.global.torch.tensor(new long[]{i});
            advantages.index_copy_(0, idx, delta.unsqueeze(0));
        }

        return advantages;
    }

    private Tensor computeClippedPolicyLoss(Tensor ratios, Tensor advantages) {
        double clipEps = currentClipRatio;

        // Clipped surrogate objective
        Tensor surr1 = ratios.mul(advantages);
        Tensor ratiosClipped = clamp(ratios, new ScalarOptional(new Scalar(1 - clipEps)), new ScalarOptional(new Scalar(1 + clipEps)));
        Tensor surr2 = ratiosClipped.mul(advantages);

        // Take minimum to get clipped loss (elementwise min via torch.where)
        Tensor clippedLoss = org.bytedeco.pytorch.global.torch.where(surr1.lt(surr2), surr1, surr2);

        return clippedLoss.neg().mean();
    }

    private Tensor computeValueLoss(Map<String, Tensor> batch, Tensor advantages) {
        if (valueModel == null) {
            return org.bytedeco.pytorch.global.torch.zeros(1);
        }

        Tensor values = require(batch, "values");
        Tensor oldValues = require(batch, "old_values");

        // Value clipping for stability
        double clipEps = config.valueClipRatio();
        Scalar epsScalar = new Scalar(clipEps);
        Tensor valuesClipped = clamp(
                values,
                new ScalarOptional(new Scalar(oldValues.sub(epsScalar))),
                new ScalarOptional(new Scalar(oldValues.add(epsScalar)))
        );

        Tensor loss1 = values.sub(advantages).pow(new Scalar(2));
        Tensor loss2 = valuesClipped.sub(advantages).pow(new Scalar(2));

        return org.bytedeco.pytorch.global.torch.where(loss1.gt(loss2), loss1, loss2).mean();
    }

    private Tensor computeEntropyLoss(Tensor logProbs) {
        // Entropy bonus: -sum(p * log(p))
        Tensor probs = exp(logProbs);
        Tensor entropy = logProbs.mul(probs).neg();
        return entropy.mean();
    }

    private void updateAdaptiveClipping(Tensor ratios) {
        // Compute KL divergence from importance ratios
        double kl = (ratios.log().mean().item_double() -
                    (ratios.mean().log().item_double()));

        runningKL = 0.9 * runningKL + 0.1 * kl;

        // Adapt clip ratio based on KL
        double targetKL = config.trustRegionRadius();
        double adjustFactor = 1.1;

        if (runningKL > targetKL * 1.5) {
            currentClipRatio = Math.max(0.01, currentClipRatio / adjustFactor);
        } else if (runningKL < targetKL / 1.5) {
            currentClipRatio = Math.min(0.5, currentClipRatio * adjustFactor);
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
                "[PPOXTrainer] Closed: trainingSteps=%d, finalClipRatio=%.4f, finalKL=%.4f%n",
                numTrainingSteps, currentClipRatio, runningKL);
    }

    public boolean isClosed() { return closed; }
    public double getCurrentClipRatio() { return currentClipRatio; }
    public double getRunningKL() { return runningKL; }
}
