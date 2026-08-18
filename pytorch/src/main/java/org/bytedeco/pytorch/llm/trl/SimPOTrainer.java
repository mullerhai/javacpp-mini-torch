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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.SimPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.SimPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * SimPO (Simple Preference Optimization) trainer.
 *
 * <p>SimPO removes the reference model term from DPO and uses a target margin
 * on log probabilities instead.
 *
 * <p>Enterprise features beyond the legacy implementation:
 * <ul>
 *   <li>Loss type switch: {@code simpo} (default), {@code cpo}, {@code rlhf}</li>
 *   <li>CPO-style auxiliary term (cAlpha) integration</li>
 *   <li>Optional SFT NLL auxiliary loss</li>
 *   <li>Label smoothing-aware sigmoid</li>
 *   <li>Reward margin tracking and metric exposure</li>
 *   <li>Reference-free + optional precomputed logprob fast path</li>
 *   <li>NaN/Inf guard with fallback to margin term</li>
 * </ul>
 *
 * <p>Reference: "SimPO: Simple Preference Optimization" (Meng et al., 2024)
 * <a href="https://arxiv.org/abs/2405.14734">arXiv:2405.14734</a>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SimPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "simpo";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final SimPOConfig simpoConfig;
    private final TensorVector params;

    private double totalRewardMargin;
    private double rewardMarginCount;
    private double totalAccuracy;
    private double accuracyCount;

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
                "[SimPOTrainer v%s] beta=%.2f, targetMargin=%.2f, lengthNorm=%s, " +
                        "lossType=%s, cAlpha=%.3f, sftW=%.3f%n",
                VERSION, simpoConfig.beta(), simpoConfig.targetMargin(),
                simpoConfig.lengthNormalize(), simpoConfig.lossType(),
                simpoConfig.cAlpha(), simpoConfig.sftWeight());
    }

    public Module policy() { return policy; }
    public SimPOConfig config() { return simpoConfig; }

    @Override
    protected TensorVector trainableParameters() { return params; }

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
        if (hasKey(batch, "policy_chosen_logps") && hasKey(batch, "policy_rejected_logps")) {
            return computeFromLogps(batch);
        }

        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        Tensor chosenLp = computeLogProbs(chosenLogits, chosenLabels, chosenMask);
        Tensor rejectedLp = computeLogProbs(rejectedLogits, rejectedLabels, rejectedMask);

        // Stash for downstream
        batch.put("policy_chosen_logps", chosenLp);
        batch.put("policy_rejected_logps", rejectedLp);
        batch.put("chosen_logits", chosenLogits);
        batch.put("chosen_labels", chosenLabels);
        batch.put("chosen_attention_mask", chosenMask);

        return computeFromLogps(batch);
    }

    private Tensor computeFromLogps(Map<String, Tensor> batch) {
        Tensor chosenLogps = require(batch, "policy_chosen_logps");
        Tensor rejectedLogps = require(batch, "policy_rejected_logps");

        // Track reward margin and accuracy
        Tensor rewardMargin = chosenLogps.sub(rejectedLogps);
        totalRewardMargin += rewardMargin.mean().item_double();
        rewardMarginCount++;
        try {
            totalAccuracy += rewardMargin.gt(new Scalar(0.0)).to(org.bytedeco.pytorch.global.torch.ScalarType.Float).mean().item_double();
            accuracyCount++;
        } catch (Exception ignored) {}

        String type = simpoConfig.lossType() == null ? "simpo" : simpoConfig.lossType().toLowerCase();
        Tensor loss;
        switch (type) {
            case "cpo":
                loss = computeCpoVariant(chosenLogps, rejectedLogps);
                break;
            case "rlhf":
                loss = computeRlhfVariant(chosenLogps, rejectedLogps);
                break;
            case "simpo":
            default:
                loss = SimPOLoss.compute(
                        chosenLogps, rejectedLogps,
                        simpoConfig.beta(),
                        simpoConfig.targetMargin(),
                        simpoConfig.lengthNormalize(),
                        simpoConfig.labelSmoothing());
                break;
        }

        // CPO auxiliary term
        if (simpoConfig.cAlpha() > 0.0) {
            Tensor aux = chosenLogps.sub(rejectedLogps)
                    .neg()
                    .mean()
                    .mul(new Scalar(simpoConfig.cAlpha()));
            loss = loss.sub(aux);
        }

        // SFT NLL auxiliary
        if (simpoConfig.sftWeight() > 0.0 && hasKey(batch, "chosen_logits")) {
            Tensor chosenLogits = batch.get("chosen_logits");
            Tensor chosenLabels = batch.get("chosen_labels");
            Tensor chosenMask = batch.get("chosen_attention_mask");
            if (chosenLogits != null && chosenLogits.defined()
                    && chosenLabels != null && chosenLabels.defined()) {
                Tensor sft = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
                loss = loss.add(sft.mul(new Scalar(simpoConfig.sftWeight())));
            }
        }

        // NaN/Inf guard
        double v = loss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[SimPOTrainer] WARNING: NaN/Inf loss; falling back to base SimPO term.");
            loss = SimPOLoss.compute(
                    chosenLogps, rejectedLogps,
                    simpoConfig.beta(), simpoConfig.targetMargin(),
                    simpoConfig.lengthNormalize());
        }

        numTrainingSteps++;
        return loss;
    }

    /**
     * CPO-style variant: pair SimPO with a contrastive NLL term to further
     * encourage chosen-likelihood gain.
     */
    private Tensor computeCpoVariant(Tensor chosenLogps, Tensor rejectedLogps) {
        Tensor simpo = SimPOLoss.compute(
                chosenLogps, rejectedLogps,
                simpoConfig.beta(), simpoConfig.targetMargin(),
                simpoConfig.lengthNormalize());

        // NLL term on chosen
        Tensor nll = chosenLogps.neg().mean();
        return simpo.add(nll.mul(new Scalar(simpoConfig.cAlpha())));
    }

    /**
     * RLHF-style variant: classic policy gradient with target margin.
     */
    private Tensor computeRlhfVariant(Tensor chosenLogps, Tensor rejectedLogps) {
        // Convert log-probs to per-sample rewards with target margin
        Tensor rewardDiff = chosenLogps.sub(rejectedLogps).sub(new Scalar(simpoConfig.targetMargin()));
        Tensor policyLoss = rewardDiff.mul(new Scalar(simpoConfig.beta())).sigmoid().log().neg().mean();
        return policyLoss;
    }

    private Tensor computeLogProbs(Tensor logits, Tensor labels, Tensor mask) {
        if (simpoConfig.lengthNormalize()) {
            return LogProbUtils.sequenceMeanLogProbs(logits, labels, mask);
        } else {
            return LogProbUtils.sequenceLogProbs(logits, labels, mask);
        }
    }

    public double getAverageRewardMargin() {
        return rewardMarginCount > 0 ? totalRewardMargin / rewardMarginCount : 0.0;
    }

    public double getAverageAccuracy() {
        return accuracyCount > 0 ? totalAccuracy / accuracyCount : 0.0;
    }

    public void resetMetrics() {
        totalRewardMargin = 0.0;
        rewardMarginCount = 0.0;
        totalAccuracy = 0.0;
        accuracyCount = 0.0;
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
                "[SimPOTrainer v%s] Closed: steps=%d, avgRewardMargin=%.4f, avgAcc=%.4f%n",
                VERSION, numTrainingSteps, getAverageRewardMargin(), getAverageAccuracy());
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "SimPO v" + VERSION + " (Simple Preference Optimization)";
    }
}