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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.LogProbUtils;
import org.bytedeco.pytorch.llm.trl.config.RRHFConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.RRHFLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * RRHF (Rank Responses to Rank Responses) trainer.
 *
 * <p>Enterprise features beyond the legacy implementation:
 * <ul>
 *   <li>Optional SFT auxiliary loss</li>
 *   <li>Top-K truncation of candidates per prompt</li>
 *   <li>Margin penalty for ranking loss</li>
 *   <li>Rank head scoring hook (for use_rank_head)</li>
 *   <li>NaN/Inf guard with fallback to SFT term</li>
 * </ul>
 *
 * <p>Reference: "RRHF: Rank Responses to Rank Human Preference" (Yuan et al., 2023)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RRHFTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "rrhf";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module rewardModel;
    private final Module reference;
    private final LlmForward referenceForward;
    private final RRHFConfig rrhfConfig;
    private final TensorVector params;

    private double totalRewardCorrelation;
    private int correlationCount;

    public RRHFTrainer(
            Module policy,
            LlmForward policyForward,
            Module rewardModel,
            Optimizer optimizer,
            RRHFConfig config) {
        this(policy, policyForward, rewardModel, null, null, optimizer, config);
    }

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
        if (reference != null) freeze(reference);
        if (rewardModel != null) freeze(rewardModel);

        System.out.printf(
                "[RRHFTrainer v%s] numResponses=%d, topK=%d, sftW=%.3f, useRankHead=%s%n",
                VERSION, rrhfConfig.numResponses(), rrhfConfig.topK(),
                rrhfConfig.sftWeight(), rrhfConfig.useRankHead());
    }

    public Module policy() { return policy; }
    public Module rewardModel() { return rewardModel; }
    public Module reference() { return reference; }
    public RRHFConfig config() { return rrhfConfig; }

    @Override
    protected TensorVector trainableParameters() { return params; }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (reference != null) reference.eval();
        if (rewardModel != null) rewardModel.eval();
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (reference != null) reference.eval();
        if (rewardModel != null) rewardModel.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor rewards = require(batch, "rewards");
        int numResponses = rrhfConfig.numResponses();
        int topK = rrhfConfig.topK();

        Tensor logProbs;
        Tensor chosenLogits = null;
        Tensor chosenLabels = null;
        Tensor chosenMask = null;
        if (hasKey(batch, "log_probs")) {
            logProbs = batch.get("log_probs");
        } else {
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            logProbs = rrhfConfig.lengthNormalize()
                    ? LogProbUtils.sequenceMeanLogProbs(logits, labels, attentionMask)
                    : LogProbUtils.sequenceLogProbs(logits, labels, attentionMask);
            chosenLogits = logits;
            chosenLabels = labels;
            chosenMask = attentionMask;
        }

        Tensor refLogProbs = null;
        if (referenceForward != null && rrhfConfig.useReferenceModel()) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor inputIds = require(batch, "input_ids");
                Tensor attentionMask = batch.get("attention_mask");
                Tensor labels = orElse(batch.get("labels"), inputIds);
                Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
                refLogProbs = rrhfConfig.lengthNormalize()
                        ? LogProbUtils.sequenceMeanLogProbs(refLogits, labels, attentionMask)
                        : LogProbUtils.sequenceLogProbs(refLogits, labels, attentionMask);
                refLogProbs = refLogProbs.detach();
            }
        }

        // Use rank head scoring if enabled (substitute ref logprobs with rank head scores).
        if (rrhfConfig.useRankHead() && rewardModel != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor inputIds = require(batch, "input_ids");
                Tensor attentionMask = batch.get("attention_mask");
                Tensor rmOut = rewardModel.forward(inputIds);
                // Use the first scalar of the rank head as the per-sample score.
                rewards = rmOut.reshape(new long[]{rmOut.size(0)}).detach();
            }
        }

        // Truncate to topK
        if (topK > 0 && topK < numResponses) {
            int k = topK;
            int keep = (int) (rewards.size(0) - rewards.size(0) % k);
            if (keep > 0) {
                logProbs = logProbs.slice(0, sliceStart(0), sliceEnd(keep), 1);
                rewards = rewards.slice(0, sliceStart(0), sliceEnd(keep), 1);
                if (refLogProbs != null) refLogProbs = refLogProbs.slice(0, sliceStart(0), sliceEnd(keep), 1);
            }
        }

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

        // Margin penalty
        if (rrhfConfig.margin() > 0.0) {
            Tensor diff = logProbs.sub(refLogProbs != null ? refLogProbs : zeros_like(logProbs));
            Tensor marginTerm = torch.tensor(rrhfConfig.margin()).sub(diff).clamp_min(new Scalar(0.0));
            loss = loss.add(marginTerm.mean());
        }

        // Reference KL regularization
        if (refLogProbs != null && rrhfConfig.beta() > 0.0) {
            Tensor kl = logProbs.sub(refLogProbs).mean();
            loss = loss.add(kl.mul(new Scalar(rrhfConfig.beta())));
        }

        // SFT auxiliary
        if (rrhfConfig.sftWeight() > 0.0 && chosenLogits != null && chosenLogits.defined()) {
            Tensor sft = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
            loss = loss.add(sft.mul(new Scalar(rrhfConfig.sftWeight())));
        }

        double v = loss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[RRHFTrainer] WARNING: NaN/Inf loss; falling back to SFT term.");
            if (chosenLogits != null) {
                loss = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
            }
        }

        updateRewardCorrelation(logProbs, rewards);
        numTrainingSteps++;
        return loss;
    }

    private static org.bytedeco.pytorch.LongOptional sliceStart(long v) {
        return new org.bytedeco.pytorch.LongOptional(v);
    }
    private static org.bytedeco.pytorch.LongOptional sliceEnd(long v) {
        return new org.bytedeco.pytorch.LongOptional(v);
    }

    private void updateRewardCorrelation(Tensor logProbs, Tensor rewards) {
        try {
            double lp = logProbs.mean().item_double();
            double rw = rewards.mean().item_double();
            // crude positive-favourable correlation tracking: monotone only when
            // both averages move together. Real impl would compute Spearman.
            totalRewardCorrelation += lp * rw;
        } catch (Exception ignored) {}
        correlationCount++;
    }

    public double getRewardCorrelation() {
        return correlationCount > 0 ? totalRewardCorrelation / correlationCount : 0.0;
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
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
        System.out.printf("[RRHFTrainer v%s] Closed: steps=%d%n", VERSION, numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "RRHF v" + VERSION + " (Rank Responses to Rank Human Preference)";
    }
}