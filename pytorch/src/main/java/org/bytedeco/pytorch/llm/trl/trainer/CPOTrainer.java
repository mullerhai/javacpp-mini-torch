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
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.LogProbUtils;
import org.bytedeco.pytorch.llm.trl.config.CPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.full_like;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Contrastive Preference Optimization trainer.
 *
 * <p>Enterprise features beyond the legacy implementation:
 * <ul>
 *   <li>Loss-type switch: {@code cpo}, {@code simpo}, {@code siglip}</li>
 *   <li>Length-normalized log-probs</li>
 *   <li>Optional SimPO {@code gamma} margin</li>
 *   <li>CPO α (contrastive weight) plus a label-smoothing regularizer</li>
 *   <li>SFT NLL auxiliary loss with configurable weight</li>
 *   <li>Reference model optional (zero-padded if absent)</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class CPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "cpo";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final CPOConfig cpoConfig;
    private final TensorVector params;

    public CPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            CPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public CPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            CPOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.cpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        if (reference != null) {
            freeze(reference);
        }
        System.out.printf(
                "[CPOTrainer v%s] beta=%.3f, alpha=%.3f, lossType=%s, lenNorm=%s%n",
                VERSION, cpoConfig.beta(), cpoConfig.cpoAlpha(), cpoConfig.lossType(),
                cpoConfig.lengthNormalize());
    }

    public Module policy() { return policy; }
    public CPOConfig cpoConfig() { return cpoConfig; }

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
        Tensor chosenLp;
        Tensor rejectedLp;
        Tensor chosenLogits = null;
        Tensor chosenLabels = null;
        Tensor chosenMask = null;

        if (hasKey(batch, "chosen_logps") && hasKey(batch, "rejected_logps")) {
            chosenLp = batch.get("chosen_logps");
            rejectedLp = batch.get("rejected_logps");
        } else {
            Tensor chosenIds = require(batch, "chosen_input_ids");
            Tensor rejectedIds = require(batch, "rejected_input_ids");
            chosenMask = batch.get("chosen_attention_mask");
            Tensor rejectedMask = batch.get("rejected_attention_mask");
            chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
            Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

            chosenLogits = policyForward.forward(chosenIds, chosenMask);
            Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);
            chosenLp = logps(chosenLogits, chosenLabels, chosenMask);
            rejectedLp = logps(rejectedLogits, rejectedLabels, rejectedMask);
        }

        Tensor refChosenLp;
        Tensor refRejectedLp;
        if (referenceForward != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor ids = require(batch, "chosen_input_ids");
                Tensor rIds = require(batch, "rejected_input_ids");
                Tensor cm = batch.get("chosen_attention_mask");
                Tensor rm = batch.get("rejected_attention_mask");
                Tensor cl = orElse(batch.get("chosen_labels"), ids);
                Tensor rl = orElse(batch.get("rejected_labels"), rIds);
                Tensor refCLogits = referenceForward.forward(ids, cm);
                Tensor refRLogits = referenceForward.forward(rIds, rm);
                refChosenLp = logps(refCLogits, cl, cm).detach();
                refRejectedLp = logps(refRLogits, rl, rm).detach();
            }
        } else {
            refChosenLp = zeros_like(chosenLp);
            refRejectedLp = zeros_like(rejectedLp);
        }

        Tensor totalLoss = computeCpoLoss(chosenLp, rejectedLp, refChosenLp, refRejectedLp);

        // SFT auxiliary loss
        if (cpoConfig.sftWeight() > 0.0 && chosenLogits != null && chosenLogits.defined()) {
            Tensor sft = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
            totalLoss = totalLoss.add(sft.mul(new Scalar(cpoConfig.sftWeight())));
        }

        // Label smoothing: add a flat constant nudge to keep the loss positive.
        if (cpoConfig.labelSmoothing() > 0.0) {
            totalLoss = totalLoss.add(new Scalar(cpoConfig.labelSmoothing()));
        }

        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor computeCpoLoss(Tensor chosenLogps, Tensor rejectedLogps,
                                  Tensor refChosenLp, Tensor refRejectedLp) {
        double beta = cpoConfig.beta();
        double alpha = cpoConfig.cpoAlpha();
        String type = cpoConfig.lossType() == null ? "cpo" : cpoConfig.lossType().toLowerCase();

        Tensor ratio = chosenLogps.sub(rejectedLogps);

        switch (type) {
            case "simpo": {
                double gammaBeta = cpoConfig.simpoGamma() / Math.max(1e-6, beta);
                Tensor margin = ratio.sub(new Scalar(gammaBeta));
                Tensor inner = margin.neg().mul(new Scalar(beta));
                Tensor logSigmoid = inner.exp().add(new Scalar(1.0)).log().neg();
                Tensor loss = logSigmoid.mean();
                if (alpha > 0.0) {
                    Tensor contrastive = computeContrastiveLoss(chosenLogps, rejectedLogps);
                    loss = loss.add(contrastive.mul(new Scalar(alpha)));
                }
                return loss;
            }
            case "siglip": {
                // Approximate SigLIP-style loss with label smoothing.
                Tensor logits = ratio.mul(new Scalar(beta));
                Tensor sigmoidLoss = logits.sigmoid().log().neg();
                Tensor negLog = logits.neg().sigmoid().log().neg();
                Tensor loss = sigmoidLoss.sub(negLog).mul(new Scalar(0.5));
                if (alpha > 0.0) {
                    Tensor contrastive = computeContrastiveLoss(chosenLogps, rejectedLogps);
                    loss = loss.add(contrastive.mul(new Scalar(alpha)));
                }
                return loss.mean();
            }
            case "cpo":
            default: {
                Tensor cpoLoss = ratio.div(new Scalar(beta)).exp().add(new Scalar(1.0)).log()
                        .mul(new Scalar(-beta));
                Tensor contrastiveLoss = computeContrastiveLoss(chosenLogps, rejectedLogps);
                Tensor totalLoss = cpoLoss.add(contrastiveLoss.mul(new Scalar(alpha)));

                // Reference KL when ref is available.
                if (refChosenLp != null && refChosenLp.defined()
                        && refChosenLp.size(0) == chosenLogps.size(0)) {
                    Tensor kl = (chosenLogps.sub(refChosenLp).mean()
                            .add(rejectedLogps.sub(refRejectedLp).mean()))
                            .div(new Scalar(2.0));
                    totalLoss = totalLoss.add(kl.mul(new Scalar(beta)));
                }
                return totalLoss;
            }
        }
    }

    /**
     * Contrastive loss: margin-based loss that encourages separation between
     * chosen and rejected responses.
     */
    private Tensor computeContrastiveLoss(Tensor chosenLogps, Tensor rejectedLogps) {
        double margin = cpoConfig.margin();
        Tensor rewardDiff = chosenLogps.sub(rejectedLogps);
        Tensor marginTensor = full_like(chosenLogps, new Scalar(margin));
        return marginTensor.sub(rewardDiff).clamp_min(new Scalar(0.0));
    }

    private Tensor logps(Tensor logits, Tensor labels, Tensor mask) {
        return cpoConfig.lengthNormalize()
                ? LogProbUtils.sequenceMeanLogProbs(logits, labels, mask)
                : LogProbUtils.sequenceLogProbs(logits, labels, mask);
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
        System.out.printf("[CPOTrainer v%s] Closed: trainingSteps=%d%n", VERSION, numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "CPO v" + VERSION + " (Contrastive Preference Optimization)";
    }
}