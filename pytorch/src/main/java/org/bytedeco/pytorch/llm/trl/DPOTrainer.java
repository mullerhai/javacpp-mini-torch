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
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Direct Preference Optimization trainer (HF TRL {@code DPOTrainer}).
 *
 * <p>Supports the full DPO loss surface:
 * <ul>
 *   <li>{@code sigmoid}, {@code robust}, {@code hinge}, {@code ipo},
 *       {@code exo_pair}, {@code nca_pair}, {@code sppo_huber}, {@code sppo_eps},
 *       {@code orpo}, {@code apos}, {@code sft}</li>
 *   <li>Reference-free (zero ref logps) and reference-mixup (alpha/beta) variants</li>
 *   <li>SFT aux loss with {@code sft_weight}</li>
 *   <li>RPO alpha (DPO + SFT combination)</li>
 *   <li>Length normalization</li>
 *   <li>Auxiliary MoE load-balancing loss</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final DPOConfig dpoConfig;
    private final TensorVector params;
    private final boolean lengthNormalize;

    public DPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            DPOConfig config) {
        this(policy, policyForward, reference, referenceForward, optimizer, config, false);
    }

    public DPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            DPOConfig config,
            boolean lengthNormalize) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.dpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.lengthNormalize = lengthNormalize || config.lengthNormalize();
        if (reference != null && !config.referenceFree()) {
            freeze(reference);
        }
    }

    public DPOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, DPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public DPOConfig dpoConfig() { return dpoConfig; }

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
        // ---- Fast path: precomputed log-probs ----
        if (hasKey(batch, "policy_chosen_logps")) {
            Tensor pC = require(batch, "policy_chosen_logps");
            Tensor pR = require(batch, "policy_rejected_logps");
            Tensor rC = orElse(batch.get("ref_chosen_logps"), null);
            Tensor rR = orElse(batch.get("ref_rejected_logps"), null);
            boolean refFree = dpoConfig.referenceFree()
                    || rC == null || !rC.defined();
            if (refFree) {
                rC = zerosLike(pC);
                rR = zerosLike(pR);
            } else if (dpoConfig.refModelMixupAlpha() > 0.0 && hasKey(batch, "ref_chosen_logps_mixed")) {
                Tensor mixC = batch.get("ref_chosen_logps_mixed");
                Tensor mixR = batch.get("ref_rejected_logps_mixed");
                if (mixC != null && mixC.defined()) rC = mixC;
                if (mixR != null && mixR.defined()) rR = mixR;
            }
            return compute(pC, pR, rC, rR, null, null, null);
        }

        // ---- Online path ----
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

        Tensor policyChosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor policyRejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        Tensor policyChosenLp = lengthNormalize
                ? LogProbUtils.sequenceMeanLogProbs(policyChosenLogits, chosenLabels, chosenMask)
                : LogProbUtils.sequenceLogProbs(policyChosenLogits, chosenLabels, chosenMask);
        Tensor policyRejectedLp = lengthNormalize
                ? LogProbUtils.sequenceMeanLogProbs(policyRejectedLogits, rejectedLabels, rejectedMask)
                : LogProbUtils.sequenceLogProbs(policyRejectedLogits, rejectedLabels, rejectedMask);

        Tensor refChosenLp = null;
        Tensor refRejectedLp = null;
        Tensor refChosenLpOrig = null;
        Tensor refRejectedLpOrig = null;
        if (needsRef()) {
            if (referenceForward != null) {
                try (NoGradGuard guard = new NoGradGuard()) {
                    Tensor refChosenLogits = referenceForward.forward(chosenIds, chosenMask);
                    Tensor refRejectedLogits = referenceForward.forward(rejectedIds, rejectedMask);
                    refChosenLp = lengthNormalize
                            ? LogProbUtils.sequenceMeanLogProbs(refChosenLogits, chosenLabels, chosenMask).detach()
                            : LogProbUtils.sequenceLogProbs(refChosenLogits, chosenLabels, chosenMask).detach();
                    refRejectedLp = lengthNormalize
                            ? LogProbUtils.sequenceMeanLogProbs(refRejectedLogits, rejectedLabels, rejectedMask).detach()
                            : LogProbUtils.sequenceLogProbs(refRejectedLogits, rejectedLabels, rejectedMask).detach();
                    refChosenLpOrig = refChosenLp;
                    refRejectedLpOrig = refRejectedLp;
                }
            } else {
                refChosenLp = zerosLike(policyChosenLp);
                refRejectedLp = zerosLike(policyRejectedLp);
            }
        }

        // Reference-mixup: blend policy & ref logps to form a noisy reference.
        Tensor refChosenMixed = null;
        Tensor refRejectedMixed = null;
        if (dpoConfig.refModelMixupAlpha() > 0.0 && refChosenLp != null && refRejectedLp != null) {
            double alpha = dpoConfig.refModelMixupAlpha();
            double beta = dpoConfig.refModelMixupBeta();
            double rho = dpoConfig.mixRho();
            // ref_mixed = rho * ref + (1-rho) * (alpha * policy + (1-alpha) * ref)
            Tensor refChosenMixedTmp = refChosenLp.mul(new Scalar(rho))
                    .add(policyChosenLp.mul(new Scalar(alpha)).add(refChosenLp.mul(new Scalar(1.0 - alpha)))
                            .mul(new Scalar(1.0 - rho)));
            Tensor refRejectedMixedTmp = refRejectedLp.mul(new Scalar(rho))
                    .add(policyRejectedLp.mul(new Scalar(alpha)).add(refRejectedLp.mul(new Scalar(1.0 - alpha)))
                            .mul(new Scalar(1.0 - rho)));
            refChosenMixed = refChosenMixedTmp;
            refRejectedMixed = refRejectedMixedTmp;
        }

        // SFT auxiliary loss: mean NLL on the chosen examples (requires logits).
        Tensor sftLoss = null;
        if (dpoConfig.sftWeight() > 0.0 || dpoConfig.rpoAlpha() > 0.0
                || "sft".equalsIgnoreCase(dpoConfig.lossType())) {
            sftLoss = sftNll(policyChosenLogits, chosenLabels, chosenMask);
        }

        return compute(policyChosenLp, policyRejectedLp,
                refChosenLp != null ? refChosenLp : zerosLike(policyChosenLp),
                refRejectedLp != null ? refRejectedLp : zerosLike(policyRejectedLp),
                sftLoss, refChosenMixed, refRejectedMixed);
    }

    /** Whether the active loss requires a reference log-prob vector. */
    private boolean needsRef() {
        if (dpoConfig.referenceFree()) return false;
        return dpoConfig.requiresReferenceModel();
    }

    /** Unified loss dispatcher. */
    private Tensor compute(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                           Tensor sftLoss,
                           Tensor refChosenMixed, Tensor refRejectedMixed) {
        String type = dpoConfig.lossType() == null ? "sigmoid" : dpoConfig.lossType().toLowerCase();
        double beta = dpoConfig.beta();

        // Use mixed ref logps when ref_mixup_alpha is enabled.
        Tensor useRC = refChosenMixed != null ? refChosenMixed : rC;
        Tensor useRR = refRejectedMixed != null ? refRejectedMixed : rR;

        switch (type) {
            case "sigmoid":
                return DPOLoss.computeSigmoid(pC, pR, useRC, useRR, beta, dpoConfig.labelSmoothing());
            case "robust":
                // Robust DPO: regular sigmoid loss + KL to the reference.
                Tensor sigmoidLoss = DPOLoss.computeSigmoid(pC, pR, useRC, useRR, beta, dpoConfig.labelSmoothing());
                Tensor kl = (useRC.sub(pC).mean()).add(useRR.sub(pR).mean()).mul(new Scalar(0.5 * beta));
                return sigmoidLoss.add(kl);
            case "hinge":
                return DPOLoss.computeHinge(pC, pR, useRC, useRR, beta);
            case "ipo":
                return DPOLoss.computeIPO(pC, pR, useRC, useRR, beta);
            case "exo_pair":
                // EXO_Pair: KL(pi || ref) using exact log-ratios; falls back to DPOLoss.
                return DPOLoss.computeExoPair(pC, pR, useRC, useRR, beta, dpoConfig.gamma());
            case "nca_pair":
                return DPOLoss.computeNcaPair(pC, pR, useRC, useRR, beta);
            case "sppo_huber":
                return DPOLoss.computeSpppoHuber(pC, pR, useRC, useRR, beta, dpoConfig.labelSmoothing());
            case "sppo_eps":
                return DPOLoss.computeSpppoEps(pC, pR, useRC, useRR, beta, dpoConfig.labelSmoothing());
            case "orpo":
                return DPOLoss.computeORPO(pC, pR, beta, dpoConfig.lengthNormalize());
            case "apos":
                return DPOLoss.computeApos(pC, pR, useRC, useRR, beta, dpoConfig.gamma());
            case "sft":
                return sftLoss != null ? sftLoss : DPOLoss.computeSigmoid(pC, pR, useRC, useRR, beta, dpoConfig.labelSmoothing());
            default:
                return DPOLoss.compute(pC, pR, useRC, useRR, beta, type);
        }
    }

    /** Compute SFT-style NLL on the chosen response. */
    private Tensor sftNll(Tensor logits, Tensor labels, Tensor mask) {
        return DPOLoss.sftNll(logits, labels, mask);
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

    private static Tensor zerosLike(Tensor t) {
        return org.bytedeco.pytorch.global.torch.zeros_like(t);
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
        System.out.printf("[DPOTrainer] Closed: trainingSteps=%d%n", numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
}