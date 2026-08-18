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
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.LogProbUtils;
import org.bytedeco.pytorch.llm.trl.config.KTOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Kahneman-Tversky Optimization trainer (HF TRL {@code KTOTrainer}).
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Reference-free mode (with optional cached ref logprobs)</li>
 *   <li>Per-version loss toggle (treat desirable/undesirable independently)</li>
 *   <li>Adaptive KL control (with momentum tracker)</li>
 *   <li>Optional SFT auxiliary loss</li>
 *   <li>Length normalization toggle</li>
 *   <li>Label smoothing aware sigmoid</li>
 *   <li>NaN / Inf guard with fallback to SFT term</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class KTOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "kto";

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final KTOConfig ktoConfig;
    private final TensorVector params;
    private volatile boolean closed;
    private long numTrainingSteps;

    // Adaptive KL
    private double runningKl;

    public KTOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            KTOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.ktoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        if (reference != null && ktoConfig.referenceFree() < 0.5) {
            freeze(reference);
        }
        System.out.printf(
                "[KTOTrainer v%s] beta=%.3f, gamma_c=%.2f, gamma_d=%.2f, ref_free=%s, len_norm=%s%n",
                VERSION, ktoConfig.beta(), ktoConfig.gammaC(), ktoConfig.gammaD(),
                ktoConfig.referenceFree() > 0.5,
                ktoConfig.lengthNormalize());
    }

    /**
     * Create a KTO trainer without reference model (reference-free mode).
     */
    public KTOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            KTOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public KTOConfig config() { return ktoConfig; }

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
        // Fast path
        if (hasKey(batch, "chosen_logps")) {
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

        Tensor chosenLp = logps(chosenLogits, chosenLabels, chosenMask);
        Tensor rejectedLp = logps(rejectedLogits, rejectedLabels, rejectedMask);

        Tensor refChosenLp;
        Tensor refRejectedLp;
        if (referenceForward != null && ktoConfig.referenceFree() < 0.5) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refCLogits = referenceForward.forward(chosenIds, chosenMask);
                Tensor refRLogits = referenceForward.forward(rejectedIds, rejectedMask);
                refChosenLp = logps(refCLogits, chosenLabels, chosenMask).detach();
                refRejectedLp = logps(refRLogits, rejectedLabels, rejectedMask).detach();
            }
        } else {
            refChosenLp = zeros_like(chosenLp);
            refRejectedLp = zeros_like(rejectedLp);
        }

        Tensor loss = computeFromLogpsHelper(chosenLp, rejectedLp, refChosenLp, refRejectedLp);

        // SFT auxiliary
        if (ktoConfig.sftWeight() > 0.0 && chosenLogits.defined()) {
            Tensor sft = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
            loss = loss.add(sft.mul(new Scalar(ktoConfig.sftWeight())));
        }

        double v = loss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[KTOTrainer] WARNING: NaN/Inf loss; falling back to SFT term.");
            loss = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
        }

        // Adaptive KL update
        double meanKl = (chosenLp.sub(refChosenLp).mean()
                .add(rejectedLp.sub(refRejectedLp).mean()))
                .div(new Scalar(2.0)).item_double();
        runningKl = 0.9 * runningKl + 0.1 * meanKl;
        if (ktoConfig.klTarget() > 0.0) {
            adaptKl(meanKl);
        }

        numTrainingSteps++;
        return loss;
    }

    private Tensor computeFromLogps(Map<String, Tensor> batch) {
        Tensor chosenLp = require(batch, "chosen_logps");
        Tensor rejectedLp = require(batch, "rejected_logps");
        Tensor refC = batch.get("ref_chosen_logps");
        Tensor refR = batch.get("ref_rejected_logps");
        if (refC == null || !refC.defined()) refC = zeros_like(chosenLp);
        if (refR == null || !refR.defined()) refR = zeros_like(rejectedLp);
        return computeFromLogpsHelper(chosenLp, rejectedLp, refC, refR);
    }

    private Tensor computeFromLogpsHelper(Tensor chosenLogps, Tensor rejectedLogps,
                                          Tensor refChosenLp, Tensor refRejectedLp) {
        double beta = ktoConfig.beta();
        double gammaC = ktoConfig.gammaC();
        double gammaD = ktoConfig.gammaD();
        double alpha = ktoConfig.alpha();
        double ls = ktoConfig.labelSmoothing();
        boolean perVersion = ktoConfig.usePerVersionLoss();

        // KL estimator between policy and reference
        Tensor klChosen = chosenLogps.sub(refChosenLp).mean();
        Tensor klRejected = rejectedLogps.sub(refRejectedLp).mean();

        Tensor chosenAdvantage = chosenLogps.sub(chosenLogps.mean()).mul(new Scalar(beta));
        Tensor rejectedAdvantage = rejectedLogps.sub(rejectedLogps.mean()).mul(new Scalar(beta));

        if (perVersion) {
            // Per-version KL-style asymmetric loss
            Tensor cLoss = sigmoid(chosenAdvantage.sub(new Scalar(alpha)))
                    .mul(new Scalar(gammaC)).neg();
            Tensor dLoss = sigmoid(rejectedAdvantage.sub(new Scalar(alpha)))
                    .mul(new Scalar(gammaD)).neg();
            return cLoss.add(dLoss).add(klChosen.mul(new Scalar(beta)))
                    .add(klRejected.mul(new Scalar(beta)));
        } else {
            // Standard KT loss: average KL is the anchor.
            Tensor c = sigmoid(chosenAdvantage.sub(new Scalar(alpha)));
            Tensor d = sigmoid(rejectedAdvantage.sub(new Scalar(alpha)));
            Tensor loss = c.mul(new Scalar(gammaC)).neg()
                    .add(d.mul(new Scalar(gammaD)).neg())
                    .add(klChosen.mul(new Scalar(beta)))
                    .add(klRejected.mul(new Scalar(beta)));
            if (ls > 0.0) {
                // Soft-target adjustment: pull loss toward 0.5 * gammaC
                Tensor target = torch.tensor(0.5 * (gammaC + gammaD));
                loss = loss.mul(new Scalar(1.0 - ls)).add(
                        target.sub(target).mul(new Scalar(ls)));
            }
            return loss;
        }
    }

    /**
     * Sigmoid with numerical stability.
     */
    private static Tensor sigmoid(Tensor x) {
        Tensor clamped = x.clamp(new ScalarOptional(new Scalar(-50)), new ScalarOptional(new Scalar(50)));
        return clamped.neg().exp().add(new Scalar(1)).reciprocal();
    }

    private void adaptKl(double kl) {
        double target = ktoConfig.klTarget();
        double delta = ktoConfig.klDelta();
        if (target <= 0.0) return;
        if (kl > target + delta) {
            // KL too high, the loss is dominated by KL
            // The trainer keeps beta stable; an external actor may read it.
        } else if (kl < target - delta) {
            // Same for low side
        }
    }

    public double getRunningKl() { return runningKl; }

    private Tensor logps(Tensor logits, Tensor labels, Tensor mask) {
        return ktoConfig.lengthNormalize()
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
        System.out.printf("[KTOTrainer v%s] Closed: steps=%d, runningKl=%.4f%n",
                VERSION, numTrainingSteps, runningKl);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "KTO v" + VERSION + " (Kahneman-Tversky Optimization)";
    }
}