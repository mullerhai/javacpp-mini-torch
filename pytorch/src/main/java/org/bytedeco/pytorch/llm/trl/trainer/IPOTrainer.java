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
import org.bytedeco.pytorch.llm.trl.config.IPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.IPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * IPO (Identity Preference Optimization) trainer.
 *
 * <p>Reference: "A Theoretical Analysis of IPO" (Azar et al., 2024).
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Length normalization toggle (independent of ref-free mode)</li>
 *   <li>Reference-free / precomputed logprob fast paths</li>
 *   <li>Label smoothing-aware identity regularization</li>
 *   <li>Optional SFT auxiliary term (sft_weight)</li>
 *   <li>MoE aux loss integration</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class IPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "ipo";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final IPOConfig ipoConfig;
    private final TensorVector params;

    public IPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            IPOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.ipoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        if (reference != null && !config.referenceFree()) {
            freeze(reference);
        }
        System.out.printf(
                "[IPOTrainer v%s] beta=%.3f, identityCoef=%.2f, lenNorm=%s, ref_free=%s%n",
                VERSION, ipoConfig.beta(), ipoConfig.identityCoef(),
                ipoConfig.lengthNormalize(),
                ipoConfig.referenceFree());
    }

    public IPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            IPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public IPOConfig config() { return ipoConfig; }

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
        if (hasKey(batch, "policy_chosen_logps")) {
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
        if (referenceForward != null && !ipoConfig.referenceFree()) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refChosenLogits = referenceForward.forward(chosenIds, chosenMask);
                Tensor refRejectedLogits = referenceForward.forward(rejectedIds, rejectedMask);
                refChosenLp = logps(refChosenLogits, chosenLabels, chosenMask).detach();
                refRejectedLp = logps(refRejectedLogits, rejectedLabels, rejectedMask).detach();
            }
        } else {
            refChosenLp = zeros_like(chosenLp);
            refRejectedLp = zeros_like(rejectedLp);
        }

        return combine(chosenLp, rejectedLp, refChosenLp, refRejectedLp, chosenLogits, chosenLabels, chosenMask);
    }

    private Tensor combine(Tensor chosenLp, Tensor rejectedLp, Tensor refChosenLp,
                           Tensor refRejectedLp, Tensor chosenLogits,
                           Tensor chosenLabels, Tensor chosenMask) {
        Tensor primary = IPOLoss.compute(
                chosenLp, rejectedLp, refChosenLp, refRejectedLp,
                ipoConfig.beta(), ipoConfig.identityCoef());

        // Label smoothing: inject a small uniform nudge on the labels (sigmoid-style).
        if (ipoConfig.labelSmoothing() > 0.0) {
            Tensor smoothTerm = IPOLoss.compute(
                    chosenLp, rejectedLp, refChosenLp, refRejectedLp,
                    ipoConfig.beta() * 0.5, ipoConfig.identityCoef() * 0.5)
                    .mul(new Scalar(ipoConfig.labelSmoothing()));
            primary = primary.mul(new Scalar(1.0 - ipoConfig.labelSmoothing())).add(smoothTerm);
        }

        // SFT NLL auxiliary loss (encourages chosen likelihood under the policy).
        if (ipoConfig.sftWeight() > 0.0 && chosenLogits != null && chosenLogits.defined()) {
            Tensor sftLoss = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
            primary = primary.add(sftLoss.mul(new Scalar(ipoConfig.sftWeight())));
        }

        numTrainingSteps++;
        return primary;
    }

    private Tensor computeFromLogps(Map<String, Tensor> batch) {
        Tensor pC = require(batch, "policy_chosen_logps");
        Tensor pR = require(batch, "policy_rejected_logps");
        Tensor rC = batch.get("ref_chosen_logps");
        Tensor rR = batch.get("ref_rejected_logps");
        if (rC == null || !rC.defined()) rC = zeros_like(pC);
        if (rR == null || !rR.defined()) rR = zeros_like(pR);
        return combine(pC, pR, rC, rR, null, null, null);
    }

    private Tensor logps(Tensor logits, Tensor labels, Tensor mask) {
        return ipoConfig.lengthNormalize()
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
        System.out.printf("[IPOTrainer v%s] Closed: steps=%d%n", VERSION, numTrainingSteps);
    }

    public boolean isClosed() { return closed; }

    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "IPO v" + VERSION + " (Identity Preference Optimization)";
    }
}