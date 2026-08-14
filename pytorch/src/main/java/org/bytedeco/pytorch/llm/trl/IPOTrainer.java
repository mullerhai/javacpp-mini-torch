/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.trl.config.IPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.IPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.TensorVector;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * IPO (Identity Preference Optimization) trainer.
 *
 * <p>IPO is a theoretically grounded preference optimization algorithm that
 * adds a regularization term to DPO, providing:
 * <ul>
 *   <li>Provable convergence to optimal policy</li>
 *   <li>Better finite-sample bounds</li>
 *   <li>Reduced reward hacking</li>
 *   <li>More stable training</li>
 * </ul>
 *
 * <p>The key difference from DPO is the squared error loss:
 * <pre>
 *   L = E[(π(y_w) - π(y_l) - 1/(2β))²]
 * </pre>
 *
 * <p>Reference: "A Theoretical Analysis of IPO" (Azar et al., 2024)
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids}</li>
 *   <li>optional {@code chosen_attention_mask}, {@code rejected_attention_mask}</li>
 *   <li>or precomputed {@code policy_chosen_logps} / {@code policy_rejected_logps}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class IPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";
    public static final String ALGORITHM_ID = "ipo";

    private volatile boolean closed;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final IPOConfig ipoConfig;
    private final TensorVector params;

    /**
     * Create IPO trainer with reference model.
     */
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

        if (reference != null) {
            freeze(reference);
        }

        System.out.printf(
                "[IPOTrainer v%s] beta=%.3f, identityCoef=%.2f%n",
                VERSION, ipoConfig.beta(), ipoConfig.identityCoef());
    }

    /** Policy-only constructor (reference-free). */
    public IPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            IPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    // ==================== BaseTrainer Overrides ====================

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

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
        // Fast path: precomputed log-probs
        if (batch.containsKey("policy_chosen_logps")
                && batch.get("policy_chosen_logps") != null
                && batch.get("policy_chosen_logps").defined()) {
            return computeFromLogps(batch);
        }

        // Compute log-probs from input
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

        // Forward pass
        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        // Get log-probs
        Tensor chosenLp = logps(chosenLogits, chosenLabels, chosenMask);
        Tensor rejectedLp = logps(rejectedLogits, rejectedLabels, rejectedMask);

        // Reference log-probs
        Tensor refChosenLp, refRejectedLp;
        if (referenceForward != null) {
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

        // Compute IPO loss
        return IPOLoss.compute(
                chosenLp, rejectedLp,
                refChosenLp, refRejectedLp,
                ipoConfig.beta(),
                ipoConfig.identityCoef());
    }

    /**
     * Compute IPO loss from precomputed log-probs.
     */
    private Tensor computeFromLogps(Map<String, Tensor> batch) {
        Tensor pC = require(batch, "policy_chosen_logps");
        Tensor pR = require(batch, "policy_rejected_logps");
        Tensor rC = batch.get("ref_chosen_logps");
        Tensor rR = batch.get("ref_rejected_logps");

        if (rC == null || !rC.defined()) rC = zeros_like(pC);
        if (rR == null || !rR.defined()) rR = zeros_like(pR);

        return IPOLoss.compute(pC, pR, rC, rR,
                ipoConfig.beta(), ipoConfig.identityCoef());
    }

    private Tensor logps(Tensor logits, Tensor labels, Tensor mask) {
        return ipoConfig.lengthNormalize()
                ? LogProbUtils.sequenceMeanLogProbs(logits, labels, mask)
                : LogProbUtils.sequenceLogProbs(logits, labels, mask);
    }

    // ==================== Utility Methods ====================

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

    // ==================== Getters ====================

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public IPOConfig config() { return ipoConfig; }

//    @Override
    public String algorithm() {
        return ALGORITHM_ID;
    }

    public String algorithmName() {
        return "IPO (Identity Preference Optimization)";
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        super.close();
        System.out.printf("[IPOTrainer v%s] Closed%n", VERSION);
    }

    public boolean isClosed() { return closed; }
}
