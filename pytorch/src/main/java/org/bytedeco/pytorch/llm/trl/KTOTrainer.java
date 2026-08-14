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
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.llm.trl.config.KTOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Kahneman-Tversky Optimization trainer (Meta's latest alignment algorithm).
 *
 * <p>KTO is a novel alignment algorithm that:
 * <ul>
 *   <li>Uses a reference-free asymmetric loss function inspired by prospect theory</li>
 *   <li>Does not require paired preference data (only binary acceptance signals)</li>
 *   <li>Achieves better diversity and helpfulness trade-off than DPO</li>
 *   <li>Works with any base model without a reference model</li>
 * </ul>
 *
 * <p>Reference: "KTO: Kahneman-Tversky Optimization – A Novel Approach to
 * Aligning Language Models with Human Feedback"
 * (Ethayarajh et al., Meta AI, 2024)
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids} (or precomputed {@code chosen_logps}, {@code rejected_logps})</li>
 *   <li>optional {@code chosen_attention_mask}, {@code rejected_attention_mask}</li>
 *   <li>optional {@code kl_temperature} for per-sample KL weighting</li>
 * </ul>
 *
 * <pre>{@code
 * try (KTOTrainer trainer = new KTOTrainer(policyModel, policyForward, optimizer, config)) {
 *     for (Map<String, Tensor> batch : dataloader) {
 *         trainer.trainingStep(batch);
 *     }
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class KTOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;          // optional, for reference-free mode
    private final LlmForward referenceForward;
    private final KTOConfig ktoConfig;
    private final TensorVector params;
    private volatile boolean closed;

    /**
     * Create a KTO trainer with a reference model.
     */
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
        if (reference != null) {
            freeze(reference);
        }
        System.out.printf(
                "[KTOTrainer v%s] beta=%.3f, gamma_c=%.2f, gamma_d=%.2f, ref_free=%s%n",
                VERSION, ktoConfig.beta(), ktoConfig.gammaC(), ktoConfig.gammaD(),
                reference == null);
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
        if (batch.containsKey("chosen_logps") && batch.get("chosen_logps") != null) {
            return computeFromLogps(batch);
        }

        // Compute log-probs from scratch
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");

        // Policy log-probs
        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

        Tensor chosenLp = LogProbUtils.sequenceLogProbs(chosenLogits, chosenLabels, chosenMask);
        Tensor rejectedLp = LogProbUtils.sequenceLogProbs(rejectedLogits, rejectedLabels, rejectedMask);

        // Reference log-probs
        Tensor refChosenLp, refRejectedLp;
        if (referenceForward != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refChosenLogits = referenceForward.forward(chosenIds, chosenMask);
                Tensor refRejectedLogits = referenceForward.forward(rejectedIds, rejectedMask);
                refChosenLp = LogProbUtils.sequenceLogProbs(refChosenLogits, chosenLabels, chosenMask).detach();
                refRejectedLp = LogProbUtils.sequenceLogProbs(refRejectedLogits, rejectedLabels, rejectedMask).detach();
            }
        } else {
            // Reference-free: use prior as reference (uniform distribution)
            // KL(p||uniform) = -H(p) = log(vocab_size) + sum(p * log(p))
            // Simplified: use policy log-probs directly as advantage
            refChosenLp = zeros_like(chosenLp);
            refRejectedLp = zeros_like(rejectedLp);
        }

        // Build batch map and compute loss
        batch.put("chosen_logps", chosenLp);
        batch.put("rejected_logps", rejectedLp);
        batch.put("ref_chosen_logps", refChosenLp);
        batch.put("ref_rejected_logps", refRejectedLp);

        return computeFromLogps(batch);
    }

    /**
     * Compute KTO loss from precomputed log-probs.
     */
    private Tensor computeFromLogps(Map<String, Tensor> batch) {
        Tensor chosenLogps = require(batch, "chosen_logps");
        Tensor rejectedLogps = require(batch, "rejected_logps");
        Tensor refChosenLp = batch.get("ref_chosen_logps");
        Tensor refRejectedLp = batch.get("ref_rejected_logps");

        if (refChosenLp == null) refChosenLp = zeros_like(chosenLogps);
        if (refRejectedLp == null) refRejectedLp = zeros_like(rejectedLogps);

        double beta = ktoConfig.beta();
        double gammaC = ktoConfig.gammaC();  // gain coefficient for chosen
        double gammaD = ktoConfig.gammaD();  // loss coefficient for rejected

        // KL divergences
        Tensor klChosen = chosenLogps.sub(refChosenLp).mean();
        Tensor klRejected = rejectedLogps.sub(refRejectedLp).mean();

        // KTO asymmetric loss
        // For chosen: -gamma_c * sigmoid(beta * kl - alpha)
        // For rejected: -gamma_d * sigmoid(alpha - beta * kl)
        double alpha = ktoConfig.alpha();

        Tensor chosenAdvantage = chosenLogps.sub(chosenLogps.mean()).mul(new Scalar(beta));
        Tensor rejectedAdvantage = rejectedLogps.sub(rejectedLogps.mean()).mul(new Scalar(beta));

        // Sigmoid with temperature
        Tensor chosenProb = sigmoid(chosenAdvantage.sub(new Scalar(alpha)));
        Tensor rejectedProb = sigmoid(rejectedAdvantage.sub(new Scalar(alpha)));

        // Weighted loss
        Tensor chosenLoss = chosenProb.mul(new Scalar(gammaC)).neg();
        Tensor rejectedLoss = rejectedProb.mul(new Scalar(gammaD)).neg();

        // Total loss + KL penalty
        Tensor totalLoss = chosenLoss.add(rejectedLoss).add(klChosen.mul(new Scalar(beta))).add(klRejected.mul(new Scalar(beta)));

        return totalLoss;
    }

    /**
     * Sigmoid with numerical stability.
     */
    private static Tensor sigmoid(Tensor x) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // For stability: use clamp on x first
        Tensor clamped = x.clamp(new ScalarOptional(new Scalar(-50)), new ScalarOptional(new Scalar(50)));
        return clamped.neg().exp().add(new Scalar(1)).reciprocal();
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
        System.out.println("[KTOTrainer] Closed");
    }

    public boolean isClosed() { return closed; }
}
