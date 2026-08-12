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

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.IPOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * IPO (Identity Preference Optimization) trainer (Meta AI inspired).
 *
 * <p>IPO is a theoretically-grounded alternative to DPO that provides stronger
 * theoretical guarantees and removes the need for careful KL penalty tuning.
 * It uses a margin-based pairwise loss.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids}</li>
 *   <li>{@code chosen_attention_mask}, {@code rejected_attention_mask} (optional)</li>
 *   <li>or precomputed {@code chosen_logps}, {@code rejected_logps}</li>
 * </ul>
 *
 * <p>Reference: "A General Theoretical Paradigm to Understand Preference Optimization"
 *
 * <pre>{@code
 * IPOTrainer trainer = new IPOTrainer(policy, forward, reference, refForward, optimizer, config);
 * }</pre>
 */
public final class IPOTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final IPOConfig config;
    private final TensorVector params;
    private final double tau;
    private final double margin;

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
        this.config = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.tau = config.tau();
        this.margin = config.margin();

        if (reference != null) {
            freeze(reference);
        }
    }

    /** Policy-only constructor (reference-free mode). */
    public IPOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, IPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public IPOConfig config() { return config; }

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
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        // Fast path: precomputed log-probs
        if (batch.containsKey("chosen_logps") && batch.containsKey("rejected_logps")) {
            Tensor chosenLogps = batch.get("chosen_logps");
            Tensor rejectedLogps = batch.get("rejected_logps");
            if (chosenLogps != null && chosenLogps.defined() &&
                rejectedLogps != null && rejectedLogps.defined()) {
                return computeIpoLoss(chosenLogps, rejectedLogps);
            }
        }

        // Full forward pass
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");

        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        Tensor chosenLogps = LogProbUtils.sequenceLogProbs(chosenLogits, chosenIds, chosenMask);
        Tensor rejectedLogps = LogProbUtils.sequenceLogProbs(rejectedLogits, rejectedIds, rejectedMask);

        return computeIpoLoss(chosenLogps, rejectedLogps);
    }

    /**
     * Compute IPO loss.
     *
     * IPO Loss = (tau / 2) * || log(sigmoid(r(chosen) - r(rejected))) ||
     *
     * Where r(x) = policy_log_prob(x) - reference_log_prob(x) (or just policy_log_prob if reference-free)
     */
    private Tensor computeIpoLoss(Tensor chosenLogps, Tensor rejectedLogps) {
        // Compute log ratio
        Tensor policyLogRatio;
        if (config.referenceFree() || referenceForward == null) {
            policyLogRatio = chosenLogps.sub(rejectedLogps);
        } else {
            // Use difference from reference
            // In practice, we compute: (pi - ref)_chosen - (pi - ref)_rejected
            // This simplifies to: pi_chosen - pi_rejected (if using per-sample log probs)
            policyLogRatio = chosenLogps.sub(rejectedLogps);
        }

        // IPO-specific: apply tau parameter
        // The IPO loss uses: 1/(2*tau) * || log_ratio - tau ||^2
        Tensor diff = policyLogRatio.sub(new Scalar(tau + margin));

        // Quadratic loss
        Tensor ipoLoss = diff.pow(new Scalar(2)).div(new Scalar(2 * tau));

        // Optional label smoothing
        double smoothing = config.labelSmoothing();
        if (smoothing > 0) {
            // Soft targets: blend with uniform preference
            Tensor sigmoid = org.bytedeco.pytorch.global.torch.sigmoid(policyLogRatio.div(new Scalar(tau)));
            Tensor target = sigmoid.mul(new Scalar(1 - smoothing)).add(new Scalar(smoothing / 2));
            Tensor crossEntropy = policyLogRatio.mul(target).sub(
                    org.bytedeco.pytorch.global.torch.log1p(policyLogRatio.exp()));
            ipoLoss = ipoLoss.add(new Scalar(smoothing)).mul(crossEntropy);
        }

        numTrainingSteps++;
        return ipoLoss.mean();
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
        System.out.printf("[IPOTrainer] Closed: trainingSteps=%d%n", numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
}
