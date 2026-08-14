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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.CPOConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.exp;
import static org.bytedeco.pytorch.global.torch.log;

/**
 * Contrastive Preference Optimization trainer (Meta AI inspired).
 *
 * <p>CPO combines DPO-style preference optimization with contrastive loss to
 * better separate chosen and rejected responses. The contrastive component
 * uses a margin-based loss that pushes chosen responses away from rejected ones.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids}</li>
 *   <li>{@code chosen_attention_mask}, {@code rejected_attention_mask} (optional)</li>
 *   <li>or precomputed {@code chosen_logps}, {@code rejected_logps} as {@code [B]}</li>
 * </ul>
 *
 * <p>Loss = CPO_loss + alpha * contrastive_loss
 *
 * <p>Reference: "Contrastive Preference Learning" (Meta AI)
 */
public final class CPOTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final CPOConfig cpoConfig;
    private final TensorVector params;
    private final double beta;

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
        this.beta = config.beta();
        if (reference != null) {
            freeze(reference);
        }
    }

    /** Policy-only constructor (no reference model). */
    public CPOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, CPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public CPOConfig cpoConfig() { return cpoConfig; }

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
        // Fast path: precomputed log-probs
        if (batch.containsKey("chosen_logps") && batch.containsKey("rejected_logps")) {
            Tensor chosenLogps = batch.get("chosen_logps");
            Tensor rejectedLogps = batch.get("rejected_logps");
            if (chosenLogps != null && chosenLogps.defined() &&
                rejectedLogps != null && rejectedLogps.defined()) {
                return computeCpoLoss(chosenLogps, rejectedLogps);
            }
        }

        // Full forward pass
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");

        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        // Simple sequence log-probs (using last token logits)
        Tensor chosenLogps = LogProbUtils.sequenceLogProbs(chosenLogits, chosenIds, chosenMask);
        Tensor rejectedLogps = LogProbUtils.sequenceLogProbs(rejectedLogits, rejectedIds, rejectedMask);

        return computeCpoLoss(chosenLogps, rejectedLogps);
    }

    private Tensor computeCpoLoss(Tensor chosenLogps, Tensor rejectedLogps) {
        // CPO Loss (similar to DPO but with contrastive term)
        Tensor logRatio = chosenLogps.sub(rejectedLogps);
        // CPO loss: -beta * log(1 + exp((log_ratio) / beta))
        Tensor cpoLoss = logRatio.div(new Scalar(beta)).exp().add(new Scalar(1.0)).log().mul(new Scalar(-beta));

        // Contrastive loss: push chosen away from rejected with margin
        Tensor contrastiveLoss = computeContrastiveLoss(chosenLogps, rejectedLogps);

        // Combined loss
        double alpha = cpoConfig.contrastiveAlpha();
        Tensor totalLoss = cpoLoss.add(contrastiveLoss.mul(new Scalar(alpha)));

        numTrainingSteps++;
        return totalLoss;
    }

    /**
     * Contrastive loss: margin-based loss that encourages separation between
     * chosen and rejected responses.
     *
     * Loss = max(0, margin - (chosen_reward - rejected_reward))
     */
    private Tensor computeContrastiveLoss(Tensor chosenLogps, Tensor rejectedLogps) {
        double margin = cpoConfig.margin();
        Tensor rewardDiff = chosenLogps.sub(rejectedLogps);
        Tensor marginTensor = org.bytedeco.pytorch.global.torch.full_like(
                chosenLogps, new Scalar(margin));
        return marginTensor.sub(rewardDiff).clamp_min(new Scalar(0.0));
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
        System.out.printf("[CPOTrainer] Closed: trainingSteps=%d%n", numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
}
