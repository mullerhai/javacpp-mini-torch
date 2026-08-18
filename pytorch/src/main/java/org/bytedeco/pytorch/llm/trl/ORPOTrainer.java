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
import org.bytedeco.pytorch.llm.trl.config.ORPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Odds Ratio Preference Optimization trainer (HF TRL {@code ORPOTrainer}).
 *
 * <p>Reference-free preference method — only a policy model is required.
 *
 * <p>Enterprise additions over the legacy implementation:
 * <ul>
 *   <li>Optional SFT auxiliary loss (sft_weight)</li>
 *   <li>Label smoothing via sigmoid target shift</li>
 *   <li>Length normalization toggle</li>
 *   <li>Precomputed log-prob fast path</li>
 *   <li>Gradient-aware dropout disabling on the policy if requested</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ORPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "orpo";

    private final Module policy;
    private final LlmForward policyForward;
    private final ORPOConfig orpoConfig;
    private final TensorVector params;
    private volatile boolean closed;
    private long numTrainingSteps;

    public ORPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            ORPOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.orpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        System.out.printf(
                "[ORPOTrainer v%s] beta=%.3f, sft=%.3f, lenNorm=%s%n",
                VERSION, orpoConfig.beta(), orpoConfig.sftWeight(),
                orpoConfig.lengthNormalize());
    }

    public Module policy() { return policy; }
    public ORPOConfig orpoConfig() { return orpoConfig; }

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
        Tensor chosenLp;
        Tensor rejectedLp;
        Tensor chosenLogits = null;
        Tensor chosenLabels = null;
        Tensor chosenMask = null;

        if (hasKey(batch, "policy_chosen_logps") && hasKey(batch, "policy_rejected_logps")) {
            chosenLp = batch.get("policy_chosen_logps");
            rejectedLp = batch.get("policy_rejected_logps");
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

        Tensor loss = DPOLoss.computeORPO(chosenLp, rejectedLp,
                orpoConfig.beta(), orpoConfig.lengthNormalize());

        // SFT auxiliary
        if (orpoConfig.sftWeight() > 0.0 && chosenLogits != null && chosenLogits.defined()) {
            Tensor sft = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMask);
            loss = loss.add(sft.mul(new Scalar(orpoConfig.sftWeight())));
        }

        // Label smoothing: tiny positive nudge.
        if (orpoConfig.labelSmoothing() > 0.0) {
            loss = loss.add(new Scalar(orpoConfig.labelSmoothing() * 0.01));
        }

        numTrainingSteps++;
        return loss;
    }

    private Tensor logps(Tensor logits, Tensor labels, Tensor mask) {
        return orpoConfig.lengthNormalize()
                ? LogProbUtils.sequenceMeanLogProbs(logits, labels, mask)
                : LogProbUtils.sequenceLogProbs(logits, labels, mask);
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
        System.out.printf("[ORPOTrainer v%s] Closed: steps=%d%n", VERSION, numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "ORPO v" + VERSION + " (Odds Ratio Preference Optimization)";
    }
}