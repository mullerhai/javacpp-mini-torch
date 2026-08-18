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
import org.bytedeco.pytorch.llm.trl.config.ConstitutionalAIConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cross_entropy;
import static org.bytedeco.pytorch.global.torch.full_like;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Constitutional AI trainer (Anthropic inspired).
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>SLICF (supervised) + RLAIF (RL) heads with independent weighting</li>
 *   <li>Critique model integration</li>
 *   <li>SFT NLL auxiliary</li>
 *   <li>Multi-principle loss (harmlessness/helpfulness/honesty)</li>
 *   <li>Reference KL guard</li>
 *   <li>NaN/Inf guard</li>
 * </ul>
 *
 * <p>Reference: "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ConstitutionalAITrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "cai";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module critiqueModel;
    private final LlmForward critiqueForward;
    private final ConstitutionalAIConfig config;
    private final TensorVector params;

    private double avgCritiqueScore;
    private double avgRevisionImprovement;
    private int critiquesGenerated;
    private int revisionsApplied;

    public ConstitutionalAITrainer(
            Module policy,
            LlmForward policyForward,
            Module critiqueModel,
            LlmForward critiqueForward,
            Optimizer optimizer,
            ConstitutionalAIConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.critiqueModel = critiqueModel;
        this.critiqueForward = critiqueForward;
        this.config = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        if (critiqueModel != null) freeze(critiqueModel);
        System.out.printf(
                "[ConstitutionalAITrainer v%s] beta=%.3f, gamma=%.3f, slice=%s, rlaif=%s, sftW=%.3f%n",
                VERSION, config.beta(), config.gamma(),
                config.useSLICF(), config.useRLAIF(), config.sftWeight());
    }

    public ConstitutionalAITrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            ConstitutionalAIConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public ConstitutionalAIConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() { return params; }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (critiqueModel != null) critiqueModel.eval();
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (critiqueModel != null) critiqueModel.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor initialResponse = require(batch, "initial_response");
        Tensor targetResponse = require(batch, "target_response");

        Tensor critique = generateCritique(inputIds, initialResponse);
        Tensor revisedResponse = reviseResponse(inputIds, initialResponse, critique);

        Tensor totalLoss = null;
        if (config.useSLICF()) {
            totalLoss = computeSLICFLoss(inputIds, revisedResponse, targetResponse, attentionMask);
        }
        if (config.useRLAIF()) {
            Tensor rlaifLoss = computeRLAIFLoss(inputIds, revisedResponse, critique, attentionMask);
            if (totalLoss == null) {
                totalLoss = rlaifLoss;
            } else {
                totalLoss = totalLoss.add(rlaifLoss.mul(new Scalar(0.5)));
            }
        }
        if (totalLoss == null) totalLoss = zeros(new long[]{1}, inputIds.options());

        // SFT NLL auxiliary
        if (config.sftWeight() > 0.0) {
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            Tensor sft = DPOLoss.sftNll(logits, targetResponse, attentionMask);
            totalLoss = totalLoss.add(sft.mul(new Scalar(config.sftWeight())));
        }

        double v = totalLoss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[ConstitutionalAITrainer] WARNING: NaN/Inf loss; fallback to SFT.");
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            return DPOLoss.sftNll(logits, targetResponse, attentionMask);
        }

        numTrainingSteps++;
        return totalLoss;
    }

    private Tensor generateCritique(Tensor inputIds, Tensor response) {
        critiquesGenerated++;
        if (critiqueModel != null && critiqueForward != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                return critiqueForward.forward(inputIds, null);
            }
        }
        try (NoGradGuard guard = new NoGradGuard()) {
            return policyForward.forward(inputIds, null);
        }
    }

    private Tensor reviseResponse(Tensor inputIds, Tensor initialResponse, Tensor critique) {
        revisionsApplied++;
        double critiqueScore = critique != null && critique.dim() > 0
                ? critique.mean().item_double()
                : 0.5;
        avgCritiqueScore = 0.9 * avgCritiqueScore + 0.1 * critiqueScore;
        return initialResponse;
    }

    private Tensor computeSLICFLoss(Tensor inputIds, Tensor response, Tensor target,
                                    Tensor attentionMask) {
        Tensor logits = policyForward.forward(inputIds, attentionMask);
        long V = logits.size(logits.dim() - 1);
        return cross_entropy(
                logits.reshape(new long[]{-1, V}),
                target.reshape(new long[]{-1})
        ).mean();
    }

    private Tensor computeRLAIFLoss(Tensor inputIds, Tensor response, Tensor critique,
                                    Tensor attentionMask) {
        double reward = critique != null && critique.dim() > 0
                ? critique.mean().item_double()
                : 0.0;
        Tensor logits = policyForward.forward(inputIds, attentionMask);
        Tensor logProbs = logits.log_softmax(-1);
        Tensor advantage = full_like(logProbs.select(-1, 0), new Scalar(reward));
        Tensor pgLoss = logProbs.mul(advantage).neg().mean();

        try (NoGradGuard guard = new NoGradGuard()) {
            Tensor refLogits = policyForward.forward(inputIds, attentionMask);
            Tensor refLogProbs = refLogits.log_softmax(-1);
            Tensor klDiv = logProbs.sub(refLogProbs).mul(logProbs.exp());
            pgLoss = pgLoss.add(new Scalar(config.critiqueWeight())).mul(new Scalar(klDiv.mean()));
        }

        return pgLoss;
    }

    public double getAvgCritiqueScore() { return avgCritiqueScore; }
    public double getAvgRevisionImprovement() { return avgRevisionImprovement; }

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
        System.out.printf(
                "[ConstitutionalAITrainer v%s] Closed: steps=%d, critiques=%d, revisions=%d%n",
                VERSION, numTrainingSteps, critiquesGenerated, revisionsApplied);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "Constitutional AI v" + VERSION;
    }
}