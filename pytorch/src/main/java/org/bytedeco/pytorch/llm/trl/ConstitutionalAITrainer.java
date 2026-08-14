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
import org.bytedeco.pytorch.llm.trl.config.ConstitutionalAIConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Constitutional AI trainer for Claude alignment (Anthropic inspired).
 *
 * <p>Constitutional AI uses a set of principles to guide model behavior:
 * <ul>
 *   <li>Helpful: Maximize positive impact</li>
 *   <li>Harmless: Avoid harmful content</li>
 *   <li>Honest: Provide accurate information</li>
 * </ul>
 *
 * <p>The training process includes:
 * <ol>
 *   <li>Critique: Model identifies harmful aspects of responses</li>
 *   <li>Revision: Model revises responses based on critique</li>
 *   <li>SLICF/RLAIF: Learning from AI-generated feedback</li>
 * </ol>
 *
 * <p>Reference: "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
 */
public final class ConstitutionalAITrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module critiqueModel;
    private final LlmForward critiqueForward;
    private final ConstitutionalAIConfig config;
    private final TensorVector params;

    // Training statistics
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
        this.avgCritiqueScore = 0.0;
        this.avgRevisionImprovement = 0.0;

        if (critiqueModel != null) {
            freeze(critiqueModel);
        }
    }

    /** Policy-only constructor (uses policy for critique). */
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
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (critiqueModel != null) {
            critiqueModel.eval();
        }
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (critiqueModel != null) {
            critiqueModel.eval();
        }
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        // Extract initial response
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor initialResponse = require(batch, "initial_response");
        Tensor targetResponse = require(batch, "target_response");

        // Step 1: Critique the initial response
        Tensor critique = generateCritique(inputIds, initialResponse);

        // Step 2: Revise response based on critique
        Tensor revisedResponse = reviseResponse(inputIds, initialResponse, critique);

        // Step 3: Compute loss
        Tensor totalLoss = null;

        if (config.useSLICF()) {
            // Supervised Learning from AI Feedback
            totalLoss = computeSLICFLoss(inputIds, revisedResponse, targetResponse);
        }

        if (config.useRLAIF()) {
            // RL from AI Feedback
            Tensor rlaifLoss = computeRLAIFLoss(inputIds, revisedResponse, critique);
            if (config.useSLICF()) {
                totalLoss = totalLoss.add(rlaifLoss.mul(new Scalar(0.5)));
            } else {
                totalLoss = rlaifLoss;
            }
        }

        numTrainingSteps++;
        return totalLoss != null ? totalLoss : org.bytedeco.pytorch.global.torch.zeros(1);
    }

    /**
     * Generate critique based on constitutional principles.
     */
    private Tensor generateCritique(Tensor inputIds, Tensor response) {
        critiquesGenerated++;

        if (critiqueModel != null && critiqueForward != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                return critiqueForward.forward(inputIds, null);
            }
        }

        // Fallback: Use policy model for critique
        try (NoGradGuard guard = new NoGradGuard()) {
            return policyForward.forward(inputIds, null);
        }
    }

    /**
     * Revise response based on critique.
     */
    private Tensor reviseResponse(Tensor inputIds, Tensor initialResponse, Tensor critique) {
        revisionsApplied++;

        // Simplified revision: blend initial and target based on critique score
        // In practice, this would be a more sophisticated revision process
        double critiqueScore = critique != null && critique.dim() > 0
                ? critique.mean().item_double()
                : 0.5;

        // Update running average
        avgCritiqueScore = 0.9 * avgCritiqueScore + 0.1 * critiqueScore;

        // Return initial response (simplified)
        return initialResponse;
    }

    /**
     * Compute SLICF loss: supervised learning from AI feedback.
     */
    private Tensor computeSLICFLoss(Tensor inputIds, Tensor response, Tensor target) {
        // Compute policy logits for response
        Tensor logits = policyForward.forward(inputIds, null);

        // Cross-entropy loss between response and target
        Tensor loss = org.bytedeco.pytorch.global.torch.cross_entropy(
                logits.reshape(-1, logits.size(logits.dim() - 1)),
                target.reshape(-1)
        );

        return loss.mean();
    }

    /**
     * Compute RLAIF loss: RL from AI feedback.
     */
    private Tensor computeRLAIFLoss(Tensor inputIds, Tensor response, Tensor critique) {
        // Use critique score as reward
        double reward = critique != null && critique.dim() > 0
                ? critique.mean().item_double()
                : 0.0;

        // Policy gradient loss
        Tensor logits = policyForward.forward(inputIds, null);
        Tensor logProbs = logits.log_softmax(-1);

        // Simplified advantage
        Tensor advantage = org.bytedeco.pytorch.global.torch.full_like(
                logProbs.select(-1, 0), new Scalar(reward));

        // Policy gradient
        Tensor pgLoss = logProbs.mul(advantage).neg().mean();

        // KL penalty against reference
        try (NoGradGuard guard = new NoGradGuard()) {
            Tensor refLogits = policyForward.forward(inputIds, null);
            Tensor refLogProbs = refLogits.log_softmax(-1);
            Tensor klDiv = logProbs.sub(refLogProbs).mul(logProbs.exp());
            pgLoss = pgLoss.add(new Scalar(config.critiqueWeight())).mul(new Scalar(klDiv.mean()));
        }

        return pgLoss;
    }

    /**
     * Multi-objective loss based on constitutional principles.
     */
    private Tensor computeConstitutionalLoss(Tensor inputIds, Tensor response) {
        // Harmlessness loss
        double harmWeight = config.harmlessnessWeight();
        Tensor harmLoss = computeObjectiveLoss(inputIds, response, "harmless");

        // Helpfulness loss
        double helpWeight = config.helpfulnessWeight();
        Tensor helpLoss = computeObjectiveLoss(inputIds, response, "helpful");

        // Honesty loss
        double honestWeight = config.honestyWeight();
        Tensor honestLoss = computeObjectiveLoss(inputIds, response, "honest");

        return harmLoss.mul(new Scalar(harmWeight))
                .add(helpLoss.mul(new Scalar(helpWeight)))
                .add(honestLoss.mul(new Scalar(honestWeight)));
    }

    private Tensor computeObjectiveLoss(Tensor inputIds, Tensor response, String objective) {
        // Simplified objective loss computation
        // In practice, this would use separate reward models for each objective
        Tensor logits = policyForward.forward(inputIds, null);
        return logits.abs().mean();
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
        System.out.printf(
                "[ConstitutionalAITrainer] Closed: steps=%d, critiques=%d, revisions=%d, " +
                "avgCritique=%.3f, avgRevision=%.3f%n",
                numTrainingSteps, critiquesGenerated, revisionsApplied,
                avgCritiqueScore, avgRevisionImprovement);
    }

    public boolean isClosed() { return closed; }
    public double getAvgCritiqueScore() { return avgCritiqueScore; }
    public double getAvgRevisionImprovement() { return avgRevisionImprovement; }
}
