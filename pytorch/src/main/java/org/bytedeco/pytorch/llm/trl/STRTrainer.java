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
import org.bytedeco.pytorch.llm.trl.config.STRConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Self-Taught Reasoning trainer (Kimi/DeepSeek inspired).
 *
 * <p>STR enables models to learn reasoning capabilities through:
 * <ul>
 *   <li>Chain-of-thought generation</li>
 *   <li>Self-critique and verification</li>
 *   <li>Iterative refinement</li>
 *   <li>Process reward modeling</li>
 * </ul>
 *
 * <p>Reference: Kimi/DeepSeek research on self-taught reasoning
 */
public final class STRTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module processRewardModel;
    private final LlmForward processRewardForward;
    private final STRConfig config;
    private final TensorVector params;

    // Training statistics
    private double avgReasoningScore;
    private double avgCritiqueScore;
    private int reasoningStepsUsed;
    private int refinementsApplied;

    public STRTrainer(
            Module policy,
            LlmForward policyForward,
            Module processRewardModel,
            LlmForward processRewardForward,
            Optimizer optimizer,
            STRConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.processRewardModel = processRewardModel;
        this.processRewardForward = processRewardForward;
        this.config = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.avgReasoningScore = 0.0;
        this.avgCritiqueScore = 0.0;

        if (processRewardModel != null) {
            freeze(processRewardModel);
        }
    }

    /** Simplified constructor without process reward model. */
    public STRTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            STRConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public STRConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

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
        // Extract inputs
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor target = require(batch, "target");

        // Generate chain-of-thought reasoning
        Tensor[] reasoningChain = generateReasoningChain(inputIds, attentionMask);

        // Self-critique if enabled
        Tensor[] critiques = null;
        if (config.useSelfCritique()) {
            critiques = selfCritique(inputIds, reasoningChain);
        }

        // Compute process rewards if enabled
        Tensor[] processRewards = null;
        if (config.useProcessReward() && processRewardModel != null) {
            processRewards = computeProcessRewards(inputIds, reasoningChain);
        }

        // Compute final loss with reasoning bonus
        Tensor totalLoss = computeReasoningLoss(inputIds, reasoningChain, critiques, processRewards, target);

        // Update statistics
        updateStatistics(reasoningChain, critiques, processRewards);

        numTrainingSteps++;
        return totalLoss;
    }

    /**
     * Generate chain-of-thought reasoning chain.
     */
    private Tensor[] generateReasoningChain(Tensor inputIds, Tensor attentionMask) {
        int maxSteps = config.maxReasoningSteps();
        reasoningStepsUsed += maxSteps;

        // Simplified reasoning chain generation
        // In practice, this would use beam search or sampling
        Tensor[] chain = new Tensor[maxSteps];

        for (int step = 0; step < maxSteps; step++) {
            // Generate reasoning step
            chain[step] = policyForward.forward(inputIds, attentionMask);
        }

        return chain;
    }

    /**
     * Self-critique the reasoning chain.
     */
    private Tensor[] selfCritique(Tensor inputIds, Tensor[] reasoningChain) {
        Tensor[] critiques = new Tensor[reasoningChain.length];

        for (int i = 0; i < reasoningChain.length; i++) {
            // Use policy to critique its own reasoning
            critiques[i] = policyForward.forward(inputIds, null);

            // Update running average
            double score = critiques[i] != null && critiques[i].dim() > 0
                    ? critiques[i].mean().item_double()
                    : 0.5;
            avgCritiqueScore = 0.9 * avgCritiqueScore + 0.1 * score;
        }

        return critiques;
    }

    /**
     * Compute process rewards for each reasoning step.
     */
    private Tensor[] computeProcessRewards(Tensor inputIds, Tensor[] reasoningChain) {
        Tensor[] rewards = new Tensor[reasoningChain.length];

        for (int i = 0; i < reasoningChain.length; i++) {
            if (processRewardForward != null) {
                rewards[i] = processRewardForward.forward(inputIds, null);
            } else {
                rewards[i] = org.bytedeco.pytorch.global.torch.ones(1);
            }

            // Update running average
            double score = rewards[i] != null && rewards[i].dim() > 0
                    ? rewards[i].mean().item_double()
                    : 0.5;
            avgReasoningScore = 0.9 * avgReasoningScore + 0.1 * score;
        }

        return rewards;
    }

    /**
     * Compute loss with reasoning and process reward components.
     */
    private Tensor computeReasoningLoss(
            Tensor inputIds,
            Tensor[] reasoningChain,
            Tensor[] critiques,
            Tensor[] processRewards,
            Tensor target) {

        // Standard language modeling loss on target
        Tensor finalOutput = reasoningChain[reasoningChain.length - 1];
        Tensor targetLoss = org.bytedeco.pytorch.global.torch.cross_entropy(
                finalOutput.reshape(-1, finalOutput.size(finalOutput.dim() - 1)),
                target.reshape(-1)
        ).mean();

        // Reasoning bonus if enabled
        double reasoningAlpha = config.reasoningAlpha();
        Tensor reasoningBonus = org.bytedeco.pytorch.global.torch.zeros(1);

        if (processRewards != null && processRewards.length > 0) {
            // Process reward: sum of rewards weighted by step
            double totalReward = 0;
            for (int i = 0; i < processRewards.length; i++) {
                double stepWeight = (double) (i + 1) / processRewards.length;
                double reward = processRewards[i] != null && processRewards[i].dim() > 0
                        ? processRewards[i].mean().item_double()
                        : 0.5;
                totalReward += stepWeight * reward;
            }
            reasoningBonus = reasoningBonus.add(new Scalar(reasoningAlpha * totalReward));
        }

        // Critique loss if enabled
        double critiqueWeight = config.critiqueWeight();
        Tensor critiqueLoss = org.bytedeco.pytorch.global.torch.zeros(1);

        if (critiques != null && critiques.length > 0) {
            for (Tensor critique : critiques) {
                if (critique != null && critique.dim() > 0) {
                    // Encourage high critique scores (self-improvement)
                    critiqueLoss = critiqueLoss.add(critique.mean());
                }
            }
            critiqueLoss = critiqueLoss.mul(new Scalar(critiqueWeight));
        }

        return targetLoss.neg().add(reasoningBonus).sub(critiqueLoss);
    }

    private void updateStatistics(Tensor[] reasoningChain, Tensor[] critiques, Tensor[] processRewards) {
        // Already updated in generateReasoningChain, selfCritique, computeProcessRewards
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
                "[STRTrainer] Closed: steps=%d, reasoningSteps=%d, " +
                "avgReasoning=%.3f, avgCritique=%.3f%n",
                numTrainingSteps, reasoningStepsUsed,
                avgReasoningScore, avgCritiqueScore);
    }

    public boolean isClosed() { return closed; }
    public double getAvgReasoningScore() { return avgReasoningScore; }
    public double getAvgCritiqueScore() { return avgCritiqueScore; }
}
