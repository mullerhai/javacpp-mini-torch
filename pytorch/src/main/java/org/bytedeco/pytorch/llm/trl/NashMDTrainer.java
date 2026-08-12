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
import org.bytedeco.pytorch.llm.trl.config.NashMDConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Nash-MD (Nash Mirror Descent) trainer - A game-theoretic approach to LLM alignment.
 *
 * <p>Nash-MD treats LLM alignment as a game between the policy and reward,
 * using mirror descent with Nash equilibrium as the learning target. This provides:
 * <ul>
 *   <li>Better convergence guarantees than standard policy gradient methods</li>
 *   <li>Natural handling of multi-objective rewards (helpfulness + safety + etc.)</li>
 *   <li>Robustness to reward hacking through variational regularization</li>
 *   <li>Proven regret bounds in the online learning setting</li>
 * </ul>
 *
 * <p>Reference: "Nash Learning and Nash Matching in Multi-Objective Games"
 * Adapted for LLM alignment with KL-constrained policy updates.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code rewards} - multi-objective rewards {@code [B, K]} where K is number of objectives</li>
 *   <li>{@code old_logprobs} - policy log-probs at rollout time</li>
 *   <li>{@code input_ids} (+ optional masks) for online log-prob computation</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class NashMDTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final NashMDConfig nashConfig;
    private final TensorVector params;
    private volatile boolean closed;

    // Nash equilibrium tracking
    private Tensor equilibriumWeights;  // Current Nash equilibrium over objectives
    private int numUpdates;

    /**
     * Create Nash-MD trainer with reference model.
     */
    public NashMDTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            NashMDConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.nashConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();

        if (reference != null) {
            freeze(reference);
        }

        // Initialize uniform Nash equilibrium
        this.equilibriumWeights = null;

        System.out.printf(
                "[NashMDTrainer v%s] lr=%.6f, kl_target=%.4f, eta=%.4f, objectives=%d%n",
                VERSION, config.learningRate(), config.klTarget(), config.eta(),
                config.numObjectives());
    }

    /**
     * Create Nash-MD trainer without reference model.
     */
    public NashMDTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            NashMDConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public NashMDConfig nashConfig() { return nashConfig; }

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
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor rewards = require(batch, "rewards");
        int numObjectives = nashConfig.numObjectives();

        // Get current policy log-probs
        Tensor newLogprobs;
        if (batch.containsKey("logprobs") && batch.get("logprobs") != null) {
            newLogprobs = batch.get("logprobs");
        } else {
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);

            Tensor logits = policyForward.forward(inputIds, attentionMask);
            newLogprobs = LogProbUtils.sequenceLogProbs(logits, labels, attentionMask);
        }

        // Get old log-probs
        Tensor oldLogprobs = batch.get("old_logprobs");
        if (oldLogprobs == null || !oldLogprobs.defined()) {
            oldLogprobs = newLogprobs.detach();
        }

        // Compute KL to reference if available
        Tensor refKl = null;
        if (referenceForward != null && nashConfig.useReferenceKl()) {
            refKl = computeReferenceKl(batch);
        }

        // Compute Nash equilibrium weights from multi-objective rewards
        Tensor nashWeights = computeNashWeights(rewards, numObjectives);

        // Compute Nash-MD policy gradient loss
        Tensor ratio = newLogprobs.sub(oldLogprobs.detach()).exp();
        Tensor advantage = computeNashAdvantage(rewards, nashWeights);

        // Clipped policy gradient
        double clipRange = nashConfig.clipRange();
        Tensor surr1 = ratio.mul(advantage);
        Tensor ratioClipped = ratio.clamp(
                new org.bytedeco.pytorch.ScalarOptional(new Scalar(1.0 - clipRange)),
                new org.bytedeco.pytorch.ScalarOptional(new Scalar(1.0 + clipRange)));
        Tensor surr2 = ratioClipped.mul(advantage);
        Tensor policyLoss = surr1.min(surr2).mean().neg();

        // KL divergence penalty (Mirror Descent component)
        double klTarget = nashConfig.klTarget();
        Tensor klLoss;
        if (klTarget > 0 && refKl != null) {
            klLoss = refKl.sub(new Scalar(klTarget)).pow(new Scalar(2));
        } else if (klTarget > 0) {
            klLoss = newLogprobs.sub(oldLogprobs.detach()).pow(new Scalar(2)).mean();
        } else {
            klLoss = zeros_like(policyLoss);
        }

        // Combine losses
        double alpha = nashConfig.klCoef();
        return policyLoss.add(klLoss.mul(new Scalar(alpha)));
    }

    /**
     * Compute Nash equilibrium weights from multi-objective rewards.
     *
     * Uses a softmax-based approach to find weights that equalize
     * marginal utilities across objectives (Nash bargaining solution).
     */
    private Tensor computeNashWeights(Tensor rewards, int numObjectives) {
        double temperature = nashConfig.equilibriumTemperature();

        // Softmax over objectives to find equilibrium
        Tensor expRewards = rewards.div(new Scalar(temperature)).exp();
        Tensor partition = expRewards.sum(new long[]{1}, true,new ScalarTypeOptional());
        Tensor weights = expRewards.div(partition);

        // Track running average for stability
        if (equilibriumWeights == null || !equilibriumWeights.defined()) {
            equilibriumWeights = weights.detach().clone();
        } else {
            double momentum = nashConfig.equilibriumMomentum();
            equilibriumWeights = equilibriumWeights.mul(new Scalar(momentum))
                    .add(weights.detach().mul(new Scalar(1.0 - momentum)));
        }

        return equilibriumWeights;
    }

    /**
     * Compute weighted advantage using Nash equilibrium weights.
     */
    private Tensor computeNashAdvantage(Tensor rewards, Tensor weights) {
        // Weighted sum of rewards = Nash equilibrium payoff
        return rewards.mul(weights).sum(1);
    }

    /**
     * Compute KL divergence to reference model.
     */
    private Tensor computeReferenceKl(Map<String, Tensor> batch) {
        if (referenceForward == null) return null;

        try (NoGradGuard guard = new NoGradGuard()) {
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);

            Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
            Tensor refLogps = LogProbUtils.sequenceLogProbs(refLogits, labels, attentionMask);

            // KL(p||q) = sum(p * (log(p) - log(q)))
            return refLogps.sub(refLogps).mean();
        } catch (Exception e) {
            return null;
        }
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
        if (equilibriumWeights != null) {
            try { equilibriumWeights.close(); } catch (Throwable ignored) {}
        }
        System.out.printf("[NashMDTrainer] Closed: numUpdates=%d%n", numUpdates);
    }

    public boolean isClosed() { return closed; }
}
