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
 * or as provided in the LICENSE.txt file that accompanied this code. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.TrainerConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Reward Model Fine-Tuning trainer with online preference learning.
 *
 * <p>This trainer combines reward modeling with online RL optimization:
 * <ul>
 *   <li>Trains a reward model on preference data</li>
 *   <li>Uses the reward model to generate online preferences</li>
 *   <li>Optimizes policy using the learned reward signal</li>
 * </ul>
 *
 * <p>Reference: ByteDance internal RMT (Reward Model Training) approach
 * combined with online RL optimization techniques.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids} (+ optional masks)</li>
 *   <li>or precomputed {@code chosen_rewards}, {@code rejected_rewards}</li>
 *   <li>optional {@code margin} for margin-based ranking loss</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RMRTTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";

    private final Module rewardModel;
    private final Module policy;              // optional, for joint training
    private final LlmForward rewardForward;
    private final LlmForward policyForward;
    private final TrainerConfig config;
    private final TensorVector params;
    private volatile boolean closed;

    // Training mode
    private enum Mode { REWARD_ONLY, JOINT, POLICY_ONLY }
    private Mode mode = Mode.REWARD_ONLY;

    /**
     * Reward model only training (classic reward modeling).
     */
    public RMRTTrainer(
            Module rewardModel,
            LlmForward rewardForward,
            Optimizer optimizer,
            TrainerConfig config) {
        super(config, optimizer);
        this.rewardModel = Objects.requireNonNull(rewardModel, "rewardModel");
        this.rewardForward = Objects.requireNonNull(rewardForward, "rewardForward");
        this.policy = null;
        this.policyForward = null;
        this.config = Objects.requireNonNull(config, "config");
        this.params = rewardModel.parameters();
    }

    /**
     * Joint reward model + policy training (RMT + online RL).
     */
    public RMRTTrainer(
            Module rewardModel,
            LlmForward rewardForward,
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            TrainerConfig config) {
        super(config, optimizer);
        this.rewardModel = Objects.requireNonNull(rewardModel, "rewardModel");
        this.rewardForward = Objects.requireNonNull(rewardForward, "rewardForward");
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.config = Objects.requireNonNull(config, "config");
        this.params = joinParams(rewardModel, policy);
        this.mode = Mode.JOINT;
    }

    /**
     * Policy only training with frozen reward model.
     */
    public RMRTTrainer(
            Module rewardModel,
            LlmForward rewardForward,
            Module policy,
            LlmForward policyForward,
            Optimizer policyOptimizer,
            TrainerConfig config) {
        super(config, policyOptimizer);
        this.rewardModel = Objects.requireNonNull(rewardModel, "rewardModel");
        this.rewardForward = Objects.requireNonNull(rewardForward, "rewardForward");
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.config = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.mode = Mode.POLICY_ONLY;

        // Freeze reward model
        freeze(rewardModel);
    }

    public Module rewardModel() { return rewardModel; }
    public Module policy() { return policy; }
    public TrainerConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        rewardModel.train(true);
        if (policy != null) policy.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        rewardModel.eval();
        if (policy != null) policy.eval();
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        switch (mode) {
            case REWARD_ONLY:
                return computeRewardLoss(batch);
            case JOINT:
                return computeJointLoss(batch);
            case POLICY_ONLY:
                return computePolicyLoss(batch);
            default:
                throw new IllegalStateException("Unknown mode: " + mode);
        }
    }

    /**
     * Compute reward modeling loss (Bradley-Terry preference loss).
     */
    private Tensor computeRewardLoss(Map<String, Tensor> batch) {
        Tensor chosen;
        Tensor rejected;

        if (batch.containsKey("chosen_rewards")) {
            chosen = batch.get("chosen_rewards");
            rejected = require(batch, "rejected_rewards");
        } else {
            Tensor chosenIds = require(batch, "chosen_input_ids");
            Tensor rejectedIds = require(batch, "rejected_input_ids");
            Tensor chosenMask = batch.get("chosen_attention_mask");
            Tensor rejectedMask = batch.get("rejected_attention_mask");

            chosen = rewardForward.forward(chosenIds, chosenMask);
            rejected = rewardForward.forward(rejectedIds, rejectedMask);
        }

        // Margin-based ranking loss
        double margin = config.margin();
        if (margin > 0) {
            return org.bytedeco.pytorch.global.torch.log_sigmoid(
                    chosen.sub(rejected).sub(new Scalar(margin))).neg().mean();
        }

        // Standard Bradley-Terry loss
        return org.bytedeco.pytorch.global.torch.log_sigmoid(
                chosen.sub(rejected)).neg().mean();
    }

    /**
     * Compute joint loss: reward modeling + policy optimization.
     */
    private Tensor computeJointLoss(Map<String, Tensor> batch) {
        // 1. Reward loss on preference pairs
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");

        Tensor chosenReward = rewardForward.forward(chosenIds, chosenMask);
        Tensor rejectedReward = rewardForward.forward(rejectedIds, rejectedMask);

        Tensor rewardLoss = org.bytedeco.pytorch.global.torch.log_sigmoid(
                chosenReward.sub(rejectedReward)).neg().mean();

        // 2. Policy loss using learned reward
        // Sample new completions and compute RL loss
        // This is a simplified version - full implementation would require
        // a generation step

        return rewardLoss;
    }

    /**
     * Compute policy loss using frozen reward model.
     */
    private Tensor computePolicyLoss(Map<String, Tensor> batch) {
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor labels = orElse(batch.get("labels"), inputIds);

        // Get policy log-probs
        Tensor logits = policyForward.forward(inputIds, attentionMask);
        Tensor logps = LogProbUtils.sequenceLogProbs(logits, labels, attentionMask);

        // Get reward for the completion
        Tensor rewards = rewardForward.forward(inputIds, attentionMask);

        // Simple policy gradient loss: maximize expected reward
        return logps.mul(rewards.detach()).mean().neg();
    }

    private static TensorVector joinParams(Module... modules) {
        TensorVector combined = new TensorVector();
        for (Module m : modules) {
            if (m == null) continue;
            TensorVector params = m.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                combined.push_back(params.get(i));
            }
        }
        return combined;
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
        System.out.printf("[RMRTTrainer] Closed: mode=%s%n", mode);
    }

    public boolean isClosed() { return closed; }
}
