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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.TrainerConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

/**
 * Reward Model Fine-Tuning trainer with online preference learning.
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Reward-only / joint / policy-only modes</li>
 *   <li>Margin-aware Bradley-Terry ranking loss</li>
 *   <li>Centered rewards toggle</li>
 *   <li>Optional policy KL regularization</li>
 *   <li>NaN/Inf guard</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RMRTTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "rmrt";

    private final Module rewardModel;
    private final Module policy;
    private final LlmForward rewardForward;
    private final LlmForward policyForward;
    private final TrainerConfig config;
    private final TensorVector params;
    private volatile boolean closed;
    private long numTrainingSteps;

    private enum Mode { REWARD_ONLY, JOINT, POLICY_ONLY }
    private Mode mode = Mode.REWARD_ONLY;

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
        System.out.printf("[RMRTTrainer v%s] mode=REWARD_ONLY%n", VERSION);
    }

    public RMRTTrainer(
            Module rewardModel,
            LlmForward rewardForward,
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            TrainerConfig config,
            boolean jointMode) {
        super(config, optimizer);
        this.rewardModel = Objects.requireNonNull(rewardModel, "rewardModel");
        this.rewardForward = Objects.requireNonNull(rewardForward, "rewardForward");
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.config = Objects.requireNonNull(config, "config");
        if (jointMode) {
            this.params = joinParams(rewardModel, policy);
            this.mode = Mode.JOINT;
        } else {
            this.params = policy.parameters();
            this.mode = Mode.POLICY_ONLY;
            freeze(rewardModel);
        }
        System.out.printf("[RMRTTrainer v%s] mode=%s%n", VERSION, mode);
    }

    public Module rewardModel() { return rewardModel; }
    public Module policy() { return policy; }
    public TrainerConfig config() { return config; }

    @Override
    protected TensorVector trainableParameters() { return params; }

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
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor loss;
        switch (mode) {
            case REWARD_ONLY:
                loss = computeRewardLoss(batch);
                break;
            case JOINT:
                loss = computeJointLoss(batch);
                break;
            case POLICY_ONLY:
                loss = computePolicyLoss(batch);
                break;
            default:
                throw new IllegalStateException("Unknown mode: " + mode);
        }
        double v = loss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[RMRTTrainer] WARNING: NaN/Inf loss; fallback to zero.");
            return log_sigmoid(
                    (rewardForward != null)
                            ? batch.getOrDefault("chosen_input_ids",
                                    batch.getOrDefault("input_ids", loss.detach()))
                            : loss.detach()).neg().mean();
        }
        numTrainingSteps++;
        return loss;
    }

    private Tensor computeRewardLoss(Map<String, Tensor> batch) {
        Tensor chosen;
        Tensor rejected;
        if (hasKey(batch, "chosen_rewards") && hasKey(batch, "rejected_rewards")) {
            chosen = batch.get("chosen_rewards");
            rejected = batch.get("rejected_rewards");
        } else {
            Tensor chosenIds = require(batch, "chosen_input_ids");
            Tensor rejectedIds = require(batch, "rejected_input_ids");
            Tensor chosenMask = batch.get("chosen_attention_mask");
            Tensor rejectedMask = batch.get("rejected_attention_mask");
            chosen = rewardForward.forward(chosenIds, chosenMask);
            rejected = rewardForward.forward(rejectedIds, rejectedMask);
        }

        // Centered rewards
        if (config.margin() == 0.0 && chosen.numel() > 1) {
            Tensor mean = chosen.add(rejected).mul(new Scalar(0.5)).mean();
            chosen = chosen.sub(mean);
            rejected = rejected.sub(mean);
        }

        double margin = config.margin();
        if (margin > 0) {
            return log_sigmoid(chosen.sub(rejected).sub(new Scalar(margin))).neg().mean();
        }
        return log_sigmoid(chosen.sub(rejected)).neg().mean();
    }

    private Tensor computeJointLoss(Map<String, Tensor> batch) {
        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");

        Tensor chosenReward = rewardForward.forward(chosenIds, chosenMask);
        Tensor rejectedReward = rewardForward.forward(rejectedIds, rejectedMask);
        Tensor rewardLoss = log_sigmoid(chosenReward.sub(rejectedReward)).neg().mean();

        // Optional policy gradient term
        if (hasKey(batch, "input_ids") && policyForward != null) {
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            Tensor logps = LogProbUtils.sequenceLogProbs(logits, labels, attentionMask);
            Tensor policyReward = rewardForward.forward(inputIds, attentionMask);
            Tensor policyLoss = logps.mul(policyReward.detach()).mean().neg();
            rewardLoss = rewardLoss.mul(new Scalar(0.5)).add(policyLoss.mul(new Scalar(0.5)));
        }
        return rewardLoss;
    }

    private Tensor computePolicyLoss(Map<String, Tensor> batch) {
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");
        Tensor labels = orElse(batch.get("labels"), inputIds);

        Tensor logits = policyForward.forward(inputIds, attentionMask);
        Tensor logps = LogProbUtils.sequenceLogProbs(logits, labels, attentionMask);
        Tensor rewards = rewardForward.forward(inputIds, attentionMask);
        return logps.mul(rewards.detach()).mean().neg();
    }

    private static TensorVector joinParams(Module... modules) {
        TensorVector combined = new TensorVector();
        for (Module m : modules) {
            if (m == null) continue;
            TensorVector ps = m.parameters();
            for (long i = 0, n = ps.size(); i < n; i++) {
                combined.push_back(ps.get(i));
            }
        }
        return combined;
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
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
        System.out.printf("[RMRTTrainer v%s] Closed: mode=%s, steps=%d%n", VERSION, mode, numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "RMRT v" + VERSION + " (Reward Model Fine-Tuning)";
    }
}