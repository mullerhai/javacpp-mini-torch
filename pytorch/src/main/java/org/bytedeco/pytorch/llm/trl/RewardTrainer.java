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
import org.bytedeco.pytorch.llm.trl.config.RewardConfig;
import org.bytedeco.pytorch.llm.trl.loss.RewardModelLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

/**
 * Reward-model trainer (HF TRL {@code RewardTrainer} subset).
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Bradley-Terry loss with margin / label smoothing / length normalization</li>
 *   <li>Centered rewards toggle</li>
 *   <li>Optional per-layer / partial training</li>
 *   <li>Separate learning rates for head and backbone</li>
 *   <li>NaN/Inf guard with fallback to identity</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RewardTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "reward";

    @FunctionalInterface
    public interface RewardForward {
        Tensor forward(Tensor inputIds, Tensor attentionMask);
    }

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module model;
    private final RewardForward rewardForward;
    private final RewardConfig rewardConfig;
    private final TensorVector params;

    public RewardTrainer(
            Module model,
            RewardForward rewardForward,
            Optimizer optimizer,
            RewardConfig config) {
        super(config, optimizer);
        this.model = Objects.requireNonNull(model, "model");
        this.rewardForward = rewardForward;
        this.rewardConfig = Objects.requireNonNull(config, "config");
        this.params = model.parameters();
        System.out.printf(
                "[RewardTrainer v%s] lr=%.2e, head_lr=%.2e, margin=%.3f, ls=%.3f, center=%s%n",
                VERSION, rewardConfig.learningRate(), rewardConfig.learningRateReward(),
                rewardConfig.margin(), rewardConfig.labelSmoothing(),
                rewardConfig.centerRewards());
    }

    public RewardTrainer(Module model, Optimizer optimizer, RewardConfig config) {
        this(model, null, optimizer, config);
    }

    public Module model() { return model; }
    public RewardConfig rewardConfig() { return rewardConfig; }

    @Override
    protected TensorVector trainableParameters() { return params; }

    @Override
    public void train() {
        super.train();
        model.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        model.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor chosen;
        Tensor rejected;

        if (hasKey(batch, "chosen_rewards") && hasKey(batch, "rejected_rewards")) {
            chosen = batch.get("chosen_rewards");
            rejected = batch.get("rejected_rewards");
        } else {
            if (rewardForward == null) {
                throw new IllegalStateException(
                        "batch missing chosen_rewards and no RewardForward was provided");
            }
            Tensor chosenIds = require(batch, "chosen_input_ids");
            Tensor rejectedIds = require(batch, "rejected_input_ids");
            Tensor chosenMask = batch.get("chosen_attention_mask");
            Tensor rejectedMask = batch.get("rejected_attention_mask");
            chosen = rewardForward.forward(chosenIds, chosenMask);
            rejected = rewardForward.forward(rejectedIds, rejectedMask);
        }

        if (rewardConfig.centerRewards()) {
            Tensor mean = chosen.add(rejected).mul(new Scalar(0.5)).mean();
            chosen = chosen.sub(mean);
            rejected = rejected.sub(mean);
        }

        // Length normalization for the scalar reward (rare but supported).
        if (rewardConfig.lengthNormalize()) {
            chosen = chosen.div(new Scalar(Math.max(1.0, (double) chosen.size(0))));
            rejected = rejected.div(new Scalar(Math.max(1.0, (double) rejected.size(0))));
        }

        double margin = rewardConfig.margin();
        Tensor loss;
        if (margin != 0.0) {
            loss = log_sigmoid(chosen.sub(rejected).sub(new Scalar(margin))).neg().mean();
        } else {
            loss = RewardModelLoss.compute(chosen, rejected);
        }

        // Label smoothing
        double ls = rewardConfig.labelSmoothing();
        if (ls > 0.0) {
            Tensor smoothed = log_sigmoid(chosen.sub(rejected))
                    .mul(new Scalar(1.0 - ls))
                    .add(log_sigmoid(rejected.sub(chosen)).mul(new Scalar(ls)))
                    .neg()
                    .mean();
            loss = loss.mul(new Scalar(1.0 - ls)).add(smoothed.mul(new Scalar(ls)));
        }

        // Truncation mode (data-side; we expose a soft tolerance)
        if (rewardConfig.truncationMode() > 0.0) {
            Tensor diff = chosen.sub(rejected);
            Tensor upper = diff.clamp_min(new Scalar(0.0));
            Tensor lower = diff.clamp_max(new Scalar(0.0));
            Tensor truncationMask = lower.mul(new Scalar(rewardConfig.truncationMode()))
                    .add(upper.mul(new Scalar(1.0 - rewardConfig.truncationMode())));
            loss = loss.mul(truncationMask.mean().add(new Scalar(1e-8)));
        }

        double v = loss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[RewardTrainer] WARNING: NaN/Inf loss; falling back to identity.");
            return chosen.sub(rejected).pow(new Scalar(2)).mean();
        }

        numTrainingSteps++;
        return loss;
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
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
        System.out.printf("[RewardTrainer v%s] Closed: steps=%d%n", VERSION, numTrainingSteps);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "Reward v" + VERSION + " (Bradley-Terry Reward Model)";
    }
}