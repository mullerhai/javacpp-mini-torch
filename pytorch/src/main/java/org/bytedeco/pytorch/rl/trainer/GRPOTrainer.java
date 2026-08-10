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
package org.bytedeco.pytorch.rl.trainer;

import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;

import java.util.Objects;

/**
 * Enterprise-grade GRPO trainer with full resource management.
 *
 * <p>Reuses shared {@link org.bytedeco.pytorch.llm.trl.loss.GRPOLoss}.
 * This is <em>not</em> the guided-reward agent
 * ({@link org.bytedeco.pytorch.rl.agent.GuidedRewardPPOAgent} /
 * {@link org.bytedeco.pytorch.rl.agent.GRPOAgent}).
 */
public class GRPOTrainer implements RLTrainer, AutoCloseable {
    private static final String VERSION = "2.0";

    private final AbstractActorCritic model;
    private final Optimizer optimizer;
    private final double clipRange;
    private final int groupSize;
    private volatile boolean closed;

    // Performance metrics
    private long totalTrainingTimeMs;
    private int totalSteps;

    public GRPOTrainer(AbstractActorCritic model) {
        this(model, 1e-4, 0.2, 4);
    }

    public GRPOTrainer(AbstractActorCritic model, double lr, double clipRange, int groupSize) {
        this.model = Objects.requireNonNull(model, "model");
        AdamOptions option = new AdamOptions();
        option.lr().put(lr);
        this.optimizer = new Adam(model.parameters(), option);
        this.clipRange = clipRange;
        this.groupSize = groupSize;
    }

    public GRPOTrainer(AbstractActorCritic model, Optimizer optimizer, double clipRange, int groupSize) {
        this.model = Objects.requireNonNull(model, "model");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        this.clipRange = clipRange;
        this.groupSize = groupSize;
    }

    /** @deprecated prefer {@link #GRPOTrainer(AbstractActorCritic)} */
    @Deprecated
    public GRPOTrainer(ActorCriticNetwork model) {
        this((AbstractActorCritic) model);
    }

    /** @deprecated prefer {@link #GRPOTrainer(AbstractActorCritic, double, double, int)} */
    @Deprecated
    public GRPOTrainer(ActorCriticNetwork model, double lr, double clipRange, int groupSize) {
        this((AbstractActorCritic) model, lr, clipRange, groupSize);
    }

    /**
     * One GRPO update given a distribution over actions and group rewards.
     */
    public Tensor trainStep(Distribution dist, Tensor actions, Tensor oldLps, Tensor groupRewards) {
        if (closed) throw new IllegalStateException("Trainer is closed");

        long startTime = System.currentTimeMillis();

        Tensor flatRewards = groupRewards.dim() > 1 ? groupRewards.flatten() : groupRewards;
        int g = groupSize > 0 ? groupSize : (int) groupRewards.size(groupRewards.dim() - 1);
        if (flatRewards.numel() % g != 0) {
            g = (int) flatRewards.numel();
        }

        Tensor currLps = actionLogProb(dist, actions);
        Tensor loss;
        if (oldLps != null && oldLps.defined() && clipRange > 0.0) {
            loss = GRPOLoss.computeClipped(currLps, flat1d(oldLps), flatRewards, g, clipRange);
        } else {
            loss = GRPOLoss.compute(currLps, flatRewards, g);
        }

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        totalSteps++;
        totalTrainingTimeMs += (System.currentTimeMillis() - startTime);

        return loss.detach();
    }

    @Override
    public void trainBatch(ReplayBuffer buffer) {
        if (closed) throw new IllegalStateException("Trainer is closed");
        if (buffer == null || buffer.size() == 0) {
            return;
        }
        Tensor[] data = buffer.getAll();
        if (data == null) {
            return;
        }
        // getAll: [states, actions, oldLogProbs, advantages, returns]
        Tensor states = data[0];
        Tensor actions = data[1];
        Tensor oldLps = data[2];
        Tensor advantages = data.length > 3 ? data[3] : null;
        Tensor returns = data.length > 4 ? data[4] : null;
        // Group-relative scores = returns (raw reward often stored there) > advantages
        Tensor scores = (returns != null && returns.defined()) ? returns : advantages;
        if (scores == null || !scores.defined()) {
            return;
        }

        Distribution dist = model.getDistribution(states);
        int g = groupSize;
        if (g <= 0 || scores.numel() % g != 0) {
            g = (int) scores.numel();
        }
        Tensor currLps = actionLogProb(dist, actions);
        Tensor loss = GRPOLoss.computeClipped(currLps, flat1d(oldLps), flat1d(scores), g, clipRange);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    static Tensor actionLogProb(Distribution dist, Tensor actions) {
        Tensor lp = dist.log_prob(actions);
        return flat1d(sumActionDimsOnly(lp));
    }

    static Tensor sumActionDimsOnly(Tensor logProb) {
        if (logProb.dim() <= 1) {
            return logProb;
        }
        return logProb.sum(-1);
    }

    static Tensor flat1d(Tensor t) {
        Tensor x = t;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) {
            x = x.squeeze(x.dim() - 1);
        }
        if (x.dim() == 0) {
            x = x.reshape(1);
        }
        if (x.dim() > 1) {
            x = x.reshape(-1);
        }
        return x;
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public AbstractActorCritic model() {
        return model;
    }

    @Override
    public String algorithm() {
        return "grpo-group-relative";
    }

    public boolean isClosed() { return closed; }

    public int totalSteps() { return totalSteps; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[GRPOTrainer v%s] Closed: steps=%d, totalTime=%.2fs%n",
                VERSION, totalSteps, totalTrainingTimeMs / 1000.0);
    }
}
