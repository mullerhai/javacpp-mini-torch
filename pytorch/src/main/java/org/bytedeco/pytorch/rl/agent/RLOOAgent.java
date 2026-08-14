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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.rl.agent;

import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;

import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * RLOO (Reward-Offered-Out-Off) Agent.
 *
 * <p>RLOO is a policy gradient method similar to REINFORCE but with baseline subtraction.
 * It uses the mean of the batch returns as baseline, reducing variance while maintaining
 * the simplicity of REINFORCE.
 *
 * <p>Reference: "Variance Reduction for Deep Reinforcement Learning" (Tucker et al., 2018)
 */
public class RLOOAgent extends AbstractRLAgent {

    private float gamma;
    private float lr;
    private final RunningMeanStd obsNormalizer;

    public RLOOAgent(AbstractActorCritic model, Optimizer optimizer, ReplayBuffer buffer) {
        super(model, optimizer, buffer);
        this.gamma = 0.99f;
        this.lr = 3e-4f;
        this.obsNormalizer = new RunningMeanStd(guessStateDim(model));
    }

    public RLOOAgent(long stateDim, long actionDim) {
        this(buildModel(stateDim, actionDim));
    }

    public RLOOAgent(long stateDim, long actionDim, float lr, float gamma) {
        this(buildModel(stateDim, actionDim, lr));
        this.lr = lr;
        this.gamma = gamma;
    }

    private RLOOAgent(Object[] built) {
        super((AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer());
        this.gamma = 0.99f;
        this.lr = 3e-4f;
        this.obsNormalizer = new RunningMeanStd(guessStateDim((AbstractActorCritic) built[0]));
    }

    private static Object[] buildModel(long stateDim, long actionDim) {
        return buildModel(stateDim, actionDim, 3e-4f);
    }

    private static Object[] buildModel(long stateDim, long actionDim, float lr) {
        AbstractActorCritic model = new CartPoleActorCritic(stateDim, actionDim);
        AdamOptions opt = new AdamOptions();
        opt.lr().put(lr);
        return new Object[]{model, new Adam(model.parameters(), opt)};
    }

    private static long guessStateDim(AbstractActorCritic model) {
        try {
            return model.getStateDim();
        } catch (Throwable t) {
            return 4;
        }
    }

    @Override
    public Tensor trainStep() {
        ReplayBuffer buffer = getReplayBuffer();
        if (buffer.size() == 0) {
            throw new IllegalStateException("RLOO replay buffer is empty");
        }

        // Get batched tensors from buffer
        List<Tensor> stateList = buffer.getStateList();
        Tensor states = stackList(stateList);
        Tensor actions = buffer.getActions();
        Tensor rewards = buffer.getRewards();

        // Compute discounted returns
        long T = rewards.size(0);
        Tensor returns = zeros_like(rewards);
        double acc = 0;
        for (long t = T - 1; t >= 0; t--) {
            double r = rewards.select(0, t).item_double();
            acc = r + gamma * acc;
            returns.select(0, t).fill_(new Scalar(acc));
        }

        // Advantages = returns - mean(returns)
        Tensor advantages = returns.sub(returns.mean());

        // Normalize advantages
        advantages = normalizeAdvantages(advantages);

        // Compute policy gradient loss
        model.train(true);
        Distribution dist = model.getDistribution(states);
        Tensor logProbs = sumActionLogProb(dist.log_prob(actions));
        Tensor loss = logProbs.mul(advantages).mean().neg();

        // Optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        return loss.detach();
    }

    @Override
    public Tensor[] sample(Tensor state) {
        model.train(true);
        Tensor st = maybeNormalizeObs(state);
        Distribution dist = model.getDistribution(st);
        Tensor action = dist.sample();
        Tensor logProb = sumActionLogProb(dist.log_prob(action));
        Tensor value = model.getValue(st);
        return new Tensor[]{
                action.detach().clone(),
                logProb.detach().clone(),
                value.detach().clone()
        };
    }

    private Tensor maybeNormalizeObs(Tensor state) {
        if (obsNormalizer != null) {
            obsNormalizer.update(state);
            return obsNormalizer.normalize(state);
        }
        return state;
    }

    private static Tensor normalizeAdvantages(Tensor advantages) {
        if (advantages.numel() < 2) {
            return advantages.sub(advantages.mean());
        }
        return advantages.sub(advantages.mean()).div(advantages.std().add(new Scalar(1e-8)));
    }

    private static Tensor stackList(List<Tensor> list) {
        if (list == null || list.isEmpty()) {
            throw new IllegalStateException("Empty tensor list");
        }
        Tensor[] arr = list.toArray(new Tensor[0]);
        return stack(new TensorVector(arr));
    }

    private static Tensor sumActionLogProb(Tensor logProb) {
        if (logProb.dim() <= 1) {
            return logProb;
        }
        return logProb.sum(-1);
    }

    public float getGamma() { return gamma; }
    public float getLearningRate() { return lr; }
}
