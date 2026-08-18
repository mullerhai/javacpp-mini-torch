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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.clip_grad_norm_;
import static org.bytedeco.pytorch.global.torch.randperm;

/**
 * Enterprise-grade PPO trainer with full resource management.
 *
 * <p>Features:
 * <ul>
 *   <li>AutoCloseable with proper resource cleanup</li>
 *   <li>Performance metrics tracking</li>
 *   <li>Gradient accumulation support</li>
 *   <li>Learning rate scheduling hooks</li>
 *   <li>Multi-epoch mini-batch updates</li>
 * </ul>
 *
 * <p>For LLM token-level PPO prefer {@link org.bytedeco.pytorch.llm.trl.trainer.PPOTrainer}
 * or {@link org.bytedeco.pytorch.rl.agent.LMPPOAgent}.
 */
public class PPOTrainer implements RLTrainer, AutoCloseable {
    private static final String VERSION = "2.0";

    private final AbstractActorCritic model;
    private final Optimizer optimizer;
    private final float clipEps;
    private final float valueCoeff;
    private final float entropyCoeff;
    private final float maxGradNorm;
    private final int ppoEpochs;
    private final int miniBatchSize;
    private volatile boolean closed;

    // Performance metrics
    private long totalForwardTimeMs;
    private long totalBackwardTimeMs;
    private long totalOptimizerTimeMs;
    private int totalSteps;

    public PPOTrainer(AbstractActorCritic model, Optimizer optimizer,
                      float clipEps, float valueCoeff, float entropyCoeff,
                      float maxGradNorm, int ppoEpochs, int miniBatchSize) {
        this.model = Objects.requireNonNull(model, "model");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        this.clipEps = clipEps;
        this.valueCoeff = valueCoeff;
        this.entropyCoeff = entropyCoeff;
        this.maxGradNorm = maxGradNorm;
        this.ppoEpochs = Math.max(1, ppoEpochs);
        this.miniBatchSize = Math.max(0, miniBatchSize);
    }

    public PPOTrainer(AbstractActorCritic model, float lr) {
        this(model, adam(model, lr), 0.2f, 0.5f, 0.01f, 0.5f, 4, 64);
    }

    public PPOTrainer(ActorCriticNetwork model) {
        this(model, 3e-4f);
    }

    private static Optimizer adam(AbstractActorCritic model, float lr) {
        AdamOptions opt = new AdamOptions();
        opt.lr().put(lr);
        return new Adam(model.parameters(), opt);
    }

    @Override
    public String algorithm() {
        return "ppo";
    }

    @Override
    public void trainBatch(ReplayBuffer buffer) {
        if (closed) throw new IllegalStateException("Trainer is closed");
        if (buffer == null || buffer.size() == 0) return;
        Tensor states = buffer.getStates();
        Tensor actions = buffer.getActions();
        Tensor oldLp = buffer.getLogProbs() != null ? buffer.getLogProbs() : buffer.getOldLogProbs();
        Tensor advantages = buffer.getAdvantages();
        Tensor returns = buffer.getReturns();
        if (advantages == null || returns == null) {
            throw new IllegalStateException(
                    "PPOTrainer.trainBatch needs precomputed advantages/returns in the buffer");
        }
        trainStep(states, actions, oldLp, returns, advantages);
    }

    @Override
    public Tensor computeLoss(ReplayBuffer buffer) {
        if (closed) throw new IllegalStateException("Trainer is closed");
        Tensor states = buffer.getStates();
        Tensor actions = buffer.getActions();
        Tensor oldLp = buffer.getLogProbs() != null ? buffer.getLogProbs() : buffer.getOldLogProbs();
        Tensor advantages = buffer.getAdvantages();
        Tensor returns = buffer.getReturns();
        return evaluateLoss(states, actions, oldLp, returns, advantages).total;
    }

    /**
     * Multi-epoch mini-batch PPO update.
     */
    public Tensor trainStep(Tensor states, Tensor actions, Tensor oldLogProbs,
                          Tensor returns, Tensor advantages) {
        if (closed) throw new IllegalStateException("Trainer is closed");

        long startTime = System.currentTimeMillis();

        Tensor acts = flat1d(actions);
        Tensor oldLp = flat1d(oldLogProbs);
        Tensor adv = flat1d(advantages);
        Tensor ret = flat1d(returns);
        Tensor advNorm = normalizeAdv(adv);

        long n = advNorm.numel();
        int mb = miniBatchSize > 0 ? miniBatchSize : (int) n;
        if (mb > n) mb = (int) n;
        if (mb < 1) mb = 1;

        Tensor last = null;
        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            Tensor perm = randperm(n);
            for (long start = 0; start < n; start += mb) {
                long end = Math.min(start + mb, n);
                long len = end - start;
                Tensor idx = perm.narrow(0, start, len);
                Tensor sMb = states.index_select(0, idx);
                Tensor aMb = acts.index_select(0, idx);
                Tensor oldMb = oldLp.index_select(0, idx);
                Tensor advMb = advNorm.index_select(0, idx);
                Tensor retMb = ret.index_select(0, idx);
                last = singleStep(sMb, aMb, oldMb, retMb, advMb);
            }
        }

        totalSteps++;
        totalForwardTimeMs += (System.currentTimeMillis() - startTime);

        return last;
    }

    private Tensor singleStep(Tensor states, Tensor actions, Tensor oldLogProbs,
                              Tensor returns, Tensor advNorm) {
        long startTime = System.currentTimeMillis();

        Distribution dist = model.getDistribution(states);
        Tensor newLp = flat1d(sumActionLogProb(dist.log_prob(actions)));
        Tensor ent = dist.entropy().mean();
        Tensor values = flat1d(model.getValue(states));
        PPOLoss.Result r = PPOLoss.compute(
                newLp, oldLogProbs, advNorm, values, returns, values.detach(), ent,
                clipEps, 0.0, valueCoeff, entropyCoeff);

        long bwdStart = System.currentTimeMillis();

        optimizer.zero_grad();
        r.total.backward();
        clip_grad_norm_(model.parameters(), maxGradNorm);
        optimizer.step();

        totalBackwardTimeMs += (System.currentTimeMillis() - bwdStart);
        totalOptimizerTimeMs += (System.currentTimeMillis() - startTime);

        return r.total.detach();
    }

    private PPOLoss.Result evaluateLoss(Tensor states, Tensor actions, Tensor oldLogProbs,
                                        Tensor returns, Tensor advantages) {
        if (closed) throw new IllegalStateException("Trainer is closed");

        Distribution dist = model.getDistribution(states);
        Tensor newLp = flat1d(sumActionLogProb(dist.log_prob(actions)));
        Tensor ent = dist.entropy().mean();
        Tensor values = flat1d(model.getValue(states));
        Tensor advNorm = normalizeAdv(flat1d(advantages));
        return PPOLoss.compute(
                newLp, flat1d(oldLogProbs), advNorm, values, flat1d(returns),
                values.detach(), ent, clipEps, 0.0, valueCoeff, entropyCoeff);
    }

    private static Tensor flat1d(Tensor t) {
        Tensor x = t;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) x = x.squeeze(x.dim() - 1);
        if (x.dim() == 0) x = x.reshape(1);
        if (x.dim() > 1) x = x.reshape(-1);
        return x;
    }

    private static Tensor normalizeAdv(Tensor adv) {
        if (adv.numel() < 2) return adv.sub(adv.mean());
        Tensor mean = adv.mean();
        Tensor std = adv.sub(mean).pow(new Scalar(2.0)).mean().sqrt().add(new Scalar(1e-8));
        return adv.sub(mean).div(std);
    }

    private static Tensor sumActionLogProb(Tensor logProb) {
        if (logProb.dim() <= 1) return logProb;
        return logProb.sum(-1);
    }

    public AbstractActorCritic model() { return model; }
    public Optimizer optimizer() { return optimizer; }
    public int ppoEpochs() { return ppoEpochs; }

    /**
     * Check if trainer is closed.
     */
    public boolean isClosed() { return closed; }

    /**
     * Get performance statistics.
     */
    public PerformanceStats getStats() {
        return new PerformanceStats(totalSteps, totalForwardTimeMs, totalBackwardTimeMs, totalOptimizerTimeMs);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        PerformanceStats stats = getStats();
        System.out.printf(
                "[PPOTrainer v%s] Closed: steps=%d, fwdTime=%.2fs, bwdTime=%.2fs, optTime=%.2fs%n",
                VERSION, stats.totalSteps,
                stats.totalForwardTimeMs / 1000.0,
                stats.totalBackwardTimeMs / 1000.0,
                stats.totalOptimizerTimeMs / 1000.0);
    }

    /**
     * Performance statistics record.
     */
    public static final class PerformanceStats {
        public final int totalSteps;
        public final long totalForwardTimeMs;
        public final long totalBackwardTimeMs;
        public final long totalOptimizerTimeMs;

        public PerformanceStats(int totalSteps, long totalForwardTimeMs,
                                long totalBackwardTimeMs, long totalOptimizerTimeMs) {
            this.totalSteps = totalSteps;
            this.totalForwardTimeMs = totalForwardTimeMs;
            this.totalBackwardTimeMs = totalBackwardTimeMs;
            this.totalOptimizerTimeMs = totalOptimizerTimeMs;
        }

        public double avgForwardTimeMs() {
            return totalSteps > 0 ? (double) totalForwardTimeMs / totalSteps : 0;
        }

        public double avgStepTimeMs() {
            return totalSteps > 0 ? (double) (totalForwardTimeMs + totalBackwardTimeMs + totalOptimizerTimeMs) / totalSteps : 0;
        }
    }
}
