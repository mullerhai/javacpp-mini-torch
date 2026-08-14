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
package org.bytedeco.pytorch.rl.loss;

import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Collection of on-policy reinforcement learning loss functions.
 *
 * <p>These losses are used by various RL agents for policy optimization.
 * All functions are static utilities that can be used independently of any agent.
 *
 * <p>Supported algorithms:
 * <ul>
 *   <li><b>PPO</b>: Proximal Policy Optimization with clipped surrogate</li>
 *   <li><b>REINFORCE</b>: Vanilla policy gradient with baseline</li>
 *   <li><b>AWR</b>: Advantage-Weighted Regression</li>
 *   <li><b>V-MPO</b>: Value-Maximizing Policy Optimization</li>
 *   <li><b>TD3</b>: Twin Delayed DDPG for continuous control</li>
 *   <li><b>SAC</b>: Soft Actor-Critic loss</li>
 * </ul>
 *
 * @see org.bytedeco.pytorch.rl.GAE for GAE computation
 */
public final class OnPolicyLoss {

    private OnPolicyLoss() {} // Static utility class

    // ==================== PPO Loss ====================

    /**
     * Compute PPO clipped surrogate loss.
     *
     * @param logProbs Current policy log probabilities
     * @param oldLogProbs Old policy log probabilities
     * @param advantages Advantage estimates
     * @param clipEps Clipping epsilon (typically 0.2)
     * @return Clipped surrogate loss
     */
    public static Tensor ppo(Tensor logProbs, Tensor oldLogProbs,
                            Tensor advantages, double clipEps) {
        Tensor ratio = exp(logProbs.sub(oldLogProbs));
        Tensor surr1 = ratio.mul(advantages);
        Tensor surr2 = clamp(ratio, new ScalarOptional(new Scalar(1 - clipEps)), new ScalarOptional(new Scalar(1 + clipEps))).mul(advantages);
        return minimum(surr1, surr2).mean().neg();
    }

    /**
     * Compute full PPO loss with value and entropy terms.
     *
     * @param dist Current policy distribution
     * @param actions Actions taken
     * @param oldLogProbs Old log probabilities
     * @param advantages Advantage estimates
     * @param values Current value estimates
     * @param returns Return estimates
     * @param clipEps PPO clipping epsilon
     * @param valueCoeff Value loss coefficient
     * @param entropyCoeff Entropy bonus coefficient
     * @return Combined PPO loss
     */
    public static Tensor ppoFull(Distribution dist, Tensor actions,
                                 Tensor oldLogProbs, Tensor advantages,
                                 Tensor values, Tensor returns,
                                 double clipEps, double valueCoeff, double entropyCoeff) {
        // Policy loss
        Tensor logProbs = sumActionLogProb(dist.log_prob(actions));
        Tensor policyLoss = ppo(logProbs, oldLogProbs, advantages, clipEps);

        // Value loss
        Tensor valueLoss = mse_loss(values, returns);

        // Entropy bonus
        Tensor entropy = dist.entropy().mean();

        return policyLoss
                .add(valueLoss.mul(new Scalar(valueCoeff)))
                .sub(entropy.mul(new Scalar(entropyCoeff)));
    }

    // ==================== REINFORCE Loss ====================

    /**
     * Compute REINFORCE loss (vanilla policy gradient with baseline).
     *
     * @param logProbs Policy log probabilities
     * @param advantages Advantage estimates (already include GAE or Monte Carlo returns)
     * @return Policy gradient loss
     */
    public static Tensor reinforce(Tensor logProbs, Tensor advantages) {
        return logProbs.mul(advantages).mean().neg();
    }

    /**
     * Compute REINFORCE with entropy regularization.
     */
    public static Tensor reinforceWithEntropy(Tensor logProbs, Tensor advantages,
                                             Distribution dist, double entropyCoeff) {
        Tensor policyLoss = reinforce(logProbs, advantages);
        Tensor entropyBonus = dist.entropy().mean().mul(new Scalar(entropyCoeff));
        return policyLoss.sub(entropyBonus);
    }

    // ==================== AWR Loss ====================

    /**
     * Compute Advantage-Weighted Regression (AWR) loss.
     *
     * <p>AWR is a simple offline RL algorithm that weights samples by their advantages.
     * It performs well on continuous control tasks with batched data.
     *
     * <p>Reference: "Advantage-Weighted Regression: Simple and Scalable Off-Policy RL"
     * (Peng et al., 2019)
     *
     * @param logProbs Policy log probabilities
     * @param advantages Advantage estimates
     * @param beta Advantage weighting temperature
     * @param betaScale Scaling factor for advantages
     * @return AWR weighted policy loss
     */
    public static Tensor awr(Tensor logProbs, Tensor advantages,
                            double beta, double betaScale) {
        // Weight samples by exp(beta * advantages)
        Tensor weights = exp(advantages.mul(new Scalar(beta)).div(new Scalar(betaScale)));
        // Normalize weights
        weights = weights.div(weights.sum());
        // Weighted policy gradient
        return logProbs.mul(weights).mean().neg();
    }

    /**
     * Compute AWR with value function and regularization.
     */
    public static Tensor awrFull(Tensor logProbs, Tensor advantages,
                                 Tensor values, Tensor returns,
                                 Tensor refLogProbs,
                                 double beta, double betaScale,
                                 double valueCoeff, double klTarget) {
        // Policy loss
        Tensor policyLoss = awr(logProbs, advantages, beta, betaScale);

        // Value loss
        Tensor valueLoss = mse_loss(values, returns);

        // KL regularization (pull policy toward behavioral policy)
        Tensor klLoss = logProbs.sub(refLogProbs).mean();

        return policyLoss
                .add(valueLoss.mul(new Scalar(valueCoeff)))
                .add(klLoss);
    }

    // ==================== V-MPO Loss ====================

    /**
     * Compute Value-Maximizing Policy Optimization (V-MPO) loss.
     *
     * <p>V-MPO is an on-policy algorithm that uses target distribution
     * normalization for stable learning.
     *
     * <p>Reference: "V-MPO: On-Policy Maximum a Posteriori Policy Optimization"
     * (Sinha et al., 2022)
     *
     * @param logProbs Policy log probabilities
     * @param advantages Advantage estimates
     * @param temperature Target distribution temperature
     * @return V-MPO loss
     */
    public static Tensor vmpo(Tensor logProbs, Tensor advantages, double temperature) {
        // Compute target distribution
        Tensor advantagesNorm = advantages.sub(advantages.mean()).div(advantages.std().add(new Scalar(1e-8)));
        Tensor targetDist = exp(advantagesNorm.div(new Scalar(temperature)));
        targetDist = targetDist.div(targetDist.sum());

        // KL between current and target
        Tensor loss = targetDist.mul(logProbs).mean().neg();
        return loss;
    }

    // ==================== TD3 Loss ====================

    /**
     * Compute Twin Delayed DDPG (TD3) critic loss.
     *
     * <p>TD3 uses twin Q-networks and delayed policy updates for
     * more stable continuous control learning.
     *
     * <p>Reference: "Addressing Function Approximation Error in Actor-Critic Methods"
     * (Fujimoto et al., 2018)
     *
     * @param q1Values First Q-network values
     * @param q2Values Second Q-network values
     * @param targetQ Target Q-values from target networks
     * @return Critic loss (minimum of twin Q-values)
     */
    public static Tensor td3Critic(Tensor q1Values, Tensor q2Values, Tensor targetQ) {
        Tensor target = targetQ.detach();
        // Use MSE loss for TD3 critic
        return mse_loss(min(q1Values, q2Values), target).mul(new Scalar(0.5));
    }

    /**
     * Compute TD3 actor loss.
     */
    public static Tensor td3Actor(Tensor qValues, Tensor actions) {
        // Maximize Q-values (minimize negative Q-values)
        return qValues.mean().neg();
    }

    // ==================== SAC Loss ====================

    /**
     * Compute Soft Actor-Critic (SAC) loss.
     *
     * <p>SAC uses entropy regularization for exploration and
     * automatic temperature tuning.
     *
     * <p>Reference: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
     * (Haarnoja et al., 2018)
     *
     * @param logProbs Entropy log probabilities
     * @param qValues Q-value estimates
     * @param targetQ Target Q-values
     * @param alpha Entropy temperature
     * @return SAC critic loss
     */
    public static Tensor sacCritic(Tensor logProbs, Tensor qValues, Tensor targetQ, double alpha) {
        // J_Q = E[(Q - (r + gamma * (targetQ - alpha * logProbs)))^2]
        Tensor backup = targetQ.sub(logProbs.mul(new Scalar(alpha)));
        return mse_loss(qValues, backup);
    }

    /**
     * Compute SAC actor loss with automatic alpha tuning.
     */
    public static Tensor sacActor(Tensor logProbs, Tensor qValues, double targetEntropy) {
        // J_pi = E[alpha * logProbs - min_i Q_i]
        return exp(logProbs).mul(logProbs.add(new Scalar(targetEntropy))).mean()
                .add(qValues.mean().neg());
    }

    // ==================== Utility Methods ====================

    /**
     * Sum log probabilities over action dimensions.
     */
    private static Tensor sumActionLogProb(Tensor logProb) {
        if (logProb.dim() <= 1) {
            return logProb;
        }
        return logProb.sum(-1);
    }

    /**
     * Compute Monte Carlo returns from rewards.
     *
     * @param rewards [T] reward tensor
     * @param dones [T] done tensor (1 = episode continues, 0 = episode ended)
     * @param gamma Discount factor
     * @return [T] returns tensor
     */
    public static Tensor monteCarloReturns(Tensor rewards, Tensor dones, double gamma) {
        long T = rewards.size(0);
        Tensor returns = zeros_like(rewards);

        double acc = 0;
        for (long t = T - 1; t >= 0; t--) {
            double reward = rewards.select(0, t).item_double();
            double mask = 1.0 - dones.select(0, t).item_double(); // 1 = continue, 0 = done
            acc = reward + gamma * acc * mask;
            returns.select(0, t).fill_(new Scalar(acc));
        }
        return returns;
    }

    /**
     * Normalize advantages.
     */
    public static Tensor normalizeAdvantages(Tensor advantages) {
        if (advantages.numel() < 2) {
            return advantages.sub(advantages.mean());
        }
        return advantages.sub(advantages.mean()).div(advantages.std().add(new Scalar(1e-8)));
    }
}
