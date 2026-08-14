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
package org.bytedeco.pytorch.rl;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;
import org.bytedeco.pytorch.rl.env.SimpleTradingEnv;
import org.bytedeco.pytorch.rl.trainer.GRPOTrainer;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Trading demo using GRPO (Group Relative Policy Optimization).
 *
 * <p>Demonstrates the enterprise RL framework with:
 * <ul>
 *   <li>Unified {@link StepResult} for environment interactions</li>
 *   <li>Modern {@link org.bytedeco.pytorch.rl.env.Env} interface</li>
 *   <li>GRPOTrainer for group-relative advantage estimation</li>
 * </ul>
 *
 * @see SimpleTradingEnv for environment implementation
 * @see GRPOTrainer for training
 */
public class TradingSystem {
    public static void main(String[] args) {
        // 1. Mock prices (sine + noise)
        float[] mockPrices = new float[200];
        for (int i = 0; i < 200; i++) {
            mockPrices[i] = (float) (100 + 10 * Math.sin(i * 0.1) + Math.random() * 2);
        }

        // 2. Create environment and model
        SimpleTradingEnv env = new SimpleTradingEnv(mockPrices);
        CartPoleActorCritic model = new CartPoleActorCritic(5, 3);
        ReplayBuffer buffer = new ReplayBuffer();

        // 3. Create trainer with GRPO algorithm
        GRPOTrainer trainer = new GRPOTrainer(model, 1e-4, 0.2f, 4);

        int episodes = args != null && args.length > 0 ? Integer.parseInt(args[0]) : 20;

        System.out.println("=== Trading RL Demo with GRPO ===");
        System.out.println("Environment: SimpleTradingEnv (5-dim obs, 3 actions)");
        System.out.println("Algorithm: GRPO (Group Relative Policy Optimization)");
        System.out.println();

        for (int episode = 0; episode < episodes; episode++) {
            // Reset environment using unified interface
            StepResult result = env.reset();

            try (PointerScope scope = new PointerScope()) {
                double episodeReturn = 0.0;

                while (!result.done()) {
                    // Get observation and create batched tensor
                    float[] obs = result.legacyObservation();
                    Tensor stateTensor = tensor(obs).reshape(1, 5);

                    // Sample action from policy
                    Categorical dist = model.forward_policy(stateTensor);
                    Tensor actionTensor = dist.sample();
                    Tensor logp = dist.log_prob(actionTensor);

                    // Environment step using unified interface
                    result = env.step((int) actionTensor.item_long());
                    episodeReturn += result.reward();

                    // Push to buffer for training
                    // GRPO uses returns as group scores for normalization
                    buffer.push(
                            stateTensor.squeeze(0),
                            actionTensor.reshape(-1),
                            logp.reshape(-1),
                            zeros(new long[]{1}),
                            scalar_tensor(new Scalar((float) result.reward()),
                                    new TensorOptions().dtype(new ScalarTypeOptional(kFloat()))).reshape(1)
                    );
                }

                // Train on collected experience
                trainer.trainBatch(buffer);
                buffer.clear();

                System.out.printf("Episode %3d: return=%.4f, env_steps=%d%n",
                        episode, episodeReturn, env.episodeLength());
            }
        }

        System.out.println("\n=== Training Complete ===");
        trainer.close();
    }
}
