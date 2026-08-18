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
package org.bytedeco.pytorch.rl.trainer;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.config.AgentConfig;

import java.util.Map;

/**
 * Unified trainer interface for reinforcement learning.
 *
 * <p>This interface provides a common contract for all RL trainers,
 * enabling polymorphic usage and consistent APIs across different algorithms.
 *
 * <p>Supported algorithms:
 * <ul>
 *   <li>PPO - Proximal Policy Optimization</li>
 *   <li>GRPO - Group Relative Policy Optimization</li>
 *   <li>A2C - Advantage Actor-Critic</li>
 *   <li>RLOO - Reward-Offered-Out-Off</li>
 *   <li>DPO - Direct Preference Optimization</li>
 * </ul>
 *
 * @see PPOTrainer for enterprise PPO implementation
 * @see org.bytedeco.pytorch.llm.trl.trainer.PPOTrainer for LLM-specific PPO
 */
public interface RLTrainer extends AutoCloseable {

    // ==================== Core Methods ====================

    /**
     * Train on a batch from the replay buffer.
     *
     * @param buffer Replay buffer containing experience
     */
    void trainBatch(ReplayBuffer buffer);

    /**
     * Compute loss without performing optimization step.
     * Useful for evaluation and debugging.
     *
     * @param buffer Replay buffer
     * @return Scalar loss tensor
     */
    default Tensor computeLoss(ReplayBuffer buffer) {
        throw new UnsupportedOperationException("computeLoss not implemented");
    }

    // ==================== Algorithm Identification ====================

    /**
     * Get algorithm identifier.
     * Standard names: "ppo", "grpo", "a2c", "rloo", "dpo", "sac", "td3"
     */
    default String algorithm() {
        return "unknown";
    }

    /**
     * Get human-readable algorithm name.
     */
    default String algorithmName() {
        return algorithm().toUpperCase();
    }

    // ==================== Mode Control ====================

    /**
     * Set training mode.
     */
    default void train() {}

    /**
     * Set evaluation mode.
     */
    default void eval() {}

    /**
     * Check if in training mode.
     */
    default boolean isTraining() {
        return true;
    }

    // ==================== State & Metrics ====================

    /**
     * Get current training step.
     */
    default int globalStep() {
        return 0;
    }

    /**
     * Reset accumulated metrics.
     */
    default void resetMetrics() {}

    // ==================== Lifecycle ====================

    /**
     * Check if trainer is closed.
     */
    default boolean isClosed() {
        return false;
    }

    @Override
    default void close() {}

    // ==================== Default Implementations ====================

    /**
     * Default metrics implementation.
     */
    default Map<String, Double> getMetrics() {
        return Map.of(
            "global_step", (double) globalStep(),
            "training", isTraining() ? 1.0 : 0.0
        );
    }

    // ==================== Factory Methods ====================

    /**
     * Create a trainer for the specified algorithm.
     *
     * @param algorithm Algorithm name
     * @param config   Agent configuration
     * @return Appropriate trainer instance
     */
    static RLTrainer create(String algorithm, AgentConfig config) {
        return switch (algorithm.toLowerCase()) {
            case "ppo" -> new PPOTrainer(
                    null, // Model set separately
                    null, // Optimizer set separately
                    (float) config.clipEps(),
                    (float) config.valueCoeff(),
                    (float) config.entropyCoeff(),
                    (float) config.maxGradNorm(),
                    config.ppoEpochs(),
                    config.miniBatchSize()
            );
            default -> throw new IllegalArgumentException(
                    "Unsupported algorithm: " + algorithm);
        };
    }
}
