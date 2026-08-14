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
package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.rl.StepResult;

/**
 * Unified environment interface for reinforcement learning.
 *
 * <p>This interface provides a Gymnasium-compatible API for RL environments,
 * supporting both discrete and continuous action spaces. The interface has been
 * updated to support Tensor-based observations for PyTorch native compatibility.
 *
 * <p>Key features:
 * <ul>
 *   <li>Unified step interface returning {@link StepResult}</li>
 *   <li>Support for both Tensor and float[] observations</li>
 *   <li>Episodic return tracking</li>
 *   <li>Episode statistics</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * Env env = new CartPoleEnv();
 * StepResult result = env.reset();
 *
 * while (!result.done()) {
 *     int action = selectAction(result.observation());
 *     result = env.step(action);
 *     agent.update(result);
 * }
 * }</pre>
 *
 * @see StepResult for unified result container
 * @see <a href="https://gymnasium.farama.org/">Gymnasium RL Library</a>
 */
public interface Env {

    // ==================== Core Methods ====================

    /**
     * Reset the environment to initial state.
     *
     * @return Initial observation wrapped in {@link StepResult}
     */
    StepResult reset();

    /**
     * Execute one step in the environment.
     *
     * @param action The action to take (type depends on action space)
     * @return Result containing next observation, reward, and done flags
     */
    StepResult step(int action);

    // ==================== Observation Space ====================

    /**
     * Get the observation dimension.
     *
     * @return Number of observation features
     */
    default int observationDim() {
        float[] sample = legacyReset();
        return sample != null ? sample.length : 0;
    }

    /**
     * Get the action space size (for discrete actions).
     *
     * @return Number of possible actions
     */
    int actionSpaceSize();

    // ==================== Legacy Support ====================

    /**
     * Legacy reset method returning float array.
     *
     * @return Initial observation as float array
     * @deprecated Use {@link #reset()} instead
     */
    @Deprecated
    default float[] legacyReset() {
        StepResult result = reset();
        if (result.legacyObservation() != null) {
            return result.legacyObservation();
        }
        Tensor obs = result.observation();
        if (obs != null) {
            long n = obs.numel();
            float[] data = new float[(int) n];
            Tensor cpu = obs.contiguous().cpu();
            org.bytedeco.javacpp.FloatPointer fp = cpu.data_ptr_float();
            for (int i = 0; i < n; i++) {
                data[i] = fp.get(i);
            }
            cpu.close();
            return data;
        }
        return new float[0];
    }

    /**
     * Legacy step method for backward compatibility.
     *
     * @param action The action to take
     * @return Result as legacy float array
     * @deprecated Use {@link #step(int)} instead
     */
    @Deprecated
    default float[] legacyStep(int action) {
        StepResult result = step(action);
        return result.legacyObservation();
    }

    // ==================== Episode Tracking ====================

    /**
     * Get cumulative reward for current episode.
     *
     * @return Sum of rewards since last reset
     */
    default double episodeReturn() {
        return 0.0;
    }

    /**
     * Get current episode length.
     *
     * @return Number of steps in current episode
     */
    default int episodeLength() {
        return 0;
    }

    // ==================== Environment Properties ====================

    /**
     * Check if the environment supports continuous actions.
     *
     * @return true if continuous, false if discrete
     */
    default boolean isContinuous() {
        return false;
    }

    /**
     * Get maximum episode length.
     *
     * @return Maximum number of steps per episode, or -1 for unlimited
     */
    default int maxEpisodeLength() {
        return -1;
    }

    // ==================== Seeding ====================

    /**
     * Set random seed for reproducibility.
     *
     * @param seed The seed value
     */
    default void seed(long seed) {
        // Default implementation does nothing
    }
}
