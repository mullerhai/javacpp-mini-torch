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

import org.bytedeco.pytorch.Tensor;

import java.util.Objects;

/**
 * Unified step result container for reinforcement learning environments.
 *
 * <p>This record replaces the legacy {@link StepResult} class
 * with a unified interface that supports both Tensor-based (PyTorch native) and
 * legacy float[] observations for backward compatibility.
 *
 * <p>Supports Gymnasium-style termination/truncation separation via the {@code truncated}
 * field, enabling compatibility with modern RL training pipelines.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Modern Tensor-based usage
 * StepResult result = env.step(action);
 * Tensor obs = result.observation();
 *
 * // Legacy float[] conversion
 * float[] legacyObs = result.toFloatArray();
 * }</pre>
 *
 * @see <a href="https://gymnasium.farama.org/">Gymnasium RL Library</a>
 */
public final class StepResult {

    // ==================== Core Fields ====================

    /** Primary observation (Tensor for PyTorch native support) */
    private final Tensor observation;

    /** Legacy observation array (for backward compatibility) */
    private final float[] legacyObservation;

    /** Scalar reward for this step */
    private final double reward;

    /** Whether the episode has terminated (no more steps will be taken) */
    private final boolean terminated;

    /** Whether the episode has been truncated (time limit, safety, etc.) */
    private final boolean truncated;

    // ==================== Constructors ====================

    /**
     * Create a step result with Tensor observation (modern API).
     *
     * @param observation Tensor observation from the environment
     * @param reward     Scalar reward
     * @param terminated Whether episode terminated
     */
    public StepResult(Tensor observation, double reward, boolean terminated) {
        this(observation, null, reward, terminated, false);
    }

    /**
     * Create a step result with full Gymnasium-style fields.
     *
     * @param observation Tensor observation
     * @param reward      Scalar reward
     * @param terminated  Episode terminated (natural end)
     * @param truncated  Episode truncated (time limit, safety, etc.)
     */
    public StepResult(Tensor observation, double reward, boolean terminated, boolean truncated) {
        this(observation, null, reward, terminated, truncated);
    }

    /**
     * Create a step result with legacy float[] observation.
     * The Tensor observation will be created from the float array.
     *
     * @param legacyObservation Legacy float[] observation
     * @param reward           Scalar reward
     * @param terminated       Whether episode terminated
     */
    public StepResult(float[] legacyObservation, double reward, boolean terminated) {
        this(null, legacyObservation, reward, terminated, false);
    }

    /**
     * Create a step result with legacy float[] observation and truncation.
     *
     * @param legacyObservation Legacy float[] observation
     * @param reward           Scalar reward
     * @param terminated       Whether episode terminated
     * @param truncated        Whether episode truncated
     */
    public StepResult(float[] legacyObservation, double reward, boolean terminated, boolean truncated) {
        this(null, legacyObservation, reward, terminated, truncated);
    }

    /**
     * Full constructor with all fields.
     */
    private StepResult(Tensor observation, float[] legacyObservation,
                       double reward, boolean terminated, boolean truncated) {
        this.observation = observation;
        this.legacyObservation = legacyObservation;
        this.reward = reward;
        this.terminated = terminated;
        this.truncated = truncated;
    }

    // ==================== Factory Methods ====================

    /**
     * Create from a legacy StepResult object.
     *
     * @param legacy The legacy StepResult to convert
     * @return New StepResult with Tensor observation
     */
    public static StepResult fromLegacy(Object legacy) {
        if (legacy == null) {
            throw new IllegalArgumentException("Legacy StepResult cannot be null");
        }
        // Legacy StepResult has public fields: nextState, reward, done
        try {
            java.lang.reflect.Field nextStateField = legacy.getClass().getField("nextState");
            java.lang.reflect.Field rewardField = legacy.getClass().getField("reward");
            java.lang.reflect.Field doneField = legacy.getClass().getField("done");
            float[] nextState = (float[]) nextStateField.get(legacy);
            float reward = rewardField.getFloat(legacy);
            boolean done = doneField.getBoolean(legacy);
            return new StepResult(nextState, reward, done);
        } catch (Exception e) {
            throw new IllegalStateException("Cannot convert legacy StepResult", e);
        }
    }

    /**
     * Create a terminal step result (episode ended).
     *
     * @param observation Final observation
     * @param reward     Final reward
     * @return StepResult marked as terminated
     */
    public static StepResult terminal(Tensor observation, double reward) {
        return new StepResult(observation, reward, true, false);
    }

    /**
     * Create a truncated step result.
     *
     * @param observation Current observation
     * @param reward     Current reward
     * @return StepResult marked as truncated
     */
    public static StepResult truncated(Tensor observation, double reward) {
        return new StepResult(observation, reward, false, true);
    }

    // ==================== Accessors ====================

    /**
     * Get the Tensor observation (primary).
     *
     * @return Tensor observation, or null if only legacy observation was provided
     */
    public Tensor observation() {
        if (observation != null) {
            return observation;
        }
        if (legacyObservation != null) {
            return org.bytedeco.pytorch.global.torch.tensor(legacyObservation);
        }
        return null;
    }

    /**
     * Get the legacy float[] observation for backward compatibility.
     *
     * @return float[] observation, or null if only Tensor was provided
     */
    public float[] legacyObservation() {
        if (legacyObservation != null) {
            return legacyObservation;
        }
        if (observation != null) {
            long n = observation.numel();
            float[] data = new float[(int) n];
            Tensor cpu = observation.contiguous().cpu();
            org.bytedeco.javacpp.FloatPointer fp = cpu.data_ptr_float();
            for (int i = 0; i < n; i++) {
                data[i] = fp.get(i);
            }
            cpu.close();
            return data;
        }
        return null;
    }

    /**
     * Get the scalar reward.
     *
     * @return Reward value
     */
    public double reward() {
        return reward;
    }

    /**
     * Get the terminated flag (episode naturally ended).
     *
     * @return true if episode terminated
     */
    public boolean terminated() {
        return terminated;
    }

    /**
     * Get the truncated flag (episode ended due to external limits).
     *
     * @return true if episode truncated
     */
    public boolean truncated() {
        return truncated;
    }

    /**
     * Check if the episode is done (terminated or truncated).
     *
     * @return true if episode ended for any reason
     */
    public boolean done() {
        return terminated || truncated;
    }

    /**
     * Alias for {@link #terminated()}.
     */
    public boolean isDone() {
        return done();
    }

    /**
     * Alias for {@link #truncated()}.
     */
    public boolean isTruncated() {
        return truncated;
    }

    // ==================== Conversions ====================

    /**
     * Convert to float array.
     *
     * @return Observation as float array
     */
    public float[] toFloatArray() {
        return legacyObservation();
    }

    /**
     * Convert to a 2D Tensor with batch dimension.
     *
     * @return Observation as [1, obs_dim] Tensor
     */
    public Tensor toBatchTensor() {
        Tensor obs = observation();
        if (obs == null) return null;
        if (obs.dim() == 1) {
            return obs.unsqueeze(0);
        }
        return obs;
    }

    // ==================== Object Methods ====================

    @Override
    public String toString() {
        return String.format("StepResult{obs=%s, reward=%.4f, terminated=%s, truncated=%s}",
                observation != null ? "Tensor" : "float[]",
                reward, terminated, truncated);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StepResult that = (StepResult) o;
        return Double.compare(that.reward, reward) == 0
                && terminated == that.terminated
                && truncated == that.truncated;
    }

    @Override
    public int hashCode() {
        return Objects.hash(observation, legacyObservation, reward, terminated, truncated);
    }

    // ==================== Builder ====================

    /**
     * Create a new builder for StepResult.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for StepResult.
     */
    public static final class Builder {
        private Tensor observation;
        private float[] legacyObservation;
        private double reward = 0.0;
        private boolean terminated = false;
        private boolean truncated = false;

        public Builder observation(Tensor obs) {
            this.observation = obs;
            return this;
        }

        public Builder legacyObservation(float[] obs) {
            this.legacyObservation = obs;
            return this;
        }

        public Builder reward(double reward) {
            this.reward = reward;
            return this;
        }

        public Builder terminated(boolean terminated) {
            this.terminated = terminated;
            return this;
        }

        public Builder truncated(boolean truncated) {
            this.truncated = truncated;
            return this;
        }

        public StepResult build() {
            return new StepResult(observation, legacyObservation, reward, terminated, truncated);
        }
    }
}
