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

import java.util.stream.IntStream;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Vectorized environment for parallel data collection.
 *
 * <p>Wraps multiple environment instances and executes them in parallel
 * for efficient batch data collection. Commonly used in PPO, A2C, and
 * other on-policy algorithms.
 *
 * <p>Example:
 * <pre>{@code
 * VectorEnv venv = new VectorEnv(8, CartPoleEnv::new);
 * Tensor batchedObs = venv.getStackedObs();
 * Tensor[] actions = agent.sampleBatch(batchedObs);
 * StepResult[] results = venv.step(actions);
 * }</pre>
 *
 * @param <E> Environment type
 */
public class VectorEnv<E extends Env> {

    private E[] envs;
    private int numEnvs;
    private float[][] currentObs;

    /**
     * Create vectorized environments.
     *
     * @param numEnvs Number of parallel environments
     * @param factory Environment factory
     */
    @SuppressWarnings("unchecked")
    public VectorEnv(int numEnvs, java.util.function.Supplier<E> factory) {
        this.numEnvs = numEnvs;
        this.envs = (E[]) new Env[numEnvs];
        this.currentObs = new float[numEnvs][];

        for (int i = 0; i < numEnvs; i++) {
            envs[i] = factory.get();
            currentObs[i] = envs[i].legacyReset();
        }
    }

    /**
     * Create vectorized CartPole environments.
     *
     * @param numEnvs Number of parallel environments
     */
    @SuppressWarnings("unchecked")
    public VectorEnv(int numEnvs) {
        this.numEnvs = numEnvs;
        this.envs = (E[]) new Env[numEnvs];
        this.currentObs = new float[numEnvs][];

        for (int i = 0; i < numEnvs; i++) {
            CartPoleEnv cartPole = new CartPoleEnv();
            envs[i] = (E) cartPole;
            currentObs[i] = cartPole.legacyReset();
        }
    }

    // ==================== Observation ====================

    /**
     * Get number of parallel environments.
     */
    public int getNumEnvs() {
        return numEnvs;
    }

    /**
     * Get observation dimension.
     */
    public int observationDim() {
        return currentObs[0] != null ? currentObs[0].length : 0;
    }

    /**
     * Stack all current observations into a single Tensor [N, obs_dim].
     */
    public Tensor getStackedObs() {
        int obsDim = currentObs[0].length;
        float[] flatObs = new float[numEnvs * obsDim];
        for (int i = 0; i < numEnvs; i++) {
            System.arraycopy(currentObs[i], 0, flatObs, i * obsDim, obsDim);
        }
        return tensor(flatObs).reshape(numEnvs, obsDim);
    }

    /**
     * Get individual environment at index.
     */
    public E getEnv(int index) {
        return envs[index];
    }

    // ==================== Step ====================

    /**
     * Execute one step in all environments in parallel.
     *
     * @param actions Array of actions, one per environment
     * @return Array of step results
     */
    public StepResult[] step(int[] actions) {
        return IntStream.range(0, numEnvs)
                .parallel()
                .mapToObj(i -> {
                    StepResult res = envs[i].step(actions[i]);
                    if (res.done()) {
                        // Auto-reset on episode end
                        currentObs[i] = envs[i].legacyReset();
                    } else {
                        currentObs[i] = res.legacyObservation();
                    }
                    return res;
                }).toArray(StepResult[]::new);
    }

    /**
     * Reset all environments.
     */
    public void resetAll() {
        for (int i = 0; i < numEnvs; i++) {
            currentObs[i] = envs[i].legacyReset();
        }
    }

    /**
     * Reset specific environment.
     */
    public void reset(int index) {
        currentObs[index] = envs[index].legacyReset();
    }

    // ==================== Batch Statistics ====================

    /**
     * Compute mean reward across all environments.
     */
    public double meanReward(StepResult[] results) {
        return java.util.Arrays.stream(results)
                .mapToDouble(StepResult::reward)
                .average()
                .orElse(0.0);
    }

    /**
     * Count number of finished episodes.
     */
    public int countFinished(StepResult[] results) {
        return (int) java.util.Arrays.stream(results)
                .filter(StepResult::done)
                .count();
    }

    /**
     * Check if all environments are done.
     */
    public boolean allDone(StepResult[] results) {
        for (StepResult r : results) {
            if (!r.done()) return false;
        }
        return true;
    }
}
