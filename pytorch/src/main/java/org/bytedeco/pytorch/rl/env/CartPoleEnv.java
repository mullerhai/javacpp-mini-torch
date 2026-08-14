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

import org.bytedeco.pytorch.rl.StepResult;

import java.util.Random;

/**
 * Classic CartPole balancing task.
 *
 * <p>State: [x, x_dot, theta, theta_dot] where:
 * <ul>
 *   <li>x: cart position</li>
 *   <li>x_dot: cart velocity</li>
 *   <li>theta: pole angle</li>
 *   <li>theta_dot: pole angular velocity</li>
 * </ul>
 *
 * <p>Actions: 0 = push left, 1 = push right
 *
 * <p>Termination: x out of bounds [-2.4, 2.4] or theta out of bounds [-12°, 12°]
 *
 * @see <a href="https://gymnasium.farama.org/environments/classic_control/cart_pole/">Gymnasium CartPole</a>
 */
public class CartPoleEnv implements Env {

    // ==================== Physics Parameters ====================

    private double gravity = 9.8;
    private double masscart = 1.0;
    private double masspole = 0.1;
    private double length = 0.5; // half pole length
    private double force_mag = 10.0;
    private double tau = 0.02; // time step

    // ==================== State ====================

    private float[] state = new float[4]; // [x, x_dot, theta, theta_dot]
    private Random random = new Random();

    // ==================== Episode Tracking ====================

    private double cumulativeReward = 0.0;
    private int stepCount = 0;

    // ==================== Constants ====================

    private static final int ACTION_SPACE_SIZE = 2;
    private static final int OBSERVATION_DIM = 4;
    private static final int MAX_EPISODE_STEPS = 500;
    private static final double X_THRESHOLD = 2.4;
    private static final double THETA_THRESHOLD = 0.209; // ~12 degrees

    // ==================== Constructor ====================

    public CartPoleEnv() {}

    // ==================== Env Interface ====================

    @Override
    public StepResult reset() {
        random = new Random();
        for (int i = 0; i < OBSERVATION_DIM; i++) {
            state[i] = (random.nextFloat() - 0.5f) * 0.1f;
        }
        cumulativeReward = 0.0;
        stepCount = 0;
        return StepResult.builder()
                .legacyObservation(state.clone())
                .reward(0.0)
                .terminated(false)
                .build();
    }

    @Override
    public StepResult step(int action) {
        stepCount++;

        float x = state[0], x_dot = state[1], theta = state[2], theta_dot = state[3];
        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = Math.cos(theta);
        double sintheta = Math.sin(theta);

        double temp = (force + masspole * length * theta_dot * theta_dot * sintheta)
                / (masscart + masspole);
        double theta_acc = (gravity * sintheta - costheta * temp)
                / (length * (4.0 / 3.0 - masspole * costheta * costheta / (masscart + masspole)));
        double x_acc = temp - masspole * length * theta_acc * costheta / (masscart + masspole);

        state[0] = (float) (state[0] + x_dot * tau);
        state[1] = (float) (state[1] + x_acc * tau);
        state[2] = (float) (state[2] + theta_dot * tau);
        state[3] = (float) (state[3] + theta_acc * tau);

        boolean terminated = Math.abs(state[0]) > X_THRESHOLD
                || Math.abs(state[2]) > THETA_THRESHOLD;
        boolean truncated = stepCount >= MAX_EPISODE_STEPS;

        double reward = terminated ? 0.0 : 1.0;
        cumulativeReward += reward;

        return StepResult.builder()
                .legacyObservation(state.clone())
                .reward(reward)
                .terminated(terminated)
                .truncated(truncated)
                .build();
    }

    @Override
    public int actionSpaceSize() {
        return ACTION_SPACE_SIZE;
    }

    @Override
    public int observationDim() {
        return OBSERVATION_DIM;
    }

    @Override
    public double episodeReturn() {
        return cumulativeReward;
    }

    @Override
    public int episodeLength() {
        return stepCount;
    }

    @Override
    public int maxEpisodeLength() {
        return MAX_EPISODE_STEPS;
    }

    @Override
    public void seed(long seed) {
        this.random = new Random(seed);
    }

    // ==================== Additional Methods ====================

    /**
     * Get current state.
     */
    public float[] getState() {
        return state.clone();
    }

    /**
     * Get current x position.
     */
    public double getX() {
        return state[0];
    }

    /**
     * Get current pole angle.
     */
    public double getTheta() {
        return state[2];
    }
}
