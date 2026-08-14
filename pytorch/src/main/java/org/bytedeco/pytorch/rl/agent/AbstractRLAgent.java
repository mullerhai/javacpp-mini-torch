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

import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;

import java.util.Objects;

/**
 * Abstract base class for reinforcement learning agents.
 *
 * <p>This class unifies core behaviors across all training algorithms including
 * training, sampling, and resource management. Subclasses implement specific
 * algorithms like PPO, GRPO, A2C, etc.
 *
 * <p>Key features:
 * <ul>
 *   <li>Model and optimizer management</li>
 *   <li>Experience buffer integration</li>
 *   <li>Resource lifecycle management (AutoCloseable)</li>
 *   <li>Gradient tracking and computation</li>
 * </ul>
 *
 * @see PPOAgent
 * @see GRPOAgent
 * @see A2CAgent
 */
public abstract class AbstractRLAgent implements AutoCloseable {

    // ==================== Core Components ====================

    /** Policy-value model */
    protected final AbstractActorCritic model;

    /** Optimizer for model parameters */
    protected final Optimizer optimizer;

    /** Experience replay buffer */
    protected final ReplayBuffer replayBuffer;

    // ==================== State Tracking ====================

    /** Training step counter */
    protected int trainingStep = 0;

    /** Whether agent is in training mode */
    protected volatile boolean training = true;

    // ==================== Constructor ====================

    /**
     * Create a new RL agent.
     *
     * @param model Policy-value model
     * @param optimizer Optimizer
     * @param replayBuffer Experience buffer
     */
    protected AbstractRLAgent(AbstractActorCritic model, Optimizer optimizer, ReplayBuffer replayBuffer) {
        this.model = Objects.requireNonNull(model, "model");
        this.optimizer = optimizer;
        this.replayBuffer = replayBuffer;
    }

    // ==================== Core Methods ====================

    /**
     * Perform one training step.
     *
     * @return Training loss tensor
     */
    public abstract Tensor trainStep();

    /**
     * Sample actions from the policy.
     *
     * @param state State tensor [batch, obs_dim]
     * @return Array containing [action, log_prob, value] tensors
     */
    public abstract Tensor[] sample(Tensor state);

    /**
     * Sample a single action (for inference).
     *
     * @param state State tensor
     * @return Action tensor
     */
    public Tensor sampleAction(Tensor state) {
        Tensor[] result = sample(state);
        Tensor action = result[0];
        // Close intermediate tensors
        for (int i = 1; i < result.length; i++) {
            if (result[i] != null) result[i].close();
        }
        return action;
    }

    // ==================== Buffer Management ====================

    /**
     * Clear the experience buffer.
     */
    public void clearBuffer() {
        if (replayBuffer != null) {
            replayBuffer.clear();
        }
    }

    // ==================== Mode Control ====================

    /**
     * Set training mode.
     */
    public void train() {
        this.training = true;
        if (model != null) {
            model.train(true);
        }
    }

    /**
     * Set evaluation mode.
     */
    public void eval() {
        this.training = false;
        if (model != null) {
            model.eval();
        }
    }

    /**
     * Check if in training mode.
     */
    public boolean isTraining() {
        return training;
    }

    // ==================== Gradient Operations ====================

    /**
     * Freeze model parameters (for reference models, etc.).
     */
    protected void freezeModel(AbstractActorCritic model) {
        if (model == null) return;
        var paramsVector = model.parameters();
        var begin = paramsVector.begin();
        var end = paramsVector.end();
        while (!begin.equals(end)) {
            var param = begin.get();
            param.requires_grad_(false);
            begin.increment();
        }
        paramsVector.close();
    }

    /**
     * Zero gradients in optimizer.
     */
    protected void zeroGrad() {
        if (optimizer != null) {
            optimizer.zero_grad();
        }
    }

    /**
     * Perform optimizer step.
     */
    protected void step() {
        if (optimizer != null) {
            optimizer.step();
        }
    }

    // ==================== Lifecycle ====================

    /**
     * Close and release all resources.
     */
    @Override
    public void close() {
        training = false;
        if (model != null) {
            model.close();
        }
        if (optimizer != null) {
            optimizer.close();
        }
        if (replayBuffer != null) {
            replayBuffer.clear();
        }
    }

    // ==================== Getters ====================

    public AbstractActorCritic getModel() {
        return model;
    }

    public ReplayBuffer getReplayBuffer() {
        return replayBuffer;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public int getTrainingStep() {
        return trainingStep;
    }

    /**
     * Get algorithm identifier.
     */
    public String algorithm() {
        return getClass().getSimpleName().replace("Agent", "");
    }

    // ==================== Statistics ====================

    /**
     * Get trainable parameters.
     */
    public TensorVector parameters() {
        return model.parameters();
    }

    /**
     * Get number of parameters.
     */
    public long numParameters() {
        long count = 0;
        var params = model.parameters();
        var begin = params.begin();
        var end = params.end();
        while (!begin.equals(end)) {
            var param = begin.get();
            count += param.numel();
            begin.increment();
        }
        params.close();
        return count;
    }
}
