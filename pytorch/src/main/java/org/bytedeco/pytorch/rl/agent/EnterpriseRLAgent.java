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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.agent.RunningMeanStd;
import org.bytedeco.pytorch.rl.config.AgentConfig;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.env.Env;
import org.bytedeco.pytorch.rl.env.VectorEnv;
import org.bytedeco.pytorch.rl.sampler.ParallelSampler;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.distribution.Distribution;

import java.util.function.Consumer;

/**
 * Enterprise-grade base class for reinforcement learning agents.
 *
 * <p>Provides a comprehensive framework for enterprise RL applications with:
 * <ul>
 *   <li>Automatic observation normalization</li>
 *   <li>Integrated sampling with VectorEnv</li>
 *   <li>Callback system for monitoring</li>
 *   <li>Training statistics tracking</li>
 *   <li>Checkpointing support</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * EnterpriseRLAgent agent = EnterpriseRLAgent.builder(env)
 *     .algorithm("ppo")
 *     .learningRate(3e-4)
 *     .numEnvs(8)
 *     .onStep(metrics -> logger.info("Step: " + metrics))
 *     .build();
 *
 * while (globalStep < 1_000_000) {
 *     agent.collectExperience();
 *     agent.update();
 * }
 * }</pre>
 */
public class EnterpriseRLAgent extends AbstractRLAgent {

    // ==================== Configuration ====================

    protected final AgentConfig config;

    // ==================== Environment ====================

    protected final VectorEnv<?> vectorEnv;
    protected final int numEnvs;

    // ==================== Normalization ====================

    protected final RunningMeanStd obsNormalizer;

    // ==================== Sampling ====================

    protected ParallelSampler sampler;

    // ==================== Callbacks ====================

    protected Consumer<TrainingMetrics> onStepCallback;
    protected Consumer<TrainingMetrics> onEpisodeCallback;

    // ==================== Statistics ====================

    protected long totalEnvSteps = 0;
    protected double runningMeanReward = 0.0;
    protected int episodeCount = 0;

    // ==================== Constructors ====================

    /**
     * Create with VectorEnv.
     */
    protected EnterpriseRLAgent(
            AbstractActorCritic model,
            Optimizer optimizer,
            ReplayBuffer replayBuffer,
            VectorEnv<?> vectorEnv,
            AgentConfig config) {
        super(model, optimizer, replayBuffer);
        this.config = config;
        this.vectorEnv = vectorEnv;
        this.numEnvs = config.numEnvs();
        // Initialize normalizer with observation dimension
        if (config.normalizeObservations() && vectorEnv != null) {
            this.obsNormalizer = new RunningMeanStd(vectorEnv.observationDim());
        } else {
            this.obsNormalizer = null;
        }
    }

    /**
     * Create without VectorEnv.
     */
    protected EnterpriseRLAgent(
            AbstractActorCritic model,
            Optimizer optimizer,
            ReplayBuffer replayBuffer,
            AgentConfig config) {
        this(model, optimizer, replayBuffer, null, config);
    }

    // ==================== Builder ====================

    /**
     * Create a builder for EnterpriseRLAgent.
     */
    public static Builder builder(Env env) {
        return new Builder(env);
    }

    /**
     * Create a builder with VectorEnv.
     */
    public static Builder builder(VectorEnv<?> vectorEnv) {
        return new Builder(vectorEnv);
    }

    // ==================== Normalization ====================

    /**
     * Normalize observation using running statistics.
     */
    protected Tensor normalizeObservation(Tensor obs) {
        if (obsNormalizer == null) {
            return obs;
        }
        obsNormalizer.update(obs);
        return obsNormalizer.normalize(obs);
    }

    // ==================== Experience Collection ====================

    /**
     * Collect experience from environment.
     * Subclasses can override for custom collection logic.
     */
    public void collectExperience() {
        // Default implementation collects one step from vector env
        if (vectorEnv == null) return;

        Tensor batchedObs = vectorEnv.getStackedObs();
        Tensor normalizedObs = normalizeObservation(batchedObs);

        Tensor[] samples = sample(normalizedObs);
        // Process samples and push to buffer
        // Subclasses should implement specific logic
        for (Tensor t : samples) {
            if (t != null) t.close();
        }
        normalizedObs.close();
    }

    // ==================== Statistics ====================

    /**
     * Record reward from episode.
     */
    public void recordEpisodeReward(double reward) {
        episodeCount++;
        // Running mean update
        runningMeanReward = runningMeanReward * 0.99 + reward * 0.01;
    }

    /**
     * Get current mean reward.
     */
    public double getMeanReward() {
        return runningMeanReward;
    }

    /**
     * Get episode count.
     */
    public int getEpisodeCount() {
        return episodeCount;
    }

    // ==================== Callbacks ====================

    /**
     * Set step callback.
     */
    public void setOnStepCallback(Consumer<TrainingMetrics> callback) {
        this.onStepCallback = callback;
    }

    /**
     * Set episode callback.
     */
    public void setOnEpisodeCallback(Consumer<TrainingMetrics> callback) {
        this.onEpisodeCallback = callback;
    }

    /**
     * Fire step callback.
     */
    protected void fireStepCallback(TrainingMetrics metrics) {
        if (onStepCallback != null) {
            onStepCallback.accept(metrics);
        }
    }

    /**
     * Fire episode callback.
     */
    protected void fireEpisodeCallback(TrainingMetrics metrics) {
        if (onEpisodeCallback != null) {
            onEpisodeCallback.accept(metrics);
        }
    }

    // ==================== Configuration Access ====================

    public AgentConfig config() {
        return config;
    }

    // ==================== Abstract Methods Implementation ====================

    @Override
    public Tensor trainStep() {
        throw new UnsupportedOperationException(
            "trainStep() must be implemented by subclass");
    }

    @Override
    public Tensor[] sample(Tensor state) {
        throw new UnsupportedOperationException(
            "sample() must be implemented by subclass");
    }

    // ==================== Metrics ====================

    /**
     * Training metrics container.
     */
    public record TrainingMetrics(
            int step,
            long envSteps,
            double loss,
            double meanReward,
            int episodeCount,
            double learningRate,
            long forwardTimeMs,
            long backwardTimeMs
    ) {}

    // ==================== Builder ====================

    /**
     * Builder for EnterpriseRLAgent.
     */
    public static class Builder {
        private final Env env;
        private final VectorEnv<?> vectorEnv;

        private String algorithm = "ppo";
        private double learningRate = 3e-4;
        private int numEnvs = 8;
        private int[] hiddenLayers = {64, 64};
        private boolean normalizeObs = false;
        private double gamma = 0.99;
        private double gaeLambda = 0.95;
        private double clipEps = 0.2;
        private Consumer<TrainingMetrics> onStep;
        private Consumer<TrainingMetrics> onEpisode;

        private Builder(Env env) {
            this.env = env;
            this.vectorEnv = null;
        }

        private Builder(VectorEnv<?> vectorEnv) {
            this.env = null;
            this.vectorEnv = vectorEnv;
        }

        public Builder algorithm(String algo) {
            this.algorithm = algo;
            return this;
        }

        public Builder learningRate(double lr) {
            this.learningRate = lr;
            return this;
        }

        public Builder numEnvs(int n) {
            this.numEnvs = n;
            return this;
        }

        public Builder hiddenLayers(int... layers) {
            this.hiddenLayers = layers;
            return this;
        }

        public Builder normalizeObservations(boolean normalize) {
            this.normalizeObs = normalize;
            return this;
        }

        public Builder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public Builder gaeLambda(double lambda) {
            this.gaeLambda = lambda;
            return this;
        }

        public Builder clipEps(double eps) {
            this.clipEps = eps;
            return this;
        }

        public Builder onStep(Consumer<TrainingMetrics> callback) {
            this.onStep = callback;
            return this;
        }

        public Builder onEpisode(Consumer<TrainingMetrics> callback) {
            this.onEpisode = callback;
            return this;
        }

        public AgentConfig buildConfig() {
            return AgentConfig.builder(algorithm)
                    .learningRate(learningRate)
                    .numEnvs(numEnvs)
                    .hiddenLayers(hiddenLayers)
                    .normalizeObservations(normalizeObs)
                    .gamma(gamma)
                    .gaeLambda(gaeLambda)
                    .clipEps(clipEps)
                    .build();
        }
    }
}
