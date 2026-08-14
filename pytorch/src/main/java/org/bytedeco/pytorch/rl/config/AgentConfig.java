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
package org.bytedeco.pytorch.rl.config;

import java.util.Objects;

/**
 * Unified configuration for reinforcement learning agents.
 *
 * <p>This class provides a common configuration base for all RL algorithms,
 * with sensible defaults that can be overridden.
 *
 * <p>Example usage:
 * <pre>{@code
 * AgentConfig config = AgentConfig.builder()
 *     .algorithm("ppo")
 *     .learningRate(3e-4)
 *     .gamma(0.99)
 *     .build();
 * }</pre>
 */
public class AgentConfig {

    // ==================== Algorithm ====================

    /** Algorithm identifier */
    private final String algorithm;

    /** Device for computation (cpu, cuda) */
    private final String device;

    // ==================== Learning Rate ====================

    /** Initial learning rate */
    private final double learningRate;

    /** Learning rate schedule type */
    private final LRScheduleType lrSchedule;

    /** Number of warmup steps */
    private final int warmupSteps;

    // ==================== Discount Factor ====================

    /** Discount factor gamma */
    private final double gamma;

    /** GAE lambda for advantage estimation */
    private final double gaeLambda;

    // ==================== PPO Specific ====================

    /** PPO clip epsilon */
    private final double clipEps;

    /** Entropy coefficient for exploration */
    private final double entropyCoeff;

    /** Value function coefficient */
    private final double valueCoeff;

    /** Maximum gradient norm for clipping */
    private final double maxGradNorm;

    /** Number of PPO epochs per update */
    private final int ppoEpochs;

    /** Mini-batch size */
    private final int miniBatchSize;

    // ==================== Network ====================

    /** Hidden layer sizes */
    private final int[] hiddenLayers;

    /** Activation function */
    private final ActivationType activation;

    // ==================== Training ====================

    /** Maximum training steps */
    private final int maxSteps;

    /** Batch size for updates */
    private final int batchSize;

    /** Gradient accumulation steps */
    private final int gradientAccumulationSteps;

    // ==================== Environment ====================

    /** Number of parallel environments */
    private final int numEnvs;

    /** Maximum episode length */
    private final int maxEpisodeLength;

    // ==================== Observational Normalization ====================

    /** Use running mean/std normalization for observations */
    private final boolean normalizeObservations;

    /** Clip range for normalized observations */
    private final double observationClipRange;

    // ==================== Enums ====================

    public enum LRScheduleType {
        CONSTANT,
        LINEAR_WARMUP,
        COSINE,
        STEP
    }

    public enum ActivationType {
        RELU,
        TANH,
        GELU,
        SILU
    }

    // ==================== Constructor ====================

    private AgentConfig(Builder builder) {
        this.algorithm = builder.algorithm;
        this.device = builder.device;
        this.learningRate = builder.learningRate;
        this.lrSchedule = builder.lrSchedule;
        this.warmupSteps = builder.warmupSteps;
        this.gamma = builder.gamma;
        this.gaeLambda = builder.gaeLambda;
        this.clipEps = builder.clipEps;
        this.entropyCoeff = builder.entropyCoeff;
        this.valueCoeff = builder.valueCoeff;
        this.maxGradNorm = builder.maxGradNorm;
        this.ppoEpochs = builder.ppoEpochs;
        this.miniBatchSize = builder.miniBatchSize;
        this.hiddenLayers = builder.hiddenLayers;
        this.activation = builder.activation;
        this.maxSteps = builder.maxSteps;
        this.batchSize = builder.batchSize;
        this.gradientAccumulationSteps = builder.gradientAccumulationSteps;
        this.numEnvs = builder.numEnvs;
        this.maxEpisodeLength = builder.maxEpisodeLength;
        this.normalizeObservations = builder.normalizeObservations;
        this.observationClipRange = builder.observationClipRange;
    }

    // ==================== Getters ====================

    public String algorithm() { return algorithm; }
    public String device() { return device; }
    public double learningRate() { return learningRate; }
    public LRScheduleType lrSchedule() { return lrSchedule; }
    public int warmupSteps() { return warmupSteps; }
    public double gamma() { return gamma; }
    public double gaeLambda() { return gaeLambda; }
    public double clipEps() { return clipEps; }
    public double entropyCoeff() { return entropyCoeff; }
    public double valueCoeff() { return valueCoeff; }
    public double maxGradNorm() { return maxGradNorm; }
    public int ppoEpochs() { return ppoEpochs; }
    public int miniBatchSize() { return miniBatchSize; }
    public int[] hiddenLayers() { return hiddenLayers; }
    public ActivationType activation() { return activation; }
    public int maxSteps() { return maxSteps; }
    public int batchSize() { return batchSize; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public int numEnvs() { return numEnvs; }
    public int maxEpisodeLength() { return maxEpisodeLength; }
    public boolean normalizeObservations() { return normalizeObservations; }
    public double observationClipRange() { return observationClipRange; }

    // ==================== Builder ====================

    public static Builder builder() {
        return new Builder();
    }

    public static Builder builder(String algorithm) {
        return new Builder().algorithm(algorithm);
    }

    public static class Builder {
        private String algorithm = "ppo";
        private String device = "cpu";
        private double learningRate = 3e-4;
        private LRScheduleType lrSchedule = LRScheduleType.CONSTANT;
        private int warmupSteps = 0;
        private double gamma = 0.99;
        private double gaeLambda = 0.95;
        private double clipEps = 0.2;
        private double entropyCoeff = 0.01;
        private double valueCoeff = 0.5;
        private double maxGradNorm = 0.5;
        private int ppoEpochs = 4;
        private int miniBatchSize = 64;
        private int[] hiddenLayers = {64, 64};
        private ActivationType activation = ActivationType.RELU;
        private int maxSteps = 1_000_000;
        private int batchSize = 256;
        private int gradientAccumulationSteps = 1;
        private int numEnvs = 8;
        private int maxEpisodeLength = 500;
        private boolean normalizeObservations = false;
        private double observationClipRange = 10.0;

        public Builder algorithm(String algorithm) {
            this.algorithm = Objects.requireNonNull(algorithm);
            return this;
        }

        public Builder device(String device) {
            this.device = device;
            return this;
        }

        public Builder learningRate(double lr) {
            this.learningRate = lr;
            return this;
        }

        public Builder lrSchedule(LRScheduleType schedule) {
            this.lrSchedule = schedule;
            return this;
        }

        public Builder warmupSteps(int steps) {
            this.warmupSteps = steps;
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

        public Builder clipEps(double clipEps) {
            this.clipEps = clipEps;
            return this;
        }

        public Builder entropyCoeff(double coeff) {
            this.entropyCoeff = coeff;
            return this;
        }

        public Builder valueCoeff(double coeff) {
            this.valueCoeff = coeff;
            return this;
        }

        public Builder maxGradNorm(double norm) {
            this.maxGradNorm = norm;
            return this;
        }

        public Builder ppoEpochs(int epochs) {
            this.ppoEpochs = epochs;
            return this;
        }

        public Builder miniBatchSize(int size) {
            this.miniBatchSize = size;
            return this;
        }

        public Builder hiddenLayers(int... layers) {
            this.hiddenLayers = layers;
            return this;
        }

        public Builder activation(ActivationType activation) {
            this.activation = activation;
            return this;
        }

        public Builder maxSteps(int steps) {
            this.maxSteps = steps;
            return this;
        }

        public Builder batchSize(int size) {
            this.batchSize = size;
            return this;
        }

        public Builder gradientAccumulationSteps(int steps) {
            this.gradientAccumulationSteps = steps;
            return this;
        }

        public Builder numEnvs(int num) {
            this.numEnvs = num;
            return this;
        }

        public Builder maxEpisodeLength(int length) {
            this.maxEpisodeLength = length;
            return this;
        }

        public Builder normalizeObservations(boolean normalize) {
            this.normalizeObservations = normalize;
            return this;
        }

        public Builder observationClipRange(double range) {
            this.observationClipRange = range;
            return this;
        }

        public AgentConfig build() {
            return new AgentConfig(this);
        }
    }
}
