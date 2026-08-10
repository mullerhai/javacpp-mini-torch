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
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.config;

import java.util.Objects;

/**
 * Configuration for RLOO (REINFORCE Leave-One-Out) trainer.
 *
 * <p>RLOO is a simplified policy gradient algorithm from Meta AI that estimates
 * advantage by comparing each sample's reward to the mean of other samples in
 * the same batch. Unlike PPO, RLOO doesn't require a value function.
 *
 * <p>Reference: "A Minimalist Approach to LLM Reinforcement Learning" (Meta AI)
 *
 * <pre>{@code
 * RLOOConfig config = RLOOConfig.builder()
 *     .beta(0.01)           // KL penalty coefficient
 *     .klTarget(6.0)        // Target KL divergence
 *     .lr(1e-5)
 *     .build();
 * }</pre>
 */
public final class RLOOConfig extends TrainerConfig {
    private final double beta;
    private final double klTarget;
    private final double klEpsilon;
    private final double baselineCoeff;
    private final boolean useAdaptiveKL;
    private final int numSamples;

    private RLOOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.klTarget = b.klTarget;
        this.klEpsilon = b.klEpsilon;
        this.baselineCoeff = b.baselineCoeff;
        this.useAdaptiveKL = b.useAdaptiveKL;
        this.numSamples = b.numSamples;
    }

    /** Coefficient for KL penalty against reference model. */
    public double beta() { return beta; }

    /** Target KL divergence for adaptive KL (when enabled). */
    public double klTarget() { return klTarget; }

    /** KL epsilon for clipping (unused if useAdaptiveKL is false). */
    public double klEpsilon() { return klEpsilon; }

    /** Coefficient for baseline (mean reward) in advantage estimation. */
    public double baselineCoeff() { return baselineCoeff; }

    /** Use adaptive KL coefficient based on KL target. */
    public boolean useAdaptiveKL() { return useAdaptiveKL; }

    /** Number of samples per prompt (for LOO baseline). */
    public int numSamples() { return numSamples; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.01;
        private double klTarget = 6.0;
        private double klEpsilon = 0.2;
        private double baselineCoeff = 0.99;
        private boolean useAdaptiveKL = true;
        private int numSamples = 4;

        public Builder beta(double beta) {
            this.beta = Math.max(0, beta);
            return this;
        }

        public Builder klTarget(double klTarget) {
            this.klTarget = Math.max(0, klTarget);
            return this;
        }

        public Builder klEpsilon(double klEpsilon) {
            this.klEpsilon = Math.max(0.01, klEpsilon);
            return this;
        }

        public Builder baselineCoeff(double baselineCoeff) {
            this.baselineCoeff = Math.max(0, Math.min(1, baselineCoeff));
            return this;
        }

        public Builder useAdaptiveKL(boolean useAdaptiveKL) {
            this.useAdaptiveKL = useAdaptiveKL;
            return this;
        }

        public Builder numSamples(int numSamples) {
            this.numSamples = Math.max(2, numSamples);
            return this;
        }

        @Override
        public RLOOConfig build() {
            return new RLOOConfig(this);
        }
    }

    @Override
    public String toString() {
        return "RLOOConfig{" +
                "beta=" + beta +
                ", klTarget=" + klTarget +
                ", klEpsilon=" + klEpsilon +
                ", baselineCoeff=" + baselineCoeff +
                ", useAdaptiveKL=" + useAdaptiveKL +
                ", numSamples=" + numSamples +
                '}';
    }
}
