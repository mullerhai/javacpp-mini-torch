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
package org.bytedeco.pytorch.llm.trl.config;

/**
 * Configuration for SPO (Self-Play Preference Optimization) trainer.
 *
 * <p>SPO uses a self-play mechanism where the model competes against
 * different versions of itself. It has been shown to achieve better
 * alignment than DPO while being more robust to noise in human feedback.
 *
 * <p>Key features:
 * <ul>
 *   <li>Self-play mechanism for robust learning</li>
 *   <li>No reference model needed</li>
 *   <li>Better robustness to noisy labels</li>
 *   <li>Game-theoretic convergence properties</li>
 * </ul>
 *
 * <p>Reference: "Self-Play Preference Optimization (SPO)" (Zhao et al., 2024)
 *
 * @see org.bytedeco.pytorch.llm.trl.SPOTrainer
 */
public final class SPOConfig extends TrainerConfig {

    // ==================== SPO-specific Parameters ====================

    /** Nash equilibrium temperature (default: 1.0) */
    private final double temperature;

    /** Self-play iterations per step (default: 3) */
    private final int selfPlayIterations;

    /** Strategy update rate (default: 0.1) */
    private final double updateRate;

    /** Use mixture of policies (default: true) */
    private final boolean useMixture;

    /** Mixture coefficient for historical policies (default: 0.3) */
    private final double mixtureCoeff;

    // ==================== Constructor ====================

    private SPOConfig(Builder b) {
        super(b);
        this.temperature = b.temperature;
        this.selfPlayIterations = b.selfPlayIterations;
        this.updateRate = b.updateRate;
        this.useMixture = b.useMixture;
        this.mixtureCoeff = b.mixtureCoeff;
    }

    // ==================== Getters ====================

    /**
     * Temperature for Nash equilibrium computation.
     * Default: 1.0
     */
    public double temperature() { return temperature; }

    /**
     * Number of self-play iterations per training step.
     * Default: 3
     */
    public int selfPlayIterations() { return selfPlayIterations; }

    /**
     * Strategy update rate for mixture model.
     * Default: 0.1
     */
    public double updateRate() { return updateRate; }

    /**
     * Whether to use mixture of historical policies.
     * Default: true
     */
    public boolean useMixture() { return useMixture; }

    /**
     * Coefficient for historical policies in mixture.
     * Default: 0.3
     */
    public double mixtureCoeff() { return mixtureCoeff; }

    // ==================== Builder ====================

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {

        private double temperature = 1.0;
        private int selfPlayIterations = 3;
        private double updateRate = 0.1;
        private boolean useMixture = true;
        private double mixtureCoeff = 0.3;

        public Builder temperature(double v) {
            if (v <= 0) throw new IllegalArgumentException("temperature must be positive");
            this.temperature = v;
            return this;
        }

        public Builder selfPlayIterations(int v) {
            if (v < 1) throw new IllegalArgumentException("selfPlayIterations must be >= 1");
            this.selfPlayIterations = v;
            return this;
        }

        public Builder updateRate(double v) {
            if (v <= 0 || v > 1) throw new IllegalArgumentException("updateRate must be in (0, 1]");
            this.updateRate = v;
            return this;
        }

        public Builder useMixture(boolean v) {
            this.useMixture = v;
            return this;
        }

        public Builder mixtureCoeff(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("mixtureCoeff must be in [0, 1]");
            this.mixtureCoeff = v;
            return this;
        }

        @Override
        public SPOConfig build() { return new SPOConfig(this); }
    }

    @Override
    public String toString() {
        return "SPOConfig{" +
                "temperature=" + temperature +
                ", selfPlayIterations=" + selfPlayIterations +
                ", updateRate=" + updateRate +
                ", useMixture=" + useMixture +
                '}';
    }
}
