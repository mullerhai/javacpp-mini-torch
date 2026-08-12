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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.config;

/**
 * Configuration for Nash Mirror Descent (Nash-MD) trainer.
 *
 * <p>Nash-MD is a game-theoretic approach to LLM alignment that uses
 * mirror descent with Nash equilibrium as the learning target.
 *
 * @see org.bytedeco.pytorch.llm.trl.NashMDTrainer
 */
public class NashMDConfig extends TrainerConfig {
    private int numObjectives = 1;                    // Number of reward objectives
    private double eta = 0.1;                       // Mirror descent step size
    private double klTarget = 0.1;                   // Target KL divergence
    private double klCoef = 1.0;                    // KL loss coefficient
    private double clipRange = 0.2;                   // Policy clip range
    private double equilibriumTemperature = 1.0;       // Temperature for softmax
    private double equilibriumMomentum = 0.9;          // Momentum for running average
    private boolean useReferenceKl = true;            // Use KL to reference model

//    public NashMDConfig() {
//        super();
//    }

    public NashMDConfig(Builder b) {
        super(b);
        this.numObjectives = b.numObjectives;
        this.eta = b.eta;
        this.klTarget = b.klTarget;
        this.klCoef = b.klCoef;
        this.clipRange = b.clipRange;
        this.equilibriumTemperature = b.equilibriumTemperature;
        this.equilibriumMomentum = b.equilibriumMomentum;
        this.useReferenceKl = b.useReferenceKl;
    }

    /** Number of reward objectives (default: 1). */
    public int numObjectives() { return numObjectives; }

    /** Mirror descent step size (default: 0.1). */
    public double eta() { return eta; }

    /** Target KL divergence (default: 0.1). */
    public double klTarget() { return klTarget; }

    /** KL loss coefficient (default: 1.0). */
    public double klCoef() { return klCoef; }

    /** Policy clip range (default: 0.2). */
    public double clipRange() { return clipRange; }

    /** Temperature for softmax equilibrium computation (default: 1.0). */
    public double equilibriumTemperature() { return equilibriumTemperature; }

    /** Momentum for running average of equilibrium weights (default: 0.9). */
    public double equilibriumMomentum() { return equilibriumMomentum; }

    /** Use KL to reference model (default: true). */
    public boolean useReferenceKl() { return useReferenceKl; }

    // Builders
    public Builder toBuilder() { return new Builder(this); }
    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int numObjectives = 1;
        private double eta = 0.1;
        private double klTarget = 0.1;
        private double klCoef = 1.0;
        private double clipRange = 0.2;
        private double equilibriumTemperature = 1.0;
        private double equilibriumMomentum = 0.9;
        private boolean useReferenceKl = true;

        public Builder() {}

        public Builder(NashMDConfig config) {
//            super(config);
            this.numObjectives = config.numObjectives;
            this.eta = config.eta;
            this.klTarget = config.klTarget;
            this.klCoef = config.klCoef;
            this.clipRange = config.clipRange;
            this.equilibriumTemperature = config.equilibriumTemperature;
            this.equilibriumMomentum = config.equilibriumMomentum;
            this.useReferenceKl = config.useReferenceKl;
        }

        public Builder numObjectives(int v) { this.numObjectives = v; return this; }
        public Builder eta(double v) { this.eta = v; return this; }
        public Builder klTarget(double v) { this.klTarget = v; return this; }
        public Builder klCoef(double v) { this.klCoef = v; return this; }
        public Builder clipRange(double v) { this.clipRange = v; return this; }
        public Builder equilibriumTemperature(double v) { this.equilibriumTemperature = v; return this; }
        public Builder equilibriumMomentum(double v) { this.equilibriumMomentum = v; return this; }
        public Builder useReferenceKl(boolean v) { this.useReferenceKl = v; return this; }

        public NashMDConfig build() { return new NashMDConfig(this); }
    }
}
