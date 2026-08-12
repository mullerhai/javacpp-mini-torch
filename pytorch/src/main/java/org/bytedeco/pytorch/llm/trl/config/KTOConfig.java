/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath> exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 * Configuration for Kahneman-Tversky Optimization (KTO) trainer.
 *
 * <p>Reference: "KTO: Kahneman-Tversky Optimization" (Meta AI, 2024)
 *
 * @see org.bytedeco.pytorch.llm.trl.KTOTrainer
 */
public class KTOConfig extends TrainerConfig {
    private double beta = 0.1;           // KL penalty coefficient
    private double gammaC = 1.0;          // Gain coefficient for chosen responses
    private double gammaD = 1.0;          // Loss coefficient for rejected responses
    private double alpha = 0.0;          // Reference point (default: 0)
    private double klTarget = 0.0;       // Target average KL (0 = no target)
    private double klDelta = 0.1;        // Delta for KL target tracking
    private boolean usePerVersionLoss = false;  // Use loss_v1 or loss_v2

//    public KTOConfig() {
//        super();
//    }

    public KTOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.gammaC = b.gammaC;
        this.gammaD = b.gammaD;
        this.alpha = b.alpha;
        this.klTarget = b.klTarget;
        this.klDelta = b.klDelta;
        this.usePerVersionLoss = b.usePerVersionLoss;
    }

    /** KL penalty coefficient (default: 0.1). */
    public double beta() { return beta; }

    /** Gain coefficient for chosen responses (default: 1.0). */
    public double gammaC() { return gammaC; }

    /** Loss coefficient for rejected responses (default: 1.0). */
    public double gammaD() { return gammaD; }

    /** Reference point / loss aversion parameter (default: 0.0). */
    public double alpha() { return alpha; }

    /** Target average KL divergence (default: 0 = no target). */
    public double klTarget() { return klTarget; }

    /** Delta for KL target tracking (default: 0.1). */
    public double klDelta() { return klDelta; }

    /** Use per-version loss (loss_v2). */
    public boolean usePerVersionLoss() { return usePerVersionLoss; }

    // Builders
    public Builder toBuilder() { return new Builder(this); }
    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private double gammaC = 1.0;
        private double gammaD = 1.0;
        private double alpha = 0.0;
        private double klTarget = 0.0;
        private double klDelta = 0.1;
        private boolean usePerVersionLoss = false;

        public Builder() {}

        public Builder(KTOConfig config) {
//            super(config);
            this.beta = config.beta;
            this.gammaC = config.gammaC;
            this.gammaD = config.gammaD;
            this.alpha = config.alpha;
            this.klTarget = config.klTarget;
            this.klDelta = config.klDelta;
            this.usePerVersionLoss = config.usePerVersionLoss;
        }

        public Builder beta(double v) { this.beta = v; return this; }
        public Builder gammaC(double v) { this.gammaC = v; return this; }
        public Builder gammaD(double v) { this.gammaD = v; return this; }
        public Builder alpha(double v) { this.alpha = v; return this; }
        public Builder klTarget(double v) { this.klTarget = v; return this; }
        public Builder klDelta(double v) { this.klDelta = v; return this; }
        public Builder usePerVersionLoss(boolean v) { this.usePerVersionLoss = v; return this; }

        public KTOConfig build() { return new KTOConfig(this); }
    }
}
