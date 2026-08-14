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
 * Configuration for TDPO (Token-level Direct Preference Optimization) trainer.
 *
 * <p>TDPO extends DPO to the token level, applying preference optimization
 * at each token position rather than just at the sequence level. This provides
 * finer-grained control for tasks requiring precise token-level alignment.
 *
 * <p>Key differences from DPO:
 * <ul>
 *   <li>Token-level KL divergence computation</li>
 *   <li>Forward KL regularization at each token</li>
 *   <li>Better for sequential token prediction tasks</li>
 * </ul>
 *
 * <p>Reference: "Token-level Direct Preference Optimization" (Dong et al., 2024)
 *
 * @see org.bytedeco.pytorch.llm.trl.TDPOTrainer
 */
public final class TDPOConfig extends TrainerConfig {

    // ==================== TDPO-specific Parameters ====================

    /** KL penalty coefficient for token-level divergence (default: 0.1) */
    private final double beta;

    /** Clipping range for PPO-style clipping (default: 0.2) */
    private final double clipRange;

    /** Forward KL coefficient for regularization (default: 0.1) */
    private final double forwardKlCoef;

    /** Use token-level advantage (vs sequence-level) (default: true) */
    private final boolean tokenLevelAdvantage;

    /** Bootstrap with last token value (default: false) */
    private final boolean bootstrapFromLastToken;

    // ==================== Constructor ====================

    private TDPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.clipRange = b.clipRange;
        this.forwardKlCoef = b.forwardKlCoef;
        this.tokenLevelAdvantage = b.tokenLevelAdvantage;
        this.bootstrapFromLastToken = b.bootstrapFromLastToken;
    }

    // ==================== Getters ====================

    /**
     * KL penalty coefficient (β).
     * Controls strength of reference model regularization.
     * Default: 0.1
     */
    public double beta() { return beta; }

    /**
     * Clipping range for importance ratio.
     * Default: 0.2
     */
    public double clipRange() { return clipRange; }

    /**
     * Forward KL divergence coefficient.
     * Used for regularization when forward KL is positive.
     * Default: 0.1
     */
    public double forwardKlCoef() { return forwardKlCoef; }

    /**
     * Whether to use token-level advantage estimation.
     * When true, computes advantage at each token position.
     * Default: true
     */
    public boolean tokenLevelAdvantage() { return tokenLevelAdvantage; }

    /**
     * Whether to bootstrap value estimates from last token.
     * Default: false
     */
    public boolean bootstrapFromLastToken() { return bootstrapFromLastToken; }

    // ==================== Builder ====================

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {

        private double beta = 0.1;
        private double clipRange = 0.2;
        private double forwardKlCoef = 0.1;
        private boolean tokenLevelAdvantage = true;
        private boolean bootstrapFromLastToken = false;

        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be non-negative");
            this.beta = v;
            return this;
        }

        public Builder clipRange(double v) {
            if (v < 0) throw new IllegalArgumentException("clipRange must be non-negative");
            this.clipRange = v;
            return this;
        }

        public Builder forwardKlCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("forwardKlCoef must be non-negative");
            this.forwardKlCoef = v;
            return this;
        }

        public Builder tokenLevelAdvantage(boolean v) {
            this.tokenLevelAdvantage = v;
            return this;
        }

        public Builder bootstrapFromLastToken(boolean v) {
            this.bootstrapFromLastToken = v;
            return this;
        }

        @Override
        public TDPOConfig build() { return new TDPOConfig(this); }
    }

    @Override
    public String toString() {
        return "TDPOConfig{" +
                "beta=" + beta +
                ", clipRange=" + clipRange +
                ", forwardKlCoef=" + forwardKlCoef +
                ", tokenLevelAdvantage=" + tokenLevelAdvantage +
                '}';
    }
}
