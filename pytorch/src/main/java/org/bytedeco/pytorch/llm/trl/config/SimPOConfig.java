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
 * Configuration for SimPO (Simple Preference Optimization) trainer.
 *
 * <p>SimPO removes the reference model term from DPO and uses a target margin
 * on log probabilities instead, making it simpler and more memory-efficient.
 *
 * <p>Reference: "SimPO: Simple Preference Optimization" (Meng et al., 2024)
 * <a href="https://arxiv.org/abs/2405.14734">arXiv:2405.14734</a>
 *
 * @see org.bytedeco.pytorch.llm.trl.SimPOTrainer
 */
public final class SimPOConfig extends TrainerConfig {

    // ==================== SimPO-specific Parameters ====================

    /** Reward difference coefficient (default: 2.0) */
    private final double beta;

    /** Target reward margin to encourage separation between chosen/rejected (default: 1.0) */
    private final double targetMargin;

    /** Divide log-probs by sequence length to reduce length bias (default: true) */
    private final boolean lengthNormalize;

    /** Label smoothing for robustness (default: 0.0) */
    private final double labelSmoothing;

    // ==================== Constructor ====================

    private SimPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.targetMargin = b.targetMargin;
        this.lengthNormalize = b.lengthNormalize;
        this.labelSmoothing = b.labelSmoothing;
    }

    // ==================== Getters ====================

    /**
     * Reward difference coefficient (β).
     * Higher values increase sensitivity to reward differences.
     * Default: 2.0
     */
    public double beta() { return beta; }

    /**
     * Target reward margin (γ).
     * Encourages the model to achieve at least this much separation between
     * chosen and rejected responses.
     * Default: 1.0
     */
    public double targetMargin() { return targetMargin; }

    /**
     * Whether to normalize by sequence length.
     * Reduces length bias by dividing log-probs by |y|.
     * Default: true
     */
    public boolean lengthNormalize() { return lengthNormalize; }

    /**
     * Label smoothing factor for robustness.
     * Default: 0.0 (no smoothing)
     */
    public double labelSmoothing() { return labelSmoothing; }

    // ==================== Builder ====================

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {

        private double beta = 2.0;
        private double targetMargin = 1.0;
        private boolean lengthNormalize = true;
        private double labelSmoothing = 0.0;

        /**
         * Set reward coefficient (β).
         * Recommended range: [0.5, 4.0]
         */
        public Builder beta(double v) {
            if (v <= 0) throw new IllegalArgumentException("beta must be positive");
            this.beta = v;
            return this;
        }

        /**
         * Set target margin (γ).
         * Controls minimum separation between chosen and rejected.
         * Recommended range: [0.5, 2.0]
         */
        public Builder targetMargin(double v) {
            if (v < 0) throw new IllegalArgumentException("targetMargin must be non-negative");
            this.targetMargin = v;
            return this;
        }

        /**
         * Enable/disable length normalization.
         * When true, divides log-probs by sequence length.
         * Default: true (recommended)
         */
        public Builder lengthNormalize(boolean v) {
            this.lengthNormalize = v;
            return this;
        }

        /**
         * Set label smoothing factor.
         * Recommended range: [0.0, 0.2]
         */
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("labelSmoothing must be in [0, 1]");
            this.labelSmoothing = v;
            return this;
        }

        @Override
        public SimPOConfig build() { return new SimPOConfig(this); }
    }

    // ==================== Object Methods ====================

    @Override
    public String toString() {
        return "SimPOConfig{" +
                "beta=" + beta +
                ", targetMargin=" + targetMargin +
                ", lengthNormalize=" + lengthNormalize +
                ", labelSmoothing=" + labelSmoothing +
                '}';
    }
}
