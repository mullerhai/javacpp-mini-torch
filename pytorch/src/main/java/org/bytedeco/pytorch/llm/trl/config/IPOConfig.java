/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
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
 * Configuration for IPO (Identity Preference Optimization) trainer.
 *
 * <p>IPO adds a regularization term to DPO that makes the algorithm provably
 * converge to the optimal policy. It has better theoretical properties and
 * improved finite-sample bounds compared to DPO.
 *
 * <p>Key differences from DPO:
 * <ul>
 *   <li>Adds regularization term: π(y_c) - π(y_r) = 1</li>
 *   <li>Provably converges to optimal policy</li>
 *   <li>Better finite-sample guarantees</li>
 *   <li>Reduced reward hacking behavior</li>
 * </ul>
 *
 * <p>Reference: "A Theoretical Analysis of Identity Preference Optimization (IPO)"
 * (Azar et al., 2024)
 *
 * @see org.bytedeco.pytorch.llm.trl.IPOTrainer
 */
public final class IPOConfig extends TrainerConfig {

    // ==================== IPO-specific Parameters ====================

    /** KL penalty coefficient (default: 0.1) */
    private final double beta;

    /** Regularization coefficient for identity term (default: 1.0) */
    private final double identityCoef;

    /** Length normalization (default: false) */
    private final boolean lengthNormalize;

    // ==================== Constructor ====================

    private IPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.identityCoef = b.identityCoef;
        this.lengthNormalize = b.lengthNormalize;
    }

    // ==================== Getters ====================

    /**
     * KL penalty coefficient (β).
     * Controls strength of reference model regularization.
     * Default: 0.1
     */
    public double beta() { return beta; }

    /**
     * Identity regularization coefficient.
     * Weight for the identity term (π(y_c) - π(y_r) = 1).
     * Default: 1.0
     */
    public double identityCoef() { return identityCoef; }

    /**
     * Whether to normalize by sequence length.
     * Default: false
     */
    public boolean lengthNormalize() { return lengthNormalize; }

    // ==================== Builder ====================

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {

        private double beta = 0.1;
        private double identityCoef = 1.0;
        private boolean lengthNormalize = false;

        public Builder beta(double v) {
            if (v <= 0) throw new IllegalArgumentException("beta must be positive");
            this.beta = v;
            return this;
        }

        public Builder identityCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("identityCoef must be non-negative");
            this.identityCoef = v;
            return this;
        }

        public Builder lengthNormalize(boolean v) {
            this.lengthNormalize = v;
            return this;
        }

        @Override
        public IPOConfig build() { return new IPOConfig(this); }
    }

    @Override
    public String toString() {
        return "IPOConfig{" +
                "beta=" + beta +
                ", identityCoef=" + identityCoef +
                ", lengthNormalize=" + lengthNormalize +
                '}';
    }
}
