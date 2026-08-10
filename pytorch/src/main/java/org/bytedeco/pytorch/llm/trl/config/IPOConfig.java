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
 * Configuration for IPO (Identity Preference Optimization) trainer.
 *
 * <p>IPO is a theoretically-grounded alternative to DPO that removes the need for
 * careful KL penalty tuning. It uses a margin-based loss that has stronger
 * theoretical guarantees.
 *
 * <p>Reference: "A General Theoretical Paradigm to Understand Preference Optimization"
 * (ITISA, Meta AI)
 *
 * <pre>{@code
 * IPOConfig config = IPOConfig.builder()
 *     .tau(0.1)           // IPO tau parameter
 *     .margin(0.0)        // additional margin
 *     .referenceFree(false)
 *     .build();
 * }</pre>
 */
public final class IPOConfig extends TrainerConfig {
    private final double tau;
    private final double margin;
    private final boolean referenceFree;
    private final double labelSmoothing;

    private IPOConfig(Builder b) {
        super(b);
        this.tau = b.tau;
        this.margin = b.margin;
        this.referenceFree = b.referenceFree;
        this.labelSmoothing = b.labelSmoothing;
    }

    /** IPO tau parameter (controls margin strength). */
    public double tau() { return tau; }

    /** Additional margin for IPO loss. */
    public double margin() { return margin; }

    /** Use reference-free mode (no reference model needed). */
    public boolean referenceFree() { return referenceFree; }

    /** Label smoothing for IPO loss. */
    public double labelSmoothing() { return labelSmoothing; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double tau = 0.1;
        private double margin = 0.0;
        private boolean referenceFree = false;
        private double labelSmoothing = 0.0;

        public Builder tau(double tau) {
            this.tau = Math.max(0.001, tau);
            return this;
        }

        public Builder margin(double margin) {
            this.margin = margin;
            return this;
        }

        public Builder referenceFree(boolean referenceFree) {
            this.referenceFree = referenceFree;
            return this;
        }

        public Builder labelSmoothing(double labelSmoothing) {
            this.labelSmoothing = Math.max(0, Math.min(0.5, labelSmoothing));
            return this;
        }

        @Override
        public IPOConfig build() {
            return new IPOConfig(this);
        }
    }

    @Override
    public String toString() {
        return "IPOConfig{" +
                "tau=" + tau +
                ", margin=" + margin +
                ", referenceFree=" + referenceFree +
                ", labelSmoothing=" + labelSmoothing +
                '}';
    }
}
