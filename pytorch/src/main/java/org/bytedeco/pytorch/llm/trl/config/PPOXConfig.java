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
 * Configuration for PPO-X trainer (ByteDance inspired).
 *
 * <p>PPO-X is an enhanced PPO variant from ByteDance research that includes:
 * <ul>
 *   <li>Adaptive clipping with trust region management</li>
 *   <li>Importance weight clipping for stability</li>
 *   <li>Reward normalization and shaping</li>
 *   <li>Efficient advantage estimation</li>
 * </ul>
 *
 * <p>Reference: ByteDance internal research on efficient RLHF
 *
 * <pre>{@code
 * PPOXConfig config = PPOXConfig.builder()
 *     .clipRatio(0.2)
 *     .valueClipRatio(0.2)
 *     .trustRegionRadius(0.1)
 *     .adaptiveClipping(true)
 *     .build();
 * }</pre>
 */
public final class PPOXConfig extends TrainerConfig {
    private final double clipRatio;
    private final double valueClipRatio;
    private final double trustRegionRadius;
    private final boolean adaptiveClipping;
    private final double advantageLambda;
    private final double valueLossCoeff;
    private final boolean useGAE;
    private final double gaeGamma;
    private final double gaeTau;

    private PPOXConfig(Builder b) {
        super(b);
        this.clipRatio = b.clipRatio;
        this.valueClipRatio = b.valueClipRatio;
        this.trustRegionRadius = b.trustRegionRadius;
        this.adaptiveClipping = b.adaptiveClipping;
        this.advantageLambda = b.advantageLambda;
        this.valueLossCoeff = b.valueLossCoeff;
        this.useGAE = b.useGAE;
        this.gaeGamma = b.gaeGamma;
        this.gaeTau = b.gaeTau;
    }

    public double clipRatio() { return clipRatio; }
    public double valueClipRatio() { return valueClipRatio; }
    public double trustRegionRadius() { return trustRegionRadius; }
    public boolean adaptiveClipping() { return adaptiveClipping; }
    public double advantageLambda() { return advantageLambda; }
    public double valueLossCoeff() { return valueLossCoeff; }
    public boolean useGAE() { return useGAE; }
    public double gaeGamma() { return gaeGamma; }
    public double gaeTau() { return gaeTau; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double clipRatio = 0.2;
        private double valueClipRatio = 0.2;
        private double trustRegionRadius = 0.1;
        private boolean adaptiveClipping = true;
        private double advantageLambda = 0.95;
        private double valueLossCoeff = 0.5;
        private boolean useGAE = true;
        private double gaeGamma = 0.99;
        private double gaeTau = 0.95;

        public Builder clipRatio(double clipRatio) {
            this.clipRatio = Math.max(0.01, Math.min(0.5, clipRatio));
            return this;
        }

        public Builder valueClipRatio(double valueClipRatio) {
            this.valueClipRatio = Math.max(0.01, Math.min(0.5, valueClipRatio));
            return this;
        }

        public Builder trustRegionRadius(double trustRegionRadius) {
            this.trustRegionRadius = Math.max(0.01, trustRegionRadius);
            return this;
        }

        public Builder adaptiveClipping(boolean adaptiveClipping) {
            this.adaptiveClipping = adaptiveClipping;
            return this;
        }

        public Builder advantageLambda(double advantageLambda) {
            this.advantageLambda = Math.max(0, Math.min(1, advantageLambda));
            return this;
        }

        public Builder valueLossCoeff(double valueLossCoeff) {
            this.valueLossCoeff = Math.max(0, valueLossCoeff);
            return this;
        }

        public Builder useGAE(boolean useGAE) {
            this.useGAE = useGAE;
            return this;
        }

        public Builder gaeGamma(double gaeGamma) {
            this.gaeGamma = Math.max(0, Math.min(1, gaeGamma));
            return this;
        }

        public Builder gaeTau(double gaeTau) {
            this.gaeTau = Math.max(0, Math.min(1, gaeTau));
            return this;
        }

        @Override
        public PPOXConfig build() {
            return new PPOXConfig(this);
        }
    }

    @Override
    public String toString() {
        return "PPOXConfig{" +
                "clipRatio=" + clipRatio +
                ", valueClipRatio=" + valueClipRatio +
                ", trustRegionRadius=" + trustRegionRadius +
                ", adaptiveClipping=" + adaptiveClipping +
                ", advantageLambda=" + advantageLambda +
                ", useGAE=" + useGAE +
                '}';
    }
}
