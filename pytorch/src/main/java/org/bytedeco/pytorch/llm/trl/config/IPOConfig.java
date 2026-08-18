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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.config;

import org.bytedeco.pytorch.llm.trl.trainer.IPOTrainer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for IPO (Identity Preference Optimization) trainer (HF TRL {@code IPOConfig}).
 *
 * <p>IPO adds a regularization term to DPO that makes the algorithm provably
 * converge to the optimal policy. It has better theoretical properties and
 * improved finite-sample bounds compared to DPO.
 *
 * <p>Reference: "A Theoretical Analysis of Identity Preference Optimization (IPO)"
 * (Azar et al., 2024)
 *
 * @see IPOTrainer
 */
public final class IPOConfig extends TrainerConfig {
    private final double beta;
    private final double identityCoef;
    private final boolean lengthNormalize;
    private final double labelSmoothing;
    private final double gamma;
    private final double sftWeight;
    private final boolean referenceFree;
    private final double auxLossCoef;
    private final boolean disableDropout;
    private final double truncationMode;
    private final boolean precomputeRefLogits;

    private IPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.identityCoef = b.identityCoef;
        this.lengthNormalize = b.lengthNormalize;
        this.labelSmoothing = b.labelSmoothing;
        this.gamma = b.gamma;
        this.sftWeight = b.sftWeight;
        this.referenceFree = b.referenceFree;
        this.auxLossCoef = b.auxLossCoef;
        this.disableDropout = b.disableDropout;
        this.truncationMode = b.truncationMode;
        this.precomputeRefLogits = b.precomputeRefLogits;
    }

    public double beta() { return beta; }
    public double identityCoef() { return identityCoef; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double labelSmoothing() { return labelSmoothing; }
    public double gamma() { return gamma; }
    public double sftWeight() { return sftWeight; }
    public boolean referenceFree() { return referenceFree; }
    public double auxLossCoef() { return auxLossCoef; }
    public boolean disableDropout() { return disableDropout; }
    public double truncationMode() { return truncationMode; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("identity_coef", identityCoef);
        m.put("length_normalize", lengthNormalize);
        m.put("label_smoothing", labelSmoothing);
        m.put("gamma", gamma);
        m.put("sft_weight", sftWeight);
        m.put("reference_free", referenceFree);
        m.put("aux_loss_coef", auxLossCoef);
        m.put("disable_dropout", disableDropout);
        m.put("truncation_mode", truncationMode);
        m.put("precompute_ref_logits", precomputeRefLogits);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private double identityCoef = 1.0;
        private boolean lengthNormalize = false;
        private double labelSmoothing = 0.0;
        private double gamma = 0.0;
        private double sftWeight = 0.0;
        private boolean referenceFree = false;
        private double auxLossCoef = 0.0;
        private boolean disableDropout = false;
        private double truncationMode = 0.0;
        private boolean precomputeRefLogits = false;

        public Builder beta(double v) {
            if (v <= 0) throw new IllegalArgumentException("beta must be positive");
            this.beta = v; return this;
        }
        public Builder identityCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("identityCoef must be non-negative");
            this.identityCoef = v; return this;
        }
        public Builder identity_coef(double v) { return identityCoef(v); }
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("labelSmoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder referenceFree(boolean v) { this.referenceFree = v; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }
        public Builder auxLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("aux_loss_coef must be >= 0");
            this.auxLossCoef = v; return this;
        }
        public Builder aux_loss_coef(double v) { return auxLossCoef(v); }
        public Builder disableDropout(boolean v) { this.disableDropout = v; return this; }
        public Builder disable_dropout(boolean v) { return disableDropout(v); }
        public Builder truncationMode(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("truncation_mode must be in [0, 1]");
            this.truncationMode = v; return this;
        }
        public Builder truncation_mode(double v) { return truncationMode(v); }
        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }

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