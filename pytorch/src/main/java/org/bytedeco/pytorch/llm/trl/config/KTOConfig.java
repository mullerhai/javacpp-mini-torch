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

import org.bytedeco.pytorch.llm.trl.trainer.KTOTrainer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for Kahneman-Tversky Optimization (KTO) trainer (HF TRL {@code KTOConfig}).
 *
 * <p>Reference: "KTO: Kahneman-Tversky Optimization" (Meta AI, 2024)
 *
 * @see KTOTrainer
 */
public final class KTOConfig extends TrainerConfig {
    private final double beta;
    private final double gammaC;
    private final double gammaD;
    private final double alpha;
    private final double klTarget;
    private final double klDelta;
    private final boolean usePerVersionLoss;
    private final double labelSmoothing;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final double sftWeight;
    private final boolean lengthNormalize;
    private final double referenceFree; // 0 / 1; HF accepts boolean too
    private final boolean precomputeRefLogits;
    private final double auxLossCoef;
    private final double truncationMode;

    private KTOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.gammaC = b.gammaC;
        this.gammaD = b.gammaD;
        this.alpha = b.alpha;
        this.klTarget = b.klTarget;
        this.klDelta = b.klDelta;
        this.usePerVersionLoss = b.usePerVersionLoss;
        this.labelSmoothing = b.labelSmoothing;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.sftWeight = b.sftWeight;
        this.lengthNormalize = b.lengthNormalize;
        this.referenceFree = b.referenceFree;
        this.precomputeRefLogits = b.precomputeRefLogits;
        this.auxLossCoef = b.auxLossCoef;
        this.truncationMode = b.truncationMode;
    }

    public double beta() { return beta; }
    public double gammaC() { return gammaC; }
    public double gammaD() { return gammaD; }
    public double alpha() { return alpha; }
    public double klTarget() { return klTarget; }
    public double klDelta() { return klDelta; }
    public boolean usePerVersionLoss() { return usePerVersionLoss; }
    public double labelSmoothing() { return labelSmoothing; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public double sftWeight() { return sftWeight; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double referenceFree() { return referenceFree; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }
    public double auxLossCoef() { return auxLossCoef; }
    public double truncationMode() { return truncationMode; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("gamma_c", gammaC);
        m.put("gamma_d", gammaD);
        m.put("alpha", alpha);
        m.put("kl_target", klTarget);
        m.put("kl_delta", klDelta);
        m.put("use_per_version_loss", usePerVersionLoss);
        m.put("label_smoothing", labelSmoothing);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("sft_weight", sftWeight);
        m.put("length_normalize", lengthNormalize);
        m.put("reference_free", referenceFree);
        m.put("precompute_ref_logits", precomputeRefLogits);
        m.put("aux_loss_coef", auxLossCoef);
        m.put("truncation_mode", truncationMode);
        return m;
    }

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
        private double labelSmoothing = 0.0;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private double sftWeight = 0.0;
        private boolean lengthNormalize = false;
        private double referenceFree = 1.0;
        private boolean precomputeRefLogits = false;
        private double auxLossCoef = 0.0;
        private double truncationMode = 0.0;

        public Builder() {}

        public Builder(KTOConfig config) {
            this.beta = config.beta;
            this.gammaC = config.gammaC;
            this.gammaD = config.gammaD;
            this.alpha = config.alpha;
            this.klTarget = config.klTarget;
            this.klDelta = config.klDelta;
            this.usePerVersionLoss = config.usePerVersionLoss;
            this.labelSmoothing = config.labelSmoothing;
            this.maxLength = config.maxLength;
            this.maxPromptLength = config.maxPromptLength;
            this.disableDropout = config.disableDropout;
            this.sftWeight = config.sftWeight;
            this.lengthNormalize = config.lengthNormalize;
            this.referenceFree = config.referenceFree;
            this.precomputeRefLogits = config.precomputeRefLogits;
            this.auxLossCoef = config.auxLossCoef;
            this.truncationMode = config.truncationMode;
        }

        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder gammaC(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma_c must be >= 0");
            this.gammaC = v; return this;
        }
        public Builder gamma_c(double v) { return gammaC(v); }
        public Builder gammaD(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma_d must be >= 0");
            this.gammaD = v; return this;
        }
        public Builder gamma_d(double v) { return gammaD(v); }
        public Builder alpha(double v) {
            if (v < 0) throw new IllegalArgumentException("alpha must be >= 0");
            this.alpha = v; return this;
        }
        public Builder klTarget(double v) {
            if (v < 0) throw new IllegalArgumentException("kl_target must be >= 0");
            this.klTarget = v; return this;
        }
        public Builder kl_target(double v) { return klTarget(v); }
        public Builder klDelta(double v) {
            if (v <= 0) throw new IllegalArgumentException("kl_delta must be > 0");
            this.klDelta = v; return this;
        }
        public Builder kl_delta(double v) { return klDelta(v); }
        public Builder usePerVersionLoss(boolean v) { this.usePerVersionLoss = v; return this; }
        public Builder use_per_version_loss(boolean v) { return usePerVersionLoss(v); }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
        public Builder maxLength(double v) {
            if (v < 1) throw new IllegalArgumentException("max_length must be >= 1");
            this.maxLength = v; return this;
        }
        public Builder max_length(double v) { return maxLength(v); }
        public Builder maxPromptLength(double v) {
            if (v < 1) throw new IllegalArgumentException("max_prompt_length must be >= 1");
            this.maxPromptLength = v; return this;
        }
        public Builder max_prompt_length(double v) { return maxPromptLength(v); }
        public Builder disableDropout(boolean v) { this.disableDropout = v; return this; }
        public Builder disable_dropout(boolean v) { return disableDropout(v); }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder referenceFree(boolean v) { this.referenceFree = v ? 1.0 : 0.0; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }
        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }
        public Builder auxLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("aux_loss_coef must be >= 0");
            this.auxLossCoef = v; return this;
        }
        public Builder aux_loss_coef(double v) { return auxLossCoef(v); }
        public Builder truncationMode(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("truncation_mode must be in [0, 1]");
            this.truncationMode = v; return this;
        }
        public Builder truncation_mode(double v) { return truncationMode(v); }

        public KTOConfig build() { return new KTOConfig(this); }
    }
}