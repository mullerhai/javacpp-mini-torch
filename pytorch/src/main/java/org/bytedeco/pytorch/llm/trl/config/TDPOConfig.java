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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for TDPO (Token-level Direct Preference Optimization) trainer (HF TRL {@code TDPOConfig}).
 *
 * <p>Reference: "Token-level Direct Preference Optimization" (Dong et al., 2024)
 */
public final class TDPOConfig extends TrainerConfig {
    private final double beta;
    private final double clipRange;
    private final double forwardKlCoef;
    private final boolean tokenLevelAdvantage;
    private final boolean bootstrapFromLastToken;
    private final double labelSmoothing;
    private final double gamma;
    private final double sftWeight;
    private final boolean lengthNormalize;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final boolean referenceFree;
    private final double auxLossCoef;
    private final double truncationMode;
    private final boolean precomputeRefLogits;

    private TDPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.clipRange = b.clipRange;
        this.forwardKlCoef = b.forwardKlCoef;
        this.tokenLevelAdvantage = b.tokenLevelAdvantage;
        this.bootstrapFromLastToken = b.bootstrapFromLastToken;
        this.labelSmoothing = b.labelSmoothing;
        this.gamma = b.gamma;
        this.sftWeight = b.sftWeight;
        this.lengthNormalize = b.lengthNormalize;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.referenceFree = b.referenceFree;
        this.auxLossCoef = b.auxLossCoef;
        this.truncationMode = b.truncationMode;
        this.precomputeRefLogits = b.precomputeRefLogits;
    }

    public double beta() { return beta; }
    public double clipRange() { return clipRange; }
    public double forwardKlCoef() { return forwardKlCoef; }
    public boolean tokenLevelAdvantage() { return tokenLevelAdvantage; }
    public boolean bootstrapFromLastToken() { return bootstrapFromLastToken; }
    public double labelSmoothing() { return labelSmoothing; }
    public double gamma() { return gamma; }
    public double sftWeight() { return sftWeight; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public boolean referenceFree() { return referenceFree; }
    public double auxLossCoef() { return auxLossCoef; }
    public double truncationMode() { return truncationMode; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("clip_range", clipRange);
        m.put("forward_kl_coef", forwardKlCoef);
        m.put("token_level_advantage", tokenLevelAdvantage);
        m.put("bootstrap_from_last_token", bootstrapFromLastToken);
        m.put("label_smoothing", labelSmoothing);
        m.put("gamma", gamma);
        m.put("sft_weight", sftWeight);
        m.put("length_normalize", lengthNormalize);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("reference_free", referenceFree);
        m.put("aux_loss_coef", auxLossCoef);
        m.put("truncation_mode", truncationMode);
        m.put("precompute_ref_logits", precomputeRefLogits);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private double clipRange = 0.2;
        private double forwardKlCoef = 0.1;
        private boolean tokenLevelAdvantage = true;
        private boolean bootstrapFromLastToken = false;
        private double labelSmoothing = 0.0;
        private double gamma = 0.0;
        private double sftWeight = 0.0;
        private boolean lengthNormalize = false;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private boolean referenceFree = false;
        private double auxLossCoef = 0.0;
        private double truncationMode = 0.0;
        private boolean precomputeRefLogits = false;

        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be non-negative");
            this.beta = v; return this;
        }
        public Builder clipRange(double v) {
            if (v < 0) throw new IllegalArgumentException("clipRange must be non-negative");
            this.clipRange = v; return this;
        }
        public Builder clip_range(double v) { return clipRange(v); }
        public Builder forwardKlCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("forwardKlCoef must be non-negative");
            this.forwardKlCoef = v; return this;
        }
        public Builder forward_kl_coef(double v) { return forwardKlCoef(v); }
        public Builder tokenLevelAdvantage(boolean v) { this.tokenLevelAdvantage = v; return this; }
        public Builder token_level_advantage(boolean v) { return tokenLevelAdvantage(v); }
        public Builder bootstrapFromLastToken(boolean v) { this.bootstrapFromLastToken = v; return this; }
        public Builder bootstrap_from_last_token(boolean v) { return bootstrapFromLastToken(v); }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
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
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
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
        public Builder referenceFree(boolean v) { this.referenceFree = v; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }
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
        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }

        @Override
        public TDPOConfig build() { return new TDPOConfig(this); }
    }
}