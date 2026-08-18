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
 * ORPO trainer config (HF TRL {@code ORPOConfig}).
 *
 * <p>ORPO = SFT + odds-ratio penalty; reference-free by construction.
 */
public final class ORPOConfig extends TrainerConfig {
    private final double beta;
    private final boolean lengthNormalize;
    private final double sftWeight;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final double labelSmoothing;
    private final double tools;
    private final double gamma;        // margin coefficient (rarely used)
    private final boolean referenceFree; // always true for ORPO
    private final boolean precomputeRefLogits;

    private ORPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.lengthNormalize = b.lengthNormalize;
        this.sftWeight = b.sftWeight;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.labelSmoothing = b.labelSmoothing;
        this.tools = b.tools;
        this.gamma = b.gamma;
        this.referenceFree = b.referenceFree;
        this.precomputeRefLogits = b.precomputeRefLogits;
    }

    public double beta() { return beta; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double sftWeight() { return sftWeight; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public double labelSmoothing() { return labelSmoothing; }
    public double tools() { return tools; }
    public double gamma() { return gamma; }
    public boolean referenceFree() { return referenceFree; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("length_normalize", lengthNormalize);
        m.put("sft_weight", sftWeight);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("label_smoothing", labelSmoothing);
        m.put("gamma", gamma);
        m.put("reference_free", referenceFree);
        m.put("precompute_ref_logits", precomputeRefLogits);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private boolean lengthNormalize = false;
        private double sftWeight = 1.0;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private double labelSmoothing = 0.0;
        private double tools = 0.0;
        private double gamma = 0.0;
        private boolean referenceFree = true;
        private boolean precomputeRefLogits = false;

        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
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
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("labelSmoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }
        public Builder referenceFree(boolean v) { this.referenceFree = v; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }
        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }

        @Override
        public ORPOConfig build() { return new ORPOConfig(this); }
    }
}