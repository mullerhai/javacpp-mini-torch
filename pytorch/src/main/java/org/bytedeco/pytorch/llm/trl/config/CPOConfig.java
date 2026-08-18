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
 * Configuration for CPO (Contrastive Preference Optimization) trainer (HF TRL {@code CPOConfig}).
 *
 * <p>CPO combines DPO-style preference learning with a contrastive NLL term
 * (similar to SimPO) and a BT loss component.
 */
public final class CPOConfig extends TrainerConfig {
    private final double beta;
    private final double contrastiveAlpha;
    private final double margin;
    private final boolean useReferenceModel;
    private final int contrastiveSteps;
    private final double labelSmoothing;
    private final double gamma;
    private final double cpoAlpha;
    private final double simpoGamma;
    private final String lossType;             // "cpo" | "simpo" | "siglip"
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final boolean lengthNormalize;
    private final double sftWeight;            // optional SFT NLL auxiliary
    private final double referenceFree;        // 0 / 1
    private final boolean precomputeRefLogits;
    private final double truncationMode;
    private final double tools;

    private CPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.contrastiveAlpha = b.contrastiveAlpha;
        this.margin = b.margin;
        this.useReferenceModel = b.useReferenceModel;
        this.contrastiveSteps = b.contrastiveSteps;
        this.labelSmoothing = b.labelSmoothing;
        this.gamma = b.gamma;
        this.cpoAlpha = b.cpoAlpha;
        this.simpoGamma = b.simpoGamma;
        this.lossType = b.lossType;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.lengthNormalize = b.lengthNormalize;
        this.sftWeight = b.sftWeight;
        this.referenceFree = b.referenceFree;
        this.precomputeRefLogits = b.precomputeRefLogits;
        this.truncationMode = b.truncationMode;
        this.tools = b.tools;
    }

    public double beta() { return beta; }
    public double contrastiveAlpha() { return contrastiveAlpha; }
    public double margin() { return margin; }
    public boolean useReferenceModel() { return useReferenceModel; }
    public int contrastiveSteps() { return contrastiveSteps; }
    public double labelSmoothing() { return labelSmoothing; }
    public double gamma() { return gamma; }
    public double cpoAlpha() { return cpoAlpha; }
    public double simpoGamma() { return simpoGamma; }
    public String lossType() { return lossType; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double sftWeight() { return sftWeight; }
    public double referenceFree() { return referenceFree; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }
    public double truncationMode() { return truncationMode; }
    public double tools() { return tools; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("contrastive_alpha", contrastiveAlpha);
        m.put("margin", margin);
        m.put("use_reference_model", useReferenceModel);
        m.put("label_smoothing", labelSmoothing);
        m.put("gamma", gamma);
        m.put("cpo_alpha", cpoAlpha);
        m.put("simpo_gamma", simpoGamma);
        m.put("loss_type", lossType);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("length_normalize", lengthNormalize);
        m.put("sft_weight", sftWeight);
        m.put("reference_free", referenceFree);
        m.put("precompute_ref_logits", precomputeRefLogits);
        m.put("truncation_mode", truncationMode);
        m.put("tools", tools);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private double contrastiveAlpha = 0.5;
        private double margin = 0.5;
        private boolean useReferenceModel = true;
        private int contrastiveSteps = 1;
        private double labelSmoothing = 0.0;
        private double gamma = 1.0;
        private double cpoAlpha = 1.0;
        private double simpoGamma = 1.0;
        private String lossType = "cpo";
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private boolean lengthNormalize = false;
        private double sftWeight = 0.0;
        private double referenceFree = 0.0;
        private boolean precomputeRefLogits = false;
        private double truncationMode = 0.0;
        private double tools = 0.0;

        public Builder beta(double beta) { this.beta = Math.max(0, beta); return this; }
        public Builder contrastiveAlpha(double v) { this.contrastiveAlpha = Math.max(0, v); return this; }
        public Builder contrastive_alpha(double v) { return contrastiveAlpha(v); }
        public Builder margin(double margin) { this.margin = margin; return this; }
        public Builder useReferenceModel(boolean v) { this.useReferenceModel = v; return this; }
        public Builder use_reference_model(boolean v) { return useReferenceModel(v); }
        public Builder contrastiveSteps(int v) { this.contrastiveSteps = Math.max(1, v); return this; }
        public Builder contrastive_steps(int v) { return contrastiveSteps(v); }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }
        public Builder cpoAlpha(double v) {
            if (v < 0) throw new IllegalArgumentException("cpo_alpha must be >= 0");
            this.cpoAlpha = v; return this;
        }
        public Builder cpo_alpha(double v) { return cpoAlpha(v); }
        public Builder simpoGamma(double v) {
            if (v < 0) throw new IllegalArgumentException("simpo_gamma must be >= 0");
            this.simpoGamma = v; return this;
        }
        public Builder simpo_gamma(double v) { return simpoGamma(v); }
        public Builder lossType(String v) {
            if (v == null) { this.lossType = "cpo"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "cpo": case "simpo": case "siglip":
                    this.lossType = n; return this;
                default:
                    throw new IllegalArgumentException("loss_type must be 'cpo', 'simpo', or 'siglip'");
            }
        }
        public Builder loss_type(String v) { return lossType(v); }
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
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder referenceFree(boolean v) { this.referenceFree = v ? 1.0 : 0.0; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }
        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }
        public Builder truncationMode(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("truncation_mode must be in [0, 1]");
            this.truncationMode = v; return this;
        }
        public Builder truncation_mode(double v) { return truncationMode(v); }
        public Builder tools(double v) {
            if (v < 0) throw new IllegalArgumentException("tools must be >= 0");
            this.tools = v; return this;
        }

        @Override
        public CPOConfig build() { return new CPOConfig(this); }
    }

    @Override
    public String toString() {
        return "CPOConfig{" +
                "beta=" + beta +
                ", contrastiveAlpha=" + contrastiveAlpha +
                ", margin=" + margin +
                ", lossType=" + lossType +
                ", lengthNormalize=" + lengthNormalize +
                '}';
    }
}
