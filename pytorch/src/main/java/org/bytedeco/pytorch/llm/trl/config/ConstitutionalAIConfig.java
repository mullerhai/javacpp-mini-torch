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
 * Configuration for Constitutional AI (CAI) trainer (HF TRL {@code ConstitutionalAIConfig}).
 */
public final class ConstitutionalAIConfig extends TrainerConfig {
    private final String principles;
    private final double critiqueWeight;
    private final int revisionSteps;
    private final boolean useSLICF;
    private final boolean useRLAIF;
    private final double harmlessnessWeight;
    private final double helpfulnessWeight;
    private final double honestyWeight;
    private final double labelSmoothing;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final boolean lengthNormalize;
    private final double beta;
    private final double gamma;
    private final boolean useReferenceModel;
    private final double sftWeight;

    private ConstitutionalAIConfig(Builder b) {
        super(b);
        this.principles = b.principles;
        this.critiqueWeight = b.critiqueWeight;
        this.revisionSteps = b.revisionSteps;
        this.useSLICF = b.useSLICF;
        this.useRLAIF = b.useRLAIF;
        this.harmlessnessWeight = b.harmlessnessWeight;
        this.helpfulnessWeight = b.helpfulnessWeight;
        this.honestyWeight = b.honestyWeight;
        this.labelSmoothing = b.labelSmoothing;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.lengthNormalize = b.lengthNormalize;
        this.beta = b.beta;
        this.gamma = b.gamma;
        this.useReferenceModel = b.useReferenceModel;
        this.sftWeight = b.sftWeight;
    }

    public String principles() { return principles; }
    public double critiqueWeight() { return critiqueWeight; }
    public int revisionSteps() { return revisionSteps; }
    public boolean useSLICF() { return useSLICF; }
    public boolean useRLAIF() { return useRLAIF; }
    public double harmlessnessWeight() { return harmlessnessWeight; }
    public double helpfulnessWeight() { return helpfulnessWeight; }
    public double honestyWeight() { return honestyWeight; }
    public double labelSmoothing() { return labelSmoothing; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double beta() { return beta; }
    public double gamma() { return gamma; }
    public boolean useReferenceModel() { return useReferenceModel; }
    public double sftWeight() { return sftWeight; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("principles", principles);
        m.put("critique_weight", critiqueWeight);
        m.put("revision_steps", revisionSteps);
        m.put("use_slicf", useSLICF);
        m.put("use_rlaif", useRLAIF);
        m.put("harmlessness_weight", harmlessnessWeight);
        m.put("helpfulness_weight", helpfulnessWeight);
        m.put("honesty_weight", honestyWeight);
        m.put("label_smoothing", labelSmoothing);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("length_normalize", lengthNormalize);
        m.put("beta", beta);
        m.put("gamma", gamma);
        m.put("use_reference_model", useReferenceModel);
        m.put("sft_weight", sftWeight);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private String principles = "Helpful, Harmless, Honest";
        private double critiqueWeight = 0.5;
        private int revisionSteps = 2;
        private boolean useSLICF = true;
        private boolean useRLAIF = true;
        private double harmlessnessWeight = 0.4;
        private double helpfulnessWeight = 0.3;
        private double honestyWeight = 0.3;
        private double labelSmoothing = 0.0;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private boolean lengthNormalize = false;
        private double beta = 0.04;
        private double gamma = 1.0;
        private boolean useReferenceModel = false;
        private double sftWeight = 0.0;

        public Builder principles(String v) { this.principles = v; return this; }
        public Builder principles(String[] v) { this.principles = String.join(", ", v); return this; }
        public Builder critiqueWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("critique_weight must be in [0, 1]");
            this.critiqueWeight = v; return this;
        }
        public Builder critique_weight(double v) { return critiqueWeight(v); }
        public Builder revisionSteps(int v) {
            if (v < 1) throw new IllegalArgumentException("revision_steps must be >= 1");
            this.revisionSteps = v; return this;
        }
        public Builder revision_steps(int v) { return revisionSteps(v); }
        public Builder useSLICF(boolean v) { this.useSLICF = v; return this; }
        public Builder use_slicf(boolean v) { return useSLICF(v); }
        public Builder useRLAIF(boolean v) { this.useRLAIF = v; return this; }
        public Builder use_rlaif(boolean v) { return useRLAIF(v); }
        public Builder harmlessnessWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("harmlessness_weight must be in [0, 1]");
            this.harmlessnessWeight = v; return this;
        }
        public Builder harmlessness_weight(double v) { return harmlessnessWeight(v); }
        public Builder helpfulnessWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("helpfulness_weight must be in [0, 1]");
            this.helpfulnessWeight = v; return this;
        }
        public Builder helpfulness_weight(double v) { return helpfulnessWeight(v); }
        public Builder honestyWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("honesty_weight must be in [0, 1]");
            this.honestyWeight = v; return this;
        }
        public Builder honesty_weight(double v) { return honestyWeight(v); }
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
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }
        public Builder useReferenceModel(boolean v) { this.useReferenceModel = v; return this; }
        public Builder use_reference_model(boolean v) { return useReferenceModel(v); }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }

        @Override
        public ConstitutionalAIConfig build() {
            // Normalize weights
            double total = harmlessnessWeight + helpfulnessWeight + honestyWeight;
            if (Math.abs(total - 1.0) > 0.01 && total > 1e-8) {
                harmlessnessWeight /= total;
                helpfulnessWeight /= total;
                honestyWeight /= total;
            }
            return new ConstitutionalAIConfig(this);
        }
    }
}