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

import org.bytedeco.pytorch.llm.trl.trainer.SimPOTrainer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for SimPO (Simple Preference Optimization) trainer (HF TRL {@code SimPOConfig}).
 *
 * <p>SimPO removes the reference model term from DPO and uses a target margin
 * on log probabilities instead, making it simpler and more memory-efficient.
 *
 * <p>Reference: "SimPO: Simple Preference Optimization" (Meng et al., 2024)
 * <a href="https://arxiv.org/abs/2405.14734">arXiv:2405.14734</a>
 *
 * @see SimPOTrainer
 */
public final class SimPOConfig extends TrainerConfig {
    // SimPO-specific
    private final double beta;
    private final double targetMargin;
    private final boolean lengthNormalize;
    private final double labelSmoothing;
    private final double cAlpha;            // CPO-style auxiliary term (when used)
    private final double sftWeight;          // optional SFT loss weight
    private final String lossType;           // "simpo" | "cpo" | "rlhf"
    private final boolean disableDropout;
    private final boolean referenceFree;     // always true for SimPO
    private final double gamma;             // margin coefficient for APOS variant
    private final double truncationMode;
    private final boolean precomputeRefLogits;
    private final double tools;
    private final double maxLength;
    private final double maxPromptLength;
    // Auxiliary SFT NLL
    private final double auxiliaryLossCoef;
    // Number of reward signals
    private final double numRewardSamples;

    private SimPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.targetMargin = b.targetMargin;
        this.lengthNormalize = b.lengthNormalize;
        this.labelSmoothing = b.labelSmoothing;
        this.cAlpha = b.cAlpha;
        this.sftWeight = b.sftWeight;
        this.lossType = b.lossType;
        this.disableDropout = b.disableDropout;
        this.referenceFree = b.referenceFree;
        this.gamma = b.gamma;
        this.truncationMode = b.truncationMode;
        this.precomputeRefLogits = b.precomputeRefLogits;
        this.tools = b.tools;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.auxiliaryLossCoef = b.auxiliaryLossCoef;
        this.numRewardSamples = b.numRewardSamples;
    }

    public double beta() { return beta; }
    public double targetMargin() { return targetMargin; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double labelSmoothing() { return labelSmoothing; }
    public double cAlpha() { return cAlpha; }
    public double sftWeight() { return sftWeight; }
    public String lossType() { return lossType; }
    public boolean disableDropout() { return disableDropout; }
    public boolean referenceFree() { return referenceFree; }
    public double gamma() { return gamma; }
    public double truncationMode() { return truncationMode; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }
    public double tools() { return tools; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public double auxiliaryLossCoef() { return auxiliaryLossCoef; }
    public double numRewardSamples() { return numRewardSamples; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("target_margin", targetMargin);
        m.put("length_normalize", lengthNormalize);
        m.put("label_smoothing", labelSmoothing);
        m.put("c_alpha", cAlpha);
        m.put("sft_weight", sftWeight);
        m.put("loss_type", lossType);
        m.put("disable_dropout", disableDropout);
        m.put("reference_free", referenceFree);
        m.put("gamma", gamma);
        m.put("truncation_mode", truncationMode);
        m.put("precompute_ref_logits", precomputeRefLogits);
        m.put("tools", tools);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("auxiliary_loss_coef", auxiliaryLossCoef);
        m.put("num_reward_samples", numRewardSamples);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 2.0;
        private double targetMargin = 1.0;
        private boolean lengthNormalize = true;
        private double labelSmoothing = 0.0;
        private double cAlpha = 0.0;
        private double sftWeight = 0.0;
        private String lossType = "simpo";
        private boolean disableDropout = false;
        private boolean referenceFree = true;
        private double gamma = 1.0;
        private double truncationMode = 0.0;
        private boolean precomputeRefLogits = false;
        private double tools = 0.0;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private double auxiliaryLossCoef = 0.0;
        private double numRewardSamples = 1.0;

        public Builder beta(double v) {
            if (v <= 0) throw new IllegalArgumentException("beta must be positive");
            this.beta = v; return this;
        }
        public Builder targetMargin(double v) {
            if (v < 0) throw new IllegalArgumentException("target_margin must be non-negative");
            this.targetMargin = v; return this;
        }
        public Builder target_margin(double v) { return targetMargin(v); }

        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }

        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }

        public Builder cAlpha(double v) {
            if (v < 0) throw new IllegalArgumentException("c_alpha must be >= 0");
            this.cAlpha = v; return this;
        }
        public Builder c_alpha(double v) { return cAlpha(v); }

        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }

        public Builder lossType(String v) {
            if (v == null) { this.lossType = "simpo"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "simpo": case "cpo": case "rlhf":
                    this.lossType = n; return this;
                default:
                    throw new IllegalArgumentException("loss_type must be 'simpo', 'cpo', or 'rlhf'");
            }
        }
        public Builder loss_type(String v) { return lossType(v); }

        public Builder disableDropout(boolean v) { this.disableDropout = v; return this; }
        public Builder disable_dropout(boolean v) { return disableDropout(v); }

        public Builder referenceFree(boolean v) { this.referenceFree = v; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }

        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }

        public Builder truncationMode(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("truncation_mode must be in [0, 1]");
            this.truncationMode = v; return this;
        }
        public Builder truncation_mode(double v) { return truncationMode(v); }

        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }

        public Builder tools(double v) {
            if (v < 0) throw new IllegalArgumentException("tools must be >= 0");
            this.tools = v; return this;
        }

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

        public Builder auxiliaryLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("auxiliary_loss_coef must be >= 0");
            this.auxiliaryLossCoef = v; return this;
        }
        public Builder auxiliary_loss_coef(double v) { return auxiliaryLossCoef(v); }

        public Builder numRewardSamples(double v) {
            if (v < 1) throw new IllegalArgumentException("num_reward_samples must be >= 1");
            this.numRewardSamples = v; return this;
        }
        public Builder num_reward_samples(double v) { return numRewardSamples(v); }

        @Override
        public SimPOConfig build() { return new SimPOConfig(this); }
    }

    @Override
    public String toString() {
        return "SimPOConfig{" +
                "beta=" + beta +
                ", targetMargin=" + targetMargin +
                ", lossType=" + lossType +
                ", lengthNormalize=" + lengthNormalize +
                ", labelSmoothing=" + labelSmoothing +
                '}';
    }
}
