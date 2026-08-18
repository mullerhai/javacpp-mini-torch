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
 * Configuration for RLOO (REINFORCE Leave-One-Out) trainer (HF TRL {@code RLOOConfig}).
 */
public final class RLOOConfig extends TrainerConfig {
    private final double beta;
    private final double klTarget;
    private final double klEpsilon;
    private final double baselineCoeff;
    private final boolean useAdaptiveKL;
    private final int numSamples;
    private final double clipRange;
    private final double gamma;
    private final double entCoef;
    private final double vfCoef;
    private final int miniBatchSize;
    private final int ppoEpochs;
    private final double entropyBonus;
    private final double rewardClip;
    private final double advantageClip;
    private final String scaleRewards;
    private final boolean whitenAdvantages;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean useReferenceModel;
    private final double referenceBeta;
    private final double gaeLambda;

    private RLOOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.klTarget = b.klTarget;
        this.klEpsilon = b.klEpsilon;
        this.baselineCoeff = b.baselineCoeff;
        this.useAdaptiveKL = b.useAdaptiveKL;
        this.numSamples = b.numSamples;
        this.clipRange = b.clipRange;
        this.gamma = b.gamma;
        this.entCoef = b.entCoef;
        this.vfCoef = b.vfCoef;
        this.miniBatchSize = b.miniBatchSize;
        this.ppoEpochs = b.ppoEpochs;
        this.entropyBonus = b.entropyBonus;
        this.rewardClip = b.rewardClip;
        this.advantageClip = b.advantageClip;
        this.scaleRewards = b.scaleRewards;
        this.whitenAdvantages = b.whitenAdvantages;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.useReferenceModel = b.useReferenceModel;
        this.referenceBeta = b.referenceBeta;
        this.gaeLambda = b.gaeLambda;
    }

    public double beta() { return beta; }
    public double klTarget() { return klTarget; }
    public double klEpsilon() { return klEpsilon; }
    public double baselineCoeff() { return baselineCoeff; }
    public boolean useAdaptiveKL() { return useAdaptiveKL; }
    public int numSamples() { return numSamples; }
    public double clipRange() { return clipRange; }
    public double gamma() { return gamma; }
    public double entCoef() { return entCoef; }
    public double vfCoef() { return vfCoef; }
    public int miniBatchSize() { return miniBatchSize; }
    public int ppoEpochs() { return ppoEpochs; }
    public double entropyBonus() { return entropyBonus; }
    public double rewardClip() { return rewardClip; }
    public double advantageClip() { return advantageClip; }
    public String scaleRewards() { return scaleRewards; }
    public boolean whitenAdvantages() { return whitenAdvantages; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean useReferenceModel() { return useReferenceModel; }
    public double referenceBeta() { return referenceBeta; }
    public double gaeLambda() { return gaeLambda; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("kl_target", klTarget);
        m.put("kl_epsilon", klEpsilon);
        m.put("baseline_coeff", baselineCoeff);
        m.put("use_adaptive_kl", useAdaptiveKL);
        m.put("num_samples", numSamples);
        m.put("clip_range", clipRange);
        m.put("gamma", gamma);
        m.put("ent_coef", entCoef);
        m.put("vf_coef", vfCoef);
        m.put("mini_batch_size", miniBatchSize);
        m.put("ppo_epochs", ppoEpochs);
        m.put("entropy_bonus", entropyBonus);
        m.put("reward_clip", rewardClip);
        m.put("advantage_clip", advantageClip);
        m.put("scale_rewards", scaleRewards);
        m.put("whiten_advantages", whitenAdvantages);
        m.put("use_reference_model", useReferenceModel);
        m.put("reference_beta", referenceBeta);
        m.put("gae_lambda", gaeLambda);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.01;
        private double klTarget = 6.0;
        private double klEpsilon = 0.2;
        private double baselineCoeff = 0.99;
        private boolean useAdaptiveKL = true;
        private int numSamples = 4;
        private double clipRange = 0.2;
        private double gamma = 0.99;
        private double entCoef = 0.01;
        private double vfCoef = 0.5;
        private int miniBatchSize = 64;
        private int ppoEpochs = 1;
        private double entropyBonus = 0.01;
        private double rewardClip = 10.0;
        private double advantageClip = 0.0;
        private String scaleRewards = "none";
        private boolean whitenAdvantages = false;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean useReferenceModel = true;
        private double referenceBeta = 0.04;
        private double gaeLambda = 0.95;

        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder klTarget(double v) {
            if (v < 0) throw new IllegalArgumentException("kl_target must be >= 0");
            this.klTarget = v; return this;
        }
        public Builder kl_target(double v) { return klTarget(v); }
        public Builder klEpsilon(double v) {
            if (v <= 0) throw new IllegalArgumentException("kl_epsilon must be > 0");
            this.klEpsilon = v; return this;
        }
        public Builder kl_epsilon(double v) { return klEpsilon(v); }
        public Builder baselineCoeff(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("baseline_coeff must be in [0, 1]");
            this.baselineCoeff = v; return this;
        }
        public Builder baseline_coeff(double v) { return baselineCoeff(v); }
        public Builder useAdaptiveKL(boolean v) { this.useAdaptiveKL = v; return this; }
        public Builder use_adaptive_kl(boolean v) { return useAdaptiveKL(v); }
        public Builder numSamples(int v) {
            if (v < 2) throw new IllegalArgumentException("num_samples must be >= 2");
            this.numSamples = v; return this;
        }
        public Builder num_samples(int v) { return numSamples(v); }
        public Builder clipRange(double v) {
            if (v < 0) throw new IllegalArgumentException("clip_range must be >= 0");
            this.clipRange = v; return this;
        }
        public Builder clip_range(double v) { return clipRange(v); }
        public Builder gamma(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
            this.gamma = v; return this;
        }
        public Builder entCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("ent_coef must be >= 0");
            this.entCoef = v; return this;
        }
        public Builder ent_coef(double v) { return entCoef(v); }
        public Builder vfCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("vf_coef must be >= 0");
            this.vfCoef = v; return this;
        }
        public Builder vf_coef(double v) { return vfCoef(v); }
        public Builder miniBatchSize(int v) {
            if (v < 1) throw new IllegalArgumentException("mini_batch_size must be >= 1");
            this.miniBatchSize = v; return this;
        }
        public Builder mini_batch_size(int v) { return miniBatchSize(v); }
        public Builder ppoEpochs(int v) {
            if (v < 1) throw new IllegalArgumentException("ppo_epochs must be >= 1");
            this.ppoEpochs = v; return this;
        }
        public Builder ppo_epochs(int v) { return ppoEpochs(v); }
        public Builder entropyBonus(double v) {
            if (v < 0) throw new IllegalArgumentException("entropy_bonus must be >= 0");
            this.entropyBonus = v; return this;
        }
        public Builder entropy_bonus(double v) { return entropyBonus(v); }
        public Builder rewardClip(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_clip must be >= 0");
            this.rewardClip = v; return this;
        }
        public Builder reward_clip(double v) { return rewardClip(v); }
        public Builder advantageClip(double v) {
            if (v < 0) throw new IllegalArgumentException("advantage_clip must be >= 0");
            this.advantageClip = v; return this;
        }
        public Builder advantage_clip(double v) { return advantageClip(v); }
        public Builder scaleRewards(String v) {
            if (v == null) { this.scaleRewards = "none"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "none": case "group": case "batch":
                    this.scaleRewards = n; return this;
                default:
                    throw new IllegalArgumentException("scale_rewards must be none/group/batch");
            }
        }
        public Builder scale_rewards(String v) { return scaleRewards(v); }
        public Builder whitenAdvantages(boolean v) { this.whitenAdvantages = v; return this; }
        public Builder whiten_advantages(boolean v) { return whitenAdvantages(v); }
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
        public Builder useReferenceModel(boolean v) { this.useReferenceModel = v; return this; }
        public Builder use_reference_model(boolean v) { return useReferenceModel(v); }
        public Builder referenceBeta(double v) {
            if (v < 0) throw new IllegalArgumentException("reference_beta must be >= 0");
            this.referenceBeta = v; return this;
        }
        public Builder reference_beta(double v) { return referenceBeta(v); }
        public Builder gaeLambda(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gae_lambda must be in [0, 1]");
            this.gaeLambda = v; return this;
        }
        public Builder gae_lambda(double v) { return gaeLambda(v); }

        @Override
        public RLOOConfig build() { return new RLOOConfig(this); }
    }

    @Override
    public String toString() {
        return "RLOOConfig{" +
                "beta=" + beta +
                ", klTarget=" + klTarget +
                ", baselineCoeff=" + baselineCoeff +
                ", numSamples=" + numSamples +
                '}';
    }
}