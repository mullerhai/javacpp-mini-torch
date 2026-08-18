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
 * Configuration for PPO-X trainer (ByteDance inspired; HF TRL {@code PPOXConfig}).
 */
public final class PPOXConfig extends TrainerConfig {
    private final double clipRatio;
    private final double valueClipRatio;
    private final double trustRegionRadius;
    private final boolean adaptiveClipping;
    private final double advantageLambda;
    private final double valueLossCoeff;
    private final double entropyCoefficient;
    private final boolean useGAE;
    private final double gaeGamma;
    private final double gaeTau;
    private final double gamma;
    private final double initKlCoef;
    private final double targetKl;
    private final boolean adapKlCtrl;
    private final double cliprangeReward;
    private final String klEstimator;
    private final String scaleRewards;
    private final boolean whitenAdvantages;
    private final double maxLength;
    private final double maxPromptLength;
    private final int ppoEpochs;
    private final int miniBatchSize;
    private final double beta;
    private final double sftWeight;
    private final boolean useReferenceKl;

    private PPOXConfig(Builder b) {
        super(b);
        this.clipRatio = b.clipRatio;
        this.valueClipRatio = b.valueClipRatio;
        this.trustRegionRadius = b.trustRegionRadius;
        this.adaptiveClipping = b.adaptiveClipping;
        this.advantageLambda = b.advantageLambda;
        this.valueLossCoeff = b.valueLossCoeff;
        this.entropyCoefficient = b.entropyCoefficient;
        this.useGAE = b.useGAE;
        this.gaeGamma = b.gaeGamma;
        this.gaeTau = b.gaeTau;
        this.gamma = b.gamma;
        this.initKlCoef = b.initKlCoef;
        this.targetKl = b.targetKl;
        this.adapKlCtrl = b.adapKlCtrl;
        this.cliprangeReward = b.cliprangeReward;
        this.klEstimator = b.klEstimator;
        this.scaleRewards = b.scaleRewards;
        this.whitenAdvantages = b.whitenAdvantages;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.ppoEpochs = b.ppoEpochs;
        this.miniBatchSize = b.miniBatchSize;
        this.beta = b.beta;
        this.sftWeight = b.sftWeight;
        this.useReferenceKl = b.useReferenceKl;
    }

    public double clipRatio() { return clipRatio; }
    public double valueClipRatio() { return valueClipRatio; }
    public double trustRegionRadius() { return trustRegionRadius; }
    public boolean adaptiveClipping() { return adaptiveClipping; }
    public double advantageLambda() { return advantageLambda; }
    public double valueLossCoeff() { return valueLossCoeff; }
    public double entropyCoefficient() { return entropyCoefficient; }
    public boolean useGAE() { return useGAE; }
    public double gaeGamma() { return gaeGamma; }
    public double gaeTau() { return gaeTau; }
    public double gamma() { return gamma; }
    public double initKlCoef() { return initKlCoef; }
    public double targetKl() { return targetKl; }
    public boolean adapKlCtrl() { return adapKlCtrl; }
    public double cliprangeReward() { return cliprangeReward; }
    public String klEstimator() { return klEstimator; }
    public String scaleRewards() { return scaleRewards; }
    public boolean whitenAdvantages() { return whitenAdvantages; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public int ppoEpochs() { return ppoEpochs; }
    public int miniBatchSize() { return miniBatchSize; }
    public double beta() { return beta; }
    public double sftWeight() { return sftWeight; }
    public boolean useReferenceKl() { return useReferenceKl; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("clip_ratio", clipRatio);
        m.put("value_clip_ratio", valueClipRatio);
        m.put("trust_region_radius", trustRegionRadius);
        m.put("adaptive_clipping", adaptiveClipping);
        m.put("advantage_lambda", advantageLambda);
        m.put("value_loss_coeff", valueLossCoeff);
        m.put("entropy_coefficient", entropyCoefficient);
        m.put("use_gae", useGAE);
        m.put("gae_gamma", gaeGamma);
        m.put("gae_tau", gaeTau);
        m.put("gamma", gamma);
        m.put("init_kl_coef", initKlCoef);
        m.put("target_kl", targetKl);
        m.put("adap_kl_ctrl", adapKlCtrl);
        m.put("cliprange_reward", cliprangeReward);
        m.put("kl_estimator", klEstimator);
        m.put("scale_rewards", scaleRewards);
        m.put("whiten_advantages", whitenAdvantages);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("ppo_epochs", ppoEpochs);
        m.put("mini_batch_size", miniBatchSize);
        m.put("beta", beta);
        m.put("sft_weight", sftWeight);
        m.put("use_reference_kl", useReferenceKl);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double clipRatio = 0.2;
        private double valueClipRatio = 0.2;
        private double trustRegionRadius = 0.1;
        private boolean adaptiveClipping = true;
        private double advantageLambda = 0.95;
        private double valueLossCoeff = 0.5;
        private double entropyCoefficient = 0.01;
        private boolean useGAE = true;
        private double gaeGamma = 0.99;
        private double gaeTau = 0.95;
        private double gamma = 0.99;
        private double initKlCoef = 0.01;
        private double targetKl = 6.0;
        private boolean adapKlCtrl = true;
        private double cliprangeReward = 10.0;
        private String klEstimator = "kl";
        private String scaleRewards = "none";
        private boolean whitenAdvantages = false;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private int ppoEpochs = 4;
        private int miniBatchSize = 64;
        private double beta = 0.04;
        private double sftWeight = 0.0;
        private boolean useReferenceKl = false;

        public Builder clipRatio(double v) {
            if (v <= 0 || v > 0.5) throw new IllegalArgumentException("clip_ratio must be in (0, 0.5]");
            this.clipRatio = v; return this;
        }
        public Builder clip_ratio(double v) { return clipRatio(v); }
        public Builder valueClipRatio(double v) {
            if (v <= 0 || v > 0.5) throw new IllegalArgumentException("value_clip_ratio must be in (0, 0.5]");
            this.valueClipRatio = v; return this;
        }
        public Builder value_clip_ratio(double v) { return valueClipRatio(v); }
        public Builder trustRegionRadius(double v) {
            if (v <= 0) throw new IllegalArgumentException("trust_region_radius must be > 0");
            this.trustRegionRadius = v; return this;
        }
        public Builder trust_region_radius(double v) { return trustRegionRadius(v); }
        public Builder adaptiveClipping(boolean v) { this.adaptiveClipping = v; return this; }
        public Builder adaptive_clipping(boolean v) { return adaptiveClipping(v); }
        public Builder advantageLambda(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("advantage_lambda must be in [0, 1]");
            this.advantageLambda = v; return this;
        }
        public Builder advantage_lambda(double v) { return advantageLambda(v); }
        public Builder valueLossCoeff(double v) {
            if (v < 0) throw new IllegalArgumentException("value_loss_coeff must be >= 0");
            this.valueLossCoeff = v; return this;
        }
        public Builder value_loss_coeff(double v) { return valueLossCoeff(v); }
        public Builder entropyCoefficient(double v) {
            if (v < 0) throw new IllegalArgumentException("entropy_coefficient must be >= 0");
            this.entropyCoefficient = v; return this;
        }
        public Builder entropy_coefficient(double v) { return entropyCoefficient(v); }
        public Builder useGAE(boolean v) { this.useGAE = v; return this; }
        public Builder use_gae(boolean v) { return useGAE(v); }
        public Builder gaeGamma(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gae_gamma must be in [0, 1]");
            this.gaeGamma = v; return this;
        }
        public Builder gae_gamma(double v) { return gaeGamma(v); }
        public Builder gaeTau(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gae_tau must be in [0, 1]");
            this.gaeTau = v; return this;
        }
        public Builder gae_tau(double v) { return gaeTau(v); }
        public Builder gamma(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
            this.gamma = v; return this;
        }
        public Builder initKlCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("init_kl_coef must be >= 0");
            this.initKlCoef = v; return this;
        }
        public Builder init_kl_coef(double v) { return initKlCoef(v); }
        public Builder targetKl(double v) {
            if (v < 0) throw new IllegalArgumentException("target_kl must be >= 0");
            this.targetKl = v; return this;
        }
        public Builder target_kl(double v) { return targetKl(v); }
        public Builder adapKlCtrl(boolean v) { this.adapKlCtrl = v; return this; }
        public Builder adap_kl_ctrl(boolean v) { return adapKlCtrl(v); }
        public Builder cliprangeReward(double v) {
            if (v < 0) throw new IllegalArgumentException("cliprange_reward must be >= 0");
            this.cliprangeReward = v; return this;
        }
        public Builder cliprange_reward(double v) { return cliprangeReward(v); }
        public Builder klEstimator(String v) {
            if (v == null) { this.klEstimator = "kl"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "kl": case "k1": case "k2": case "k3":
                    this.klEstimator = n; return this;
                default:
                    throw new IllegalArgumentException("kl_estimator must be kl/k1/k2/k3");
            }
        }
        public Builder kl_estimator(String v) { return klEstimator(v); }
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
        public Builder ppoEpochs(int v) {
            if (v < 1) throw new IllegalArgumentException("ppo_epochs must be >= 1");
            this.ppoEpochs = v; return this;
        }
        public Builder ppo_epochs(int v) { return ppoEpochs(v); }
        public Builder miniBatchSize(int v) {
            if (v < 1) throw new IllegalArgumentException("mini_batch_size must be >= 1");
            this.miniBatchSize = v; return this;
        }
        public Builder mini_batch_size(int v) { return miniBatchSize(v); }
        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder useReferenceKl(boolean v) { this.useReferenceKl = v; return this; }
        public Builder use_reference_kl(boolean v) { return useReferenceKl(v); }

        @Override
        public PPOXConfig build() { return new PPOXConfig(this); }
    }
}