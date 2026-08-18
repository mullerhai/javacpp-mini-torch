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
 * Configuration for Nash Mirror Descent (Nash-MD) trainer (HF TRL {@code NashMDConfig}).
 *
 * <p>Nash-MD is a game-theoretic approach to LLM alignment that uses
 * mirror descent with Nash equilibrium as the learning target.
 */
public final class NashMDConfig extends TrainerConfig {
    private final int numObjectives;
    private final double eta;
    private final double klTarget;
    private final double klCoef;
    private final double clipRange;
    private final double equilibriumTemperature;
    private final double equilibriumMomentum;
    private final boolean useReferenceKl;
    private final double gamma;
    private final double gaeLambda;
    private final int miniBatchSize;
    private final int ppoEpochs;
    private final double entCoef;
    private final double vfCoef;
    private final double maxLength;
    private final double maxPromptLength;
    private final String scaleRewards;
    private final boolean whitenAdvantages;
    private final double advantageClip;
    private final double rewardClip;
    private final double beta;
    private final double rewardBaseline;
    private final String strategyUpdate;

    private NashMDConfig(Builder b) {
        super(b);
        this.numObjectives = b.numObjectives;
        this.eta = b.eta;
        this.klTarget = b.klTarget;
        this.klCoef = b.klCoef;
        this.clipRange = b.clipRange;
        this.equilibriumTemperature = b.equilibriumTemperature;
        this.equilibriumMomentum = b.equilibriumMomentum;
        this.useReferenceKl = b.useReferenceKl;
        this.gamma = b.gamma;
        this.gaeLambda = b.gaeLambda;
        this.miniBatchSize = b.miniBatchSize;
        this.ppoEpochs = b.ppoEpochs;
        this.entCoef = b.entCoef;
        this.vfCoef = b.vfCoef;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.scaleRewards = b.scaleRewards;
        this.whitenAdvantages = b.whitenAdvantages;
        this.advantageClip = b.advantageClip;
        this.rewardClip = b.rewardClip;
        this.beta = b.beta;
        this.rewardBaseline = b.rewardBaseline;
        this.strategyUpdate = b.strategyUpdate;
    }

    public int numObjectives() { return numObjectives; }
    public double eta() { return eta; }
    public double klTarget() { return klTarget; }
    public double klCoef() { return klCoef; }
    public double clipRange() { return clipRange; }
    public double equilibriumTemperature() { return equilibriumTemperature; }
    public double equilibriumMomentum() { return equilibriumMomentum; }
    public boolean useReferenceKl() { return useReferenceKl; }
    public double gamma() { return gamma; }
    public double gaeLambda() { return gaeLambda; }
    public int miniBatchSize() { return miniBatchSize; }
    public int ppoEpochs() { return ppoEpochs; }
    public double entCoef() { return entCoef; }
    public double vfCoef() { return vfCoef; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public String scaleRewards() { return scaleRewards; }
    public boolean whitenAdvantages() { return whitenAdvantages; }
    public double advantageClip() { return advantageClip; }
    public double rewardClip() { return rewardClip; }
    public double beta() { return beta; }
    public double rewardBaseline() { return rewardBaseline; }
    public String strategyUpdate() { return strategyUpdate; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("num_objectives", numObjectives);
        m.put("eta", eta);
        m.put("kl_target", klTarget);
        m.put("kl_coef", klCoef);
        m.put("clip_range", clipRange);
        m.put("equilibrium_temperature", equilibriumTemperature);
        m.put("equilibrium_momentum", equilibriumMomentum);
        m.put("use_reference_kl", useReferenceKl);
        m.put("gamma", gamma);
        m.put("gae_lambda", gaeLambda);
        m.put("mini_batch_size", miniBatchSize);
        m.put("ppo_epochs", ppoEpochs);
        m.put("ent_coef", entCoef);
        m.put("vf_coef", vfCoef);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("scale_rewards", scaleRewards);
        m.put("whiten_advantages", whitenAdvantages);
        m.put("advantage_clip", advantageClip);
        m.put("reward_clip", rewardClip);
        m.put("beta", beta);
        m.put("reward_baseline", rewardBaseline);
        m.put("strategy_update", strategyUpdate);
        return m;
    }

    public Builder toBuilder() { return new Builder(this); }
    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int numObjectives = 1;
        private double eta = 0.1;
        private double klTarget = 0.1;
        private double klCoef = 1.0;
        private double clipRange = 0.2;
        private double equilibriumTemperature = 1.0;
        private double equilibriumMomentum = 0.9;
        private boolean useReferenceKl = true;
        private double gamma = 0.99;
        private double gaeLambda = 0.95;
        private int miniBatchSize = 64;
        private int ppoEpochs = 4;
        private double entCoef = 0.01;
        private double vfCoef = 0.5;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private String scaleRewards = "none";
        private boolean whitenAdvantages = false;
        private double advantageClip = 0.0;
        private double rewardClip = 0.0;
        private double beta = 0.04;
        private double rewardBaseline = 0.0;
        private String strategyUpdate = "softmax";

        public Builder() {}

        public Builder(NashMDConfig config) {
            this.numObjectives = config.numObjectives;
            this.eta = config.eta;
            this.klTarget = config.klTarget;
            this.klCoef = config.klCoef;
            this.clipRange = config.clipRange;
            this.equilibriumTemperature = config.equilibriumTemperature;
            this.equilibriumMomentum = config.equilibriumMomentum;
            this.useReferenceKl = config.useReferenceKl;
            this.gamma = config.gamma;
            this.gaeLambda = config.gaeLambda;
            this.miniBatchSize = config.miniBatchSize;
            this.ppoEpochs = config.ppoEpochs;
            this.entCoef = config.entCoef;
            this.vfCoef = config.vfCoef;
            this.maxLength = config.maxLength;
            this.maxPromptLength = config.maxPromptLength;
            this.scaleRewards = config.scaleRewards;
            this.whitenAdvantages = config.whitenAdvantages;
            this.advantageClip = config.advantageClip;
            this.rewardClip = config.rewardClip;
            this.beta = config.beta;
            this.rewardBaseline = config.rewardBaseline;
            this.strategyUpdate = config.strategyUpdate;
        }

        public Builder numObjectives(int v) {
            if (v < 1) throw new IllegalArgumentException("num_objectives must be >= 1");
            this.numObjectives = v; return this;
        }
        public Builder num_objectives(int v) { return numObjectives(v); }
        public Builder eta(double v) {
            if (v <= 0) throw new IllegalArgumentException("eta must be > 0");
            this.eta = v; return this;
        }
        public Builder klTarget(double v) {
            if (v < 0) throw new IllegalArgumentException("kl_target must be >= 0");
            this.klTarget = v; return this;
        }
        public Builder kl_target(double v) { return klTarget(v); }
        public Builder klCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("kl_coef must be >= 0");
            this.klCoef = v; return this;
        }
        public Builder kl_coef(double v) { return klCoef(v); }
        public Builder clipRange(double v) {
            if (v < 0) throw new IllegalArgumentException("clip_range must be >= 0");
            this.clipRange = v; return this;
        }
        public Builder clip_range(double v) { return clipRange(v); }
        public Builder equilibriumTemperature(double v) {
            if (v <= 0) throw new IllegalArgumentException("equilibrium_temperature must be > 0");
            this.equilibriumTemperature = v; return this;
        }
        public Builder equilibrium_temperature(double v) { return equilibriumTemperature(v); }
        public Builder equilibriumMomentum(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("equilibrium_momentum must be in [0, 1]");
            this.equilibriumMomentum = v; return this;
        }
        public Builder equilibrium_momentum(double v) { return equilibriumMomentum(v); }
        public Builder useReferenceKl(boolean v) { this.useReferenceKl = v; return this; }
        public Builder use_reference_kl(boolean v) { return useReferenceKl(v); }
        public Builder gamma(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
            this.gamma = v; return this;
        }
        public Builder gaeLambda(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gae_lambda must be in [0, 1]");
            this.gaeLambda = v; return this;
        }
        public Builder gae_lambda(double v) { return gaeLambda(v); }
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
        public Builder advantageClip(double v) {
            if (v < 0) throw new IllegalArgumentException("advantage_clip must be >= 0");
            this.advantageClip = v; return this;
        }
        public Builder advantage_clip(double v) { return advantageClip(v); }
        public Builder rewardClip(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_clip must be >= 0");
            this.rewardClip = v; return this;
        }
        public Builder reward_clip(double v) { return rewardClip(v); }
        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder rewardBaseline(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_baseline must be >= 0");
            this.rewardBaseline = v; return this;
        }
        public Builder reward_baseline(double v) { return rewardBaseline(v); }
        public Builder strategyUpdate(String v) {
            if (v == null) { this.strategyUpdate = "softmax"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "softmax": case "linear": case "none":
                    this.strategyUpdate = n; return this;
                default:
                    throw new IllegalArgumentException("strategy_update must be softmax/linear/none");
            }
        }
        public Builder strategy_update(String v) { return strategyUpdate(v); }

        public NashMDConfig build() { return new NashMDConfig(this); }
    }
}