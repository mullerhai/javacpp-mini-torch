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
 * Configuration for MultiModal GRPO trainer (HF TRL {@code MultiModalGRPOConfig}).
 */
public final class MultiModalGRPOConfig extends TrainerConfig {
    private final int groupSize;
    private final double beta;
    private final double rewardNormScale;
    private final double entropyCoeff;
    private final double modalityDropout;
    private final boolean crossModalReward;
    private final String modalityConfig;
    private final double imageWeight;
    private final double audioWeight;
    private final double textWeight;
    private final boolean adaptiveGrouping;
    private final double rewardClip;
    private final double advantageClip;
    private final boolean whitenAdvantages;
    private final String scaleRewards;
    private final double maskTruncatedCompletions; // 0.0/1.0
    private final boolean mask_truncated_completions;
    private final double epsilon;
    private final double epsilonHigh;
    private final int numIterations;
    private final int miniBatchSize;
    private final double temperature;
    private final int maxCompletionLength;
    private final double routerAuxLossCoef;
    private final String lossType;             // "grpo" | "dapo"

    private MultiModalGRPOConfig(Builder b) {
        super(b);
        this.groupSize = b.groupSize;
        this.beta = b.beta;
        this.rewardNormScale = b.rewardNormScale;
        this.entropyCoeff = b.entropyCoeff;
        this.modalityDropout = b.modalityDropout;
        this.crossModalReward = b.crossModalReward;
        this.modalityConfig = b.modalityConfig;
        this.imageWeight = b.imageWeight;
        this.audioWeight = b.audioWeight;
        this.textWeight = b.textWeight;
        this.adaptiveGrouping = b.adaptiveGrouping;
        this.rewardClip = b.rewardClip;
        this.advantageClip = b.advantageClip;
        this.whitenAdvantages = b.whitenAdvantages;
        this.scaleRewards = b.scaleRewards;
        this.maskTruncatedCompletions = b.maskTruncatedCompletions;
        this.mask_truncated_completions = b.maskTruncatedCompletions >= 1.0;
        this.epsilon = b.epsilon;
        this.epsilonHigh = b.epsilonHigh;
        this.numIterations = b.numIterations;
        this.miniBatchSize = b.miniBatchSize;
        this.temperature = b.temperature;
        this.maxCompletionLength = b.maxCompletionLength;
        this.routerAuxLossCoef = b.routerAuxLossCoef;
        this.lossType = b.lossType;
    }

    public int groupSize() { return groupSize; }
    public double beta() { return beta; }
    public double rewardNormScale() { return rewardNormScale; }
    public double entropyCoeff() { return entropyCoeff; }
    public double modalityDropout() { return modalityDropout; }
    public boolean crossModalReward() { return crossModalReward; }
    public String modalityConfig() { return modalityConfig; }
    public double imageWeight() { return imageWeight; }
    public double audioWeight() { return audioWeight; }
    public double textWeight() { return textWeight; }
    public boolean adaptiveGrouping() { return adaptiveGrouping; }
    public double rewardClip() { return rewardClip; }
    public double advantageClip() { return advantageClip; }
    public boolean whitenAdvantages() { return whitenAdvantages; }
    public String scaleRewards() { return scaleRewards; }
    public double maskTruncatedCompletions() { return maskTruncatedCompletions; }
    public boolean mask_truncated_completions() { return mask_truncated_completions; }
    public double epsilon() { return epsilon; }
    public double epsilonHigh() { return epsilonHigh; }
    public int numIterations() { return numIterations; }
    public int miniBatchSize() { return miniBatchSize; }
    public double temperature() { return temperature; }
    public int maxCompletionLength() { return maxCompletionLength; }
    public double routerAuxLossCoef() { return routerAuxLossCoef; }
    public String lossType() { return lossType; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("group_size", groupSize);
        m.put("beta", beta);
        m.put("reward_norm_scale", rewardNormScale);
        m.put("entropy_coeff", entropyCoeff);
        m.put("modality_dropout", modalityDropout);
        m.put("cross_modal_reward", crossModalReward);
        m.put("modality_config", modalityConfig);
        m.put("image_weight", imageWeight);
        m.put("audio_weight", audioWeight);
        m.put("text_weight", textWeight);
        m.put("adaptive_grouping", adaptiveGrouping);
        m.put("reward_clip", rewardClip);
        m.put("advantage_clip", advantageClip);
        m.put("whiten_advantages", whitenAdvantages);
        m.put("scale_rewards", scaleRewards);
        m.put("mask_truncated_completions", mask_truncated_completions);
        m.put("epsilon", epsilon);
        m.put("epsilon_high", epsilonHigh);
        m.put("num_iterations", numIterations);
        m.put("mini_batch_size", miniBatchSize);
        m.put("temperature", temperature);
        m.put("max_completion_length", maxCompletionLength);
        m.put("router_aux_loss_coef", routerAuxLossCoef);
        m.put("loss_type", lossType);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int groupSize = 4;
        private double beta = 0.1;
        private double rewardNormScale = 0.0;
        private double entropyCoeff = 0.01;
        private double modalityDropout = 0.1;
        private boolean crossModalReward = true;
        private String modalityConfig = "text,image";
        private double imageWeight = 0.5;
        private double audioWeight = 0.25;
        private double textWeight = 0.25;
        private boolean adaptiveGrouping = true;
        private double rewardClip = 0.0;
        private double advantageClip = 0.0;
        private boolean whitenAdvantages = true;
        private String scaleRewards = "group";
        private double maskTruncatedCompletions = 0.0;
        private double epsilon = 0.2;
        private double epsilonHigh = 0.28;
        private int numIterations = 1;
        private int miniBatchSize = 256;
        private double temperature = 0.9;
        private int maxCompletionLength = 256;
        private double routerAuxLossCoef = 0.0;
        private String lossType = "grpo";

        public Builder groupSize(int v) {
            if (v < 2) throw new IllegalArgumentException("group_size must be >= 2");
            this.groupSize = v; return this;
        }
        public Builder group_size(int v) { return groupSize(v); }
        public Builder beta(double v) {
            if (v < 0.001) throw new IllegalArgumentException("beta must be >= 0.001");
            this.beta = v; return this;
        }
        public Builder rewardNormScale(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_norm_scale must be >= 0");
            this.rewardNormScale = v; return this;
        }
        public Builder reward_norm_scale(double v) { return rewardNormScale(v); }
        public Builder entropyCoeff(double v) {
            if (v < 0) throw new IllegalArgumentException("entropy_coeff must be >= 0");
            this.entropyCoeff = v; return this;
        }
        public Builder entropy_coeff(double v) { return entropyCoeff(v); }
        public Builder modalityDropout(double v) {
            if (v < 0 || v > 0.5) throw new IllegalArgumentException("modality_dropout must be in [0, 0.5]");
            this.modalityDropout = v; return this;
        }
        public Builder modality_dropout(double v) { return modalityDropout(v); }
        public Builder crossModalReward(boolean v) { this.crossModalReward = v; return this; }
        public Builder cross_modal_reward(boolean v) { return crossModalReward(v); }
        public Builder modalityConfig(String v) { this.modalityConfig = v; return this; }
        public Builder modality_config(String v) { return modalityConfig(v); }
        public Builder imageWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("image_weight must be in [0, 1]");
            this.imageWeight = v; return this;
        }
        public Builder image_weight(double v) { return imageWeight(v); }
        public Builder audioWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("audio_weight must be in [0, 1]");
            this.audioWeight = v; return this;
        }
        public Builder audio_weight(double v) { return audioWeight(v); }
        public Builder textWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("text_weight must be in [0, 1]");
            this.textWeight = v; return this;
        }
        public Builder text_weight(double v) { return textWeight(v); }
        public Builder adaptiveGrouping(boolean v) { this.adaptiveGrouping = v; return this; }
        public Builder adaptive_grouping(boolean v) { return adaptiveGrouping(v); }
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
        public Builder whitenAdvantages(boolean v) { this.whitenAdvantages = v; return this; }
        public Builder whiten_advantages(boolean v) { return whitenAdvantages(v); }
        public Builder scaleRewards(String v) {
            if (v == null) { this.scaleRewards = "group"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "none": case "group": case "batch":
                    this.scaleRewards = n; return this;
                default:
                    throw new IllegalArgumentException("scale_rewards must be none/group/batch");
            }
        }
        public Builder scale_rewards(String v) { return scaleRewards(v); }
        public Builder maskTruncatedCompletions(boolean v) {
            this.maskTruncatedCompletions = v ? 1.0 : 0.0; return this;
        }
        public Builder mask_truncated_completions(boolean v) { return maskTruncatedCompletions(v); }
        public Builder epsilon(double v) {
            if (v < 0) throw new IllegalArgumentException("epsilon must be >= 0");
            this.epsilon = v; return this;
        }
        public Builder epsilonHigh(double v) {
            if (v < 0) throw new IllegalArgumentException("epsilon_high must be >= 0");
            this.epsilonHigh = v; return this;
        }
        public Builder epsilon_high(double v) { return epsilonHigh(v); }
        public Builder numIterations(int v) {
            if (v < 1) throw new IllegalArgumentException("num_iterations must be >= 1");
            this.numIterations = v; return this;
        }
        public Builder num_iterations(int v) { return numIterations(v); }
        public Builder miniBatchSize(int v) {
            if (v < 1) throw new IllegalArgumentException("mini_batch_size must be >= 1");
            this.miniBatchSize = v; return this;
        }
        public Builder mini_batch_size(int v) { return miniBatchSize(v); }
        public Builder temperature(double v) {
            if (v < 0) throw new IllegalArgumentException("temperature must be >= 0");
            this.temperature = v; return this;
        }
        public Builder maxCompletionLength(int v) {
            if (v < 1) throw new IllegalArgumentException("max_completion_length must be >= 1");
            this.maxCompletionLength = v; return this;
        }
        public Builder max_completion_length(int v) { return maxCompletionLength(v); }
        public Builder routerAuxLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("router_aux_loss_coef must be >= 0");
            this.routerAuxLossCoef = v; return this;
        }
        public Builder router_aux_loss_coef(double v) { return routerAuxLossCoef(v); }
        public Builder lossType(String v) {
            if (v == null) { this.lossType = "grpo"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "grpo": case "dapo":
                    this.lossType = n; return this;
                default:
                    throw new IllegalArgumentException("loss_type must be 'grpo' or 'dapo'");
            }
        }
        public Builder loss_type(String v) { return lossType(v); }

        @Override
        public MultiModalGRPOConfig build() {
            double total = imageWeight + audioWeight + textWeight;
            if (Math.abs(total - 1.0) > 0.01 && total > 1e-8) {
                imageWeight /= total;
                audioWeight /= total;
                textWeight /= total;
            }
            return new MultiModalGRPOConfig(this);
        }
    }
}