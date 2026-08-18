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
 * Reward-model trainer config (HF TRL {@code RewardConfig}).
 */
public final class RewardConfig extends TrainerConfig {
    private final double margin;
    private final boolean centerRewardsCoefficient;
    private final double labelSmoothing;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final boolean lengthNormalize;
    private final double truncationMode;
    private final double learningRateLM;
    private final double learningRateReward;
    private final boolean trainAllLayers;
    private final int layersToTrain;
    private final boolean rewardHfDatasetText;
    private final double gamma;

    private RewardConfig(Builder b) {
        super(b);
        this.margin = b.margin;
        this.centerRewardsCoefficient = b.centerRewardsCoefficient;
        this.labelSmoothing = b.labelSmoothing;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.lengthNormalize = b.lengthNormalize;
        this.truncationMode = b.truncationMode;
        this.learningRateLM = b.learningRateLM;
        this.learningRateReward = b.learningRateReward;
        this.trainAllLayers = b.trainAllLayers;
        this.layersToTrain = b.layersToTrain;
        this.rewardHfDatasetText = b.rewardHfDatasetText;
        this.gamma = b.gamma;
    }

    public double margin() { return margin; }
    public boolean centerRewards() { return centerRewardsCoefficient; }
    public double labelSmoothing() { return labelSmoothing; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double truncationMode() { return truncationMode; }
    public double learningRateLM() { return learningRateLM; }
    public double learningRateReward() { return learningRateReward; }
    public boolean trainAllLayers() { return trainAllLayers; }
    public int layersToTrain() { return layersToTrain; }
    public boolean rewardHfDatasetText() { return rewardHfDatasetText; }
    public double gamma() { return gamma; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("margin", margin);
        m.put("center_rewards_coefficient", centerRewardsCoefficient);
        m.put("label_smoothing", labelSmoothing);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("length_normalize", lengthNormalize);
        m.put("truncation_mode", truncationMode);
        m.put("learning_rate_lm", learningRateLM);
        m.put("learning_rate_reward", learningRateReward);
        m.put("train_all_layers", trainAllLayers);
        m.put("layers_to_train", layersToTrain);
        m.put("gamma", gamma);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double margin = 0.0;
        private boolean centerRewardsCoefficient = false;
        private double labelSmoothing = 0.0;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private boolean lengthNormalize = false;
        private double truncationMode = 0.0;
        private double learningRateLM = 0.0;
        private double learningRateReward = 3e-4;
        private boolean trainAllLayers = true;
        private int layersToTrain = -1;
        private boolean rewardHfDatasetText = true;
        private double gamma = 0.99;

        public Builder margin(double v) { this.margin = v; return this; }
        public Builder centerRewards(boolean v) { this.centerRewardsCoefficient = v; return this; }
        public Builder center_rewards(boolean v) { return centerRewards(v); }
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
        public Builder truncationMode(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("truncation_mode must be in [0, 1]");
            this.truncationMode = v; return this;
        }
        public Builder truncation_mode(double v) { return truncationMode(v); }
        public Builder learningRateLM(double v) {
            if (v < 0) throw new IllegalArgumentException("learning_rate_lm must be >= 0");
            this.learningRateLM = v; return this;
        }
        public Builder learning_rate_lm(double v) { return learningRateLM(v); }
        public Builder learningRateReward(double v) {
            if (v < 0) throw new IllegalArgumentException("learning_rate_reward must be >= 0");
            this.learningRateReward = v; return this;
        }
        public Builder learning_rate_reward(double v) { return learningRateReward(v); }
        public Builder trainAllLayers(boolean v) { this.trainAllLayers = v; return this; }
        public Builder train_all_layers(boolean v) { return trainAllLayers(v); }
        public Builder layersToTrain(int v) {
            if (v < -1) throw new IllegalArgumentException("layers_to_train must be >= -1");
            this.layersToTrain = v; return this;
        }
        public Builder layers_to_train(int v) { return layersToTrain(v); }
        public Builder rewardHfDatasetText(boolean v) { this.rewardHfDatasetText = v; return this; }
        public Builder gamma(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
            this.gamma = v; return this;
        }

        @Override
        public RewardConfig build() { return new RewardConfig(this); }
    }
}