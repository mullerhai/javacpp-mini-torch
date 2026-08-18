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
 * Configuration for Ensemble Reward Trainer (HF TRL {@code EnsembleRewardConfig}).
 */
public final class EnsembleRewardConfig extends TrainerConfig {
    private final int numRewards;
    private final String rewardNames;
    private final double[] initialWeights;
    private final boolean adaptiveWeighting;
    private final double weightUpdateRate;
    private final boolean useParetoTraining;
    private final boolean useUncertainty;
    private final double uncertaintyThreshold;
    private final double labelSmoothing;
    private final double maxLength;
    private final double maxPromptLength;
    private final boolean disableDropout;
    private final double margin;
    private final boolean centerRewards;

    private EnsembleRewardConfig(Builder b) {
        super(b);
        this.numRewards = b.numRewards;
        this.rewardNames = b.rewardNames;
        this.initialWeights = b.initialWeights == null ? null : b.initialWeights.clone();
        this.adaptiveWeighting = b.adaptiveWeighting;
        this.weightUpdateRate = b.weightUpdateRate;
        this.useParetoTraining = b.useParetoTraining;
        this.useUncertainty = b.useUncertainty;
        this.uncertaintyThreshold = b.uncertaintyThreshold;
        this.labelSmoothing = b.labelSmoothing;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
        this.disableDropout = b.disableDropout;
        this.margin = b.margin;
        this.centerRewards = b.centerRewards;
    }

    public int numRewards() { return numRewards; }
    public String rewardNames() { return rewardNames; }
    public double[] initialWeights() { return initialWeights == null ? null : initialWeights.clone(); }
    public boolean adaptiveWeighting() { return adaptiveWeighting; }
    public double weightUpdateRate() { return weightUpdateRate; }
    public boolean useParetoTraining() { return useParetoTraining; }
    public boolean useUncertainty() { return useUncertainty; }
    public double uncertaintyThreshold() { return uncertaintyThreshold; }
    public double labelSmoothing() { return labelSmoothing; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }
    public boolean disableDropout() { return disableDropout; }
    public double margin() { return margin; }
    public boolean centerRewards() { return centerRewards; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("num_rewards", numRewards);
        m.put("reward_names", rewardNames);
        m.put("adaptive_weighting", adaptiveWeighting);
        m.put("weight_update_rate", weightUpdateRate);
        m.put("use_pareto_training", useParetoTraining);
        m.put("use_uncertainty", useUncertainty);
        m.put("uncertainty_threshold", uncertaintyThreshold);
        m.put("label_smoothing", labelSmoothing);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        m.put("disable_dropout", disableDropout);
        m.put("margin", margin);
        m.put("center_rewards", centerRewards);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int numRewards = 2;
        private String rewardNames = "reward0,reward1";
        private double[] initialWeights = new double[]{0.5, 0.5};
        private boolean adaptiveWeighting = true;
        private double weightUpdateRate = 0.01;
        private boolean useParetoTraining = false;
        private boolean useUncertainty = true;
        private double uncertaintyThreshold = 0.1;
        private double labelSmoothing = 0.0;
        private double maxLength = 1024;
        private double maxPromptLength = 512;
        private boolean disableDropout = false;
        private double margin = 0.0;
        private boolean centerRewards = false;

        public Builder numRewards(int v) {
            if (v < 1) throw new IllegalArgumentException("num_rewards must be >= 1");
            this.numRewards = v; return this;
        }
        public Builder num_rewards(int v) { return numRewards(v); }
        public Builder rewardNames(String v) { this.rewardNames = v; return this; }
        public Builder reward_names(String v) { return rewardNames(v); }
        public Builder initialWeights(double[] v) { this.initialWeights = v == null ? null : v.clone(); return this; }
        public Builder initial_weights(double[] v) { return initialWeights(v); }
        public Builder adaptiveWeighting(boolean v) { this.adaptiveWeighting = v; return this; }
        public Builder adaptive_weighting(boolean v) { return adaptiveWeighting(v); }
        public Builder weightUpdateRate(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("weight_update_rate must be in [0, 1]");
            this.weightUpdateRate = v; return this;
        }
        public Builder weight_update_rate(double v) { return weightUpdateRate(v); }
        public Builder useParetoTraining(boolean v) { this.useParetoTraining = v; return this; }
        public Builder use_pareto_training(boolean v) { return useParetoTraining(v); }
        public Builder useUncertainty(boolean v) { this.useUncertainty = v; return this; }
        public Builder use_uncertainty(boolean v) { return useUncertainty(v); }
        public Builder uncertaintyThreshold(double v) {
            if (v < 0) throw new IllegalArgumentException("uncertainty_threshold must be >= 0");
            this.uncertaintyThreshold = v; return this;
        }
        public Builder uncertainty_threshold(double v) { return uncertaintyThreshold(v); }
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
        public Builder margin(double v) {
            if (v < 0) throw new IllegalArgumentException("margin must be >= 0");
            this.margin = v; return this;
        }
        public Builder centerRewards(boolean v) { this.centerRewards = v; return this; }
        public Builder center_rewards(boolean v) { return centerRewards(v); }

        @Override
        public EnsembleRewardConfig build() {
            if (initialWeights == null || initialWeights.length != numRewards) {
                initialWeights = new double[numRewards];
                double w = 1.0 / numRewards;
                for (int i = 0; i < numRewards; i++) initialWeights[i] = w;
            }
            double sum = 0;
            for (double w : initialWeights) sum += w;
            if (Math.abs(sum - 1.0) > 0.01 && sum > 1e-8) {
                for (int i = 0; i < initialWeights.length; i++) initialWeights[i] /= sum;
            }
            return new EnsembleRewardConfig(this);
        }
    }
}