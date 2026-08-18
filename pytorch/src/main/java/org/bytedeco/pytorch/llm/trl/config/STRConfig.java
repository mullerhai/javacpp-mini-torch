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
 * Configuration for Self-Taught Reasoning (STR) trainer (HF TRL {@code STRConfig}).
 */
public final class STRConfig extends TrainerConfig {
    private final int maxReasoningSteps;
    private final double reasoningTemperature;
    private final boolean useProcessReward;
    private final boolean useSelfCritique;
    private final double critiqueWeight;
    private final int beamSize;
    private final double reasoningAlpha;
    private final boolean useCoT;
    private final double beta;
    private final double labelSmoothing;
    private final double rewardWeight;
    private final double sftWeight;
    private final double prmThreshold;
    private final boolean lengthNormalize;
    private final boolean useReferenceModel;
    private final double referenceBeta;
    private final int maxCritiqueRounds;
    private final double topP;
    private final int topK;

    private STRConfig(Builder b) {
        super(b);
        this.maxReasoningSteps = b.maxReasoningSteps;
        this.reasoningTemperature = b.reasoningTemperature;
        this.useProcessReward = b.useProcessReward;
        this.useSelfCritique = b.useSelfCritique;
        this.critiqueWeight = b.critiqueWeight;
        this.beamSize = b.beamSize;
        this.reasoningAlpha = b.reasoningAlpha;
        this.useCoT = b.useCoT;
        this.beta = b.beta;
        this.labelSmoothing = b.labelSmoothing;
        this.rewardWeight = b.rewardWeight;
        this.sftWeight = b.sftWeight;
        this.prmThreshold = b.prmThreshold;
        this.lengthNormalize = b.lengthNormalize;
        this.useReferenceModel = b.useReferenceModel;
        this.referenceBeta = b.referenceBeta;
        this.maxCritiqueRounds = b.maxCritiqueRounds;
        this.topP = b.topP;
        this.topK = b.topK;
    }

    public int maxReasoningSteps() { return maxReasoningSteps; }
    public double reasoningTemperature() { return reasoningTemperature; }
    public boolean useProcessReward() { return useProcessReward; }
    public boolean processReward() { return useProcessReward; }
    public boolean useSelfCritique() { return useSelfCritique; }
    public double critiqueWeight() { return critiqueWeight; }
    public int beamSize() { return beamSize; }
    public double reasoningAlpha() { return reasoningAlpha; }
    public boolean useCoT() { return useCoT; }
    public double beta() { return beta; }
    public double labelSmoothing() { return labelSmoothing; }
    public double rewardWeight() { return rewardWeight; }
    public double sftWeight() { return sftWeight; }
    public double prmThreshold() { return prmThreshold; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public boolean useReferenceModel() { return useReferenceModel; }
    public double referenceBeta() { return referenceBeta; }
    public int maxCritiqueRounds() { return maxCritiqueRounds; }
    public double topP() { return topP; }
    public int topK() { return topK; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("max_reasoning_steps", maxReasoningSteps);
        m.put("reasoning_temperature", reasoningTemperature);
        m.put("use_process_reward", useProcessReward);
        m.put("use_self_critique", useSelfCritique);
        m.put("critique_weight", critiqueWeight);
        m.put("beam_size", beamSize);
        m.put("reasoning_alpha", reasoningAlpha);
        m.put("use_cot", useCoT);
        m.put("beta", beta);
        m.put("label_smoothing", labelSmoothing);
        m.put("reward_weight", rewardWeight);
        m.put("sft_weight", sftWeight);
        m.put("prm_threshold", prmThreshold);
        m.put("length_normalize", lengthNormalize);
        m.put("use_reference_model", useReferenceModel);
        m.put("reference_beta", referenceBeta);
        m.put("max_critique_rounds", maxCritiqueRounds);
        m.put("top_p", topP);
        m.put("top_k", topK);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int maxReasoningSteps = 8;
        private double reasoningTemperature = 0.8;
        private boolean useProcessReward = true;
        private boolean useSelfCritique = true;
        private double critiqueWeight = 0.3;
        private int beamSize = 4;
        private double reasoningAlpha = 0.5;
        private boolean useCoT = true;
        private double beta = 0.04;
        private double labelSmoothing = 0.0;
        private double rewardWeight = 1.0;
        private double sftWeight = 0.0;
        private double prmThreshold = 0.5;
        private boolean lengthNormalize = false;
        private boolean useReferenceModel = false;
        private double referenceBeta = 0.04;
        private int maxCritiqueRounds = 2;
        private double topP = 1.0;
        private int topK = 0;

        public Builder maxReasoningSteps(int v) {
            if (v < 1) throw new IllegalArgumentException("max_reasoning_steps must be >= 1");
            this.maxReasoningSteps = v; return this;
        }
        public Builder max_reasoning_steps(int v) { return maxReasoningSteps(v); }
        public Builder reasoningTemperature(double v) {
            if (v <= 0) throw new IllegalArgumentException("reasoning_temperature must be > 0");
            this.reasoningTemperature = v; return this;
        }
        public Builder reasoning_temperature(double v) { return reasoningTemperature(v); }
        public Builder useProcessReward(boolean v) { this.useProcessReward = v; return this; }
        public Builder use_process_reward(boolean v) { return useProcessReward(v); }
        public Builder useSelfCritique(boolean v) { this.useSelfCritique = v; return this; }
        public Builder use_self_critique(boolean v) { return useSelfCritique(v); }
        public Builder critiqueWeight(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("critique_weight must be in [0, 1]");
            this.critiqueWeight = v; return this;
        }
        public Builder critique_weight(double v) { return critiqueWeight(v); }
        public Builder beamSize(int v) {
            if (v < 1) throw new IllegalArgumentException("beam_size must be >= 1");
            this.beamSize = v; return this;
        }
        public Builder beam_size(int v) { return beamSize(v); }
        public Builder reasoningAlpha(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("reasoning_alpha must be in [0, 1]");
            this.reasoningAlpha = v; return this;
        }
        public Builder reasoning_alpha(double v) { return reasoningAlpha(v); }
        public Builder useCoT(boolean v) { this.useCoT = v; return this; }
        public Builder use_cot(boolean v) { return useCoT(v); }
        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
        public Builder rewardWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_weight must be >= 0");
            this.rewardWeight = v; return this;
        }
        public Builder reward_weight(double v) { return rewardWeight(v); }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder prmThreshold(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("prm_threshold must be in [0, 1]");
            this.prmThreshold = v; return this;
        }
        public Builder prm_threshold(double v) { return prmThreshold(v); }
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder useReferenceModel(boolean v) { this.useReferenceModel = v; return this; }
        public Builder use_reference_model(boolean v) { return useReferenceModel(v); }
        public Builder referenceBeta(double v) {
            if (v < 0) throw new IllegalArgumentException("reference_beta must be >= 0");
            this.referenceBeta = v; return this;
        }
        public Builder reference_beta(double v) { return referenceBeta(v); }
        public Builder maxCritiqueRounds(int v) {
            if (v < 0) throw new IllegalArgumentException("max_critique_rounds must be >= 0");
            this.maxCritiqueRounds = v; return this;
        }
        public Builder max_critique_rounds(int v) { return maxCritiqueRounds(v); }
        public Builder topP(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("top_p must be in [0, 1]");
            this.topP = v; return this;
        }
        public Builder top_p(double v) { return topP(v); }
        public Builder topK(int v) {
            if (v < 0) throw new IllegalArgumentException("top_k must be >= 0");
            this.topK = v; return this;
        }
        public Builder top_k(int v) { return topK(v); }

        @Override
        public STRConfig build() { return new STRConfig(this); }
    }
}