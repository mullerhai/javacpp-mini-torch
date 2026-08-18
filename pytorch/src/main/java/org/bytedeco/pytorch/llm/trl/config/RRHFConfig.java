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
 * Configuration for RRHF (Rank Responses to Rank Responses) trainer (HF TRL {@code RRHFConfig}).
 */
public final class RRHFConfig extends TrainerConfig {
    private final double rewardWeight;
    private final double ratioWeight;
    private final int numResponses;
    private final boolean sampleLevelRanking;
    private final boolean pairwiseLoss;
    private final double rewardTemperature;
    private final double labelSmoothing;
    private final boolean lengthNormalize;
    private final double beta;
    private final double gamma;
    private final double sftWeight;
    private final int topK;
    private final double margin;
    private final boolean useRankHead;
    private final boolean useReferenceModel;
    private final double maxLength;
    private final double maxPromptLength;

    private RRHFConfig(Builder b) {
        super(b);
        this.rewardWeight = b.rewardWeight;
        this.ratioWeight = b.ratioWeight;
        this.numResponses = b.numResponses;
        this.sampleLevelRanking = b.sampleLevelRanking;
        this.pairwiseLoss = b.pairwiseLoss;
        this.rewardTemperature = b.rewardTemperature;
        this.labelSmoothing = b.labelSmoothing;
        this.lengthNormalize = b.lengthNormalize;
        this.beta = b.beta;
        this.gamma = b.gamma;
        this.sftWeight = b.sftWeight;
        this.topK = b.topK;
        this.margin = b.margin;
        this.useRankHead = b.useRankHead;
        this.useReferenceModel = b.useReferenceModel;
        this.maxLength = b.maxLength;
        this.maxPromptLength = b.maxPromptLength;
    }

    public double rewardWeight() { return rewardWeight; }
    public double ratioWeight() { return ratioWeight; }
    public int numResponses() { return numResponses; }
    public boolean sampleLevelRanking() { return sampleLevelRanking; }
    public boolean pairwiseLoss() { return pairwiseLoss; }
    public double rewardTemperature() { return rewardTemperature; }
    public double labelSmoothing() { return labelSmoothing; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public double beta() { return beta; }
    public double gamma() { return gamma; }
    public double sftWeight() { return sftWeight; }
    public int topK() { return topK; }
    public double margin() { return margin; }
    public boolean useRankHead() { return useRankHead; }
    public boolean useReferenceModel() { return useReferenceModel; }
    public double maxLength() { return maxLength; }
    public double maxPromptLength() { return maxPromptLength; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("reward_weight", rewardWeight);
        m.put("ratio_weight", ratioWeight);
        m.put("num_responses", numResponses);
        m.put("sample_level_ranking", sampleLevelRanking);
        m.put("pairwise_loss", pairwiseLoss);
        m.put("reward_temperature", rewardTemperature);
        m.put("label_smoothing", labelSmoothing);
        m.put("length_normalize", lengthNormalize);
        m.put("beta", beta);
        m.put("gamma", gamma);
        m.put("sft_weight", sftWeight);
        m.put("top_k", topK);
        m.put("margin", margin);
        m.put("use_rank_head", useRankHead);
        m.put("use_reference_model", useReferenceModel);
        m.put("max_length", maxLength);
        m.put("max_prompt_length", maxPromptLength);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double rewardWeight = 1.0;
        private double ratioWeight = 1.0;
        private int numResponses = 4;
        private boolean sampleLevelRanking = true;
        private boolean pairwiseLoss = true;
        private double rewardTemperature = 1.0;
        private double labelSmoothing = 0.0;
        private boolean lengthNormalize = false;
        private double beta = 0.04;
        private double gamma = 1.0;
        private double sftWeight = 0.0;
        private int topK = 0;
        private double margin = 0.0;
        private boolean useRankHead = false;
        private boolean useReferenceModel = false;
        private double maxLength = 1024;
        private double maxPromptLength = 512;

        public Builder rewardWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("rewardWeight must be non-negative");
            this.rewardWeight = v; return this;
        }
        public Builder reward_weight(double v) { return rewardWeight(v); }
        public Builder ratioWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("ratioWeight must be non-negative");
            this.ratioWeight = v; return this;
        }
        public Builder ratio_weight(double v) { return ratioWeight(v); }
        public Builder numResponses(int v) {
            if (v < 2) throw new IllegalArgumentException("numResponses must be >= 2");
            this.numResponses = v; return this;
        }
        public Builder num_responses(int v) { return numResponses(v); }
        public Builder sampleLevelRanking(boolean v) { this.sampleLevelRanking = v; return this; }
        public Builder sample_level_ranking(boolean v) { return sampleLevelRanking(v); }
        public Builder pairwiseLoss(boolean v) { this.pairwiseLoss = v; return this; }
        public Builder pairwise_loss(boolean v) { return pairwiseLoss(v); }
        public Builder rewardTemperature(double v) {
            if (v <= 0) throw new IllegalArgumentException("rewardTemperature must be positive");
            this.rewardTemperature = v; return this;
        }
        public Builder reward_temperature(double v) { return rewardTemperature(v); }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
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
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder topK(int v) {
            if (v < 0) throw new IllegalArgumentException("top_k must be >= 0");
            this.topK = v; return this;
        }
        public Builder top_k(int v) { return topK(v); }
        public Builder margin(double v) {
            if (v < 0) throw new IllegalArgumentException("margin must be >= 0");
            this.margin = v; return this;
        }
        public Builder useRankHead(boolean v) { this.useRankHead = v; return this; }
        public Builder use_rank_head(boolean v) { return useRankHead(v); }
        public Builder useReferenceModel(boolean v) { this.useReferenceModel = v; return this; }
        public Builder use_reference_model(boolean v) { return useReferenceModel(v); }
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

        @Override
        public RRHFConfig build() { return new RRHFConfig(this); }
    }

    @Override
    public String toString() {
        return "RRHFConfig{" +
                "rewardWeight=" + rewardWeight +
                ", ratioWeight=" + ratioWeight +
                ", numResponses=" + numResponses +
                ", sampleLevelRanking=" + sampleLevelRanking +
                '}';
    }
}