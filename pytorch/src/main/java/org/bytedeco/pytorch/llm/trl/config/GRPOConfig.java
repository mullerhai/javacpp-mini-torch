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
 * GRPO (Group Relative Policy Optimization) config (HF TRL {@code GRPOConfig}).
 *
 * <p>Mirrors HF TRL — samples {@code num_generations} completions per prompt
 * and group-normalizes rewards to form advantages. Supports {@code vLLM}-style
 * generation knobs, masked truncated completions, multi-iteration policy
 * updates, and reward clipping/scaling.
 */
public final class GRPOConfig extends TrainerConfig {
    // Generation
    private final int numGenerations;
    private final double temperature;
    private final int maxCompletionLength;
    private final double topP;
    private final double topK;
    private final double minP;
    private final int maxPromptLength;
    private final String logitBias;
    private final String generation_kwargs;

    // Reward normalization / clipping
    private final String scaleRewards;
    private final double rewardClip;
    private final double rewardWeights;

    // KL / advantage
    private final double beta;
    private final double epsilon;
    private final double epsilonHigh;
    private final double delta;
    private final double klEstimator;
    private final double advantageClip;
    private final boolean maskTruncatedCompletions;
    private final boolean whitenAdvantages;
    private final double groupNormScale;
    private final double routerAuxLossCoef;

    // Policy update
    private final double clipRange;
    private final boolean useClipping;
    private final int numIterations;
    private final double lowerTopk;
    private final double miniBatchSize;
    private final boolean lossType; // false=grpo, true=dapo  (placeholder name)
    private final String lossTypeStr;

    // vLLM
    private final boolean useVllm;
    private final String vllmMode;
    private final String vllmServerHost;
    private final int vllmServerPort;
    private final int vllmServerTimeout;

    private final boolean logCompletions;
    private final int numSampleGenerations;

    private GRPOConfig(Builder b) {
        super(b);
        this.numGenerations = b.numGenerations;
        this.temperature = b.temperature;
        this.maxCompletionLength = b.maxCompletionLength;
        this.topP = b.topP;
        this.topK = b.topK;
        this.minP = b.minP;
        this.maxPromptLength = b.maxPromptLength;
        this.logitBias = b.logitBias;
        this.generation_kwargs = b.generation_kwargs;
        this.scaleRewards = b.scaleRewards;
        this.rewardClip = b.rewardClip;
        this.rewardWeights = b.rewardWeights;
        this.beta = b.beta;
        this.epsilon = b.epsilon;
        this.epsilonHigh = b.epsilonHigh;
        this.delta = b.delta;
        this.klEstimator = b.klEstimator;
        this.advantageClip = b.advantageClip;
        this.maskTruncatedCompletions = b.maskTruncatedCompletions;
        this.whitenAdvantages = b.whitenAdvantages;
        this.groupNormScale = b.groupNormScale;
        this.routerAuxLossCoef = b.routerAuxLossCoef;
        this.clipRange = b.clipRange;
        this.useClipping = b.useClipping;
        this.numIterations = b.numIterations;
        this.lowerTopk = b.lowerTopk;
        this.miniBatchSize = b.miniBatchSize;
        this.lossType = b.lossType;
        this.lossTypeStr = b.lossTypeStr;
        this.useVllm = b.useVllm;
        this.vllmMode = b.vllmMode;
        this.vllmServerHost = b.vllmServerHost;
        this.vllmServerPort = b.vllmServerPort;
        this.vllmServerTimeout = b.vllmServerTimeout;
        this.logCompletions = b.logCompletions;
        this.numSampleGenerations = b.numSampleGenerations;
    }

    public int numGenerations() { return numGenerations; }
    public double temperature() { return temperature; }
    public int maxCompletionLength() { return maxCompletionLength; }
    public double topP() { return topP; }
    public double topK() { return topK; }
    public double minP() { return minP; }
    public int maxPromptLength() { return maxPromptLength; }
    public String logitBias() { return logitBias; }
    public String generationKwargs() { return generation_kwargs; }
    public String scaleRewards() { return scaleRewards; }
    public double rewardClip() { return rewardClip; }
    public double rewardWeights() { return rewardWeights; }
    public double beta() { return beta; }
    public double epsilon() { return epsilon; }
    public double epsilonHigh() { return epsilonHigh; }
    public double delta() { return delta; }
    public double klEstimator() { return klEstimator; }
    public double advantageClip() { return advantageClip; }
    public boolean maskTruncatedCompletions() { return maskTruncatedCompletions; }
    public boolean whitenAdvantages() { return whitenAdvantages; }
    public double groupNormScale() { return groupNormScale; }
    public double routerAuxLossCoef() { return routerAuxLossCoef; }
    public double clipRange() { return clipRange; }
    public boolean useClipping() { return useClipping; }
    public int numIterations() { return numIterations; }
    public double lowerTopk() { return lowerTopk; }
    public double miniBatchSize() { return miniBatchSize; }
    public boolean lossTypeIsDapo() { return lossType; }
    public String lossTypeStr() { return lossTypeStr; }
    public boolean useVllm() { return useVllm; }
    public String vllmMode() { return vllmMode; }
    public String vllmServerHost() { return vllmServerHost; }
    public int vllmServerPort() { return vllmServerPort; }
    public int vllmServerTimeout() { return vllmServerTimeout; }
    public boolean logCompletions() { return logCompletions; }
    public int numSampleGenerations() { return numSampleGenerations; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("num_generations", numGenerations);
        m.put("temperature", temperature);
        m.put("max_completion_length", maxCompletionLength);
        m.put("top_p", topP);
        m.put("top_k", topK);
        m.put("min_p", minP);
        m.put("max_prompt_length", maxPromptLength);
        m.put("logit_bias", logitBias);
        m.put("generation_kwargs", generation_kwargs);
        m.put("scale_rewards", scaleRewards);
        m.put("reward_clip", rewardClip);
        m.put("reward_weights", rewardWeights);
        m.put("beta", beta);
        m.put("epsilon", epsilon);
        m.put("epsilon_high", epsilonHigh);
        m.put("delta", delta);
        m.put("kl_estimator", klEstimator);
        m.put("advantage_clip", advantageClip);
        m.put("mask_truncated_completions", maskTruncatedCompletions);
        m.put("whiten_advantages", whitenAdvantages);
        m.put("group_norm_scale", groupNormScale);
        m.put("router_aux_loss_coef", routerAuxLossCoef);
        m.put("clip_range", clipRange);
        m.put("num_iterations", numIterations);
        m.put("mini_batch_size", miniBatchSize);
        m.put("use_vllm", useVllm);
        m.put("vllm_mode", vllmMode);
        m.put("log_completions", logCompletions);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int numGenerations = 4;
        private double temperature = 0.9;
        private int maxCompletionLength = 256;
        private double topP = 1.0;
        private double topK = -1; // -1 disables top-k
        private double minP = 0.0;
        private int maxPromptLength = 512;
        private String logitBias = "";
        private String generation_kwargs = "{}";
        private String scaleRewards = "group";
        private double rewardClip = 0.0;
        private double rewardWeights = 1.0;
        private double beta = 0.04;
        private double epsilon = 0.2;
        private double epsilonHigh = 0.28;
        private double delta = 1.0;
        private double klEstimator = 1.0; // HF accepts either "kl"/"k1" (str) or 1.0
        private double advantageClip = 0.0;
        private boolean maskTruncatedCompletions = false;
        private boolean whitenAdvantages = true;
        private double groupNormScale = 0.0;
        private double routerAuxLossCoef = 0.0;
        private double clipRange = 0.2;
        private boolean useClipping = true;
        private int numIterations = 1;
        private double lowerTopk = 0.0;
        private double miniBatchSize = 256;
        private boolean lossType = false; // false=grpo (with clipping), true=dapo (no clipping)
        private String lossTypeStr = "grpo";
        private boolean useVllm = false;
        private String vllmMode = "server"; // "server" | "colocate"
        private String vllmServerHost = "0.0.0.0";
        private int vllmServerPort = 8000;
        private int vllmServerTimeout = 240;
        private boolean logCompletions = false;
        private int numSampleGenerations = 0;

        // Generation
        public Builder numGenerations(int v) {
            if (v < 2) throw new IllegalArgumentException("num_generations must be >= 2");
            this.numGenerations = v; return this;
        }
        public Builder num_generations(int v) { return numGenerations(v); }

        public Builder temperature(double v) {
            if (v < 0) throw new IllegalArgumentException("temperature must be >= 0");
            this.temperature = v; return this;
        }

        public Builder maxCompletionLength(int v) {
            if (v < 1) throw new IllegalArgumentException("max_completion_length must be >= 1");
            this.maxCompletionLength = v; return this;
        }
        public Builder max_completion_length(int v) { return maxCompletionLength(v); }

        public Builder topP(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("top_p must be in [0, 1]");
            this.topP = v; return this;
        }
        public Builder top_p(double v) { return topP(v); }

        public Builder topK(double v) { this.topK = v; return this; }
        public Builder top_k(double v) { return topK(v); }

        public Builder minP(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("min_p must be in [0, 1]");
            this.minP = v; return this;
        }
        public Builder min_p(double v) { return minP(v); }

        public Builder maxPromptLength(int v) {
            if (v < 1) throw new IllegalArgumentException("max_prompt_length must be >= 1");
            this.maxPromptLength = v; return this;
        }
        public Builder max_prompt_length(int v) { return maxPromptLength(v); }

        public Builder logitBias(String v) { this.logitBias = v; return this; }
        public Builder logit_bias(String v) { return logitBias(v); }

        public Builder generationKwargs(String json) { this.generation_kwargs = json; return this; }
        public Builder generation_kwargs(String v) { return generationKwargs(v); }

        // Reward normalization
        public Builder scaleRewards(String v) {
            if (v == null) { this.scaleRewards = "group"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "none": case "group": case "batch":
                    this.scaleRewards = norm; return this;
                default:
                    throw new IllegalArgumentException("scale_rewards must be none/group/batch");
            }
        }
        public Builder scale_rewards(String v) { return scaleRewards(v); }

        public Builder rewardClip(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_clip must be >= 0");
            this.rewardClip = v; return this;
        }
        public Builder reward_clip(double v) { return rewardClip(v); }

        public Builder rewardWeights(double v) {
            if (v < 0) throw new IllegalArgumentException("reward_weights must be >= 0");
            this.rewardWeights = v; return this;
        }
        public Builder reward_weights(double v) { return rewardWeights(v); }

        // KL / advantage
        public Builder beta(double v) {
            this.beta = v; return this;
        }
        public Builder epsilon(double v) {
            if (v < 0) throw new IllegalArgumentException("epsilon must be >= 0");
            this.epsilon = v; return this;
        }
        public Builder epsilonHigh(double v) {
            if (v < 0) throw new IllegalArgumentException("epsilon_high must be >= 0");
            this.epsilonHigh = v; return this;
        }
        public Builder epsilon_high(double v) { return epsilonHigh(v); }
        public Builder delta(double v) {
            if (v < 0) throw new IllegalArgumentException("delta must be >= 0");
            this.delta = v; return this;
        }
        public Builder klEstimator(double v) {
            if (v < 0) throw new IllegalArgumentException("kl_estimator must be >= 0");
            this.klEstimator = v; return this;
        }
        public Builder kl_estimator(double v) { return klEstimator(v); }
        public Builder advantageClip(double v) {
            if (v < 0) throw new IllegalArgumentException("advantage_clip must be >= 0");
            this.advantageClip = v; return this;
        }
        public Builder advantage_clip(double v) { return advantageClip(v); }

        public Builder maskTruncatedCompletions(boolean v) { this.maskTruncatedCompletions = v; return this; }
        public Builder mask_truncated_completions(boolean v) { return maskTruncatedCompletions(v); }

        public Builder whitenAdvantages(boolean v) { this.whitenAdvantages = v; return this; }
        public Builder whiten_advantages(boolean v) { return whitenAdvantages(v); }

        public Builder groupNormScale(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("group_norm_scale must be in [0, 1]");
            this.groupNormScale = v; return this;
        }
        public Builder group_norm_scale(double v) { return groupNormScale(v); }

        public Builder routerAuxLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("router_aux_loss_coef must be >= 0");
            this.routerAuxLossCoef = v; return this;
        }
        public Builder router_aux_loss_coef(double v) { return routerAuxLossCoef(v); }

        // Policy
        public Builder clipRange(double v) {
            if (v < 0) throw new IllegalArgumentException("clip_range must be >= 0");
            this.clipRange = v; return this;
        }
        public Builder clip_range(double v) { return clipRange(v); }

        public Builder useClipping(boolean v) { this.useClipping = v; return this; }
        public Builder use_clipping(boolean v) { return useClipping(v); }

        public Builder numIterations(int v) {
            if (v < 1) throw new IllegalArgumentException("num_iterations must be >= 1");
            this.numIterations = v; return this;
        }
        public Builder num_iterations(int v) { return numIterations(v); }

        public Builder lowerTopk(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("lower_topk must be in [0, 1]");
            this.lowerTopk = v; return this;
        }
        public Builder lower_topk(double v) { return lowerTopk(v); }

        public Builder miniBatchSize(double v) {
            if (v < 1) throw new IllegalArgumentException("mini_batch_size must be >= 1");
            this.miniBatchSize = v; return this;
        }
        public Builder mini_batch_size(double v) { return miniBatchSize(v); }

        public Builder output_dir(String v) { return outputDir(v); }
        public Builder learning_rate(double v) { return learningRate(v); }
        public Builder max_steps(int v) { return maxSteps(v); }
        public Builder per_device_train_batch_size(int v) { return perDeviceTrainBatchSize(v); }

        public Builder lossTypeDapo(boolean v) { this.lossType = v; this.lossTypeStr = v ? "dapo" : "grpo"; return this; }
        public Builder lossType(String v) {
            if (v == null) { this.lossType = false; this.lossTypeStr = "grpo"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "grpo": this.lossType = false; this.lossTypeStr = "grpo"; return this;
                case "dapo": this.lossType = true; this.lossTypeStr = "dapo"; return this;
                default:
                    throw new IllegalArgumentException("loss_type must be grpo or dapo");
            }
        }
        public Builder loss_type(String v) { return lossType(v); }

        // vLLM
        public Builder useVllm(boolean v) { this.useVllm = v; return this; }
        public Builder use_vllm(boolean v) { return useVllm(v); }

        public Builder vllmMode(String v) {
            if (v == null) { this.vllmMode = "server"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "server": case "colocate":
                    this.vllmMode = norm; return this;
                default:
                    throw new IllegalArgumentException("vllm_mode must be server or colocate");
            }
        }
        public Builder vllm_mode(String v) { return vllmMode(v); }

        public Builder vllmServerHost(String v) { this.vllmServerHost = v; return this; }
        public Builder vllm_server_host(String v) { return vllmServerHost(v); }

        public Builder vllmServerPort(int v) {
            if (v < 1 || v > 65535) throw new IllegalArgumentException("port out of range");
            this.vllmServerPort = v; return this;
        }
        public Builder vllm_server_port(int v) { return vllmServerPort(v); }

        public Builder vllmServerTimeout(int v) {
            if (v < 1) throw new IllegalArgumentException("timeout must be >= 1");
            this.vllmServerTimeout = v; return this;
        }
        public Builder vllm_server_timeout(int v) { return vllmServerTimeout(v); }

        public Builder logCompletions(boolean v) { this.logCompletions = v; return this; }
        public Builder log_completions(boolean v) { return logCompletions(v); }

        public Builder numSampleGenerations(int v) {
            if (v < 0) throw new IllegalArgumentException("num_sample_generations must be >= 0");
            this.numSampleGenerations = v; return this;
        }
        public Builder num_sample_generations(int v) { return numSampleGenerations(v); }

        @Override
        public GRPOConfig build() { return new GRPOConfig(this); }
    }
}