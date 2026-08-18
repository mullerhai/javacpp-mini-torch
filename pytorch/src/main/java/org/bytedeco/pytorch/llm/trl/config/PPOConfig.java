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
 * PPO trainer config (HF TRL {@code PPOConfig}).
 *
 * <p>Mirrors the Python implementation with:
 * <ul>
 *   <li>Clipping: {@code clip_range}, {@code clip_range_vf}, {@code clip_range_ratio}</li>
 *   <li>Advantage normalization: {@code whiten_advantages}, {@code normalize_advantages}, {@code scale_rewards}</li>
 *   <li>KL estimation: {@code kl_estimator} ("kl" | "k1" | "k2" | "k3"), {@code init_kl_coef}, {@code target_kl}, {@code adap_kl_ctrl}</li>
 *   <li>Loss weights: {@code vf_coef}, {@code ent_coef}, {@code early_stopping_coef}, {@code target}</li>
 *   <li>Discounting: {@code gamma}, {@code gae_lambda}, {@code lam}, {@code cliprange_reward}</li>
 *   <li>Minibatching: {@code mini_batch_size}, {@code ppo_epochs}, {@code ratio_threshold}</li>
 *   <li>Misc: {@code log_with}, {@code use_amp}, {@code use_cache}</li>
 * </ul>
 */
public final class PPOConfig extends TrainerConfig {
    private final double clipRange;
    private final double clipRangeVf;
    private final double clipRangeRatio;
    private final double vfCoef;
    private final double entCoef;
    private final double gamma;
    private final double gaeLambda;
    private final double lam;
    private final int ppoEpochs;
    private final int miniBatchSize;
    private final double initKlCoef;
    private final double targetKl;
    private final double cliprangeReward;
    private final String klEstimator;
    private final double earlyStoppingCoef;
    private final double target;
    private final boolean adapKlCtrl;
    private final double whitenAdvantages;
    private final boolean normalizeAdvantages;
    private final String scaleRewards;
    private final double ratioThreshold;
    private final boolean useAmp;
    private final boolean useCache;
    private final String logWith;

    private PPOConfig(Builder b) {
        super(b);
        this.clipRange = b.clipRange;
        this.clipRangeVf = b.clipRangeVf;
        this.clipRangeRatio = b.clipRangeRatio;
        this.vfCoef = b.vfCoef;
        this.entCoef = b.entCoef;
        this.gamma = b.gamma;
        this.gaeLambda = b.gaeLambda;
        this.lam = b.lam;
        this.ppoEpochs = b.ppoEpochs;
        this.miniBatchSize = b.miniBatchSize;
        this.initKlCoef = b.initKlCoef;
        this.targetKl = b.targetKl;
        this.cliprangeReward = b.cliprangeReward;
        this.klEstimator = b.klEstimator;
        this.earlyStoppingCoef = b.earlyStoppingCoef;
        this.target = b.target;
        this.adapKlCtrl = b.adapKlCtrl;
        this.whitenAdvantages = b.whitenAdvantages;
        this.normalizeAdvantages = b.normalizeAdvantages;
        this.scaleRewards = b.scaleRewards;
        this.ratioThreshold = b.ratioThreshold;
        this.useAmp = b.useAmp;
        this.useCache = b.useCache;
        this.logWith = b.logWith;
    }

    public double clipRange() { return clipRange; }
    public double clipRangeVf() { return clipRangeVf; }
    public double clipRangeRatio() { return clipRangeRatio; }
    public double vfCoef() { return vfCoef; }
    public double entCoef() { return entCoef; }
    public double gamma() { return gamma; }
    public double gaeLambda() { return gaeLambda; }
    public double lam() { return lam; }
    public int ppoEpochs() { return ppoEpochs; }
    public int miniBatchSize() { return miniBatchSize; }
    public double initKlCoef() { return initKlCoef; }
    public double targetKl() { return targetKl; }
    public double cliprangeReward() { return cliprangeReward; }
    public String klEstimator() { return klEstimator; }
    public double earlyStoppingCoef() { return earlyStoppingCoef; }
    public double target() { return target; }
    public boolean adapKlCtrl() { return adapKlCtrl; }
    public double whitenAdvantages() { return whitenAdvantages; }
    public boolean normalizeAdvantages() { return normalizeAdvantages; }
    public String scaleRewards() { return scaleRewards; }
    public double ratioThreshold() { return ratioThreshold; }
    public boolean useAmp() { return useAmp; }
    public boolean useCache() { return useCache; }
    public String logWith() { return logWith; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("clip_range", clipRange);
        m.put("clip_range_vf", clipRangeVf);
        m.put("clip_range_ratio", clipRangeRatio);
        m.put("vf_coef", vfCoef);
        m.put("ent_coef", entCoef);
        m.put("gamma", gamma);
        m.put("gae_lambda", gaeLambda);
        m.put("lam", lam);
        m.put("ppo_epochs", ppoEpochs);
        m.put("mini_batch_size", miniBatchSize);
        m.put("init_kl_coef", initKlCoef);
        m.put("target_kl", targetKl);
        m.put("cliprange_reward", cliprangeReward);
        m.put("kl_estimator", klEstimator);
        m.put("early_stopping_coef", earlyStoppingCoef);
        m.put("target", target);
        m.put("adap_kl_ctrl", adapKlCtrl);
        m.put("whiten_advantages", whitenAdvantages);
        m.put("normalize_advantages", normalizeAdvantages);
        m.put("scale_rewards", scaleRewards);
        m.put("ratio_threshold", ratioThreshold);
        m.put("use_amp", useAmp);
        m.put("use_cache", useCache);
        m.put("log_with", logWith);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double clipRange = 0.2;
        private double clipRangeVf = 0.2;
        private double clipRangeRatio = 0.2;
        private double vfCoef = 0.5;
        private double entCoef = 0.01;
        private double gamma = 0.99;
        private double gaeLambda = 0.95;
        private double lam = 0.95;
        private int ppoEpochs = 4;
        private int miniBatchSize = 64;
        private double initKlCoef = 0.01;
        private double targetKl = 6.0;
        private double cliprangeReward = 10.0;
        private String klEstimator = "kl";
        private double earlyStoppingCoef = 0.5;
        private double target = 6.0;
        private boolean adapKlCtrl = true;
        private double whitenAdvantages = true ? 1.0 : 0.0;
        private boolean normalizeAdvantages = true;
        private String scaleRewards = "none"; // "none" | "group" | "batch"
        private double ratioThreshold = 10.0;
        private boolean useAmp = false;
        private boolean useCache = true;
        private String logWith = "wandb"; // "wandb" | "tensorboard" | "none"

        public Builder clipRange(double v) {
            if (v <= 0) throw new IllegalArgumentException("clip_range must be > 0");
            this.clipRange = v; return this;
        }
        public Builder clip_range(double v) { return clipRange(v); }

        public Builder clipRangeVf(double v) {
            if (v <= 0) throw new IllegalArgumentException("clip_range_vf must be > 0");
            this.clipRangeVf = v; return this;
        }
        public Builder clip_range_vf(double v) { return clipRangeVf(v); }

        public Builder clipRangeRatio(double v) {
            if (v <= 0) throw new IllegalArgumentException("clip_range_ratio must be > 0");
            this.clipRangeRatio = v; return this;
        }
        public Builder clip_range_ratio(double v) { return clipRangeRatio(v); }

        public Builder vfCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("vf_coef must be >= 0");
            this.vfCoef = v; return this;
        }
        public Builder vf_coef(double v) { return vfCoef(v); }

        public Builder entCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("ent_coef must be >= 0");
            this.entCoef = v; return this;
        }
        public Builder ent_coef(double v) { return entCoef(v); }

        public Builder gamma(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gamma must be in [0, 1]");
            this.gamma = v; return this;
        }
        public Builder gaeLambda(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("gae_lambda must be in [0, 1]");
            this.gaeLambda = v; return this;
        }
        public Builder gae_lambda(double v) { return gaeLambda(v); }

        public Builder lam(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("lam must be in [0, 1]");
            this.lam = v; return this;
        }

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

        public Builder cliprangeReward(double v) {
            if (v < 0) throw new IllegalArgumentException("cliprange_reward must be >= 0");
            this.cliprangeReward = v; return this;
        }
        public Builder cliprange_reward(double v) { return cliprangeReward(v); }

        public Builder klEstimator(String v) {
            if (v == null) { this.klEstimator = "kl"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "kl": case "k1": case "k2": case "k3":
                    this.klEstimator = norm; return this;
                default:
                    throw new IllegalArgumentException("kl_estimator must be one of kl/k1/k2/k3");
            }
        }
        public Builder kl_estimator(String v) { return klEstimator(v); }

        public Builder earlyStoppingCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("early_stopping_coef must be >= 0");
            this.earlyStoppingCoef = v; return this;
        }
        public Builder early_stopping_coef(double v) { return earlyStoppingCoef(v); }

        public Builder target(double v) {
            if (v < 0) throw new IllegalArgumentException("target must be >= 0");
            this.target = v; return this;
        }

        public Builder adapKlCtrl(boolean v) { this.adapKlCtrl = v; return this; }
        public Builder adap_kl_ctrl(boolean v) { return adapKlCtrl(v); }

        public Builder whitenAdvantages(boolean v) {
            this.whitenAdvantages = v ? 1.0 : 0.0; return this;
        }
        public Builder whiten_advantages(boolean v) { return whitenAdvantages(v); }
        public Builder whitenAdvantages(double v) {
            if (v != 0.0 && v != 1.0) throw new IllegalArgumentException("whiten_advantages must be 0.0 or 1.0");
            this.whitenAdvantages = v; return this;
        }
        public Builder whiten_advantages(double v) { return whitenAdvantages(v); }

        public Builder normalizeAdvantages(boolean v) { this.normalizeAdvantages = v; return this; }
        public Builder normalize_advantages(boolean v) { return normalizeAdvantages(v); }

        public Builder scaleRewards(String v) {
            if (v == null) { this.scaleRewards = "none"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "none": case "group": case "batch":
                    this.scaleRewards = norm; return this;
                default:
                    throw new IllegalArgumentException("scale_rewards must be none/group/batch");
            }
        }
        public Builder scale_rewards(String v) { return scaleRewards(v); }

        public Builder ratioThreshold(double v) {
            if (v <= 0) throw new IllegalArgumentException("ratio_threshold must be > 0");
            this.ratioThreshold = v; return this;
        }
        public Builder ratio_threshold(double v) { return ratioThreshold(v); }

        public Builder useAmp(boolean v) { this.useAmp = v; return this; }
        public Builder use_amp(boolean v) { return useAmp(v); }

        public Builder useCache(boolean v) { this.useCache = v; return this; }
        public Builder use_cache(boolean v) { return useCache(v); }

        public Builder logWith(String v) {
            if (v == null) { this.logWith = "none"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "wandb": case "tensorboard": case "none":
                    this.logWith = norm; return this;
                default:
                    throw new IllegalArgumentException("log_with must be wandb/tensorboard/none");
            }
        }
        public Builder log_with(String v) { return logWith(v); }

        @Override
        public PPOConfig build() { return new PPOConfig(this); }
    }
}