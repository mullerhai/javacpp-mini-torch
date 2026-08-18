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

import org.bytedeco.pytorch.llm.trl.trainer.SPOTrainer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Configuration for SPO (Self-Play Preference Optimization) trainer (HF TRL {@code SPOConfig}).
 *
 * <p>SPO uses a self-play mechanism where the model competes against different
 * versions of itself, achieving more robust alignment. Mirrors the Python TRL
 * surface with additional game-theory and mixture knobs.
 *
 * <p>Reference: "Self-Play Preference Optimization (SPO)" (Zhao et al., 2024)
 *
 * @see SPOTrainer
 */
public final class SPOConfig extends TrainerConfig {
    // Core SPO
    private final double temperature;
    private final int selfPlayIterations;
    private final double updateRate;
    private final boolean useMixture;
    private final double mixtureCoeff;

    // Game-theoretic
    private final double nashGapThreshold;
    private final double equilibriumMomentum;
    private final double strategyEntropyFloor;
    private final double initialNashGap;

    // Loss variants
    private final double beta;
    private final double gamma;
    private final double labelSmoothing;
    private final double sftWeight;
    private final double lengthNormalizeCoef;
    private final double cpoLossCoef;
    private final double simpoGamma;
    private final double rpoAlpha;
    private final double margin;
    private final String lossType;            // "spo" | "simpo" | "cpo"

    // Mixture-of-historical-policies
    private final int maxHistorySize;
    private final double historyDecay;
    private final boolean keepBestSnapshot;

    // Sampling / regularization
    private final double dropoutRate;
    private final double entropyBonus;
    private final double topP;
    private final int topK;
    private final double pathLengthNorm;

    // Bookkeeping
    private final boolean lengthNormalize;
    private final boolean disableDropout;
    private final boolean useReferenceModel;
    private final double referenceBeta;

    // Optional KL guard
    private final double klTarget;
    private final double klEpsilon;
    private final boolean adapKlCtrl;
    private final double initKlCoef;

    private SPOConfig(Builder b) {
        super(b);
        this.temperature = b.temperature;
        this.selfPlayIterations = b.selfPlayIterations;
        this.updateRate = b.updateRate;
        this.useMixture = b.useMixture;
        this.mixtureCoeff = b.mixtureCoeff;
        this.nashGapThreshold = b.nashGapThreshold;
        this.equilibriumMomentum = b.equilibriumMomentum;
        this.strategyEntropyFloor = b.strategyEntropyFloor;
        this.initialNashGap = b.initialNashGap;
        this.beta = b.beta;
        this.gamma = b.gamma;
        this.labelSmoothing = b.labelSmoothing;
        this.sftWeight = b.sftWeight;
        this.lengthNormalizeCoef = b.lengthNormalizeCoef;
        this.cpoLossCoef = b.cpoLossCoef;
        this.simpoGamma = b.simpoGamma;
        this.rpoAlpha = b.rpoAlpha;
        this.margin = b.margin;
        this.lossType = b.lossType;
        this.maxHistorySize = b.maxHistorySize;
        this.historyDecay = b.historyDecay;
        this.keepBestSnapshot = b.keepBestSnapshot;
        this.dropoutRate = b.dropoutRate;
        this.entropyBonus = b.entropyBonus;
        this.topP = b.topP;
        this.topK = b.topK;
        this.pathLengthNorm = b.pathLengthNorm;
        this.lengthNormalize = b.lengthNormalize;
        this.disableDropout = b.disableDropout;
        this.useReferenceModel = b.useReferenceModel;
        this.referenceBeta = b.referenceBeta;
        this.klTarget = b.klTarget;
        this.klEpsilon = b.klEpsilon;
        this.adapKlCtrl = b.adapKlCtrl;
        this.initKlCoef = b.initKlCoef;
    }

    // ----- core -----
    public double temperature() { return temperature; }
    public int selfPlayIterations() { return selfPlayIterations; }
    public double updateRate() { return updateRate; }
    public boolean useMixture() { return useMixture; }
    public double mixtureCoeff() { return mixtureCoeff; }

    // ----- game theory -----
    public double nashGapThreshold() { return nashGapThreshold; }
    public double equilibriumMomentum() { return equilibriumMomentum; }
    public double strategyEntropyFloor() { return strategyEntropyFloor; }
    public double initialNashGap() { return initialNashGap; }

    // ----- loss -----
    public double beta() { return beta; }
    public double gamma() { return gamma; }
    public double labelSmoothing() { return labelSmoothing; }
    public double sftWeight() { return sftWeight; }
    public double lengthNormalizeCoef() { return lengthNormalizeCoef; }
    public double cpoLossCoef() { return cpoLossCoef; }
    public double simpoGamma() { return simpoGamma; }
    public double rpoAlpha() { return rpoAlpha; }
    public double margin() { return margin; }
    public String lossType() { return lossType; }

    // ----- history -----
    public int maxHistorySize() { return maxHistorySize; }
    public double historyDecay() { return historyDecay; }
    public boolean keepBestSnapshot() { return keepBestSnapshot; }

    // ----- misc -----
    public double dropoutRate() { return dropoutRate; }
    public double entropyBonus() { return entropyBonus; }
    public double topP() { return topP; }
    public int topK() { return topK; }
    public double pathLengthNorm() { return pathLengthNorm; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public boolean disableDropout() { return disableDropout; }
    public boolean useReferenceModel() { return useReferenceModel; }
    public double referenceBeta() { return referenceBeta; }
    public double klTarget() { return klTarget; }
    public double klEpsilon() { return klEpsilon; }
    public boolean adapKlCtrl() { return adapKlCtrl; }
    public double initKlCoef() { return initKlCoef; }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("temperature", temperature);
        m.put("self_play_iterations", selfPlayIterations);
        m.put("update_rate", updateRate);
        m.put("use_mixture", useMixture);
        m.put("mixture_coeff", mixtureCoeff);
        m.put("nash_gap_threshold", nashGapThreshold);
        m.put("equilibrium_momentum", equilibriumMomentum);
        m.put("strategy_entropy_floor", strategyEntropyFloor);
        m.put("initial_nash_gap", initialNashGap);
        m.put("beta", beta);
        m.put("gamma", gamma);
        m.put("label_smoothing", labelSmoothing);
        m.put("sft_weight", sftWeight);
        m.put("length_normalize_coef", lengthNormalizeCoef);
        m.put("cpo_loss_coef", cpoLossCoef);
        m.put("simpo_gamma", simpoGamma);
        m.put("rpo_alpha", rpoAlpha);
        m.put("margin", margin);
        m.put("loss_type", lossType);
        m.put("max_history_size", maxHistorySize);
        m.put("history_decay", historyDecay);
        m.put("keep_best_snapshot", keepBestSnapshot);
        m.put("dropout_rate", dropoutRate);
        m.put("entropy_bonus", entropyBonus);
        m.put("top_p", topP);
        m.put("top_k", topK);
        m.put("length_normalize", lengthNormalize);
        m.put("disable_dropout", disableDropout);
        m.put("use_reference_model", useReferenceModel);
        m.put("reference_beta", referenceBeta);
        m.put("kl_target", klTarget);
        m.put("kl_epsilon", klEpsilon);
        m.put("adap_kl_ctrl", adapKlCtrl);
        m.put("init_kl_coef", initKlCoef);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double temperature = 1.0;
        private int selfPlayIterations = 3;
        private double updateRate = 0.1;
        private boolean useMixture = true;
        private double mixtureCoeff = 0.3;
        private double nashGapThreshold = 0.05;
        private double equilibriumMomentum = 0.9;
        private double strategyEntropyFloor = 1e-3;
        private double initialNashGap = 1.0;
        private double beta = 0.04;
        private double gamma = 1.0;
        private double labelSmoothing = 0.0;
        private double sftWeight = 0.0;
        private double lengthNormalizeCoef = 1.0;
        private double cpoLossCoef = 0.0;
        private double simpoGamma = 1.0;
        private double rpoAlpha = 0.1;
        private double margin = 0.0;
        private String lossType = "spo";
        private int maxHistorySize = 5;
        private double historyDecay = 0.99;
        private boolean keepBestSnapshot = true;
        private double dropoutRate = 0.0;
        private double entropyBonus = 0.0;
        private double topP = 1.0;
        private int topK = 0;
        private double pathLengthNorm = 0.0;
        private boolean lengthNormalize = false;
        private boolean disableDropout = false;
        private boolean useReferenceModel = false;
        private double referenceBeta = 0.04;
        private double klTarget = 6.0;
        private double klEpsilon = 0.2;
        private boolean adapKlCtrl = true;
        private double initKlCoef = 0.01;

        // ----- core -----
        public Builder temperature(double v) {
            if (v <= 0) throw new IllegalArgumentException("temperature must be positive");
            this.temperature = v; return this;
        }
        public Builder selfPlayIterations(int v) {
            if (v < 1) throw new IllegalArgumentException("self_play_iterations must be >= 1");
            this.selfPlayIterations = v; return this;
        }
        public Builder self_play_iterations(int v) { return selfPlayIterations(v); }
        public Builder updateRate(double v) {
            if (v <= 0 || v > 1) throw new IllegalArgumentException("update_rate must be in (0, 1]");
            this.updateRate = v; return this;
        }
        public Builder update_rate(double v) { return updateRate(v); }
        public Builder useMixture(boolean v) { this.useMixture = v; return this; }
        public Builder use_mixture(boolean v) { return useMixture(v); }
        public Builder mixtureCoeff(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("mixture_coeff must be in [0, 1]");
            this.mixtureCoeff = v; return this;
        }
        public Builder mixture_coeff(double v) { return mixtureCoeff(v); }

        // ----- game theory -----
        public Builder nashGapThreshold(double v) {
            if (v < 0) throw new IllegalArgumentException("nash_gap_threshold must be >= 0");
            this.nashGapThreshold = v; return this;
        }
        public Builder nash_gap_threshold(double v) { return nashGapThreshold(v); }
        public Builder equilibriumMomentum(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("equilibrium_momentum must be in [0, 1]");
            this.equilibriumMomentum = v; return this;
        }
        public Builder equilibrium_momentum(double v) { return equilibriumMomentum(v); }
        public Builder strategyEntropyFloor(double v) {
            if (v < 0) throw new IllegalArgumentException("strategy_entropy_floor must be >= 0");
            this.strategyEntropyFloor = v; return this;
        }
        public Builder strategy_entropy_floor(double v) { return strategyEntropyFloor(v); }
        public Builder initialNashGap(double v) {
            if (v < 0) throw new IllegalArgumentException("initial_nash_gap must be >= 0");
            this.initialNashGap = v; return this;
        }
        public Builder initial_nash_gap(double v) { return initialNashGap(v); }

        // ----- loss -----
        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }
        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("label_smoothing must be in [0, 1]");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }
        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }
        public Builder lengthNormalizeCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("length_normalize_coef must be >= 0");
            this.lengthNormalizeCoef = v; return this;
        }
        public Builder length_normalize_coef(double v) { return lengthNormalizeCoef(v); }
        public Builder cpoLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("cpo_loss_coef must be >= 0");
            this.cpoLossCoef = v; return this;
        }
        public Builder cpo_loss_coef(double v) { return cpoLossCoef(v); }
        public Builder simpoGamma(double v) {
            if (v < 0) throw new IllegalArgumentException("simpo_gamma must be >= 0");
            this.simpoGamma = v; return this;
        }
        public Builder simpo_gamma(double v) { return simpoGamma(v); }
        public Builder rpoAlpha(double v) {
            if (v < 0) throw new IllegalArgumentException("rpo_alpha must be >= 0");
            this.rpoAlpha = v; return this;
        }
        public Builder rpo_alpha(double v) { return rpoAlpha(v); }
        public Builder margin(double v) {
            if (v < 0) throw new IllegalArgumentException("margin must be >= 0");
            this.margin = v; return this;
        }
        public Builder lossType(String v) {
            if (v == null) { this.lossType = "spo"; return this; }
            String n = v.toLowerCase();
            switch (n) {
                case "spo": case "simpo": case "cpo":
                    this.lossType = n; return this;
                default:
                    throw new IllegalArgumentException("loss_type must be spo/simpo/cpo");
            }
        }
        public Builder loss_type(String v) { return lossType(v); }

        // ----- history -----
        public Builder maxHistorySize(int v) {
            if (v < 0) throw new IllegalArgumentException("max_history_size must be >= 0");
            this.maxHistorySize = v; return this;
        }
        public Builder max_history_size(int v) { return maxHistorySize(v); }
        public Builder historyDecay(double v) {
            if (v <= 0 || v > 1) throw new IllegalArgumentException("history_decay must be in (0, 1]");
            this.historyDecay = v; return this;
        }
        public Builder history_decay(double v) { return historyDecay(v); }
        public Builder keepBestSnapshot(boolean v) { this.keepBestSnapshot = v; return this; }
        public Builder keep_best_snapshot(boolean v) { return keepBestSnapshot(v); }

        // ----- misc -----
        public Builder dropoutRate(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("dropout_rate must be in [0, 1]");
            this.dropoutRate = v; return this;
        }
        public Builder dropout_rate(double v) { return dropoutRate(v); }
        public Builder entropyBonus(double v) {
            if (v < 0) throw new IllegalArgumentException("entropy_bonus must be >= 0");
            this.entropyBonus = v; return this;
        }
        public Builder entropy_bonus(double v) { return entropyBonus(v); }
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
        public Builder pathLengthNorm(double v) {
            if (v < 0) throw new IllegalArgumentException("path_length_norm must be >= 0");
            this.pathLengthNorm = v; return this;
        }
        public Builder path_length_norm(double v) { return pathLengthNorm(v); }
        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }
        public Builder disableDropout(boolean v) { this.disableDropout = v; return this; }
        public Builder disable_dropout(boolean v) { return disableDropout(v); }
        public Builder useReferenceModel(boolean v) { this.useReferenceModel = v; return this; }
        public Builder use_reference_model(boolean v) { return useReferenceModel(v); }
        public Builder referenceBeta(double v) {
            if (v < 0) throw new IllegalArgumentException("reference_beta must be >= 0");
            this.referenceBeta = v; return this;
        }
        public Builder reference_beta(double v) { return referenceBeta(v); }
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
        public Builder adapKlCtrl(boolean v) { this.adapKlCtrl = v; return this; }
        public Builder adap_kl_ctrl(boolean v) { return adapKlCtrl(v); }
        public Builder initKlCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("init_kl_coef must be >= 0");
            this.initKlCoef = v; return this;
        }
        public Builder init_kl_coef(double v) { return initKlCoef(v); }

        @Override
        public SPOConfig build() { return new SPOConfig(this); }
    }

    @Override
    public String toString() {
        return "SPOConfig{" +
                "temperature=" + temperature +
                ", selfPlayIterations=" + selfPlayIterations +
                ", updateRate=" + updateRate +
                ", useMixture=" + useMixture +
                '}';
    }
}