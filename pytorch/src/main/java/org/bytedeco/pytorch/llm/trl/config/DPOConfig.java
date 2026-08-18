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
 * DPO trainer config (Hugging Face TRL {@code DPOConfig}).
 *
 * <p>Mirrors every parameter exposed by the Python TRL implementation:
 * <ul>
 *   <li>Loss variants: {@code sigmoid}, {@code robust}, {@code hinge}, {@code ipo},
 *       {@code exo_pair}, {@code nca_pair}, {@code sppo_huber}, {@code sppo_eps},
 *       {@code orpo}, {@code apos}, {@code sft}.</li>
 *   <li>Reference handling: {@code reference_free}, {@code ref_model_mixup_alpha}, {@code mix_rho}.</li>
 *   <li>Reward weighting: {@code gamma}, {@code rpo_alpha}, {@code use_weighting}, {@code label_smoothing}.</li>
 *   <li>Length normalization: {@code length_normalize}, {@code static_kl}, {@code sft_weight}.</li>
 *   <li>Generation-time logging: {@code disable_dropout}.</li>
 * </ul>
 */
public final class DPOConfig extends TrainerConfig {
    // Core
    private final double beta;
    private final String lossType;
    // Loss extensions
    private final double gamma;
    private final double labelSmoothing;
    private final boolean referenceFree;
    private final double sftWeight;
    private final double rpoAlpha;
    private final boolean useWeighting;
    private final double truncation_mode; // HF accepts float probability (e.g. 0.0/0.5)
    private final boolean disableDropout;
    private final boolean precomputeRefLogits;
    private final String forceUseRefModel;
    // Reference mix-up
    private final double refModelMixupAlpha;
    private final double refModelMixupBeta;
    private final double mixRho;
    // Auxiliary losses
    private final double auxLossCoef;
    private final double tools;
    // Length
    private final boolean lengthNormalize;
    private final boolean normalizeLogProbs;

    private DPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.lossType = b.lossType;
        this.gamma = b.gamma;
        this.labelSmoothing = b.labelSmoothing;
        this.referenceFree = b.referenceFree;
        this.sftWeight = b.sftWeight;
        this.rpoAlpha = b.rpoAlpha;
        this.useWeighting = b.useWeighting;
        this.truncation_mode = b.truncation_mode;
        this.disableDropout = b.disableDropout;
        this.precomputeRefLogits = b.precomputeRefLogits;
        this.forceUseRefModel = b.forceUseRefModel;
        this.refModelMixupAlpha = b.refModelMixupAlpha;
        this.refModelMixupBeta = b.refModelMixupBeta;
        this.mixRho = b.mixRho;
        this.auxLossCoef = b.auxLossCoef;
        this.tools = b.tools;
        this.lengthNormalize = b.lengthNormalize;
        this.normalizeLogProbs = b.normalizeLogProbs;
    }

    // ----- Core accessors -----
    public double beta() { return beta; }
    public String lossType() { return lossType; }
    public double gamma() { return gamma; }
    public double labelSmoothing() { return labelSmoothing; }
    public boolean referenceFree() { return referenceFree; }
    public double sftWeight() { return sftWeight; }
    public double rpoAlpha() { return rpoAlpha; }
    public boolean useWeighting() { return useWeighting; }
    public double truncationMode() { return truncation_mode; }
    public boolean disableDropout() { return disableDropout; }
    public boolean precomputeRefLogits() { return precomputeRefLogits; }
    public String forceUseRefModel() { return forceUseRefModel; }
    public double refModelMixupAlpha() { return refModelMixupAlpha; }
    public double refModelMixupBeta() { return refModelMixupBeta; }
    public double mixRho() { return mixRho; }
    public double auxLossCoef() { return auxLossCoef; }
    public boolean lengthNormalize() { return lengthNormalize; }
    public boolean normalizeLogProbs() { return normalizeLogProbs; }
    public double tools() { return tools; }

    /** Whether the requested loss requires a reference model. */
    public boolean requiresReferenceModel() {
        if (referenceFree) return false;
        switch (lossType == null ? "" : lossType.toLowerCase()) {
            case "orpo":
            case "sft":
                return false;
            default:
                return true;
        }
    }

    /** Map representation. */
    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        m.put("beta", beta);
        m.put("loss_type", lossType);
        m.put("gamma", gamma);
        m.put("label_smoothing", labelSmoothing);
        m.put("reference_free", referenceFree);
        m.put("sft_weight", sftWeight);
        m.put("rpo_alpha", rpoAlpha);
        m.put("use_weighting", useWeighting);
        m.put("truncation_mode", truncation_mode);
        m.put("disable_dropout", disableDropout);
        m.put("precompute_ref_logits", precomputeRefLogits);
        m.put("force_use_ref_model", forceUseRefModel);
        m.put("ref_model_mixup_alpha", refModelMixupAlpha);
        m.put("ref_model_mixup_beta", refModelMixupBeta);
        m.put("mix_rho", mixRho);
        m.put("aux_loss_coef", auxLossCoef);
        m.put("length_normalize", lengthNormalize);
        m.put("normalize_log_probs", normalizeLogProbs);
        return m;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private String lossType = "sigmoid";
        private double gamma = 0.0;
        private double labelSmoothing = 0.0;
        private boolean referenceFree = false;
        private double sftWeight = 0.0;
        private double rpoAlpha = 0.1;
        private boolean useWeighting = false;
        private double truncation_mode = 0.0;
        private boolean disableDropout = false;
        private boolean precomputeRefLogits = false;
        private String forceUseRefModel = "none"; // "always" | "never" | "none"
        private double refModelMixupAlpha = 0.0;
        private double refModelMixupBeta = 0.0;
        private double mixRho = 0.5;
        private double auxLossCoef = 0.0;
        private double tools = 0.0;
        private boolean lengthNormalize = false;
        private boolean normalizeLogProbs = true;

        // ----- core -----
        public Builder beta(double v) {
            if (v < 0) throw new IllegalArgumentException("beta must be >= 0");
            this.beta = v; return this;
        }

        public Builder lossType(String v) {
            if (v == null) { this.lossType = "sigmoid"; return this; }
            String norm = v.toLowerCase();
            switch (norm) {
                case "sigmoid":
                case "robust":
                case "hinge":
                case "ipo":
                case "exo_pair":
                case "nca_pair":
                case "sppo_huber":
                case "sppo_eps":
                case "orpo":
                case "apos":
                case "sft":
                    this.lossType = norm; return this;
                default:
                    throw new IllegalArgumentException("Unsupported lossType: " + v);
            }
        }
        public Builder loss_type(String v) { return lossType(v); }

        public Builder gamma(double v) {
            if (v < 0) throw new IllegalArgumentException("gamma must be >= 0");
            this.gamma = v; return this;
        }
        public Builder labelSmoothing(double v) {
            if (v < 0 || v >= 0.5) throw new IllegalArgumentException("label_smoothing must be in [0, 0.5)");
            this.labelSmoothing = v; return this;
        }
        public Builder label_smoothing(double v) { return labelSmoothing(v); }

        public Builder referenceFree(boolean v) { this.referenceFree = v; return this; }
        public Builder reference_free(boolean v) { return referenceFree(v); }

        public Builder sftWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("sft_weight must be >= 0");
            this.sftWeight = v; return this;
        }
        public Builder sft_weight(double v) { return sftWeight(v); }

        public Builder rpoAlpha(double v) {
            if (v < 0) throw new IllegalArgumentException("rpo_alpha must be >= 0");
            this.rpoAlpha = v; return this;
        }
        public Builder rpo_alpha(double v) { return rpoAlpha(v); }

        public Builder useWeighting(boolean v) { this.useWeighting = v; return this; }
        public Builder use_weighting(boolean v) { return useWeighting(v); }

        public Builder truncationMode(double v) {
            if (v < 0 || v > 1) throw new IllegalArgumentException("truncation_mode must be in [0, 1]");
            this.truncation_mode = v; return this;
        }
        public Builder truncation_mode(double v) { return truncationMode(v); }

        public Builder disableDropout(boolean v) { this.disableDropout = v; return this; }
        public Builder disable_dropout(boolean v) { return disableDropout(v); }

        public Builder precomputeRefLogits(boolean v) { this.precomputeRefLogits = v; return this; }
        public Builder precompute_ref_logits(boolean v) { return precomputeRefLogits(v); }

        public Builder forceUseRefModel(String v) {
            if (v == null) { this.forceUseRefModel = "none"; return this; }
            String norm = v.toLowerCase();
            if (!norm.equals("always") && !norm.equals("never") && !norm.equals("none")) {
                throw new IllegalArgumentException("force_use_ref_model must be one of always/never/none");
            }
            this.forceUseRefModel = norm; return this;
        }
        public Builder force_use_ref_model(String v) { return forceUseRefModel(v); }

        public Builder refModelMixupAlpha(double v) {
            if (v < 0) throw new IllegalArgumentException("ref_model_mixup_alpha must be >= 0");
            this.refModelMixupAlpha = v; return this;
        }
        public Builder ref_model_mixup_alpha(double v) { return refModelMixupAlpha(v); }

        public Builder refModelMixupBeta(double v) {
            if (v < 0) throw new IllegalArgumentException("ref_model_mixup_beta must be >= 0");
            this.refModelMixupBeta = v; return this;
        }
        public Builder ref_model_mixup_beta(double v) { return refModelMixupBeta(v); }

        public Builder mixRho(double v) {
            if (v < 0) throw new IllegalArgumentException("mix_rho must be >= 0");
            this.mixRho = v; return this;
        }
        public Builder mix_rho(double v) { return mixRho(v); }

        public Builder auxLossCoef(double v) {
            if (v < 0) throw new IllegalArgumentException("aux_loss_coef must be >= 0");
            this.auxLossCoef = v; return this;
        }
        public Builder aux_loss_coef(double v) { return auxLossCoef(v); }

        public Builder lengthNormalize(boolean v) { this.lengthNormalize = v; return this; }
        public Builder length_normalize(boolean v) { return lengthNormalize(v); }

        public Builder normalizeLogProbs(boolean v) { this.normalizeLogProbs = v; return this; }
        public Builder normalize_log_probs(boolean v) { return normalizeLogProbs(v); }

        @Override
        public DPOConfig build() { return new DPOConfig(this); }
    }
}