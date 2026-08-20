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
package org.bytedeco.pytorch.llm.transformers.configuration;

import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.util.Map;

/**
 * HuggingFace {@code MixtralConfig} — sparse Mixture-of-Experts variant of Mistral.
 *
 * <p>Adds {@code num_local_experts} (default 8) and {@code num_experts_per_tok} (default 2).
 *
 * <p>Reference: transformers/models/mixtral/configuration_mixtral.py
 */
public final class MixtralConfig extends Config {

    public static final String MODEL_TYPE = "mixtral";

    public static final int    DEFAULT_VOCAB_SIZE          = 32000;
    public static final int    DEFAULT_HIDDEN_SIZE          = 4096;
    public static final int    DEFAULT_INTERMEDIATE_SIZE     = 14336;
    public static final int    DEFAULT_NUM_HIDDEN_LAYERS    = 32;
    public static final int    DEFAULT_NUM_ATTENTION_HEADS  = 32;
    public static final int    DEFAULT_NUM_KV_HEADS        = 8;
    public static final int    DEFAULT_ROPE_THETA          = 10000;
    public static final double DEFAULT_RMS_NORM_EPS        = 1e-5;
    public static final String DEFAULT_HIDDEN_ACT           = "silu";
    public static final double DEFAULT_INITIALIZER_RANGE     = 0.02;
    public static final int    DEFAULT_MAX_POS_EMBED       = 131072;
    public static final int    DEFAULT_SLIDING_WINDOW      = 0;  // Mixtral uses sliding_window=null by default
    public static final int    DEFAULT_NUM_LOCAL_EXPERTS    = 8;
    public static final int    DEFAULT_NUM_EXPERTS_PER_TOK  = 2;
    public static final double DEFAULT_ROUTER_AUX_LOSS_COEF = 0.001;
    public static final double DEFAULT_ROUTER_JITTER_NOISE  = 0.0;

    private final String hiddenAct;
    private final double initializerRange;
    private final int    maxPositionEmbeddings;
    private final int    slidingWindow;
    private final int    numLocalExperts;
    private final int    numExpertsPerTok;
    private final boolean outputRouterLogits;
    private final double routerAuxLossCoef;
    private final double routerJitterNoise;

    public MixtralConfig(PretrainedConfig base) {
        super(base);
        this.hiddenAct          = base.hiddenActivation() != null ? base.hiddenActivation() : DEFAULT_HIDDEN_ACT;
        this.initializerRange   = base.initializerRange() > 0 ? base.initializerRange() : DEFAULT_INITIALIZER_RANGE;
        this.maxPositionEmbeddings = base.maxPositionEmbeddings() > 0 ? base.maxPositionEmbeddings() : DEFAULT_MAX_POS_EMBED;
        this.slidingWindow     = base.slidingWindow();
        this.numLocalExperts  = base.numLocalExperts() > 0 ? base.numLocalExperts() : DEFAULT_NUM_LOCAL_EXPERTS;
        this.numExpertsPerTok = base.numExpertsPerTok() > 0 ? base.numExpertsPerTok() : DEFAULT_NUM_EXPERTS_PER_TOK;
        Map<String,Object> ex = base.extra();
        this.outputRouterLogits = ex.get("output_router_logits") == Boolean.TRUE;
        this.routerAuxLossCoef = toDouble(ex.get("router_aux_loss_coef"), DEFAULT_ROUTER_AUX_LOSS_COEF);
        this.routerJitterNoise = toDouble(ex.get("router_jitter_noise"), DEFAULT_ROUTER_JITTER_NOISE);
    }

    public MixtralConfig(int vocabSize, int hiddenSize, int intermediateSize,
                       int numHiddenLayers, int numAttentionHeads, int numKeyValueHeads,
                       String hiddenAct, int maxPositionEmbeddings, double ropeTheta,
                       double rmsNormEps, double initializerRange,
                       int bosTokenId, int eosTokenId, int padTokenId,
                       boolean tieWordEmbeddings, boolean attentionBias,
                       double attentionDropout, int slidingWindow,
                       int numLocalExperts, int numExpertsPerTok,
                       boolean outputRouterLogits, double routerAuxLossCoef,
                       double routerJitterNoise) {
        super(PretrainedConfig.builder()
                .modelType(PretrainedConfig.ModelType.MIXTRAL)
                .vocabSize(vocabSize)
                .hiddenSize(hiddenSize)
                .intermediateSize(intermediateSize)
                .numHiddenLayers(numHiddenLayers)
                .numAttentionHeads(numAttentionHeads)
                .numKeyValueHeads(numKeyValueHeads)
                .maxPositionEmbeddings(maxPositionEmbeddings)
                .ropeTheta(ropeTheta)
                .rmsNormEps(rmsNormEps)
                .bosTokenId(bosTokenId)
                .eosTokenId(eosTokenId)
                .padTokenId(padTokenId)
                .tieWordEmbeddings(tieWordEmbeddings)
                .attentionBias(attentionBias)
                .attentionDropout(attentionDropout)
                .hiddenActivation(hiddenAct)
                .initializerRange(initializerRange)
                .slidingWindow(slidingWindow)
                .numLocalExperts(numLocalExperts)
                .numExpertsPerTok(numExpertsPerTok)
                .extra("output_router_logits", outputRouterLogits)
                .extra("router_aux_loss_coef", routerAuxLossCoef)
                .extra("router_jitter_noise", routerJitterNoise)
                .build());
        this.hiddenAct = hiddenAct;
        this.initializerRange = initializerRange;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.slidingWindow = slidingWindow;
        this.numLocalExperts = numLocalExperts;
        this.numExpertsPerTok = numExpertsPerTok;
        this.outputRouterLogits = outputRouterLogits;
        this.routerAuxLossCoef = routerAuxLossCoef;
        this.routerJitterNoise = routerJitterNoise;
    }

    public String hiddenAct() { return hiddenAct; }
    public double initializerRange() { return initializerRange; }
    public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
    public int slidingWindow() { return slidingWindow; }
    public int numLocalExperts() { return numLocalExperts; }
    public int numExpertsPerTok() { return numExpertsPerTok; }
    public boolean outputRouterLogits() { return outputRouterLogits; }
    public double routerAuxLossCoef() { return routerAuxLossCoef; }
    public double routerJitterNoise() { return routerJitterNoise; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return MixtralConfig.class; }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
