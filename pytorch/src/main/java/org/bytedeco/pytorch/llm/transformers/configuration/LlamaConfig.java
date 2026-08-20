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

/**
 * HuggingFace {@code LlamaConfig}.
 *
 * <p>Llama 1/2/3/4 share the same config schema with the same defaults:
 * {@code hidden_act="silu"}, {@code rope_theta=10000}, {@code rms_norm_eps=1e-6},
 * GQA enabled (num_key_value_heads ≤ num_attention_heads).
 *
 * <p>Reference: transformers/models/llama/configuration_llama.py
 */
public final class LlamaConfig extends Config {

    public static final String MODEL_TYPE = "llama";

    // ---- Canonical HF defaults ----
    public static final int    DEFAULT_VOCAB_SIZE            = 32000;
    public static final int    DEFAULT_HIDDEN_SIZE            = 4096;
    public static final int    DEFAULT_INTERMEDIATE_SIZE      = 11008;
    public static final int    DEFAULT_NUM_HIDDEN_LAYERS      = 32;
    public static final int    DEFAULT_NUM_ATTENTION_HEADS    = 32;
    public static final int    DEFAULT_ROPE_THETA             = 10000;
    public static final double DEFAULT_RMS_NORM_EPS           = 1e-6;
    public static final String DEFAULT_HIDDEN_ACT             = "silu";
    public static final double DEFAULT_INITIALIZER_RANGE      = 0.02;
    public static final int    DEFAULT_MAX_POS_EMBED          = 2048;
    public static final int    DEFAULT_BOS_TOKEN_ID          = 1;
    public static final int    DEFAULT_EOS_TOKEN_ID          = 2;

    private final int    pretrainingTp;
    private final boolean mlpBias;

    public LlamaConfig(PretrainedConfig base) {
        super(base);
        Object pt = base.extra().get("pretraining_tp");
        this.pretrainingTp = pt instanceof Number ? ((Number) pt).intValue() : 1;
        Object mb = base.extra().get("mlp_bias");
        this.mlpBias = mb instanceof Boolean ? (Boolean) mb : false;
    }

    public LlamaConfig(int vocabSize, int hiddenSize, int intermediateSize,
                       int numHiddenLayers, int numAttentionHeads, int numKeyValueHeads,
                       String hiddenAct, int maxPositionEmbeddings, double ropeTheta,
                       double rmsNormEps, double initializerRange,
                       int bosTokenId, int eosTokenId, int padTokenId,
                       boolean tieWordEmbeddings, boolean attentionBias,
                       double attentionDropout, int pretrainingTp, boolean mlpBias) {
        super(PretrainedConfig.builder()
                .modelType(PretrainedConfig.ModelType.LLAMA)
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
                .extra("pretraining_tp", pretrainingTp)
                .extra("mlp_bias", mlpBias)
                .build());
        this.pretrainingTp = pretrainingTp;
        this.mlpBias = mlpBias;
    }

    /** Hidden activation function. Default: {@code "silu"}. */
    public String hiddenAct() {
        String a = base().hiddenActivation();
        return a != null ? a : DEFAULT_HIDDEN_ACT;
    }

    /** Initializer range for weights. Default: {@code 0.02}. */
    public double initializerRange() {
        double r = base().initializerRange();
        return r > 0 ? r : DEFAULT_INITIALIZER_RANGE;
    }

    /** Max position embeddings. Default: {@code 2048}. */
    public int maxPositionEmbeddings() {
        int p = base().maxPositionEmbeddings();
        return p > 0 ? p : DEFAULT_MAX_POS_EMBED;
    }

    /** Pretraining TP value. Default: {@code 1}. */
    public int pretrainingTp() { return pretrainingTp; }

    /** MLP bias in attention. Default: {@code false}. */
    public boolean mlpBias() { return mlpBias; }

    /**
     * Standard Llama-3 8B config.
     * Reference: meta-llama/Meta-Llama-3-8B-Instruct
     */
    public static LlamaConfig standard() {
        return new LlamaConfig(
                128256, 4096, 14336, 32, 32, 8,
                "silu", 8192, 500000.0, 1e-5, 0.02,
                128000, 128001, 128004,
                false, false, 0.0, 1, false);
    }

    /** Standard Llama-2 7B config. */
    public static LlamaConfig llama2() {
        return new LlamaConfig(
                32000, 4096, 11008, 32, 32, 32,
                "silu", 4096, 10000.0, 1e-6, 0.02,
                1, 2, 0,
                false, false, 0.0, 1, false);
    }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return LlamaConfig.class; }
}
