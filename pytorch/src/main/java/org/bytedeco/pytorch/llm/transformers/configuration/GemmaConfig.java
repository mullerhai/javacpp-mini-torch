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
 * HuggingFace {@code GemmaConfig} (Gemma v1 — 2B / 7B).
 *
 * <p>Standards: {@code hidden_act="gelu_pytorch_tanh"}, RMSNorm, qk-norm,
 * {@code attention_bias=False}, {@code tie_word_embeddings=True}.
 *
 * <p>Reference: transformers/models/gemma/configuration_gemma.py
 */
public final class GemmaConfig extends Config {

    public static final String MODEL_TYPE = "gemma";

    public static final int    DEFAULT_VOCAB_SIZE           = 256000;
    public static final int    DEFAULT_HIDDEN_SIZE           = 2048;
    public static final int    DEFAULT_INTERMEDIATE_SIZE      = 16384;
    public static final int    DEFAULT_NUM_HIDDEN_LAYERS     = 18;
    public static final int    DEFAULT_NUM_ATTENTION_HEADS  = 8;
    public static final int    DEFAULT_NUM_KV_HEADS         = 1;
    public static final int    DEFAULT_HEAD_DIM             = 256;
    public static final int    DEFAULT_ROPE_THETA           = 10000;
    public static final double DEFAULT_RMS_NORM_EPS          = 1e-6;
    public static final String DEFAULT_HIDDEN_ACT            = "gelu_pytorch_tanh";
    public static final double DEFAULT_INITIALIZER_RANGE      = 0.02;
    public static final int    DEFAULT_MAX_POS_EMBED       = 8192;
    public static final int    DEFAULT_BOS_TOKEN_ID        = 2;
    public static final int    DEFAULT_EOS_TOKEN_ID        = 1;
    public static final int    DEFAULT_PAD_TOKEN_ID        = 0;

    private final String hiddenActivation;
    private final int    headDim;
    private final double initializerRange;
    private final int    maxPositionEmbeddings;
    private final double attnLogitSoftcap;
    private final double finalLogitSoftcap;
    private final double attentionDropout;

    public GemmaConfig(PretrainedConfig base) {
        super(base);
        this.hiddenActivation = base.hiddenActivation() != null ? base.hiddenActivation() : DEFAULT_HIDDEN_ACT;
        this.headDim = base.headDim() > 0 ? base.headDim() : DEFAULT_HEAD_DIM;
        this.initializerRange = base.initializerRange() > 0 ? base.initializerRange() : DEFAULT_INITIALIZER_RANGE;
        this.maxPositionEmbeddings = base.maxPositionEmbeddings() > 0 ? base.maxPositionEmbeddings() : DEFAULT_MAX_POS_EMBED;
        this.attnLogitSoftcap = base.attnLogitSoftcap();
        this.finalLogitSoftcap = base.finalLogitSoftcap();
        this.attentionDropout = base.attentionDropout();
    }

    public GemmaConfig(int vocabSize, int hiddenSize, int intermediateSize,
                     int numHiddenLayers, int numAttentionHeads, int numKeyValueHeads, int headDim,
                     String hiddenActivation, int maxPositionEmbeddings, double ropeTheta,
                     double rmsNormEps, double initializerRange,
                     int bosTokenId, int eosTokenId, int padTokenId,
                     boolean tieWordEmbeddings, boolean attentionBias,
                     double attentionDropout,
                     double attnLogitSoftcap, double finalLogitSoftcap) {
        super(PretrainedConfig.builder()
                .modelType(PretrainedConfig.ModelType.GEMMA)
                .vocabSize(vocabSize)
                .hiddenSize(hiddenSize)
                .intermediateSize(intermediateSize)
                .numHiddenLayers(numHiddenLayers)
                .numAttentionHeads(numAttentionHeads)
                .numKeyValueHeads(numKeyValueHeads)
                .headDim(headDim)
                .maxPositionEmbeddings(maxPositionEmbeddings)
                .ropeTheta(ropeTheta)
                .rmsNormEps(rmsNormEps)
                .bosTokenId(bosTokenId)
                .eosTokenId(eosTokenId)
                .padTokenId(padTokenId)
                .tieWordEmbeddings(tieWordEmbeddings)
                .attentionBias(attentionBias)
                .attentionDropout(attentionDropout)
                .hiddenActivation(hiddenActivation)
                .initializerRange(initializerRange)
                .attnLogitSoftcap(attnLogitSoftcap)
                .finalLogitSoftcap(finalLogitSoftcap)
                .build());
        this.hiddenActivation = hiddenActivation;
        this.headDim = headDim;
        this.initializerRange = initializerRange;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.attnLogitSoftcap = attnLogitSoftcap;
        this.finalLogitSoftcap = finalLogitSoftcap;
        this.attentionDropout = attentionDropout;
    }

    public String hiddenActivation() {
        String a = base().hiddenActivation();
        return a == null ? DEFAULT_HIDDEN_ACT : a;
    }
    public int headDim() { return headDim; }
    public double initializerRange() { return initializerRange; }
    public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
    public double attnLogitSoftcap() { return attnLogitSoftcap; }
    public double finalLogitSoftcap() { return finalLogitSoftcap; }
    public double attentionDropout() { return attentionDropout; }

    /** Standard Gemma-2B config. */
    public static GemmaConfig standard() {
        return new GemmaConfig(
                256000, 2048, 16384, 18, 8, 1, 256,
                "gelu_pytorch_tanh", 8192, 10000.0, 1e-6, 0.02,
                2, 1, 0, true, false, 0.0, 0.0, 0.0);
    }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return GemmaConfig.class; }
}
