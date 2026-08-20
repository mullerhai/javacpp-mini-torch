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
 * HuggingFace {@code Qwen3Config}.
 *
 * <p>Qwen3 differs from Qwen2 by:
 * <ul>
 *   <li>{@code attention_bias=False} (no bias in QKV projections).</li>
 *   <li>Explicit {@code head_dim} (e.g. 128 with hidden=2048, heads=16).</li>
 *   <li>Optionally {@code tie_word_embeddings=False} for non-embedding-sharing variants.</li>
 * </ul>
 *
 * <p>Reference: transformers/models/qwen3/configuration_qwen3.py
 */
public final class Qwen3Config extends Config {

    public static final String MODEL_TYPE = "qwen3";

    public static final int    DEFAULT_VOCAB_SIZE           = 151936;
    public static final int    DEFAULT_HIDDEN_SIZE           = 4096;
    public static final int    DEFAULT_INTERMEDIATE_SIZE      = 22016;
    public static final int    DEFAULT_NUM_HIDDEN_LAYERS     = 32;
    public static final int    DEFAULT_NUM_ATTENTION_HEADS   = 32;
    public static final int    DEFAULT_HEAD_DIM              = 128;
    public static final int    DEFAULT_ROPE_THETA           = 1000000;
    public static final double DEFAULT_RMS_NORM_EPS         = 1e-6;
    public static final String DEFAULT_HIDDEN_ACT           = "silu";
    public static final double DEFAULT_INITIALIZER_RANGE     = 0.02;
    public static final int    DEFAULT_MAX_POS_EMBED        = 32768;
    public static final int    DEFAULT_SLIDING_WINDOW       = 4096;

    private final int    headDim;
    private final String hiddenAct;
    private final double initializerRange;
    private final int    maxPositionEmbeddings;
    private final boolean useSlidingWindow;
    private final int    slidingWindow;
    private final int    maxWindowLayers;

    public Qwen3Config(PretrainedConfig base) {
        super(base);
        this.headDim = base.headDim() > 0 ? base.headDim() : DEFAULT_HEAD_DIM;
        this.hiddenAct = base.hiddenActivation() != null ? base.hiddenActivation() : DEFAULT_HIDDEN_ACT;
        this.initializerRange = base.initializerRange() > 0 ? base.initializerRange() : DEFAULT_INITIALIZER_RANGE;
        this.maxPositionEmbeddings = base.maxPositionEmbeddings() > 0 ? base.maxPositionEmbeddings() : DEFAULT_MAX_POS_EMBED;
        Object usw = base.extra().get("use_sliding_window");
        this.useSlidingWindow = usw instanceof Boolean ? (Boolean) usw : false;
        Object sw = base.extra().get("sliding_window");
        this.slidingWindow = sw instanceof Number ? ((Number) sw).intValue() : DEFAULT_SLIDING_WINDOW;
        Object mwl = base.extra().get("max_window_layers");
        this.maxWindowLayers = mwl instanceof Number ? ((Number) mwl).intValue() : base.numHiddenLayers();
    }

    public Qwen3Config(int vocabSize, int hiddenSize, int intermediateSize,
                      int numHiddenLayers, int numAttentionHeads, int numKeyValueHeads, int headDim,
                      String hiddenAct, int maxPositionEmbeddings, double ropeTheta,
                      double rmsNormEps, double initializerRange,
                      int bosTokenId, int eosTokenId, int padTokenId,
                      boolean tieWordEmbeddings, boolean attentionBias,
                      double attentionDropout, boolean useSlidingWindow,
                      int slidingWindow, int maxWindowLayers) {
        super(PretrainedConfig.builder()
                .modelType(PretrainedConfig.ModelType.QWEN3)
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
                .hiddenActivation(hiddenAct)
                .initializerRange(initializerRange)
                .extra("use_sliding_window", useSlidingWindow)
                .extra("sliding_window", slidingWindow)
                .extra("max_window_layers", maxWindowLayers)
                .extra("model_type", "qwen3")
                .build());
        this.headDim = headDim;
        this.hiddenAct = hiddenAct;
        this.initializerRange = initializerRange;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.useSlidingWindow = useSlidingWindow;
        this.slidingWindow = slidingWindow;
        this.maxWindowLayers = maxWindowLayers;
    }

    /** Explicit head dimension (default: 128). */
    public int headDim() { return headDim; }
    public String hiddenAct() { return hiddenAct; }
    public double initializerRange() { return initializerRange; }
    public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
    public boolean useSlidingWindow() { return useSlidingWindow; }
    public int slidingWindow() { return slidingWindow; }
    public int maxWindowLayers() { return maxWindowLayers; }

    /** Standard Qwen3 8B config. Reference: Qwen/Qwen3-8B */
    public static Qwen3Config standard() {
        return new Qwen3Config(
                151936, 4096, 22016, 32, 32, 8, 128,
                "silu", 32768, 1000000.0, 1e-6, 0.02,
                151643, 151645, 151643,
                false, false, 0.0,
                false, 4096, 28);
    }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Qwen3Config.class; }
}
