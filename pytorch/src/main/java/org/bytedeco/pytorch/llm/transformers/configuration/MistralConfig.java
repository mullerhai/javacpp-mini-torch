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
 * HuggingFace {@code MistralConfig}.
 *
 * <p>Sliding-window attention enabled by default (window=4096) and
 * GQA with {@code num_key_value_heads=8}.
 *
 * <p>Reference: transformers/models/mistral/configuration_mistral.py
 */
public final class MistralConfig extends Config {

    public static final String MODEL_TYPE = "mistral";

    public static final int    DEFAULT_VOCAB_SIZE          = 32000;
    public static final int    DEFAULT_HIDDEN_SIZE          = 4096;
    public static final int    DEFAULT_INTERMEDIATE_SIZE     = 14336;
    public static final int    DEFAULT_NUM_HIDDEN_LAYERS    = 32;
    public static final int    DEFAULT_NUM_ATTENTION_HEADS  = 32;
    public static final int    DEFAULT_NUM_KV_HEADS        = 8;
    public static final int    DEFAULT_ROPE_THETA          = 10000;
    public static final double DEFAULT_RMS_NORM_EPS        = 1e-6;
    public static final String DEFAULT_HIDDEN_ACT          = "silu";
    public static final double DEFAULT_INITIALIZER_RANGE     = 0.02;
    public static final int    DEFAULT_MAX_POS_EMBED      = 131072;
    public static final int    DEFAULT_SLIDING_WINDOW      = 4096;

    private final String hiddenAct;
    private final double initializerRange;
    private final int    maxPositionEmbeddings;
    private final boolean useSlidingWindow;
    private final int    slidingWindow;

    public MistralConfig(PretrainedConfig base) {
        super(base);
        this.hiddenAct          = base.hiddenActivation() != null ? base.hiddenActivation() : DEFAULT_HIDDEN_ACT;
        this.initializerRange   = base.initializerRange() > 0 ? base.initializerRange() : DEFAULT_INITIALIZER_RANGE;
        this.maxPositionEmbeddings = base.maxPositionEmbeddings() > 0 ? base.maxPositionEmbeddings() : DEFAULT_MAX_POS_EMBED;
        this.useSlidingWindow  = base.useSlidingWindow();
        this.slidingWindow     = base.slidingWindow() > 0 ? base.slidingWindow() : DEFAULT_SLIDING_WINDOW;
    }

    public MistralConfig(int vocabSize, int hiddenSize, int intermediateSize,
                       int numHiddenLayers, int numAttentionHeads, int numKeyValueHeads,
                       String hiddenAct, int maxPositionEmbeddings, double ropeTheta,
                       double rmsNormEps, double initializerRange,
                       int bosTokenId, int eosTokenId, int padTokenId,
                       boolean tieWordEmbeddings, boolean attentionBias,
                       double attentionDropout, int slidingWindow) {
        super(PretrainedConfig.builder()
                .modelType(PretrainedConfig.ModelType.MISTRAL)
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
                .useSlidingWindow(slidingWindow > 0)
                .build());
        this.hiddenAct = hiddenAct;
        this.initializerRange = initializerRange;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.useSlidingWindow = slidingWindow > 0;
        this.slidingWindow = slidingWindow;
    }

    public String hiddenAct() { return hiddenAct; }
    public double initializerRange() { return initializerRange; }
    public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
    public boolean useSlidingWindow() { return useSlidingWindow; }
    public int slidingWindow() { return slidingWindow; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return MistralConfig.class; }
}
