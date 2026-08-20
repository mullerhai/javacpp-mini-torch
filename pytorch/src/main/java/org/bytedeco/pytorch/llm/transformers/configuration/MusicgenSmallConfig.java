/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Herve Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 * HuggingFace <code>musicgen_smallConfig</code>.
 */
public final class MusicgenSmallConfig extends Config {

    public static final String MODEL_TYPE = "musicgen_small";

    private final int maxPositionEmbeddings;
    private final int ffnDim;
    private final double layerdrop;
    private final String activationFunction;
    private final double dropout;
    private final double attentionDropout;
    private final double activationDropout;
    private final double initializerFactor;
    private final boolean scaleEmbedding;
    private final int numCodebooks;
    private final int audioChannels;
    private final boolean isDecoder;
    private final boolean addCrossAttention;
    private final int crossAttentionHiddenSize;

    public MusicgenSmallConfig(PretrainedConfig base) {
        super(base);
        this.maxPositionEmbeddings = toInt(base.extra().get("max_position_embeddings"), 2048);
        this.ffnDim = toInt(base.extra().get("ffn_dim"), 4096);
        this.layerdrop = toDouble(base.extra().get("layerdrop"), 0.0);
        this.activationFunction = String.valueOf(base.extra().get("activation_function"));
        this.dropout = toDouble(base.extra().get("dropout"), 0.1);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.activationDropout = toDouble(base.extra().get("activation_dropout"), 0.0);
        this.initializerFactor = toDouble(base.extra().get("initializer_factor"), 0.02);
        this.scaleEmbedding = base.extra().get("scale_embedding") == Boolean.TRUE;
        this.numCodebooks = toInt(base.extra().get("num_codebooks"), 4);
        this.audioChannels = toInt(base.extra().get("audio_channels"), 1);
        this.isDecoder = base.extra().get("is_decoder") == Boolean.TRUE;
        this.addCrossAttention = base.extra().get("add_cross_attention") == Boolean.TRUE;
        this.crossAttentionHiddenSize = toInt(base.extra().get("cross_attention_hidden_size"), 0);
    }

    public int maxPositionEmbeddings() { return toInt(base().extra().get("max_position_embeddings"), 2048); }
    public int ffnDim() { return toInt(base().extra().get("ffn_dim"), 4096); }
    public double layerdrop() { return toDouble(base().extra().get("layerdrop"), 0.0); }
    public String activationFunction() { Object v = base().extra().get("activation_function"); return v == null ? "gelu" : String.valueOf(v); }
    public double dropout() { return toDouble(base().extra().get("dropout"), 0.1); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double activationDropout() { return toDouble(base().extra().get("activation_dropout"), 0.0); }
    public double initializerFactor() { return toDouble(base().extra().get("initializer_factor"), 0.02); }
    public boolean scaleEmbedding() { return base().extra().get("scale_embedding") == Boolean.TRUE; }
    public int numCodebooks() { return toInt(base().extra().get("num_codebooks"), 4); }
    public int audioChannels() { return toInt(base().extra().get("audio_channels"), 1); }
    public boolean isDecoder() { return base().extra().get("is_decoder") == Boolean.TRUE; }
    public boolean addCrossAttention() { return base().extra().get("add_cross_attention") == Boolean.TRUE; }
    public int crossAttentionHiddenSize() { return toInt(base().extra().get("cross_attention_hidden_size"), 0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return MusicgenSmallConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}