/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option) any later version (collectively, the "License");
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
 * HuggingFace <code>t5Config</code>.
 * Reference: transformers/models/t5/configuration_t5.py
 */
public final class T5Config extends Config {

    public static final String MODEL_TYPE = "t5";

    private final int dModel;
    private final int dKv;
    private final int dFf;
    private final int numLayers;
    private final int numDecoderLayers;
    private final int numHeads;
    private final int relativeAttentionNumBuckets;
    private final int relativeAttentionMaxDistance;
    private final double dropoutRate;
    private final double layerNormEpsilon;
    private final double initializerFactor;
    private final String feedForwardProj;
    private final boolean tieWordEmbeddings;
    private final boolean isEncoderDecoder;
    private final boolean useCache;
    private final double classifierDropout;

    public T5Config(PretrainedConfig base) {
        super(base);
        this.dModel = toInt(base.extra().get("d_model"), 512);
        this.dKv = toInt(base.extra().get("d_kv"), 64);
        this.dFf = toInt(base.extra().get("d_ff"), 2048);
        this.numLayers = toInt(base.extra().get("num_layers"), 6);
        this.numDecoderLayers = toInt(base.extra().get("num_decoder_layers"), 0);
        this.numHeads = toInt(base.extra().get("num_heads"), 8);
        this.relativeAttentionNumBuckets = toInt(base.extra().get("relative_attention_num_buckets"), 32);
        this.relativeAttentionMaxDistance = toInt(base.extra().get("relative_attention_max_distance"), 128);
        this.dropoutRate = toDouble(base.extra().get("dropout_rate"), 0.1);
        this.layerNormEpsilon = toDouble(base.extra().get("layer_norm_epsilon"), 1e-06);
        this.initializerFactor = toDouble(base.extra().get("initializer_factor"), 1.0);
        this.feedForwardProj = String.valueOf(base.extra().get("feed_forward_proj"));
        this.tieWordEmbeddings = base.extra().get("tie_word_embeddings") == Boolean.TRUE;
        this.isEncoderDecoder = base.extra().get("is_encoder_decoder") == Boolean.TRUE;
        this.useCache = base.extra().get("use_cache") == Boolean.TRUE;
        this.classifierDropout = toDouble(base.extra().get("classifier_dropout"), 0.0);
    }

    public int dModel() { return toInt(base().extra().get("d_model"), 512); }
    public int dKv() { return toInt(base().extra().get("d_kv"), 64); }
    public int dFf() { return toInt(base().extra().get("d_ff"), 2048); }
    public int numLayers() { return toInt(base().extra().get("num_layers"), 6); }
    public int numDecoderLayers() { return toInt(base().extra().get("num_decoder_layers"), 0); }
    public int numHeads() { return toInt(base().extra().get("num_heads"), 8); }
    public int relativeAttentionNumBuckets() { return toInt(base().extra().get("relative_attention_num_buckets"), 32); }
    public int relativeAttentionMaxDistance() { return toInt(base().extra().get("relative_attention_max_distance"), 128); }
    public double dropoutRate() { return toDouble(base().extra().get("dropout_rate"), 0.1); }
    public double layerNormEpsilon() { return toDouble(base().extra().get("layer_norm_epsilon"), 1e-06); }
    public double initializerFactor() { return toDouble(base().extra().get("initializer_factor"), 1.0); }
    public String feedForwardProj() { Object v = base().extra().get("feed_forward_proj"); return v == null ? "gated-gelu" : String.valueOf(v); }
    public boolean tieWordEmbeddings() { return base().extra().get("tie_word_embeddings") == Boolean.TRUE; }
    public boolean isEncoderDecoder() { return base().extra().get("is_encoder_decoder") == Boolean.TRUE; }
    public boolean useCache() { return base().extra().get("use_cache") == Boolean.TRUE; }
    public double classifierDropout() { return toDouble(base().extra().get("classifier_dropout"), 0.0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return T5Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}