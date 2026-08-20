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
 * HuggingFace <code>bartConfig</code>.
 * Reference: transformers/models/bart/configuration_bart.py
 */
public final class BartConfig extends Config {

    public static final String MODEL_TYPE = "bart";

    private final int dModel;
    private final int encoderLayers;
    private final int decoderLayers;
    private final int encoderFfnDim;
    private final int decoderFfnDim;
    private final int encoderAttentionHeads;
    private final int decoderAttentionHeads;
    private final double encoderLayerdrop;
    private final double decoderLayerdrop;
    private final String activationFunction;
    private final double dropout;
    private final double attentionDropout;
    private final double activationDropout;
    private final double initStd;
    private final double classifierDropout;
    private final boolean scaleEmbedding;
    private final boolean isEncoderDecoder;
    private final int decoderStartTokenId;
    private final int forcedEosTokenId;

    public BartConfig(PretrainedConfig base) {
        super(base);
        this.dModel = toInt(base.extra().get("d_model"), 1024);
        this.encoderLayers = toInt(base.extra().get("encoder_layers"), 12);
        this.decoderLayers = toInt(base.extra().get("decoder_layers"), 12);
        this.encoderFfnDim = toInt(base.extra().get("encoder_ffn_dim"), 4096);
        this.decoderFfnDim = toInt(base.extra().get("decoder_ffn_dim"), 4096);
        this.encoderAttentionHeads = toInt(base.extra().get("encoder_attention_heads"), 16);
        this.decoderAttentionHeads = toInt(base.extra().get("decoder_attention_heads"), 16);
        this.encoderLayerdrop = toDouble(base.extra().get("encoder_layerdrop"), 0.0);
        this.decoderLayerdrop = toDouble(base.extra().get("decoder_layerdrop"), 0.0);
        this.activationFunction = String.valueOf(base.extra().get("activation_function"));
        this.dropout = toDouble(base.extra().get("dropout"), 0.1);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.activationDropout = toDouble(base.extra().get("activation_dropout"), 0.0);
        this.initStd = toDouble(base.extra().get("init_std"), 0.02);
        this.classifierDropout = toDouble(base.extra().get("classifier_dropout"), 0.0);
        this.scaleEmbedding = base.extra().get("scale_embedding") == Boolean.TRUE;
        this.isEncoderDecoder = base.extra().get("is_encoder_decoder") == Boolean.TRUE;
        this.decoderStartTokenId = toInt(base.extra().get("decoder_start_token_id"), 2);
        this.forcedEosTokenId = toInt(base.extra().get("forced_eos_token_id"), 2);
    }

    public int dModel() { return toInt(base().extra().get("d_model"), 1024); }
    public int encoderLayers() { return toInt(base().extra().get("encoder_layers"), 12); }
    public int decoderLayers() { return toInt(base().extra().get("decoder_layers"), 12); }
    public int encoderFfnDim() { return toInt(base().extra().get("encoder_ffn_dim"), 4096); }
    public int decoderFfnDim() { return toInt(base().extra().get("decoder_ffn_dim"), 4096); }
    public int encoderAttentionHeads() { return toInt(base().extra().get("encoder_attention_heads"), 16); }
    public int decoderAttentionHeads() { return toInt(base().extra().get("decoder_attention_heads"), 16); }
    public double encoderLayerdrop() { return toDouble(base().extra().get("encoder_layerdrop"), 0.0); }
    public double decoderLayerdrop() { return toDouble(base().extra().get("decoder_layerdrop"), 0.0); }
    public String activationFunction() { Object v = base().extra().get("activation_function"); return v == null ? "gelu" : String.valueOf(v); }
    public double dropout() { return toDouble(base().extra().get("dropout"), 0.1); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double activationDropout() { return toDouble(base().extra().get("activation_dropout"), 0.0); }
    public double initStd() { return toDouble(base().extra().get("init_std"), 0.02); }
    public double classifierDropout() { return toDouble(base().extra().get("classifier_dropout"), 0.0); }
    public boolean scaleEmbedding() { return base().extra().get("scale_embedding") == Boolean.TRUE; }
    public boolean isEncoderDecoder() { return base().extra().get("is_encoder_decoder") == Boolean.TRUE; }
    public int decoderStartTokenId() { return toInt(base().extra().get("decoder_start_token_id"), 2); }
    public int forcedEosTokenId() { return toInt(base().extra().get("forced_eos_token_id"), 2); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return BartConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}