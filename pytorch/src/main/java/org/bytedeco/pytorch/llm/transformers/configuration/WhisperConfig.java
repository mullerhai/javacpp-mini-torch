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
 * HuggingFace <code>whisperConfig</code>.
 * Reference: transformers/models/whisper/configuration_whisper.py
 */
public final class WhisperConfig extends Config {

    public static final String MODEL_TYPE = "whisper";

    private final int numMelBins;
    private final int encoderLayers;
    private final int decoderLayers;
    private final int encoderAttentionHeads;
    private final int decoderAttentionHeads;
    private final int encoderFfnDim;
    private final int decoderFfnDim;
    private final double encoderLayerdrop;
    private final double decoderLayerdrop;
    private final int decoderStartTokenId;
    private final int dModel;
    private final double dropout;
    private final double attentionDropout;
    private final double activationDropout;
    private final boolean scaleEmbedding;
    private final int maxSourcePositions;
    private final int maxTargetPositions;
    private final boolean isEncoderDecoder;
    private final String suppressTokens;
    private final String beginSuppressTokens;
    private final boolean useWeightedLayerSum;
    private final int classifierProjSize;
    private final boolean applySpecAugment;
    private final double maskTimeProb;
    private final int maskTimeLength;
    private final int maskTimeMinMasks;
    private final double maskFeatureProb;
    private final int maskFeatureLength;
    private final int maskFeatureMinMasks;
    private final int medianFilterWidth;

    public WhisperConfig(PretrainedConfig base) {
        super(base);
        this.numMelBins = toInt(base.extra().get("num_mel_bins"), 80);
        this.encoderLayers = toInt(base.extra().get("encoder_layers"), 4);
        this.decoderLayers = toInt(base.extra().get("decoder_layers"), 4);
        this.encoderAttentionHeads = toInt(base.extra().get("encoder_attention_heads"), 6);
        this.decoderAttentionHeads = toInt(base.extra().get("decoder_attention_heads"), 6);
        this.encoderFfnDim = toInt(base.extra().get("encoder_ffn_dim"), 1536);
        this.decoderFfnDim = toInt(base.extra().get("decoder_ffn_dim"), 1536);
        this.encoderLayerdrop = toDouble(base.extra().get("encoder_layerdrop"), 0.0);
        this.decoderLayerdrop = toDouble(base.extra().get("decoder_layerdrop"), 0.0);
        this.decoderStartTokenId = toInt(base.extra().get("decoder_start_token_id"), 50257);
        this.dModel = toInt(base.extra().get("d_model"), 384);
        this.dropout = toDouble(base.extra().get("dropout"), 0.0);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.activationDropout = toDouble(base.extra().get("activation_dropout"), 0.0);
        this.scaleEmbedding = base.extra().get("scale_embedding") == Boolean.TRUE;
        this.maxSourcePositions = toInt(base.extra().get("max_source_positions"), 1500);
        this.maxTargetPositions = toInt(base.extra().get("max_target_positions"), 448);
        this.isEncoderDecoder = base.extra().get("is_encoder_decoder") == Boolean.TRUE;
        this.suppressTokens = String.valueOf(base.extra().get("suppress_tokens"));
        this.beginSuppressTokens = String.valueOf(base.extra().get("begin_suppress_tokens"));
        this.useWeightedLayerSum = base.extra().get("use_weighted_layer_sum") == Boolean.TRUE;
        this.classifierProjSize = toInt(base.extra().get("classifier_proj_size"), 256);
        this.applySpecAugment = base.extra().get("apply_spec_augment") == Boolean.TRUE;
        this.maskTimeProb = toDouble(base.extra().get("mask_time_prob"), 0.05);
        this.maskTimeLength = toInt(base.extra().get("mask_time_length"), 10);
        this.maskTimeMinMasks = toInt(base.extra().get("mask_time_min_masks"), 2);
        this.maskFeatureProb = toDouble(base.extra().get("mask_feature_prob"), 0.0);
        this.maskFeatureLength = toInt(base.extra().get("mask_feature_length"), 10);
        this.maskFeatureMinMasks = toInt(base.extra().get("mask_feature_min_masks"), 0);
        this.medianFilterWidth = toInt(base.extra().get("median_filter_width"), 7);
    }

    public int numMelBins() { return toInt(base().extra().get("num_mel_bins"), 80); }
    public int encoderLayers() { return toInt(base().extra().get("encoder_layers"), 4); }
    public int decoderLayers() { return toInt(base().extra().get("decoder_layers"), 4); }
    public int encoderAttentionHeads() { return toInt(base().extra().get("encoder_attention_heads"), 6); }
    public int decoderAttentionHeads() { return toInt(base().extra().get("decoder_attention_heads"), 6); }
    public int encoderFfnDim() { return toInt(base().extra().get("encoder_ffn_dim"), 1536); }
    public int decoderFfnDim() { return toInt(base().extra().get("decoder_ffn_dim"), 1536); }
    public double encoderLayerdrop() { return toDouble(base().extra().get("encoder_layerdrop"), 0.0); }
    public double decoderLayerdrop() { return toDouble(base().extra().get("decoder_layerdrop"), 0.0); }
    public int decoderStartTokenId() { return toInt(base().extra().get("decoder_start_token_id"), 50257); }
    public int dModel() { return toInt(base().extra().get("d_model"), 384); }
    public double dropout() { return toDouble(base().extra().get("dropout"), 0.0); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double activationDropout() { return toDouble(base().extra().get("activation_dropout"), 0.0); }
    public boolean scaleEmbedding() { return base().extra().get("scale_embedding") == Boolean.TRUE; }
    public int maxSourcePositions() { return toInt(base().extra().get("max_source_positions"), 1500); }
    public int maxTargetPositions() { return toInt(base().extra().get("max_target_positions"), 448); }
    public boolean isEncoderDecoder() { return base().extra().get("is_encoder_decoder") == Boolean.TRUE; }
    public String suppressTokens() { Object v = base().extra().get("suppress_tokens"); return v == null ? null : String.valueOf(v); }
    public String beginSuppressTokens() { Object v = base().extra().get("begin_suppress_tokens"); return v == null ? "(220, 50256)" : String.valueOf(v); }
    public boolean useWeightedLayerSum() { return base().extra().get("use_weighted_layer_sum") == Boolean.TRUE; }
    public int classifierProjSize() { return toInt(base().extra().get("classifier_proj_size"), 256); }
    public boolean applySpecAugment() { return base().extra().get("apply_spec_augment") == Boolean.TRUE; }
    public double maskTimeProb() { return toDouble(base().extra().get("mask_time_prob"), 0.05); }
    public int maskTimeLength() { return toInt(base().extra().get("mask_time_length"), 10); }
    public int maskTimeMinMasks() { return toInt(base().extra().get("mask_time_min_masks"), 2); }
    public double maskFeatureProb() { return toDouble(base().extra().get("mask_feature_prob"), 0.0); }
    public int maskFeatureLength() { return toInt(base().extra().get("mask_feature_length"), 10); }
    public int maskFeatureMinMasks() { return toInt(base().extra().get("mask_feature_min_masks"), 0); }
    public int medianFilterWidth() { return toInt(base().extra().get("median_filter_width"), 7); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return WhisperConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}