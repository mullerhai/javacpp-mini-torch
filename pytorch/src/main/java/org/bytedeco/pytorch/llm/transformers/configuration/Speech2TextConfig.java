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
 * HuggingFace <code>speech_to_textConfig</code>.
 * Reference: transformers/models/speech_to_text/configuration_speech_to_text.py
 */
public final class Speech2TextConfig extends Config {

    public static final String MODEL_TYPE = "speech_to_text";

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
    private final boolean scaleEmbedding;
    private final int maxSourcePositions;
    private final int maxTargetPositions;
    private final int numConvLayers;
    private final String convKernelSizes;
    private final int convChannels;
    private final int inputFeatPerChannel;
    private final int inputChannels;

    public Speech2TextConfig(PretrainedConfig base) {
        super(base);
        this.dModel = toInt(base.extra().get("d_model"), 256);
        this.encoderLayers = toInt(base.extra().get("encoder_layers"), 12);
        this.decoderLayers = toInt(base.extra().get("decoder_layers"), 6);
        this.encoderFfnDim = toInt(base.extra().get("encoder_ffn_dim"), 2048);
        this.decoderFfnDim = toInt(base.extra().get("decoder_ffn_dim"), 2048);
        this.encoderAttentionHeads = toInt(base.extra().get("encoder_attention_heads"), 4);
        this.decoderAttentionHeads = toInt(base.extra().get("decoder_attention_heads"), 4);
        this.encoderLayerdrop = toDouble(base.extra().get("encoder_layerdrop"), 0.0);
        this.decoderLayerdrop = toDouble(base.extra().get("decoder_layerdrop"), 0.0);
        this.activationFunction = String.valueOf(base.extra().get("activation_function"));
        this.dropout = toDouble(base.extra().get("dropout"), 0.1);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.activationDropout = toDouble(base.extra().get("activation_dropout"), 0.0);
        this.initStd = toDouble(base.extra().get("init_std"), 0.02);
        this.scaleEmbedding = base.extra().get("scale_embedding") == Boolean.TRUE;
        this.maxSourcePositions = toInt(base.extra().get("max_source_positions"), 6000);
        this.maxTargetPositions = toInt(base.extra().get("max_target_positions"), 1024);
        this.numConvLayers = toInt(base.extra().get("num_conv_layers"), 2);
        this.convKernelSizes = String.valueOf(base.extra().get("conv_kernel_sizes"));
        this.convChannels = toInt(base.extra().get("conv_channels"), 1024);
        this.inputFeatPerChannel = toInt(base.extra().get("input_feat_per_channel"), 80);
        this.inputChannels = toInt(base.extra().get("input_channels"), 1);
    }

    public int dModel() { return toInt(base().extra().get("d_model"), 256); }
    public int encoderLayers() { return toInt(base().extra().get("encoder_layers"), 12); }
    public int decoderLayers() { return toInt(base().extra().get("decoder_layers"), 6); }
    public int encoderFfnDim() { return toInt(base().extra().get("encoder_ffn_dim"), 2048); }
    public int decoderFfnDim() { return toInt(base().extra().get("decoder_ffn_dim"), 2048); }
    public int encoderAttentionHeads() { return toInt(base().extra().get("encoder_attention_heads"), 4); }
    public int decoderAttentionHeads() { return toInt(base().extra().get("decoder_attention_heads"), 4); }
    public double encoderLayerdrop() { return toDouble(base().extra().get("encoder_layerdrop"), 0.0); }
    public double decoderLayerdrop() { return toDouble(base().extra().get("decoder_layerdrop"), 0.0); }
    public String activationFunction() { Object v = base().extra().get("activation_function"); return v == null ? "relu" : String.valueOf(v); }
    public double dropout() { return toDouble(base().extra().get("dropout"), 0.1); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double activationDropout() { return toDouble(base().extra().get("activation_dropout"), 0.0); }
    public double initStd() { return toDouble(base().extra().get("init_std"), 0.02); }
    public boolean scaleEmbedding() { return base().extra().get("scale_embedding") == Boolean.TRUE; }
    public int maxSourcePositions() { return toInt(base().extra().get("max_source_positions"), 6000); }
    public int maxTargetPositions() { return toInt(base().extra().get("max_target_positions"), 1024); }
    public int numConvLayers() { return toInt(base().extra().get("num_conv_layers"), 2); }
    public String convKernelSizes() { Object v = base().extra().get("conv_kernel_sizes"); return v == null ? "(5, 5)" : String.valueOf(v); }
    public int convChannels() { return toInt(base().extra().get("conv_channels"), 1024); }
    public int inputFeatPerChannel() { return toInt(base().extra().get("input_feat_per_channel"), 80); }
    public int inputChannels() { return toInt(base().extra().get("input_channels"), 1); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Speech2TextConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}