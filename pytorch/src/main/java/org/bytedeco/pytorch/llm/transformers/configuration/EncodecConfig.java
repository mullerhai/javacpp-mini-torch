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
 * HuggingFace <code>encodecConfig</code>.
 * Reference: transformers/models/encodec/configuration_encodec.py
 */
public final class EncodecConfig extends Config {

    public static final String MODEL_TYPE = "encodec";

    private final double targetBandwidths;
    private final int samplingRate;
    private final int audioChannels;
    private final boolean normalize;
    private final int chunkLengthS;
    private final int overlap;
    private final int numFilters;
    private final int numResidualLayers;
    private final String upsamplingRatios;
    private final String normType;
    private final int kernelSize;
    private final int lastKernelSize;
    private final int residualKernelSize;
    private final int dilationGrowthRate;
    private final boolean useCausalConv;
    private final String padMode;
    private final int compress;
    private final int numLstmLayers;
    private final double trimRightRatio;
    private final int codebookSize;
    private final int codebookDim;
    private final boolean useConvShortcut;

    public EncodecConfig(PretrainedConfig base) {
        super(base);
        this.targetBandwidths = toDouble(base.extra().get("target_bandwidths"), 1.5);
        this.samplingRate = toInt(base.extra().get("sampling_rate"), 24000);
        this.audioChannels = toInt(base.extra().get("audio_channels"), 1);
        this.normalize = base.extra().get("normalize") == Boolean.TRUE;
        this.chunkLengthS = toInt(base.extra().get("chunk_length_s"), 0);
        this.overlap = toInt(base.extra().get("overlap"), 0);
        this.numFilters = toInt(base.extra().get("num_filters"), 32);
        this.numResidualLayers = toInt(base.extra().get("num_residual_layers"), 1);
        this.upsamplingRatios = String.valueOf(base.extra().get("upsampling_ratios"));
        this.normType = String.valueOf(base.extra().get("norm_type"));
        this.kernelSize = toInt(base.extra().get("kernel_size"), 7);
        this.lastKernelSize = toInt(base.extra().get("last_kernel_size"), 7);
        this.residualKernelSize = toInt(base.extra().get("residual_kernel_size"), 3);
        this.dilationGrowthRate = toInt(base.extra().get("dilation_growth_rate"), 2);
        this.useCausalConv = base.extra().get("use_causal_conv") == Boolean.TRUE;
        this.padMode = String.valueOf(base.extra().get("pad_mode"));
        this.compress = toInt(base.extra().get("compress"), 2);
        this.numLstmLayers = toInt(base.extra().get("num_lstm_layers"), 2);
        this.trimRightRatio = toDouble(base.extra().get("trim_right_ratio"), 1.0);
        this.codebookSize = toInt(base.extra().get("codebook_size"), 1024);
        this.codebookDim = toInt(base.extra().get("codebook_dim"), 0);
        this.useConvShortcut = base.extra().get("use_conv_shortcut") == Boolean.TRUE;
    }

    public double targetBandwidths() { return toDouble(base().extra().get("target_bandwidths"), 1.5); }
    public int samplingRate() { return toInt(base().extra().get("sampling_rate"), 24000); }
    public int audioChannels() { return toInt(base().extra().get("audio_channels"), 1); }
    public boolean normalize() { return base().extra().get("normalize") == Boolean.TRUE; }
    public int chunkLengthS() { return toInt(base().extra().get("chunk_length_s"), 0); }
    public int overlap() { return toInt(base().extra().get("overlap"), 0); }
    public int numFilters() { return toInt(base().extra().get("num_filters"), 32); }
    public int numResidualLayers() { return toInt(base().extra().get("num_residual_layers"), 1); }
    public String upsamplingRatios() { Object v = base().extra().get("upsampling_ratios"); return v == null ? "(8, 5, 4, 2)" : String.valueOf(v); }
    public String normType() { Object v = base().extra().get("norm_type"); return v == null ? "weight_norm" : String.valueOf(v); }
    public int kernelSize() { return toInt(base().extra().get("kernel_size"), 7); }
    public int lastKernelSize() { return toInt(base().extra().get("last_kernel_size"), 7); }
    public int residualKernelSize() { return toInt(base().extra().get("residual_kernel_size"), 3); }
    public int dilationGrowthRate() { return toInt(base().extra().get("dilation_growth_rate"), 2); }
    public boolean useCausalConv() { return base().extra().get("use_causal_conv") == Boolean.TRUE; }
    public String padMode() { Object v = base().extra().get("pad_mode"); return v == null ? "reflect" : String.valueOf(v); }
    public int compress() { return toInt(base().extra().get("compress"), 2); }
    public int numLstmLayers() { return toInt(base().extra().get("num_lstm_layers"), 2); }
    public double trimRightRatio() { return toDouble(base().extra().get("trim_right_ratio"), 1.0); }
    public int codebookSize() { return toInt(base().extra().get("codebook_size"), 1024); }
    public int codebookDim() { return toInt(base().extra().get("codebook_dim"), 0); }
    public boolean useConvShortcut() { return base().extra().get("use_conv_shortcut") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return EncodecConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}