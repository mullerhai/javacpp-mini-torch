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
 * HuggingFace <code>clapConfig</code>.
 * Reference: transformers/models/clap/configuration_clap.py
 */
public final class ClapConfig extends Config {

    public static final String MODEL_TYPE = "clap";

    private final int projectionDim;
    private final int windowSize;
    private final int numMelBins;
    private final int specSize;
    private final int patchSize;
    private final int patchStride;
    private final int numClasses;
    private final String depths;
    private final double hiddenDropoutProb;
    private final String fusionType;
    private final int patchEmbedInputChannels;
    private final boolean flattenPatchEmbeds;
    private final int patchEmbedsHiddenSize;
    private final boolean enablePatchLayerNorm;
    private final double dropPathRate;
    private final boolean qkvBias;
    private final double mlpRatio;
    private final int affBlockR;

    public ClapConfig(PretrainedConfig base) {
        super(base);
        this.projectionDim = toInt(base.extra().get("projection_dim"), 512);
        this.windowSize = toInt(base.extra().get("window_size"), 8);
        this.numMelBins = toInt(base.extra().get("num_mel_bins"), 64);
        this.specSize = toInt(base.extra().get("spec_size"), 256);
        this.patchSize = toInt(base.extra().get("patch_size"), 4);
        this.patchStride = toInt(base.extra().get("patch_stride"), 4);
        this.numClasses = toInt(base.extra().get("num_classes"), 527);
        this.depths = String.valueOf(base.extra().get("depths"));
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.1);
        this.fusionType = String.valueOf(base.extra().get("fusion_type"));
        this.patchEmbedInputChannels = toInt(base.extra().get("patch_embed_input_channels"), 1);
        this.flattenPatchEmbeds = base.extra().get("flatten_patch_embeds") == Boolean.TRUE;
        this.patchEmbedsHiddenSize = toInt(base.extra().get("patch_embeds_hidden_size"), 96);
        this.enablePatchLayerNorm = base.extra().get("enable_patch_layer_norm") == Boolean.TRUE;
        this.dropPathRate = toDouble(base.extra().get("drop_path_rate"), 0.0);
        this.qkvBias = base.extra().get("qkv_bias") == Boolean.TRUE;
        this.mlpRatio = toDouble(base.extra().get("mlp_ratio"), 4.0);
        this.affBlockR = toInt(base.extra().get("aff_block_r"), 4);
    }

    public int projectionDim() { return toInt(base().extra().get("projection_dim"), 512); }
    public int windowSize() { return toInt(base().extra().get("window_size"), 8); }
    public int numMelBins() { return toInt(base().extra().get("num_mel_bins"), 64); }
    public int specSize() { return toInt(base().extra().get("spec_size"), 256); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 4); }
    public int patchStride() { return toInt(base().extra().get("patch_stride"), 4); }
    public int numClasses() { return toInt(base().extra().get("num_classes"), 527); }
    public String depths() { Object v = base().extra().get("depths"); return v == null ? "(2, 2, 6, 2)" : String.valueOf(v); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.1); }
    public String fusionType() { Object v = base().extra().get("fusion_type"); return v == null ? null : String.valueOf(v); }
    public int patchEmbedInputChannels() { return toInt(base().extra().get("patch_embed_input_channels"), 1); }
    public boolean flattenPatchEmbeds() { return base().extra().get("flatten_patch_embeds") == Boolean.TRUE; }
    public int patchEmbedsHiddenSize() { return toInt(base().extra().get("patch_embeds_hidden_size"), 96); }
    public boolean enablePatchLayerNorm() { return base().extra().get("enable_patch_layer_norm") == Boolean.TRUE; }
    public double dropPathRate() { return toDouble(base().extra().get("drop_path_rate"), 0.0); }
    public boolean qkvBias() { return base().extra().get("qkv_bias") == Boolean.TRUE; }
    public double mlpRatio() { return toDouble(base().extra().get("mlp_ratio"), 4.0); }
    public int affBlockR() { return toInt(base().extra().get("aff_block_r"), 4); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return ClapConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}