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
 * HuggingFace <code>mllamaConfig</code>.
 * Reference: transformers/models/mllama/configuration_mllama.py
 */
public final class MllamaConfig extends Config {

    public static final String MODEL_TYPE = "mllama";

    private final int numHiddenLayers;
    private final int numGlobalLayers;
    private final int attentionHeads;
    private final int numChannels;
    private final int intermediateSize;
    private final int visionOutputDim;
    private final int imageSize;
    private final int patchSize;
    private final double normEps;
    private final int maxNumTiles;
    private final String intermediateLayersIndices;
    private final String supportedAspectRatios;
    private final int imageTokenIndex;

    public MllamaConfig(PretrainedConfig base) {
        super(base);
        this.numHiddenLayers = toInt(base.extra().get("num_hidden_layers"), 32);
        this.numGlobalLayers = toInt(base.extra().get("num_global_layers"), 8);
        this.attentionHeads = toInt(base.extra().get("attention_heads"), 16);
        this.numChannels = toInt(base.extra().get("num_channels"), 3);
        this.intermediateSize = toInt(base.extra().get("intermediate_size"), 5120);
        this.visionOutputDim = toInt(base.extra().get("vision_output_dim"), 7680);
        this.imageSize = toInt(base.extra().get("image_size"), 448);
        this.patchSize = toInt(base.extra().get("patch_size"), 14);
        this.normEps = toDouble(base.extra().get("norm_eps"), 1e-05);
        this.maxNumTiles = toInt(base.extra().get("max_num_tiles"), 4);
        this.intermediateLayersIndices = String.valueOf(base.extra().get("intermediate_layers_indices"));
        this.supportedAspectRatios = String.valueOf(base.extra().get("supported_aspect_ratios"));
        this.imageTokenIndex = toInt(base.extra().get("image_token_index"), 128256);
    }

    public int numHiddenLayers() { return toInt(base().extra().get("num_hidden_layers"), 32); }
    public int numGlobalLayers() { return toInt(base().extra().get("num_global_layers"), 8); }
    public int attentionHeads() { return toInt(base().extra().get("attention_heads"), 16); }
    public int numChannels() { return toInt(base().extra().get("num_channels"), 3); }
    public int intermediateSize() { return toInt(base().extra().get("intermediate_size"), 5120); }
    public int visionOutputDim() { return toInt(base().extra().get("vision_output_dim"), 7680); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 448); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 14); }
    public double normEps() { return toDouble(base().extra().get("norm_eps"), 1e-05); }
    public int maxNumTiles() { return toInt(base().extra().get("max_num_tiles"), 4); }
    public String intermediateLayersIndices() { Object v = base().extra().get("intermediate_layers_indices"); return v == null ? null : String.valueOf(v); }
    public String supportedAspectRatios() { Object v = base().extra().get("supported_aspect_ratios"); return v == null ? null : String.valueOf(v); }
    public int imageTokenIndex() { return toInt(base().extra().get("image_token_index"), 128256); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return MllamaConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}