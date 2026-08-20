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
 * HuggingFace <code>qwen3_vlConfig</code>.
 * Reference: transformers/models/qwen3_vl/configuration_qwen3_vl.py
 */
public final class Qwen3VLConfig extends Config {

    public static final String MODEL_TYPE = "qwen3_vl";

    private final int depth;
    private final String hiddenAct;
    private final int intermediateSize;
    private final int mlpRatio;
    private final int numHeads;
    private final int inChannels;
    private final int patchSize;
    private final int spatialMergeSize;
    private final int temporalPatchSize;
    private final int outHiddenSize;
    private final int numPositionEmbeddings;
    private final String deepstackVisualIndexes;
    private final int imageTokenId;
    private final int videoTokenId;
    private final int visionStartTokenId;
    private final int visionEndTokenId;
    private final boolean tieWordEmbeddings;

    public Qwen3VLConfig(PretrainedConfig base) {
        super(base);
        this.depth = toInt(base.extra().get("depth"), 27);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.intermediateSize = toInt(base.extra().get("intermediate_size"), 4304);
        this.mlpRatio = toInt(base.extra().get("mlp_ratio"), 4);
        this.numHeads = toInt(base.extra().get("num_heads"), 16);
        this.inChannels = toInt(base.extra().get("in_channels"), 3);
        this.patchSize = toInt(base.extra().get("patch_size"), 16);
        this.spatialMergeSize = toInt(base.extra().get("spatial_merge_size"), 2);
        this.temporalPatchSize = toInt(base.extra().get("temporal_patch_size"), 2);
        this.outHiddenSize = toInt(base.extra().get("out_hidden_size"), 3584);
        this.numPositionEmbeddings = toInt(base.extra().get("num_position_embeddings"), 2304);
        this.deepstackVisualIndexes = String.valueOf(base.extra().get("deepstack_visual_indexes"));
        this.imageTokenId = toInt(base.extra().get("image_token_id"), 151655);
        this.videoTokenId = toInt(base.extra().get("video_token_id"), 151656);
        this.visionStartTokenId = toInt(base.extra().get("vision_start_token_id"), 151652);
        this.visionEndTokenId = toInt(base.extra().get("vision_end_token_id"), 151653);
        this.tieWordEmbeddings = base.extra().get("tie_word_embeddings") == Boolean.TRUE;
    }

    public int depth() { return toInt(base().extra().get("depth"), 27); }
    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "gelu_pytorch_tanh" : String.valueOf(v); }
    public int intermediateSize() { return toInt(base().extra().get("intermediate_size"), 4304); }
    public int mlpRatio() { return toInt(base().extra().get("mlp_ratio"), 4); }
    public int numHeads() { return toInt(base().extra().get("num_heads"), 16); }
    public int inChannels() { return toInt(base().extra().get("in_channels"), 3); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 16); }
    public int spatialMergeSize() { return toInt(base().extra().get("spatial_merge_size"), 2); }
    public int temporalPatchSize() { return toInt(base().extra().get("temporal_patch_size"), 2); }
    public int outHiddenSize() { return toInt(base().extra().get("out_hidden_size"), 3584); }
    public int numPositionEmbeddings() { return toInt(base().extra().get("num_position_embeddings"), 2304); }
    public String deepstackVisualIndexes() { Object v = base().extra().get("deepstack_visual_indexes"); return v == null ? "(8, 16, 24)" : String.valueOf(v); }
    public int imageTokenId() { return toInt(base().extra().get("image_token_id"), 151655); }
    public int videoTokenId() { return toInt(base().extra().get("video_token_id"), 151656); }
    public int visionStartTokenId() { return toInt(base().extra().get("vision_start_token_id"), 151652); }
    public int visionEndTokenId() { return toInt(base().extra().get("vision_end_token_id"), 151653); }
    public boolean tieWordEmbeddings() { return base().extra().get("tie_word_embeddings") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Qwen3VLConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}