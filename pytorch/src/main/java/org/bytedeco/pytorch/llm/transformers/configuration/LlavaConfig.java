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
 * HuggingFace <code>llavaConfig</code>.
 * Reference: transformers/models/llava/configuration_llava.py
 */
public final class LlavaConfig extends Config {

    public static final String MODEL_TYPE = "llava";

    private final String visionConfig;
    private final String textConfig;
    private final int imageTokenIndex;
    private final int imageSeqLength;
    private final String projectorHiddenAct;
    private final String visionFeatureSelectStrategy;
    private final int visionFeatureLayer;
    private final boolean multimodalProjectorBias;
    private final boolean tieWordEmbeddings;

    public LlavaConfig(PretrainedConfig base) {
        super(base);
        this.visionConfig = String.valueOf(base.extra().get("vision_config"));
        this.textConfig = String.valueOf(base.extra().get("text_config"));
        this.imageTokenIndex = toInt(base.extra().get("image_token_index"), 32000);
        this.imageSeqLength = toInt(base.extra().get("image_seq_length"), 576);
        this.projectorHiddenAct = String.valueOf(base.extra().get("projector_hidden_act"));
        this.visionFeatureSelectStrategy = String.valueOf(base.extra().get("vision_feature_select_strategy"));
        this.visionFeatureLayer = toInt(base.extra().get("vision_feature_layer"), -2);
        this.multimodalProjectorBias = base.extra().get("multimodal_projector_bias") == Boolean.TRUE;
        this.tieWordEmbeddings = base.extra().get("tie_word_embeddings") == Boolean.TRUE;
    }

    public String visionConfig() { Object v = base().extra().get("vision_config"); return v == null ? null : String.valueOf(v); }
    public String textConfig() { Object v = base().extra().get("text_config"); return v == null ? null : String.valueOf(v); }
    public int imageTokenIndex() { return toInt(base().extra().get("image_token_index"), 32000); }
    public int imageSeqLength() { return toInt(base().extra().get("image_seq_length"), 576); }
    public String projectorHiddenAct() { Object v = base().extra().get("projector_hidden_act"); return v == null ? "gelu" : String.valueOf(v); }
    public String visionFeatureSelectStrategy() { Object v = base().extra().get("vision_feature_select_strategy"); return v == null ? "default" : String.valueOf(v); }
    public int visionFeatureLayer() { return toInt(base().extra().get("vision_feature_layer"), -2); }
    public boolean multimodalProjectorBias() { return base().extra().get("multimodal_projector_bias") == Boolean.TRUE; }
    public boolean tieWordEmbeddings() { return base().extra().get("tie_word_embeddings") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return LlavaConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}