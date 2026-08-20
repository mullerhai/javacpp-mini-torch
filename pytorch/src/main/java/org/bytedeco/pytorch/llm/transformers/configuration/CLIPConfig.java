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
 * HuggingFace <code>clipConfig</code>.
 * Reference: transformers/models/clip/configuration_clip.py
 */
public final class CLIPConfig extends Config {

    public static final String MODEL_TYPE = "clip";

    private final int projectionDim;
    private final int maxPositionEmbeddings;
    private final String hiddenAct;
    private final double layerNormEps;
    private final double attentionDropout;
    private final double initializerRange;
    private final double initializerFactor;

    public CLIPConfig(PretrainedConfig base) {
        super(base);
        this.projectionDim = toInt(base.extra().get("projection_dim"), 512);
        this.maxPositionEmbeddings = toInt(base.extra().get("max_position_embeddings"), 77);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-05);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.initializerRange = toDouble(base.extra().get("initializer_range"), 0.02);
        this.initializerFactor = toDouble(base.extra().get("initializer_factor"), 1.0);
    }

    public int projectionDim() { return toInt(base().extra().get("projection_dim"), 512); }
    public int maxPositionEmbeddings() { return toInt(base().extra().get("max_position_embeddings"), 77); }
    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "quick_gelu" : String.valueOf(v); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-05); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double initializerRange() { return toDouble(base().extra().get("initializer_range"), 0.02); }
    public double initializerFactor() { return toDouble(base().extra().get("initializer_factor"), 1.0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return CLIPConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}