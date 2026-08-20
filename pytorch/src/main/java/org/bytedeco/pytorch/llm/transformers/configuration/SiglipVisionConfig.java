/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Herve Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 * HuggingFace <code>siglip_vision_modelConfig</code>.
 */
public final class SiglipVisionConfig extends Config {

    public static final String MODEL_TYPE = "siglip_vision_model";

    private final String hiddenAct;
    private final double layerNormEps;
    private final double attentionDropout;
    private final int numPatches;
    private final int projectionSize;

    public SiglipVisionConfig(PretrainedConfig base) {
        super(base);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-06);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.numPatches = toInt(base.extra().get("num_patches"), 256);
        this.projectionSize = toInt(base.extra().get("projection_size"), 0);
    }

    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "gelu_pytorch_tanh" : String.valueOf(v); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-06); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public int numPatches() { return toInt(base().extra().get("num_patches"), 256); }
    public int projectionSize() { return toInt(base().extra().get("projection_size"), 0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return SiglipVisionConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}