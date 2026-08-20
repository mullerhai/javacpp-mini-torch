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
 * HuggingFace <code>gpt_neoConfig</code>.
 * Reference: transformers/models/gpt_neo/configuration_gpt_neo.py
 */
public final class GPTNeoConfig extends Config {

    public static final String MODEL_TYPE = "gpt_neo";

    private final int numLayers;
    private final int numHeads;
    private final int intermediateSize;
    private final int windowSize;
    private final String activationFunction;
    private final double residDropout;
    private final double embedDropout;
    private final double attentionDropout;
    private final double classifierDropout;
    private final double layerNormEpsilon;

    public GPTNeoConfig(PretrainedConfig base) {
        super(base);
        this.numLayers = toInt(base.extra().get("num_layers"), 24);
        this.numHeads = toInt(base.extra().get("num_heads"), 16);
        this.intermediateSize = toInt(base.extra().get("intermediate_size"), 0);
        this.windowSize = toInt(base.extra().get("window_size"), 256);
        this.activationFunction = String.valueOf(base.extra().get("activation_function"));
        this.residDropout = toDouble(base.extra().get("resid_dropout"), 0.0);
        this.embedDropout = toDouble(base.extra().get("embed_dropout"), 0.0);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.classifierDropout = toDouble(base.extra().get("classifier_dropout"), 0.1);
        this.layerNormEpsilon = toDouble(base.extra().get("layer_norm_epsilon"), 1e-05);
    }

    public int numLayers() { return toInt(base().extra().get("num_layers"), 24); }
    public int numHeads() { return toInt(base().extra().get("num_heads"), 16); }
    public int intermediateSize() { return toInt(base().extra().get("intermediate_size"), 0); }
    public int windowSize() { return toInt(base().extra().get("window_size"), 256); }
    public String activationFunction() { Object v = base().extra().get("activation_function"); return v == null ? "gelu_new" : String.valueOf(v); }
    public double residDropout() { return toDouble(base().extra().get("resid_dropout"), 0.0); }
    public double embedDropout() { return toDouble(base().extra().get("embed_dropout"), 0.0); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double classifierDropout() { return toDouble(base().extra().get("classifier_dropout"), 0.1); }
    public double layerNormEpsilon() { return toDouble(base().extra().get("layer_norm_epsilon"), 1e-05); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return GPTNeoConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}