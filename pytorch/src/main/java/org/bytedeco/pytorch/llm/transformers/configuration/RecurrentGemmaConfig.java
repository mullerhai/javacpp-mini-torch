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
 * HuggingFace <code>recurrent_gemmaConfig</code>.
 * Reference: transformers/models/recurrent_gemma/configuration_recurrent_gemma.py
 */
public final class RecurrentGemmaConfig extends Config {

    public static final String MODEL_TYPE = "recurrent_gemma";

    private final int intermediateSize;
    private final int lruWidth;
    private final int attentionWindowSize;
    private final int conv1dWidth;
    private final double logitsSoftCap;
    private final String blockTypes;
    private final double attentionDropout;
    private final double wInitVarianceScale;
    private final boolean attentionBias;

    public RecurrentGemmaConfig(PretrainedConfig base) {
        super(base);
        this.intermediateSize = toInt(base.extra().get("intermediate_size"), 7680);
        this.lruWidth = toInt(base.extra().get("lru_width"), 0);
        this.attentionWindowSize = toInt(base.extra().get("attention_window_size"), 2048);
        this.conv1dWidth = toInt(base.extra().get("conv1d_width"), 4);
        this.logitsSoftCap = toDouble(base.extra().get("logits_soft_cap"), 30.0);
        this.blockTypes = String.valueOf(base.extra().get("block_types"));
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.wInitVarianceScale = toDouble(base.extra().get("w_init_variance_scale"), 0.01);
        this.attentionBias = base.extra().get("attention_bias") == Boolean.TRUE;
    }

    public int intermediateSize() { return toInt(base().extra().get("intermediate_size"), 7680); }
    public int lruWidth() { return toInt(base().extra().get("lru_width"), 0); }
    public int attentionWindowSize() { return toInt(base().extra().get("attention_window_size"), 2048); }
    public int conv1dWidth() { return toInt(base().extra().get("conv1d_width"), 4); }
    public double logitsSoftCap() { return toDouble(base().extra().get("logits_soft_cap"), 30.0); }
    public String blockTypes() { Object v = base().extra().get("block_types"); return v == null ? "(recurrent, recurrent, attention)" : String.valueOf(v); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double wInitVarianceScale() { return toDouble(base().extra().get("w_init_variance_scale"), 0.01); }
    public boolean attentionBias() { return base().extra().get("attention_bias") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return RecurrentGemmaConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}