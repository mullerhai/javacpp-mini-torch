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
 * HuggingFace <code>graniteConfig</code>.
 * Reference: transformers/models/granite/configuration_granite.py
 */
public final class GraniteConfig extends Config {

    public static final String MODEL_TYPE = "granite";

    private final double embeddingMultiplier;
    private final double logitsScaling;
    private final double residualMultiplier;
    private final double attentionMultiplier;
    private final boolean attentionBias;
    private final boolean mlpBias;

    public GraniteConfig(PretrainedConfig base) {
        super(base);
        this.embeddingMultiplier = toDouble(base.extra().get("embedding_multiplier"), 1.0);
        this.logitsScaling = toDouble(base.extra().get("logits_scaling"), 1.0);
        this.residualMultiplier = toDouble(base.extra().get("residual_multiplier"), 1.0);
        this.attentionMultiplier = toDouble(base.extra().get("attention_multiplier"), 1.0);
        this.attentionBias = base.extra().get("attention_bias") == Boolean.TRUE;
        this.mlpBias = base.extra().get("mlp_bias") == Boolean.TRUE;
    }

    public double embeddingMultiplier() { return toDouble(base().extra().get("embedding_multiplier"), 1.0); }
    public double logitsScaling() { return toDouble(base().extra().get("logits_scaling"), 1.0); }
    public double residualMultiplier() { return toDouble(base().extra().get("residual_multiplier"), 1.0); }
    public double attentionMultiplier() { return toDouble(base().extra().get("attention_multiplier"), 1.0); }
    public boolean attentionBias() { return base().extra().get("attention_bias") == Boolean.TRUE; }
    public boolean mlpBias() { return base().extra().get("mlp_bias") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return GraniteConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}