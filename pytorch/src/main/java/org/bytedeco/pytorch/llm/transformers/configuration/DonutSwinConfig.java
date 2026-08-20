/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
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
import java.util.List;

/**
 * HuggingFace <code>DonutSwinConfig</code> — Swin Transformer backbone for Donut OCR/VQA.
 * Reference: transformers/models/swin/configuration_swin.py
 */
public final class DonutSwinConfig extends Config {

    public static final String MODEL_TYPE = "donut_swin";

    private final int embedDim;
    private final List<Integer> depths;
    private final List<Integer> numHeads;
    private final int windowSize;
    private final double mlpRatio;

    public DonutSwinConfig(PretrainedConfig base) {
        super(base);
        this.embedDim = toInt(base.extra().get("embed_dim"), 96);
        this.depths = parseIntList(base.extra().get("depths"));
        this.numHeads = parseIntList(base.extra().get("num_heads"));
        this.windowSize = toInt(base.extra().get("window_size"), 7);
        this.mlpRatio = toDouble(base.extra().get("mlp_ratio"), 4.0);
    }

    public int embedDim() { return toInt(base().extra().get("embed_dim"), 96); }
    public List<Integer> depths() { return parseIntList(base().extra().get("depths")); }
    public List<Integer> numHeads() { return parseIntList(base().extra().get("num_heads")); }
    public int windowSize() { return toInt(base().extra().get("window_size"), 7); }
    public double mlpRatio() { return toDouble(base().extra().get("mlp_ratio"), 4.0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DonutSwinConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }

    private static List<Integer> parseIntList(Object o) {
        if (o instanceof List<?>) {
            @SuppressWarnings("unchecked")
            List<Object> raw = (List<Object>) o;
            java.util.List<Integer> result = new java.util.ArrayList<>();
            for (Object item : raw) result.add(((Number) item).intValue());
            return result;
        }
        return java.util.Collections.emptyList();
    }
}
