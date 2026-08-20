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
 * HuggingFace <code>ayaConfig</code>.
 */
public final class AyaConfig extends Config {

    public static final String MODEL_TYPE = "aya";

    private final double attentionDropout;
    private final int slidingWindow;

    public AyaConfig(PretrainedConfig base) {
        super(base);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.slidingWindow = toInt(base.extra().get("sliding_window"), 4096);
    }

    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public int slidingWindow() { return toInt(base().extra().get("sliding_window"), 4096); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return AyaConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}