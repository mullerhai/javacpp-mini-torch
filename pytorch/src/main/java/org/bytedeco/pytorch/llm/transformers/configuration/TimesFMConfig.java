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
 * HuggingFace <code>TimesFMConfig</code>.
 * Reference: transformers/models/times_fm/configuration_times_fm.py
 */
public final class TimesFMConfig extends Config {

    public static final String MODEL_TYPE = "times_fm";

    private final int patchLen;
    private final int contextLen;
    private final int horizonLen;
    private final int numLayers;
    private final int imageSize;
    private final int patchSize;
    private final double attentionDropout;
    private final double hiddenDropoutProb;

    public TimesFMConfig(PretrainedConfig base) {
        super(base);
        this.patchLen = toInt(base.extra().get("patch_len"), 24);
        this.contextLen = toInt(base.extra().get("context_len"), 512);
        this.horizonLen = toInt(base.extra().get("horizon_len"), 128);
        this.numLayers = toInt(base.extra().get("num_layers"), 12);
        this.imageSize = toInt(base.extra().get("image_size"), 224);
        this.patchSize = toInt(base.extra().get("patch_size"), 16);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.0);
    }

    public int patchLen() { return toInt(base().extra().get("patch_len"), 24); }
    public int contextLen() { return toInt(base().extra().get("context_len"), 512); }
    public int horizonLen() { return toInt(base().extra().get("horizon_len"), 128); }
    public int numLayers() { return toInt(base().extra().get("num_layers"), 12); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 224); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 16); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return TimesFMConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
