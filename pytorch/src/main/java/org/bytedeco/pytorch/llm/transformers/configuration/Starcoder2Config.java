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
 * HuggingFace <code>starcoder2Config</code>.
 * Reference: transformers/models/starcoder2/configuration_starcoder2.py
 */
public final class Starcoder2Config extends Config {

    public static final String MODEL_TYPE = "starcoder2";

    private final double normEpsilon;
    private final int slidingWindow;
    private final double attentionDropout;
    private final double residualDropout;
    private final double embeddingDropout;
    private final boolean useBias;

    public Starcoder2Config(PretrainedConfig base) {
        super(base);
        this.normEpsilon = toDouble(base.extra().get("norm_epsilon"), 1e-05);
        this.slidingWindow = toInt(base.extra().get("sliding_window"), 0);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.residualDropout = toDouble(base.extra().get("residual_dropout"), 0.0);
        this.embeddingDropout = toDouble(base.extra().get("embedding_dropout"), 0.0);
        this.useBias = base.extra().get("use_bias") == Boolean.TRUE;
    }

    public double normEpsilon() { return toDouble(base().extra().get("norm_epsilon"), 1e-05); }
    public int slidingWindow() { return toInt(base().extra().get("sliding_window"), 0); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double residualDropout() { return toDouble(base().extra().get("residual_dropout"), 0.0); }
    public double embeddingDropout() { return toDouble(base().extra().get("embedding_dropout"), 0.0); }
    public boolean useBias() { return base().extra().get("use_bias") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Starcoder2Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}