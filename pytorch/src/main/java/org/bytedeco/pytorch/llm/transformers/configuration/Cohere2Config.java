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
 * HuggingFace <code>cohere2Config</code>.
 * Reference: transformers/models/cohere2/configuration_cohere2.py
 */
public final class Cohere2Config extends Config {

    public static final String MODEL_TYPE = "cohere2";

    private final double logitScale;
    private final int slidingWindow;
    private final String layerTypes;
    private final boolean useQkNorm;

    public Cohere2Config(PretrainedConfig base) {
        super(base);
        this.logitScale = toDouble(base.extra().get("logit_scale"), 0.0625);
        this.slidingWindow = toInt(base.extra().get("sliding_window"), 4096);
        this.layerTypes = String.valueOf(base.extra().get("layer_types"));
        this.useQkNorm = base.extra().get("use_qk_norm") == Boolean.TRUE;
    }

    public double logitScale() { return toDouble(base().extra().get("logit_scale"), 0.0625); }
    public int slidingWindow() { return toInt(base().extra().get("sliding_window"), 4096); }
    public String layerTypes() { Object v = base().extra().get("layer_types"); return v == null ? null : String.valueOf(v); }
    public boolean useQkNorm() { return base().extra().get("use_qk_norm") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Cohere2Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}