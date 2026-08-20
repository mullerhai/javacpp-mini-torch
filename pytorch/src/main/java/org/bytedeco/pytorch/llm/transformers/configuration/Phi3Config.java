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
 * HuggingFace <code>phi3Config</code>.
 * Reference: transformers/models/phi3/configuration_phi3.py
 */
public final class Phi3Config extends Config {

    public static final String MODEL_TYPE = "phi3";

    private final int originalMaxPositionEmbeddings;
    private final int slidingWindow;
    private final double residPdrop;
    private final double embdPdrop;

    public Phi3Config(PretrainedConfig base) {
        super(base);
        this.originalMaxPositionEmbeddings = toInt(base.extra().get("original_max_position_embeddings"), 4096);
        this.slidingWindow = toInt(base.extra().get("sliding_window"), 0);
        this.residPdrop = toDouble(base.extra().get("resid_pdrop"), 0.0);
        this.embdPdrop = toDouble(base.extra().get("embd_pdrop"), 0.0);
    }

    public int originalMaxPositionEmbeddings() { return toInt(base().extra().get("original_max_position_embeddings"), 4096); }
    public int slidingWindow() { return toInt(base().extra().get("sliding_window"), 0); }
    public double residPdrop() { return toDouble(base().extra().get("resid_pdrop"), 0.0); }
    public double embdPdrop() { return toDouble(base().extra().get("embd_pdrop"), 0.0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Phi3Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}