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
 * HuggingFace <code>pixtralConfig</code>.
 * Reference: transformers/models/pixtral/configuration_pixtral.py
 */
public final class PixtralConfig extends Config {

    public static final String MODEL_TYPE = "pixtral";

    private final int numChannels;
    private final int imageSize;
    private final int patchSize;
    private final String ropeParameters;

    public PixtralConfig(PretrainedConfig base) {
        super(base);
        this.numChannels = toInt(base.extra().get("num_channels"), 3);
        this.imageSize = toInt(base.extra().get("image_size"), 1024);
        this.patchSize = toInt(base.extra().get("patch_size"), 16);
        this.ropeParameters = String.valueOf(base.extra().get("rope_parameters"));
    }

    public int numChannels() { return toInt(base().extra().get("num_channels"), 3); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 1024); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 16); }
    public String ropeParameters() { Object v = base().extra().get("rope_parameters"); return v == null ? null : String.valueOf(v); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return PixtralConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}