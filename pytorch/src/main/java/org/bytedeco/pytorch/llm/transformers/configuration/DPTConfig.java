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

/**
 * HuggingFace <code>DPTConfig</code> — Dense Prediction Transformer.
 * Reference: transformers/models/dpt/configuration_dpt.py
 */
public final class DPTConfig extends Config {

    public static final String MODEL_TYPE = "dpt";

    private final int patchSize;
    private final int imageSize;
    private final int fusionLayers;
    private final boolean usePretrainedBackbone;

    public DPTConfig(PretrainedConfig base) {
        super(base);
        this.patchSize = toInt(base.extra().get("patch_size"), 16);
        this.imageSize = toInt(base.extra().get("image_size"), 384);
        this.fusionLayers = toInt(base.extra().get("fusion_layers"), 4);
        this.usePretrainedBackbone = base.extra().get("use_pretrained_backbone") == Boolean.TRUE;
    }

    public int patchSize() { return toInt(base().extra().get("patch_size"), 16); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 384); }
    public int fusionLayers() { return toInt(base().extra().get("fusion_layers"), 4); }
    public boolean usePretrainedBackbone() { return base().extra().get("use_pretrained_backbone") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DPTConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }
}
