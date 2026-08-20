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
 * HuggingFace <code>LayoutLMv3Config</code> — improved document understanding with
 * unified visual and textual embeddings.
 * Reference: transformers/models/layoutlmv3/configuration_layoutlmv3.py
 */
public final class LayoutLMv3Config extends Config {

    public static final String MODEL_TYPE = "layoutlmv3";

    private final int max2dPositionEmbeddings;
    private final int coordinateSize;
    private final int shapeSize;

    public LayoutLMv3Config(PretrainedConfig base) {
        super(base);
        this.max2dPositionEmbeddings = toInt(base.extra().get("max_2d_position_embeddings"), 1024);
        this.coordinateSize = toInt(base.extra().get("coordinate_size"), 128);
        this.shapeSize = toInt(base.extra().get("shape_size"), 128);
    }

    public int max2dPositionEmbeddings() { return toInt(base().extra().get("max_2d_position_embeddings"), 1024); }
    public int coordinateSize() { return toInt(base().extra().get("coordinate_size"), 128); }
    public int shapeSize() { return toInt(base().extra().get("shape_size"), 128); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return LayoutLMv3Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }
}
