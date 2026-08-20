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
package org.bytedeco.pytorch.llm.transformers.modeling.output;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.transformers.generation.cache.Cache;

import java.util.Map;
import java.util.List;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.io.IOException;
import org.bytedeco.pytorch.utils.json.Json;

/**
 * Base class for all model outputs.
 * Mimics HuggingFace transformers' ModelOutput hierarchy.
 */
public abstract class ModelOutput {
    private final Map<String, Object> extraData;

    protected ModelOutput() {
        this.extraData = new LinkedHashMap<>();
    }

    protected ModelOutput(Map<String, Object> extraData) {
        this.extraData = extraData != null ? new LinkedHashMap<>(extraData) : new LinkedHashMap<>();
    }

    public Map<String, Object> extra() { return Collections.unmodifiableMap(extraData); }

    public Map<String, Object> toMap() {
        return new LinkedHashMap<>(extraData);
    }

    public String toJson() {
        return Json.encode(toMap());
    }

    @Override public String toString() {
        return getClass().getSimpleName() + toMap();
    }
}
