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
package org.bytedeco.pytorch.llm.transformers.feature_extraction;

import org.bytedeco.pytorch.Tensor;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Container for a batch of extracted features.
 *
 * <p>Wraps a map of feature name to tensor.
 * Reference: HuggingFace transformers {@code BatchFeature}.
 */
public final class BatchFeature {

    private final Map<String, Tensor> features;

    public BatchFeature() {
        this.features = new HashMap<>();
    }

    public BatchFeature(Map<String, Tensor> features) {
        this.features = new HashMap<>(features);
    }

    /**
     * Get a feature tensor by key.
     *
     * @param key feature name
     * @return the tensor, or null if not present
     */
    public Tensor get(String key) {
        return features.get(key);
    }

    /**
     * Put a feature tensor.
     *
     * @param key   feature name
     * @param value tensor value
     * @return this (fluent)
     */
    public BatchFeature put(String key, Tensor value) {
        features.put(key, value);
        return this;
    }

    /**
     * Get all feature names.
     */
    public Set<String> keys() {
        return features.keySet();
    }

    /**
     * Get an unmodifiable view of the underlying map.
     */
    public Map<String, Tensor> toMap() {
        return Map.copyOf(features);
    }
}
