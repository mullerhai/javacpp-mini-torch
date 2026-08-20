/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
package org.bytedeco.pytorch.llm.transformers.tokenization_utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Container for batch tokenization output.
 *
 * <p>Reference: HuggingFace transformers {@code tokenization_utils_base.BatchEncoding}.
 */
public final class BatchEncoding {

    private final Map<String, Object> data;

    public BatchEncoding() {
        this.data = new HashMap<>();
    }

    public BatchEncoding(Map<String, Object> data) {
        this.data = new HashMap<>(data);
    }

    public int[] input_ids() {
        return get("input_ids", int[].class);
    }

    public BatchEncoding input_ids(int[] ids) {
        put("input_ids", ids);
        return this;
    }

    public int[] attention_mask() {
        return get("attention_mask", int[].class);
    }

    public BatchEncoding attention_mask(int[] mask) {
        put("attention_mask", mask);
        return this;
    }

    public int[] token_type_ids() {
        return get("token_type_ids", int[].class);
    }

    public BatchEncoding token_type_ids(int[] ids) {
        put("token_type_ids", ids);
        return this;
    }

    @SuppressWarnings("unchecked")
    public <T> T get(String key) {
        return (T) data.get(key);
    }

    public <T> T get(String key, Class<T> type) {
        Object v = data.get(key);
        if (v != null && !type.isInstance(v)) {
            throw new ClassCastException("Expected " + type + " but got " + v.getClass());
        }
        return type.cast(v);
    }

    public BatchEncoding put(String key, Object value) {
        data.put(key, value);
        return this;
    }

    public Set<String> keys() {
        return data.keySet();
    }

    public Map<String, Object> toMap() {
        return Map.copyOf(data);
    }
}
