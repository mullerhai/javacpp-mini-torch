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
package org.bytedeco.pytorch.llm.transformers.data;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Dataset backed by an in-memory {@link List} of example maps.
 *
 * <p>Mirrors HuggingFace {@code MapDataset}.
 */
public class MapDataset extends Dataset {

    private final List<Map<String, Object>> data;

    public MapDataset(List<Map<String, Object>> data) {
        this.data = new ArrayList<>(Objects.requireNonNull(data, "data"));
    }

    @Override
    public int size() { return data.size(); }

    @Override
    public Map<String, Object> get(int index) {
        if (index < 0 || index >= data.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + data.size());
        }
        return Collections.unmodifiableMap(data.get(index));
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        List<Map<String, Object>> mapped = data.stream()
                .map(fn)
                .collect(Collectors.toList());
        return new MapDataset(mapped);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Map<String, Object>> filtered = data.stream()
                .filter(predicate)
                .collect(Collectors.toList());
        return new MapDataset(filtered);
    }

    @Override
    public Dataset shuffle(long seed) {
        List<Map<String, Object>> copy = new ArrayList<>(data);
        Collections.shuffle(copy, new Random(seed));
        return new MapDataset(copy);
    }

    @Override
    public Dataset select(int[] indices) {
        List<Map<String, Object>> selected = Arrays.stream(indices)
                .mapToObj(data::get)
                .collect(Collectors.toList());
        return new MapDataset(selected);
    }
}
