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

/**
 * Dataset backed by an external index (e.g. a FAISS index or memory-mapped file).
 *
 * <p>Only loads individual examples on demand rather than materializing
 * the full dataset in memory.
 */
public class IndexedDataset extends Dataset {

    private final int size;
    private final java.util.function.Function<Integer, Map<String, Object>> loader;
    private final Map<Integer, Map<String, Object>> cache;

    /**
     * @param size   number of examples
     * @param loader function to load example at given index
     */
    public IndexedDataset(int size, java.util.function.Function<Integer, Map<String, Object>> loader) {
        if (size < 0) throw new IllegalArgumentException("size must be non-negative");
        this.size = size;
        this.loader = Objects.requireNonNull(loader, "loader");
        this.cache = new LinkedHashMap<Integer, Map<String, Object>>() {
            private static final int MAX_ENTRIES = 1000;
            @Override protected boolean removeEldestEntry(Map.Entry eldest) {
                return size() > MAX_ENTRIES;
            }
        };
    }

    @Override
    public int size() { return size; }

    @Override
    public Map<String, Object> get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }
        return cache.computeIfAbsent(index, loader::apply);
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        // Materialize for map since loader is opaque
        List<Map<String, Object>> items = new ArrayList<>(size);
        for (int i = 0; i < size; i++) items.add(get(i));
        List<Map<String, Object>> mapped = new ArrayList<>(size);
        for (Map<String, Object> ex : items) mapped.add(fn.apply(ex));
        return new MapDataset(mapped);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Map<String, Object>> filtered = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            Map<String, Object> ex = get(i);
            if (predicate.test(ex)) filtered.add(ex);
        }
        return new MapDataset(filtered);
    }

    @Override
    public Dataset shuffle(long seed) {
        List<Map<String, Object>> items = new ArrayList<>(size);
        for (int i = 0; i < size; i++) items.add(get(i));
        Collections.shuffle(items, new Random(seed));
        return new MapDataset(items);
    }

    @Override
    public Dataset select(int[] indices) {
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < size) selected.add(get(idx));
        }
        return new MapDataset(selected);
    }
}
