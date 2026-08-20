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
 * A slice / window of a parent dataset.
 *
 * <p>Provides efficient indexing into a contiguous range of a parent dataset
 * without copying underlying data.
 */
public class SubDataset extends Dataset {

    private final Dataset parent;
    private final int offset;
    private final int length;

    /**
     * Slice from offset to offset+length.
     *
     * @param parent the underlying dataset
     * @param offset starting index in the parent
     * @param length number of elements to include
     */
    public SubDataset(Dataset parent, int offset, int length) {
        this.parent = Objects.requireNonNull(parent, "parent");
        if (offset < 0 || offset + length > parent.size()) {
            throw new IllegalArgumentException(
                    "offset=" + offset + " length=" + length + " out of bounds for parent size=" + parent.size());
        }
        this.offset = offset;
        this.length = length;
    }

    @Override
    public int size() { return length; }

    @Override
    public Map<String, Object> get(int index) {
        if (index < 0 || index >= length) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + length);
        }
        return parent.get(offset + index);
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        // Materialize and rewrap
        List<Map<String, Object>> items = new ArrayList<>(length);
        for (int i = 0; i < length; i++) items.add(parent.get(offset + i));
        List<Map<String, Object>> mapped = new ArrayList<>(length);
        for (Map<String, Object> ex : items) mapped.add(fn.apply(ex));
        return new MapDataset(mapped);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Map<String, Object>> filtered = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            Map<String, Object> ex = parent.get(offset + i);
            if (predicate.test(ex)) filtered.add(ex);
        }
        return new MapDataset(filtered);
    }

    @Override
    public Dataset shuffle(long seed) {
        List<Map<String, Object>> items = new ArrayList<>(length);
        for (int i = 0; i < length; i++) items.add(parent.get(offset + i));
        Collections.shuffle(items, new Random(seed));
        return new MapDataset(items);
    }

    @Override
    public Dataset select(int[] indices) {
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < length) selected.add(parent.get(offset + idx));
        }
        return new MapDataset(selected);
    }
}
