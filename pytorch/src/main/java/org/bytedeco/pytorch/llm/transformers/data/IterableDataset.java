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
 * Dataset backed by a lazy {@link Iterator} — processed one element at a time.
 *
 * <p>Mirrors HuggingFace {@code IterableDataset}.
 */
public class IterableDataset extends Dataset {

    private final Iterator<Map<String, Object>> iterator;
    private List<Map<String, Object>> cached;
    private int index;

    public IterableDataset(Iterator<Map<String, Object>> iterator) {
        this.iterator = Objects.requireNonNull(iterator, "iterator");
        this.cached = new ArrayList<>();
        this.index = 0;
    }

    @Override
    public int size() {
        // Exhaust iterator if needed
        while (iterator.hasNext()) {
            cached.add(iterator.next());
        }
        return cached.size();
    }

    @Override
    public Map<String, Object> get(int idx) {
        while (idx >= cached.size() && iterator.hasNext()) {
            cached.add(iterator.next());
        }
        if (idx >= cached.size()) {
            throw new IndexOutOfBoundsException("Index: " + idx + ", cached size: " + cached.size());
        }
        return Collections.unmodifiableMap(cached.get(idx));
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        return new IterableDataset(
                new Iterator<Map<String, Object>>() {
                    @Override public boolean hasNext() {
                        return index < cached.size() || iterator.hasNext();
                    }
                    @Override public Map<String, Object> next() {
                        while (index >= cached.size() && iterator.hasNext()) {
                            cached.add(iterator.next());
                        }
                        return fn.apply(cached.get(index++));
                    }
                });
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        return new IterableDataset(
                new Iterator<Map<String, Object>>() {
                    private Map<String, Object> next;

                    { advance(); }

                    private void advance() {
                        next = null;
                        while ((next == null || !predicate.test(next)) &&
                               (index < cached.size() || iterator.hasNext())) {
                            while (index >= cached.size() && iterator.hasNext()) {
                                cached.add(iterator.next());
                            }
                            if (index < cached.size()) {
                                next = cached.get(index++);
                            }
                        }
                        if (next != null && !predicate.test(next)) next = null;
                    }

                    @Override public boolean hasNext() { return next != null; }
                    @Override public Map<String, Object> next() {
                        Map<String, Object> cur = next;
                        advance();
                        return cur;
                    }
                });
    }

    @Override
    public Dataset shuffle(long seed) {
        size(); // fully materialize
        List<Map<String, Object>> copy = new ArrayList<>(cached);
        Collections.shuffle(copy, new Random(seed));
        return new MapDataset(copy);
    }

    @Override
    public Dataset select(int[] indices) {
        size(); // fully materialize
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < cached.size()) {
                selected.add(cached.get(idx));
            }
        }
        return new MapDataset(selected);
    }
}
