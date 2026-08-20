/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * You may not use this file except in compliance with the License.
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

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Dataset loaded from Apache Arrow IPC ({@code .arrow}) files.
 *
 * <p>Arrow IPC format enables zero-copy reads of large datasets without
 * loading everything into heap memory.
 *
 * <p>This is a stub — full implementation depends on an Arrow IPC reader.
 */
public class ArrowDataset extends Dataset {

    private final List<Map<String, Object>> rows;
    private final Path path;

    /**
     * Load an Arrow IPC file.
     *
     * @param path path to the Arrow IPC file
     */
    public ArrowDataset(Path path) {
        this.path = Objects.requireNonNull(path, "path");
        // TODO: Replace with real Arrow IPC reader (org.apache.arrow.*)
        this.rows = new ArrayList<>();
        // Stub: no data loaded until real reader is wired
    }

    @Override
    public int size() { return rows.size(); }

    @Override
    public Map<String, Object> get(int index) {
        if (index < 0 || index >= rows.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + rows.size());
        }
        return Collections.unmodifiableMap(rows.get(index));
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        List<Map<String, Object>> mapped = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) mapped.add(fn.apply(row));
        return new MapDataset(mapped);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Map<String, Object>> filtered = new ArrayList<>();
        for (Map<String, Object> row : rows) {
            if (predicate.test(row)) filtered.add(row);
        }
        return new MapDataset(filtered);
    }

    @Override
    public Dataset shuffle(long seed) {
        List<Map<String, Object>> copy = new ArrayList<>(rows);
        Collections.shuffle(copy, new Random(seed));
        return new MapDataset(copy);
    }

    @Override
    public Dataset select(int[] indices) {
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < rows.size()) selected.add(rows.get(idx));
        }
        return new MapDataset(selected);
    }
}
