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
 * Concatenation of multiple datasets along the example axis.
 *
 * <p>The resulting dataset has size equal to the sum of all constituent datasets.
 * Iteration proceeds through each child dataset in order.
 */
public class ConcatenatedDataset extends Dataset {

    private final List<Dataset> datasets;
    private final int totalSize;

    public ConcatenatedDataset(List<Dataset> datasets) {
        this.datasets = new ArrayList<>(Objects.requireNonNull(datasets, "datasets"));
        int total = 0;
        for (Dataset d : this.datasets) total += d.size();
        this.totalSize = total;
    }

    public ConcatenatedDataset(Dataset... datasets) {
        this(Arrays.asList(datasets));
    }

    @Override
    public int size() { return totalSize; }

    @Override
    public Map<String, Object> get(int index) {
        int remaining = index;
        for (Dataset ds : datasets) {
            int sz = ds.size();
            if (remaining < sz) {
                return ds.get(remaining);
            }
            remaining -= sz;
        }
        throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + totalSize);
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        List<Dataset> mapped = datasets.stream()
                .map(d -> d.map(fn))
                .collect(Collectors.toList());
        return new ConcatenatedDataset(mapped);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Dataset> filtered = datasets.stream()
                .map(d -> d.filter(predicate))
                .collect(Collectors.toList());
        return new ConcatenatedDataset(filtered);
    }

    @Override
    public Dataset shuffle(long seed) {
        List<Map<String, Object>> all = new ArrayList<>(totalSize);
        for (Dataset d : datasets) {
            for (Map<String, Object> ex : d) all.add(ex);
        }
        Collections.shuffle(all, new Random(seed));
        return new MapDataset(all);
    }

    @Override
    public Dataset select(int[] indices) {
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < totalSize) selected.add(get(idx));
        }
        return new MapDataset(selected);
    }
}
