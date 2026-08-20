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
 * Interleaves multiple datasets, cycling through each child in round-robin order.
 *
 * <p>Example: interleaving [A,B,C] and [X,Y] yields: A, X, B, Y, C, A, X, ...
 *
 * <p>This mirrors HuggingFace {@code interleave_datasets}.
 */
public class InterleaveDataset extends Dataset {

    private final List<Dataset> datasets;
    private final long seed;
    private final int totalSize;

    /**
     * @param datasets the datasets to interleave
     */
    public InterleaveDataset(List<Dataset> datasets) {
        this(datasets, -1);
    }

    /**
     * @param datasets the datasets to interleave
     * @param seed     random seed for shuffling (-1 = no shuffle)
     */
    public InterleaveDataset(List<Dataset> datasets, long seed) {
        this.datasets = new ArrayList<>(Objects.requireNonNull(datasets, "datasets"));
        if (this.datasets.isEmpty()) throw new IllegalArgumentException("datasets must not be empty");
        this.seed = seed;
        // Estimate total size from smallest child (cyclic)
        this.totalSize = this.datasets.stream().mapToInt(Dataset::size).min().orElse(0) * this.datasets.size();
    }

    @Override
    public int size() { return totalSize; }

    @Override
    public Map<String, Object> get(int index) {
        int n = datasets.size();
        int childIndex = index % n;
        int withinChild = index / n;
        if (withinChild >= datasets.get(childIndex).size()) {
            withinChild %= datasets.get(childIndex).size();
        }
        return datasets.get(childIndex).get(withinChild);
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        List<Dataset> mapped = new ArrayList<>();
        for (Dataset d : datasets) mapped.add(d.map(fn));
        return new InterleaveDataset(mapped, seed);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Dataset> filtered = new ArrayList<>();
        for (Dataset d : datasets) filtered.add(d.filter(predicate));
        return new InterleaveDataset(filtered, seed);
    }

    @Override
    public Dataset shuffle(long newSeed) {
        List<Dataset> shuffled = new ArrayList<>();
        for (Dataset d : datasets) shuffled.add(d.shuffle(newSeed));
        return new InterleaveDataset(shuffled, newSeed);
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
