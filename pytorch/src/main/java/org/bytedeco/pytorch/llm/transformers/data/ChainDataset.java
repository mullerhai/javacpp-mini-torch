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
 * Chain of multiple datasets processed sequentially.
 *
 * <p>Unlike {@link ConcatenatedDataset} which interleaves child datasets,
 * {@code ChainDataset} applies a transformation chain to the same underlying data.
 *
 * <p>This is a utility wrapper for applying a pipeline of transforms.
 */
public class ChainDataset extends Dataset {

    private final List<Dataset> chain;
    private Dataset current;

    /**
     * Build a chain of datasets by applying transforms in order.
     *
     * @param datasets first dataset, then subsequent transform functions
     */
    public ChainDataset(List<Dataset> datasets) {
        if (datasets == null || datasets.isEmpty()) {
            throw new IllegalArgumentException("At least one dataset required");
        }
        this.chain = new ArrayList<>(datasets);
        this.current = chain.get(0);
    }

    /**
     * Apply another transform to the chain.
     *
     * @param fn the transform to apply
     * @return this (fluent)
     */
    public ChainDataset then(java.util.function.Function<Dataset, Dataset> fn) {
        current = fn.apply(current);
        return this;
    }

    /** Build and return the final dataset. */
    public Dataset build() {
        return current;
    }

    @Override
    public int size() { return current.size(); }

    @Override
    public Map<String, Object> get(int index) { return current.get(index); }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        return current.map(fn);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        return current.filter(predicate);
    }

    @Override
    public Dataset shuffle(long seed) {
        return current.shuffle(seed);
    }

    @Override
    public Dataset select(int[] indices) {
        return current.select(indices);
    }
}
