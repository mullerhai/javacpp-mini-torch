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
import java.util.stream.IntStream;

/**
 * Abstract base for datasets compatible with HuggingFace-style data loading.
 *
 * <p>Each element is a {@link Map} of feature name to feature value (e.g.
 * {@code "input_ids" -> long[]}, {@code "attention_mask" -> int[]}).
 *
 * <p>This mirrors the HF {@code Dataset} contract used by {@code Trainer}.
 */
public abstract class Dataset implements Iterable<Map<String, Object>> {

    /**
     * Number of examples in this dataset.
     */
    public abstract int size();

    /**
     * Get a single example by index.
     *
     * @param index zero-based index
     * @return the example as a feature map
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public abstract Map<String, Object> get(int index);

    /**
     * Apply a transformation to every example.
     *
     * @param fn a function from example to example
     * @return a new dataset with the transformed data
     */
    public abstract Dataset map(Function<Map<String, Object>, Map<String, Object>> fn);

    /**
     * Filter examples.
     *
     * @param predicate keeps examples where predicate returns true
     * @return a new filtered dataset
     */
    public abstract Dataset filter(Predicate<Map<String, Object>> predicate);

    /**
     * Shuffle the dataset with a given seed.
     *
     * @param seed random seed for reproducibility
     * @return a new shuffled dataset
     */
    public abstract Dataset shuffle(long seed);

    /**
     * Select a subset of indices.
     *
     * @param indices the indices to keep
     * @return a new dataset with only the selected examples
     */
    public abstract Dataset select(int[] indices);

    /**
     * Split this dataset into train and test subsets.
     *
     * @param testRatio fraction of data to use for the test set (e.g. 0.1)
     * @param seed      random seed for reproducibility
     * @return a {@link SplitDataset} containing train and test parts
     */
    public SplitDataset train_test_split(double testRatio, long seed) {
        int total = size();
        int testSize = Math.max(1, (int) Math.round(total * testRatio));
        int trainSize = total - testSize;

        List<Integer> idx = IntStream.range(0, total)
                .boxed()
                .collect(Collectors.toList());
        Collections.shuffle(idx, new Random(seed));

        int[] trainIndices = idx.subList(0, trainSize).stream().mapToInt(Integer::intValue).toArray();
        int[] testIndices = idx.subList(trainSize, total).stream().mapToInt(Integer::intValue).toArray();

        return new SplitDataset(select(trainIndices), select(testIndices));
    }

    @Override
    public Iterator<Map<String, Object>> iterator() {
        return new Iterator<Map<String, Object>>() {
            private int pos = 0;
            @Override public boolean hasNext() { return pos < size(); }
            @Override public Map<String, Object> next() { return get(pos++); }
        };
    }
}
