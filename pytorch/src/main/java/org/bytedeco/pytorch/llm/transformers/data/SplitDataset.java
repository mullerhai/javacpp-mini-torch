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
 * Holds a train-test (or train-val) split of a dataset.
 *
 * <p>Mirrors HuggingFace {@code DatasetDict} used for train/test splits.
 */
public class SplitDataset extends Dataset {

    private final Dataset train;
    private final Dataset test;

    public SplitDataset(Dataset train, Dataset test) {
        this.train = Objects.requireNonNull(train, "train");
        this.test = Objects.requireNonNull(test, "test");
    }

    public Dataset getTrain() { return train; }
    public Dataset getTest() { return test; }

    @Override
    public int size() { return train.size() + test.size(); }

    @Override
    public Map<String, Object> get(int index) {
        int trainSize = train.size();
        if (index < trainSize) return train.get(index);
        return test.get(index - trainSize);
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        return new SplitDataset(train.map(fn), test.map(fn));
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        return new SplitDataset(train.filter(predicate), test.filter(predicate));
    }

    @Override
    public Dataset shuffle(long seed) {
        long seed1 = seed;
        long seed2 = seed + 1;
        return new SplitDataset(train.shuffle(seed1), test.shuffle(seed2));
    }

    @Override
    public Dataset select(int[] indices) {
        int trainSize = train.size();
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < size()) selected.add(get(idx));
        }
        return new MapDataset(selected);
    }
}
