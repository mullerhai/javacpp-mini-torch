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

/**
 * Dictionary of named datasets (e.g. train / validation / test splits).
 *
 * <p>Mirrors HuggingFace {@code DatasetDict}.
 *
 * <p>Typical keys: {@code "train"}, {@code "validation"}, {@code "test"}.
 */
public class DatasetDict {

    private final Map<String, Dataset> splits;

    public DatasetDict(Map<String, Dataset> splits) {
        this.splits = new LinkedHashMap<>(Objects.requireNonNull(splits, "splits"));
    }

    public static DatasetDict of(String key, Dataset value) {
        return new DatasetDict(Collections.singletonMap(key, value));
    }

    public static DatasetDict of(String key1, Dataset value1, String key2, Dataset value2) {
        Map<String, Dataset> m = new LinkedHashMap<>();
        m.put(key1, value1);
        m.put(key2, value2);
        return new DatasetDict(m);
    }

    /**
     * Get the dataset for a named split.
     *
     * @param key e.g. "train", "validation", "test"
     * @return the dataset, or null if key not found
     */
    public Dataset get(String key) {
        return splits.get(key);
    }

    /**
     * Get the dataset for a named split.
     *
     * @param key e.g. "train"
     * @return the dataset
     * @throws NoSuchElementException if key not found
     */
    public Dataset require(String key) {
        Dataset ds = splits.get(key);
        if (ds == null) throw new NoSuchElementException("No split named: " + key);
        return ds;
    }

    /** Returns true if this dict contains the given split key. */
    public boolean contains(String key) { return splits.containsKey(key); }

    /** Returns all split names. */
    public Set<String> keys() { return Collections.unmodifiableSet(splits.keySet()); }

    /** Returns all datasets. */
    public Collection<Dataset> values() { return Collections.unmodifiableCollection(splits.values()); }

    /** Returns the number of splits. */
    public int size() { return splits.size(); }

    /** Convenience: returns the "train" split. */
    public Dataset train() { return require("train"); }

    /** Convenience: returns the "test" split. */
    public Dataset test() { return require("test"); }

    /** Convenience: returns the "validation" split. */
    public Dataset validation() { return require("validation"); }
}
