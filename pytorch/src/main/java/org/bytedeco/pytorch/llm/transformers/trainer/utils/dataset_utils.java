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
package org.bytedeco.pytorch.llm.transformers.trainer.utils;

import java.util.List;
import java.util.Map;

/**
 * Dataset helpers mirroring HF's {@code transformers.trainer_utils.distributed_utils}.
 *
 * <p>Provides {@link LengthGroupedSampler}, {@link SortByLength},
 * and {@link remove_unused_columns}.
 */
public final class dataset_utils {

    private dataset_utils() {}

    /**
     * Stub sampler that groups batches by sequence length to minimise padding.
     *
     * @param size         dataset size
     * @param lengths      per-example lengths
     * @param batchSize    target batch size
     * @return list of shuffled indices grouped by length
     */
    public static List<Integer> LengthGroupedSampler(int size, List<Integer> lengths, int batchSize) {
        // TODO: implement length-grouped sampling
        return new java.util.ArrayList<>(java.util.stream.IntStream.range(0, size).boxed().toList());
    }

    /**
     * Sort examples by length for bucketing in batching.
     *
     * @param features list of feature maps
     * @param lengthsKey key of the length field in each feature map
     * @return features sorted by length (descending)
     */
    public static List<Map<String, Object>> SortByLength(List<Map<String, Object>> features, String lengthsKey) {
        List<Map<String, Object>> sorted = new java.util.ArrayList<>(features);
        sorted.sort((a, b) -> {
            Object la = a.get(lengthsKey);
            Object lb = b.get(lengthsKey);
            if (la == null || lb == null) return 0;
            int ia = (la instanceof Number) ? ((Number) la).intValue() : 0;
            int ib = (lb instanceof Number) ? ((Number) lb).intValue() : 0;
            return Integer.compare(ib, ia); // descending
        });
        return sorted;
    }

    /**
     * Remove columns that are not used by the model forward signature.
     *
     * @param features   input feature maps
     * @param usedNames  column names the model actually consumes
     * @return features with unused columns stripped
     */
    public static List<Map<String, Object>> remove_unused_columns(
            List<Map<String, Object>> features, List<String> usedNames) {
        if (usedNames == null || usedNames.isEmpty()) return features;
        java.util.Set<String> keep = new java.util.HashSet<>(usedNames);
        return features.stream().map(row -> {
            Map<String, Object> filtered = new java.util.LinkedHashMap<>();
            for (Map.Entry<String, Object> e : row.entrySet()) {
                if (keep.contains(e.getKey())) filtered.put(e.getKey(), e.getValue());
            }
            return filtered;
        }).toList();
    }
}
