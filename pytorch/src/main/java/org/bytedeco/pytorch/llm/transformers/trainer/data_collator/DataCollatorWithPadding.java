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
package org.bytedeco.pytorch.llm.transformers.trainer.data_collator;

import java.util.List;
import java.util.Map;

/**
 * Generic collator that pads all variable-length tensor fields to the batch maximum.
 *
 * <p>Scans all keys in the first feature map and pads each tensor field
 * (identified by being a 1-D int[] / long[] / float[]) to the max length in the batch.
 */
public final class DataCollatorWithPadding implements DataCollator {

    private final int padTokenId;
    private final int padLabelId;

    public DataCollatorWithPadding(int padTokenId, int padLabelId) {
        this.padTokenId = padTokenId;
        this.padLabelId = padLabelId;
    }

    public DataCollatorWithPadding() { this(0, -100); }

    @Override
    public List<Map<String, Object>> collate_batch(List<Map<String, Object>> features) {
        if (features.isEmpty()) return List.of();
        if (features.size() == 1) return features;

        Map<String, Object> first = features.get(0);

        // Compute max lengths per key
        java.util.Map<String, Integer> maxLens = new java.util.HashMap<>();
        for (Map<String, Object> f : features) {
            for (Map.Entry<String, Object> e : f.entrySet()) {
                String key = e.getKey();
                int len = arrayLength(e.getValue());
                maxLens.merge(key, len, Math::max);
            }
        }

        // Build output map
        java.util.Map<String, Object> batch = new java.util.LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : maxLens.entrySet()) {
            String key = e.getKey();
            int maxLen = e.getValue();
            boolean isLabel = key.toLowerCase().contains("label");
            long padId = isLabel ? padLabelId : padTokenId;

            Object firstVal = first.get(key);
            if (firstVal instanceof int[] || firstVal instanceof long[]) {
                long[][] out = new long[features.size()][maxLen];
                for (int i = 0; i < features.size(); i++) {
                    fill(out[i], features.get(i).get(key), padId);
                }
                batch.put(key, out);
            } else if (firstVal instanceof float[]) {
                float[][] out = new float[features.size()][maxLen];
                for (int i = 0; i < features.size(); i++) {
                    fillFloat(out[i], features.get(i).get(key));
                }
                batch.put(key, out);
            } else {
                batch.put(key, firstVal);
            }
        }

        return List.of(batch);
    }

    private static int arrayLength(Object o) {
        if (o instanceof Object[] a) return a.length;
        if (o instanceof int[] a) return a.length;
        if (o instanceof long[] a) return a.length;
        if (o instanceof float[] a) return a.length;
        return 0;
    }

    private static void fill(long[] out, Object src, long padId) {
        if (src instanceof int[] s) {
            int n = Math.min(s.length, out.length);
            for (int i = 0; i < n; i++) out[i] = s[i];
            for (int i = n; i < out.length; i++) out[i] = padId;
        } else if (src instanceof long[] s) {
            int n = Math.min(s.length, out.length);
            System.arraycopy(s, 0, out, 0, n);
            for (int i = n; i < out.length; i++) out[i] = padId;
        } else {
            for (int i = 0; i < out.length; i++) out[i] = padId;
        }
    }

    private static void fillFloat(float[] out, Object src) {
        if (src instanceof float[] s) {
            int n = Math.min(s.length, out.length);
            System.arraycopy(s, 0, out, 0, n);
        }
    }
}
