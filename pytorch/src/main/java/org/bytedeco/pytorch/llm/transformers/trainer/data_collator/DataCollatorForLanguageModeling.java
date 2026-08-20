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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Collator for causal / masked language modeling tasks.
 *
 * <p>Stacks token ids, optionally masks a fraction of tokens (default 15%),
 * and creates corresponding labels (positions set to -100 for masked/unmasked tokens
 * that should be ignored in cross-entropy loss).
 */
public final class DataCollatorForLanguageModeling implements DataCollator {

    private final int vocabSize;
    private final float maskProbability;
    private final Random rng;

    /**
     * @param maskProbability fraction of tokens to mask (e.g. 0.15 for 15%); 0 to disable masking
     */
    public DataCollatorForLanguageModeling(int vocabSize, float maskProbability) {
        this.vocabSize = vocabSize;
        this.maskProbability = maskProbability;
        this.rng = new Random();
    }

    public DataCollatorForLanguageModeling(int vocabSize) {
        this(vocabSize, 0.15f);
    }

    @Override
    public List<Map<String, Object>> collate_batch(List<Map<String, Object>> features) {
        if (features.isEmpty()) return List.of();

        // Find max length
        int maxLen = 0;
        for (Map<String, Object> f : features) {
            Object ids = f.get("input_ids");
            if (ids instanceof int[] arr) maxLen = Math.max(maxLen, arr.length);
            else if (ids instanceof long[] arr) maxLen = Math.max(maxLen, arr.length);
        }
        if (maxLen == 0) return features;

        int batchSize = features.size();
        long[][] inputIds = new long[batchSize][maxLen];
        long[][] labels  = new long[batchSize][maxLen];
        long padId = 0; // default pad token

        for (int i = 0; i < batchSize; i++) {
            Map<String, Object> f = features.get(i);
            int[] ids = getIntArray(f.get("input_ids"));
            if (ids == null) ids = new int[0];

            int len = Math.min(ids.length, maxLen);
            for (int j = 0; j < len; j++) {
                inputIds[i][j] = ids[j];
                labels[i][j]   = ids[j];
            }
            // Pad positions
            for (int j = len; j < maxLen; j++) {
                inputIds[i][j] = padId;
                labels[i][j]   = -100; // ignore in CE loss
            }

            // Optional masking: replace 15% of non-pad tokens with [MASK]
            if (maskProbability > 0 && len > 0) {
                int maskCount = Math.max(1, (int) (len * maskProbability));
                List<Integer> positions = new ArrayList<>();
                for (int j = 0; j < len; j++) positions.add(j);
                java.util.Collections.shuffle(positions, rng);
                for (int k = 0; k < maskCount && k < positions.size(); k++) {
                    int pos = positions.get(k);
                    inputIds[i][pos] = 4; // assume token_id 4 = [MASK] (HF convention)
                    labels[i][pos] = ids[pos]; // label is the true token
                }
            }
        }

        // Return batch as list of one map (mirrors HF convention)
        return List.of(Map.of(
                "input_ids", inputIds,
                "labels", labels
        ));
    }

    private static int[] getIntArray(Object o) {
        if (o instanceof int[] a) return a;
        if (o instanceof long[] a) {
            int[] r = new int[a.length];
            for (int i = 0; i < a.length; i++) r[i] = (int) a[i];
            return r;
        }
        return null;
    }
}
