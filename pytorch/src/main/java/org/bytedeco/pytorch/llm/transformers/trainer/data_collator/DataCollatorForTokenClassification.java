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
 * Collator for token-classification tasks (NER, POS tagging).
 *
 * <p>Stacks input token ids and label ids, pads to the longest sequence
 * in the batch, and sets labels for padding tokens to -100.
 */
public final class DataCollatorForTokenClassification implements DataCollator {

    private final int padLabelId;

    public DataCollatorForTokenClassification(int padLabelId) {
        this.padLabelId = padLabelId;
    }

    public DataCollatorForTokenClassification() { this(-100); }

    @Override
    public List<Map<String, Object>> collate_batch(List<Map<String, Object>> features) {
        if (features.isEmpty()) return List.of();

        int maxLen = 0;
        for (Map<String, Object> f : features) {
            Object ids = f.get("input_ids");
            if (ids instanceof int[] a) maxLen = Math.max(maxLen, a.length);
            if (ids instanceof long[] a) maxLen = Math.max(maxLen, a.length);
        }

        int bs = features.size();
        long[][] inputIds = new long[bs][maxLen];
        long[][] labelIds = new long[bs][maxLen];
        long[][] attentionMask = new long[bs][maxLen];

        for (int i = 0; i < bs; i++) {
            Map<String, Object> f = features.get(i);
            int[] ids = getIntArray(f.get("input_ids"));
            int[] labs = getIntArray(f.get("labels"));
            int len = ids != null ? ids.length : 0;
            for (int j = 0; j < len && j < maxLen; j++) {
                inputIds[i][j] = ids[j];
                labelIds[i][j] = labs != null && j < labs.length ? labs[j] : padLabelId;
                attentionMask[i][j] = 1;
            }
            for (int j = len; j < maxLen; j++) {
                labelIds[i][j] = padLabelId;
            }
        }

        return List.of(Map.of(
                "input_ids", inputIds,
                "labels", labelIds,
                "attention_mask", attentionMask
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
