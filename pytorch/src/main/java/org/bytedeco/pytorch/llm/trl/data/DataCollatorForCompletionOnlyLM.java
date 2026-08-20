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
package org.bytedeco.pytorch.llm.trl.data;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * HuggingFace TRL {@code DataCollatorForCompletionOnlyLM}.
 *
 * <p>Masks every token before (and not including) the last occurrence of
 * {@code responseTemplate} so the causal-LM loss is computed only on the
 * completion.
 */
public final class DataCollatorForCompletionOnlyLM implements TrlDataCollator {

    private final FastTokenizer tokenizer;
    private final String responseTemplate;
    private final int maxLength;
    private final long ignoreIndex;

    public DataCollatorForCompletionOnlyLM(FastTokenizer tokenizer, String responseTemplate, int maxLength) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.responseTemplate = responseTemplate == null ? "" : responseTemplate;
        this.maxLength = maxLength > 0 ? maxLength : 1024;
        this.ignoreIndex = -100L;
    }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        int[] tmpl = responseTemplate.isEmpty()
                ? new int[0]
                : tokenizer.encode(responseTemplate, false).ids();
        List<int[]> idsList = new ArrayList<>();
        List<int[]> labelList = new ArrayList<>();
        int max = 0;
        for (Map<String, Object> row : features) {
            int[] ids;
            if (row.get("input_ids") instanceof int[] already) {
                ids = already;
            } else {
                String text = String.valueOf(row.getOrDefault("text",
                        row.getOrDefault("completion", "")));
                ids = tokenizer.encode(text, true).ids();
            }
            if (ids.length > maxLength) {
                int[] t = new int[maxLength];
                System.arraycopy(ids, 0, t, 0, maxLength);
                ids = t;
            }
            int[] labels = ids.clone();
            int cut = indexOfSubsequence(ids, tmpl);
            int maskUntil = cut < 0 ? 0 : cut + tmpl.length;
            for (int i = 0; i < maskUntil && i < labels.length; i++) {
                labels[i] = (int) ignoreIndex;
            }
            idsList.add(ids);
            labelList.add(labels);
            max = Math.max(max, ids.length);
        }
        int padId = tokenizer.padTokenId();
        long[][] ids = new long[features.size()][max];
        long[][] mask = new long[features.size()][max];
        long[][] labels = new long[features.size()][max];
        for (int i = 0; i < features.size(); i++) {
            int[] a = idsList.get(i);
            int[] l = labelList.get(i);
            for (int j = 0; j < max; j++) {
                if (j < a.length) {
                    ids[i][j] = a[j];
                    mask[i][j] = 1;
                    labels[i][j] = l[j];
                } else {
                    ids[i][j] = padId;
                    labels[i][j] = ignoreIndex;
                }
            }
        }
        Map<String, Tensor> batch = new LinkedHashMap<>();
        batch.put("input_ids", tensor(ids));
        batch.put("attention_mask", tensor(mask));
        batch.put("labels", tensor(labels));
        return batch;
    }

    static int indexOfSubsequence(int[] hay, int[] needle) {
        if (needle == null || needle.length == 0) return -1;
        int last = -1;
        for (int i = 0; i + needle.length <= hay.length; i++) {
            boolean ok = true;
            for (int j = 0; j < needle.length; j++) {
                if (hay[i + j] != needle[j]) { ok = false; break; }
            }
            if (ok) last = i;
        }
        return last;
    }
}
