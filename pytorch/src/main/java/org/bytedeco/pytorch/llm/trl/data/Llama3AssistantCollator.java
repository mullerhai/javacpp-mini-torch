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
 * 1:1 port of the Llama-3 persona-chatbot notebook {@code collate_fn}:
 * scan for {@code <|start_header_id|>assistant<|end_header_id|>\\n} and unmask
 * tokens until {@code <|eot_id|>} (inclusive).
 */
public final class Llama3AssistantCollator implements TrlDataCollator {

    public static final String ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n";
    public static final String EOT = "<|eot_id|>";

    private final FastTokenizer tokenizer;
    private final int maxLength;
    private final long ignoreIndex;

    public Llama3AssistantCollator(FastTokenizer tokenizer, int maxLength) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.maxLength = maxLength > 0 ? maxLength : 8192;
        this.ignoreIndex = -100L;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        int[] assistantTokens = tokenizer.encode(ASSISTANT_PREFIX, false).ids();
        int[] eotTokens = tokenizer.encode(EOT, false).ids();
        List<int[]> idsList = new ArrayList<>();
        List<int[]> maskList = new ArrayList<>();
        List<int[]> labelList = new ArrayList<>();
        int max = 0;
        for (Map<String, Object> example : features) {
            Object msgs = example.get("messages");
            StringBuilder prompt = new StringBuilder("<|begin_of_text|>");
            if (msgs instanceof List<?> list) {
                for (Object o : list) {
                    if (!(o instanceof Map<?, ?> msg)) continue;
                    String role = String.valueOf(msg.getOrDefault("role", "user"));
                    String content = String.valueOf(msg.getOrDefault("content", "")).strip();
                    prompt.append("<|start_header_id|>").append(role)
                            .append("<|end_header_id|>\n").append(content).append("<|eot_id|>");
                }
            }
            String text = prompt.toString().strip();
            int[] inputIds = tokenizer.encode(text, false).ids();
            if (inputIds.length > maxLength) {
                int[] t = new int[maxLength];
                System.arraycopy(inputIds, 0, t, 0, maxLength);
                inputIds = t;
            }
            int[] attention = new int[inputIds.length];
            java.util.Arrays.fill(attention, 1);
            int[] labels = new int[inputIds.length];
            java.util.Arrays.fill(labels, (int) ignoreIndex);
            int i = 0;
            while (i <= inputIds.length - assistantTokens.length) {
                if (matchAt(inputIds, i, assistantTokens)) {
                    int start = i + assistantTokens.length;
                    int end = start;
                    while (end <= inputIds.length - eotTokens.length) {
                        if (matchAt(inputIds, end, eotTokens)) break;
                        end++;
                    }
                    for (int j = start; j < end && j < labels.length; j++) labels[j] = inputIds[j];
                    for (int j = end; j < end + eotTokens.length && j < labels.length; j++) {
                        labels[j] = inputIds[j];
                    }
                    i = end + eotTokens.length;
                } else {
                    i++;
                }
            }
            idsList.add(inputIds);
            maskList.add(attention);
            labelList.add(labels);
            max = Math.max(max, inputIds.length);
        }
        int padId = tokenizer.padTokenId();
        long[][] ids = new long[features.size()][max];
        long[][] mask = new long[features.size()][max];
        long[][] labels = new long[features.size()][max];
        for (int r = 0; r < features.size(); r++) {
            int[] a = idsList.get(r);
            int[] m = maskList.get(r);
            int[] l = labelList.get(r);
            for (int j = 0; j < max; j++) {
                if (j < a.length) {
                    ids[r][j] = a[j];
                    mask[r][j] = m[j];
                    labels[r][j] = l[j];
                } else {
                    ids[r][j] = padId;
                    labels[r][j] = ignoreIndex;
                }
            }
        }
        Map<String, Tensor> batch = new LinkedHashMap<>();
        batch.put("input_ids", tensor(ids));
        batch.put("attention_mask", tensor(mask));
        batch.put("labels", tensor(labels));
        return batch;
    }

    static boolean matchAt(int[] hay, int i, int[] needle) {
        if (i < 0 || i + needle.length > hay.length) return false;
        for (int k = 0; k < needle.length; k++) {
            if (hay[i + k] != needle[k]) return false;
        }
        return true;
    }
}
