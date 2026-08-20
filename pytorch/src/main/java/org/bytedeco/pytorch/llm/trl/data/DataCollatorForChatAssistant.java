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
import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Conversational collator for TRL {@code assistant_only_loss}.
 *
 * <p>Python counterpart: SFTTrainer internal path when the dataset is
 * conversational and {@code SFTConfig.assistant_only_loss=True} (generation
 * mask from {@code {% generation %}} / assistant turns).
 *
 * <p>Each row must contain a {@code messages} list of {@code {role, content}}.
 * Prompt tokens get label {@code -100}; assistant (generation) tokens keep
 * their input ids.
 */
public final class DataCollatorForChatAssistant implements TrlDataCollator {

    public static final long DEFAULT_IGNORE_INDEX = -100L;

    private final FastTokenizer tokenizer;
    private final int maxLength;
    private final long ignoreIndex;
    private final boolean addGenerationPrompt;

    public DataCollatorForChatAssistant(FastTokenizer tokenizer, int maxLength) {
        this(tokenizer, maxLength, DEFAULT_IGNORE_INDEX, false);
    }

    public DataCollatorForChatAssistant(FastTokenizer tokenizer, int maxLength,
                                        long ignoreIndex, boolean addGenerationPrompt) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.maxLength = maxLength > 0 ? maxLength : 1024;
        this.ignoreIndex = ignoreIndex;
        this.addGenerationPrompt = addGenerationPrompt;
    }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        if (features.isEmpty()) {
            throw new IllegalArgumentException("empty batch");
        }
        List<int[]> idsList = new ArrayList<>(features.size());
        List<int[]> maskList = new ArrayList<>(features.size());
        List<int[]> labelList = new ArrayList<>(features.size());
        int max = 0;
        for (Map<String, Object> row : features) {
            EncodedRow er = encodeRow(row);
            idsList.add(er.ids);
            maskList.add(er.mask);
            labelList.add(er.labels);
            max = Math.max(max, er.ids.length);
        }
        int padId = tokenizer.padTokenId();
        long[][] ids = new long[features.size()][max];
        long[][] mask = new long[features.size()][max];
        long[][] labels = new long[features.size()][max];
        for (int i = 0; i < features.size(); i++) {
            int[] a = idsList.get(i);
            int[] m = maskList.get(i);
            int[] l = labelList.get(i);
            for (int j = 0; j < max; j++) {
                if (j < a.length) {
                    ids[i][j] = a[j];
                    mask[i][j] = m[j];
                    labels[i][j] = l[j];
                } else {
                    ids[i][j] = padId;
                    mask[i][j] = 0;
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

    @SuppressWarnings("unchecked")
    EncodedRow encodeRow(Map<String, Object> row) {
        Object msgs = row.get("messages");
        if (!(msgs instanceof List<?> list) || list.isEmpty()) {
            // Fall back to a plain text / input_ids row.
            if (row.get("input_ids") instanceof int[] already) {
                return fromIds(already, row);
            }
            String text = String.valueOf(row.getOrDefault("text", ""));
            Encoding enc = tokenizer.encode(text, true);
            int[] ids = truncate(enc.ids());
            int[] labels = ids.clone();
            return new EncodedRow(ids, ones(ids.length), labels);
        }
        List<Map<String, Object>> messages = new ArrayList<>();
        for (Object o : list) {
            if (o instanceof Map<?, ?> m) {
                messages.add((Map<String, Object>) m);
            }
        }
        ChatTemplate engine = tokenizer.chatTemplateEngine();
        ChatTemplate.ChatTemplateResult result;
        if (engine != null && engine.flavor() == ChatTemplate.Flavor.CUSTOM) {
            result = engine.applyCustom(messages, addGenerationPrompt);
        } else {
            ChatTemplate flavor = engine != null ? engine : ChatTemplate.qwen();
            String text = flavor.applyObject(messages, addGenerationPrompt);
            // Treat every assistant turn as a generation span via a Qwen-style re-render.
            result = ChatTemplate.custom(
                    flavor.flavor() == ChatTemplate.Flavor.LLAMA3
                            ? "<|start_header_id|>" : "<|im_start|>")
                    .applyCustom(messages, addGenerationPrompt);
            result = new ChatTemplate.ChatTemplateResult(text, result.generationCharSpans());
        }
        Encoding enc = tokenizer.encode(result.text(), false);
        int[] ids = truncate(enc.ids());
        int[] mask = ones(ids.length);
        int[] labels = new int[ids.length];
        java.util.Arrays.fill(labels, (int) ignoreIndex);
        markGenerationTokens(result.text(), ids, labels, result.generationCharSpans());
        return new EncodedRow(ids, mask, labels);
    }

    /**
     * Map character-level generation spans onto token ids by greedily decoding
     * prefixes (works even when {@link Encoding} offsets are empty).
     */
    private void markGenerationTokens(String text, int[] ids, int[] labels, List<int[]> spans) {
        if (spans == null || spans.isEmpty() || ids.length == 0) {
            // Last-resort: train on the full sequence (legacy SFT).
            System.arraycopy(ids, 0, labels, 0, ids.length);
            return;
        }
        int[] charOfTokenEnd = new int[ids.length];
        String acc = "";
        for (int i = 0; i < ids.length; i++) {
            acc = tokenizer.decode(java.util.Arrays.copyOfRange(ids, 0, i + 1), false);
            charOfTokenEnd[i] = acc.length();
        }
        for (int[] span : spans) {
            if (span == null || span.length < 2) continue;
            int start = span[0];
            int end = span[1];
            for (int i = 0; i < ids.length; i++) {
                int tokStart = i == 0 ? 0 : charOfTokenEnd[i - 1];
                int tokEnd = charOfTokenEnd[i];
                if (tokEnd > start && tokStart < end) {
                    labels[i] = ids[i];
                }
            }
        }
    }

    private EncodedRow fromIds(int[] already, Map<String, Object> row) {
        int[] ids = truncate(already);
        int[] mask = row.get("attention_mask") instanceof int[] m ? truncate(m) : ones(ids.length);
        int[] labels;
        if (row.get("labels") instanceof int[] l) {
            labels = truncate(l);
        } else {
            labels = ids.clone();
        }
        return new EncodedRow(ids, mask, labels);
    }

    private int[] truncate(int[] ids) {
        if (ids == null) return new int[0];
        if (ids.length <= maxLength) return ids;
        int[] t = new int[maxLength];
        System.arraycopy(ids, 0, t, 0, maxLength);
        return t;
    }

    private static int[] ones(int n) {
        int[] a = new int[n];
        java.util.Arrays.fill(a, 1);
        return a;
    }

    static final class EncodedRow {
        final int[] ids;
        final int[] mask;
        final int[] labels;
        EncodedRow(int[] ids, int[] mask, int[] labels) {
            this.ids = ids;
            this.mask = mask;
            this.labels = labels;
        }
    }
}
