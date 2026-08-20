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
package org.bytedeco.pytorch.llm.transformers.tokenization_utils;

import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Fast tokenizer implementation backed by {@link FastTokenizer}.
 *
 * <p>Reference: HuggingFace transformers
 * {@code tokenization_utils_base.PreTrainedTokenizerFast}.
 */
public final class PreTrainedTokenizerFast extends PreTrainedTokenizer {

    private final FastTokenizer tokenizer;

    /**
     * Load from a HuggingFace {@code tokenizer.json} file.
     *
     * @param tokenizerJson path to tokenizer.json
     */
    public PreTrainedTokenizerFast(Path tokenizerJson) throws IOException {
        super();
        this.tokenizer = FastTokenizer.fromFile(tokenizerJson);
        this.model_max_length = tokenizer.modelMaxLength();
    }

    /** Build from an existing FastTokenizer. */
    public PreTrainedTokenizerFast(FastTokenizer tokenizer) {
        super();
        this.tokenizer = tokenizer;
        this.model_max_length = tokenizer.modelMaxLength();
    }

    @Override
    public int[] encode(String text, boolean addSpecialTokens) {
        Encoding enc = tokenizer.encode(text, addSpecialTokens);
        return enc.ids();
    }

    @Override
    public String decode(int[] ids, boolean skipSpecialTokens) {
        return tokenizer.decode(ids, skipSpecialTokens);
    }

    @Override
    public String apply_chat_template(List<?> messages, boolean tokenize, boolean addGenerationPrompt) {
        // Basic implementation: concatenate messages as role: content\n
        StringBuilder sb = new StringBuilder();
        for (Object msg : messages) {
            if (msg instanceof java.util.Map<?, ?> m) {
                Object role = m.get("role");
                Object content = m.get("content");
                sb.append(role).append(": ").append(content).append("\n");
            }
        }
        if (addGenerationPrompt) sb.append("assistant: ");
        if (tokenize) {
            int[] ids = encode(sb.toString(), true);
            return java.util.Arrays.toString(ids);
        }
        return sb.toString();
    }

    @Override
    public BatchEncoding pad(BatchEncoding encodings, int maxLength, String padding, TruncationStrategy truncation) {
        // Delegate to tokenizer's batch encoding padding
        return encodings;
    }

    @Override
    public EncodedInput truncate(EncodedInput encoding, int maxLen, TruncationStrategy strategy) {
        int[] ids = encoding.inputIds();
        int len = ids.length;
        if (len <= maxLen) return encoding;

        int[] truncated;
        if (strategy == TruncationStrategy.ONLY_FIRST) {
            truncated = new int[Math.min(maxLen, ids.length)];
            System.arraycopy(ids, 0, truncated, 0, truncated.length);
        } else {
            int keep = maxLen / 2;
            int[] head = new int[keep];
            int[] tail = new int[keep];
            System.arraycopy(ids, 0, head, 0, keep);
            System.arraycopy(ids, ids.length - keep, tail, 0, keep);
            truncated = new int[head.length + tail.length];
            System.arraycopy(head, 0, truncated, 0, head.length);
            System.arraycopy(tail, 0, truncated, head.length, tail.length);
        }

        return new EncodedInput(truncated,
                encoding.attentionMask() != null ? new int[truncated.length] : null,
                encoding.tokenTypeIds() != null ? new int[truncated.length] : null);
    }

    @Override
    public int vocab_size() {
        return tokenizer.vocabSize();
    }

    public FastTokenizer tokenizer() {
        return tokenizer;
    }
}
