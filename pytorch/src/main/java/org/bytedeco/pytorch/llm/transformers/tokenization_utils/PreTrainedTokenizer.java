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

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.List;

/**
 * Abstract pretrained tokenizer mirroring the HF transformers tokenizers API.
 *
 * <p>Reference: HuggingFace transformers
 * {@code tokenization_utils_base.PreTrainedTokenizer}.
 */
public abstract class PreTrainedTokenizer extends PreTrainedTokenizerBase {

    protected PreTrainedTokenizer() {}

    /**
     * Encode a text string to token IDs.
     *
     * @param text           input text
     * @param addSpecialTokens whether to add BOS/EOS/pad tokens
     * @return token IDs
     */
    public abstract int[] encode(String text, boolean addSpecialTokens);

    /**
     * Decode token IDs back to text.
     *
     * @param ids             token IDs
     * @param skipSpecialTokens whether to skip special tokens
     * @return decoded text
     */
    public abstract String decode(int[] ids, boolean skipSpecialTokens);

    /**
     * Apply a chat template to a list of messages.
     *
     * @param messages          list of maps with "role" and "content"
     * @param tokenize          whether to also tokenize the result
     * @param addGenerationPrompt whether to append a generation prompt
     * @return formatted string or tokenized IDs
     */
    public String apply_chat_template(List<?> messages, boolean tokenize, boolean addGenerationPrompt) {
        throw new UnsupportedOperationException("Subclasses must implement apply_chat_template");
    }

    /**
     * Pad a batch encoding.
     *
     * @param encodings  list of encodings
     * @param maxLength  target length
     * @param padding    padding strategy
     * @param truncation truncation strategy
     * @return padded batch encoding
     */
    public BatchEncoding pad(BatchEncoding encodings, int maxLength, String padding, TruncationStrategy truncation) {
        throw new UnsupportedOperationException("Subclasses must implement pad");
    }

    /**
     * Truncate sequences.
     *
     * @param encoding input encoding
     * @param maxLen   maximum length
     * @param strategy truncation strategy
     * @return truncated encoding
     */
    public EncodedInput truncate(EncodedInput encoding, int maxLen, TruncationStrategy strategy) {
        throw new UnsupportedOperationException("Subclasses must implement truncate");
    }

    /**
     * Get the vocabulary size.
     */
    public abstract int vocab_size();
}
