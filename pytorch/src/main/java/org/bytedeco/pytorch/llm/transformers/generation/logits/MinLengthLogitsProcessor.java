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
package org.bytedeco.pytorch.llm.transformers.generation.logits;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * Forbid EOS until at least {@code minLength} tokens have been generated
 * (HF {@code MinLengthLogitsProcessor}). For all EOS token ids, mask logits
 * to {@code -inf} until the sequence reaches {@code minLength}.
 */
public final class MinLengthLogitsProcessor extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final int minLength;
    private final int[] eosTokenIds;

    public MinLengthLogitsProcessor(int minLength, int[] eosTokenIds) {
        if (minLength < 0) {
            throw new IllegalArgumentException("minLength must be >= 0, got " + minLength);
        }
        if (eosTokenIds == null || eosTokenIds.length == 0) {
            throw new IllegalArgumentException("eosTokenIds must contain at least one id");
        }
        this.minLength = minLength;
        this.eosTokenIds = eosTokenIds.clone();
    }

    public int minLength() { return minLength; }
    public int[] eosTokenIds() { return eosTokenIds.clone(); }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (inputIds == null) return scores;
        long[] shape = inputIds.shape();
        int batchSize = (int) shape[0];
        int seqLen = (int) shape[1];
        if (seqLen >= minLength) return scores;

        Tensor result = scores.clone();
        for (int b = 0; b < batchSize; b++) {
            for (int eos : eosTokenIds) {
                result.select(0, b).select(-1, eos).fill_(new Scalar(FILTER_VALUE));
            }
        }
        return result;
    }
}
