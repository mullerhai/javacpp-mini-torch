/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
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

import org.bytedeco.pytorch.Tensor;

/**
 * HF {@code SuppressTokensLogitsProcessor}: hard-mask a fixed list of token
 * ids to {@code -inf} on every step.
 */
public final class SuppressTokensLogitsProcessor extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final int[] suppressTokens;

    public SuppressTokensLogitsProcessor(int[] suppressTokens) {
        if (suppressTokens == null || suppressTokens.length == 0) {
            throw new IllegalArgumentException("suppressTokens must contain at least one id");
        }
        this.suppressTokens = suppressTokens.clone();
    }

    public int[] suppressTokens() { return suppressTokens.clone(); }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (inputIds == null) return scores;
        long[] shape = inputIds.shape();
        int batchSize = (int) shape[0];
        Tensor result = scores.clone();
        for (int b = 0; b < batchSize; b++) {
            Tensor row = result.select(0, b);
            for (int t : suppressTokens) {
                if (t < 0 || t >= row.size(-1)) continue;
                row.select(-1, t).fill_(new org.bytedeco.pytorch.Scalar(FILTER_VALUE));
            }
        }
        return result;
    }
}
