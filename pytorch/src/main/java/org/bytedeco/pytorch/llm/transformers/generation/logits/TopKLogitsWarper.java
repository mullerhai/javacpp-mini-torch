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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * HF-style Top-K filtering: keep the K largest logits per row, mask the rest
 * to {@code -inf}.
 */
public final class TopKLogitsWarper extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final int topK;
    private final int minTokensToKeep;

    public TopKLogitsWarper(int topK) {
        this(topK, 1);
    }

    public TopKLogitsWarper(int topK, int minTokensToKeep) {
        if (topK < 0) {
            throw new IllegalArgumentException("topK must be non-negative, got " + topK);
        }
        this.topK = topK;
        this.minTokensToKeep = Math.max(1, minTokensToKeep);
    }

    public int topK() { return topK; }
    public int minTokensToKeep() { return minTokensToKeep; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (topK == 0) return scores;
        long[] shape = scores.shape();
        int vocab = (int) shape[shape.length - 1];
        int k = Math.min(topK, vocab);
        if (k >= vocab) return scores;

        // top-k along last dim (descending). We use topk() and pick the values.
        org.bytedeco.pytorch.T_TensorTensor_T topkPair = scores.topk((long) k, /*dim=*/shape.length - 1, /*largest=*/true, /*sorted=*/true);
        Tensor topkValues = topkPair.get0();
        Tensor kthValues = topkValues.select(/*dim=*/shape.length - 1, /*index=*/k - 1).unsqueeze(-1);

        // Mask: scores < kth_value → filter
        Tensor mask = scores.lt(kthValues);
        return scores.masked_fill(mask, new org.bytedeco.pytorch.Scalar(FILTER_VALUE));
    }
}
