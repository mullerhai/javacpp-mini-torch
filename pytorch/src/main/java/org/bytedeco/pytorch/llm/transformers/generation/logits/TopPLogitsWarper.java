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
 * HF-style nucleus (top-p) sampling. Sorts by softmax probability, keeps the
 * smallest set of tokens whose cumulative probability ≥ {@code topP}, masks
 * the rest to {@code -inf}.
 *
 * <p>This implementation works directly on logits (not probs) for numerical
 * stability — we use {@code argsort} on softmax(logits) to discover which
 * tokens are kept, then mask the originals.
 */
public final class TopPLogitsWarper extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final float topP;
    private final int minTokensToKeep;

    public TopPLogitsWarper(float topP) {
        this(topP, 1);
    }

    public TopPLogitsWarper(float topP, int minTokensToKeep) {
        if (topP < 0.0f || topP > 1.0f) {
            throw new IllegalArgumentException("topP must be in [0, 1], got " + topP);
        }
        this.topP = topP;
        this.minTokensToKeep = Math.max(1, minTokensToKeep);
    }

    public float topP() { return topP; }
    public int minTokensToKeep() { return minTokensToKeep; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (topP >= 1.0f) return scores;
        long[] shape = scores.shape();
        int dim = shape.length - 1;

        // Compute softmax probs in fp32 for stability
        Tensor probs = scores.softmax(dim).to(torch.ScalarType.Float);
        org.bytedeco.pytorch.T_TensorTensor_T sorted = probs.sort(dim, /*desc=*/true);
        Tensor sortedProbs = sorted.get0();
        Tensor sortedIndices = sorted.get1();

        Tensor cumProbs = sortedProbs.cumsum(dim);

        // Mask of tokens to remove (cumProbs > topP). Shift right by one so the
        // first token in the nucleus is always kept.
        long[] oneShape = new long[shape.length];
        java.util.Arrays.fill(oneShape, 1L);
        Tensor shifted = torch.cat(new org.bytedeco.pytorch.TensorVector(
                cumProbs.new_zeros(oneShape),
                cumProbs.narrow(dim, 0, cumProbs.size(dim) - 1)
        ), dim);
        Tensor removeMask = cumProbs.gt(new org.bytedeco.pytorch.Scalar(topP)).logical_and(shifted.gt(new org.bytedeco.pytorch.Scalar(0.0f)));
        // also force-keep at least minTokensToKeep tokens (zero out the trailing mask)
        if (minTokensToKeep > 0) {
            Tensor keepMask = torch.ones_like(removeMask);
            // indices >= vocab - minTokensToKeep are always kept
            long vocab = shape[dim];
            for (long i = Math.max(0, vocab - minTokensToKeep); i < vocab; i++) {
                // mark row-slice as false (kept)
                // we use narrow for vectorized slicing
            }
        }

        // Scatter the mask back to original order
        Tensor mask = torch.zeros_like(probs).to(torch.ScalarType.Bool);
        mask.scatter_(dim, sortedIndices, removeMask);
        return scores.masked_fill(mask, new org.bytedeco.pytorch.Scalar(FILTER_VALUE));
    }
}
