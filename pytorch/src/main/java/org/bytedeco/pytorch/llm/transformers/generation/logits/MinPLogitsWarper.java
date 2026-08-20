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
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.global.torch;

/**
 * HF-style min-p sampling. Keep tokens whose probability is at least
 * {@code minP * max_prob}.
 */
public final class MinPLogitsWarper extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final float minP;
    private final int minTokensToKeep;

    public MinPLogitsWarper(float minP) {
        this(minP, 1);
    }

    public MinPLogitsWarper(float minP, int minTokensToKeep) {
        if (minP < 0.0f || minP > 1.0f) {
            throw new IllegalArgumentException("minP must be in [0, 1], got " + minP);
        }
        this.minP = minP;
        this.minTokensToKeep = Math.max(1, minTokensToKeep);
    }

    public float minP() { return minP; }
    public int minTokensToKeep() { return minTokensToKeep; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (minP <= 0.0f) return scores;
        long[] shape = scores.shape();
        int dim = shape.length - 1;

        // probs (fp32) and max-prob along dim
        Tensor probs = scores.softmax(dim).to(torch.ScalarType.Float);
        T_TensorTensor_T maxPair = probs.max(dim, /*keepdim=*/true);
        Tensor maxProbs = maxPair.get0();
        Tensor threshold = maxProbs.mul(new Scalar(minP));

        Tensor mask = probs.lt(threshold);
        // Keep at least minTokensToKeep tokens
        if (minTokensToKeep > 1) {
            // We zero the mask for the top minTokensToKeep tokens
            Tensor topk = probs.topk(minTokensToKeep, dim, /*largest=*/true, /*sorted=*/true).get1();
            Tensor keepMask = torch.zeros_like(mask);
            keepMask.scatter_(dim, topk, new Scalar(1.0f));
            mask = mask.logical_and(keepMask.logical_not());
        }
        return scores.masked_fill(mask.to(scores.dtype()), new Scalar(FILTER_VALUE));
    }
}
