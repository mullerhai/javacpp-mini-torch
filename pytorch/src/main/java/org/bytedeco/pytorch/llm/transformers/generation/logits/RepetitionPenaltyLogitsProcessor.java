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
 * HF-style repetition penalty. For each previously emitted token:
 * <pre>{@code
 *   logits[t] *= penalty       if logits[t] > 0
 *   logits[t] /= penalty       if logits[t] < 0
 * }</pre>
 *
 * <p>Penalty == 1 is a no-op. We process the per-batch unique tokens row by
 * row for clarity (the same algorithm HF uses internally).
 */
public final class RepetitionPenaltyLogitsProcessor extends LogitsProcessor {

    private final float penalty;

    public RepetitionPenaltyLogitsProcessor(float penalty) {
        if (penalty <= 0.0f || Float.isNaN(penalty)) {
            throw new IllegalArgumentException("penalty must be positive, got " + penalty);
        }
        this.penalty = penalty;
    }

    public float penalty() { return penalty; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (penalty == 1.0f) return scores;
        if (inputIds == null) return scores;

        long[] shape = inputIds.shape();
        int batchSize = (int) shape[0];
        int seqLen = (int) shape[1];
        if (seqLen <= 1) return scores;

        Tensor result = scores.clone();
        for (int b = 0; b < batchSize; b++) {
            // collect unique tokens (HF uses set() on the row).
            java.util.Set<Long> seen = new java.util.HashSet<>();
            for (int s = 0; s < seqLen; s++) {
                long t = inputIds.select(0, b).select(0, s).item().toLong();
                seen.add(t);
            }
            // Apply penalty to logits at each seen token id.
            for (Long tid : seen) {
                int t = tid.intValue();
                float cur = result.select(0, b).select(-1, t).item().toFloat();
                float next;
                if (cur > 0.0f) {
                    next = cur / penalty;
                } else {
                    next = cur * penalty;
                }
                // In-place update via .fill_ on a single-element view
                Tensor slot = result.select(0, b).select(-1, t);
                slot.fill_(new org.bytedeco.pytorch.Scalar(next));
            }
        }
        return result;
    }
}
