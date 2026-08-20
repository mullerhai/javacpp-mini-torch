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

/**
 * Encoder-side repetition penalty (HF
 * {@code EncoderRepetitionPenaltyLogitsProcessor}).
 *
 * <p>Penalises tokens that already appear in the encoder input — useful for
 * encoder-decoder models that would otherwise echo source tokens.
 *
 * <p>If {@code encoderInputIds} is {@code null} we degrade to a no-op rather
 * than throwing — callers may pass null when only decoder-side penalty is
 * wanted.
 */
public final class EncoderRepetitionPenaltyLogitsProcessor extends LogitsProcessor {

    private final float penalty;
    private final Tensor encoderInputIds;

    public EncoderRepetitionPenaltyLogitsProcessor(float penalty, Tensor encoderInputIds) {
        if (penalty <= 0.0f || Float.isNaN(penalty)) {
            throw new IllegalArgumentException("penalty must be positive, got " + penalty);
        }
        this.penalty = penalty;
        this.encoderInputIds = encoderInputIds;
    }

    public float penalty() { return penalty; }
    public Tensor encoderInputIds() { return encoderInputIds; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (penalty == 1.0f || encoderInputIds == null) return scores;
        long[] shape = encoderInputIds.shape();
        int batchSize = (int) shape[0];
        int encLen = (int) shape[1];
        if (encLen == 0) return scores;

        Tensor result = scores.clone();
        for (int b = 0; b < batchSize; b++) {
            java.util.Set<Long> seen = new java.util.HashSet<>();
            for (int s = 0; s < encLen; s++) {
                long t = encoderInputIds.select(0, b).select(0, s).item().toLong();
                seen.add(t);
            }
            for (Long tid : seen) {
                int t = tid.intValue();
                float cur = result.select(0, b).select(-1, t).item().toFloat();
                float next = (cur > 0.0f) ? (cur / penalty) : (cur * penalty);
                result.select(0, b).select(-1, t).fill_(new Scalar(next));
            }
        }
        return result;
    }
}
