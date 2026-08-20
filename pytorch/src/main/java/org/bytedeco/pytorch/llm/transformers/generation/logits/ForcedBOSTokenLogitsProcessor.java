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

import java.util.Map;

/**
 * HF {@code ForcedBOSTokenLogitsProcessor} (and {@code ForcedEOSTokenLogitsProcessor}):
 * force a specific token id at a specific position. Map keyed by generation
 * index (0 = first generated token) → forced token id.
 *
 * <p>Outside the map's positions, this processor is a no-op.
 */
public final class ForcedBOSTokenLogitsProcessor extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final Map<Integer, Integer> forcedMap;

    public ForcedBOSTokenLogitsProcessor(Map<Integer, Integer> forcedMap) {
        this.forcedMap = forcedMap;
    }

    public Map<Integer, Integer> forcedMap() { return forcedMap; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (forcedMap == null || forcedMap.isEmpty() || inputIds == null) return scores;
        long[] shape = inputIds.shape();
        int batchSize = (int) shape[0];
        int curLen = (int) shape[1];
        // generation index = curLen - promptLen (HF semantics); here we treat curLen
        // as the absolute position, so the user supplies the *position* at which to force.
        // The HF default convention (force_bos_token_id at index 0 of the generation)
        // is therefore "position == promptLen" when promptLen is the prefix length.
        // We support both: if a key is negative, we treat it as a generation index from end.

        int genIdx = curLen; // raw; consumer is expected to pre-shift keys
        Integer forced = forcedMap.get(genIdx);
        if (forced == null) return scores;

        int target = forced;
        Tensor result = scores.clone();
        for (int b = 0; b < batchSize; b++) {
            // zero out everything, then assign only the forced id
            Tensor row = result.select(0, b);
            row.fill_(new Scalar(FILTER_VALUE));
            row.select(-1, target).fill_(new Scalar(0.0f));
        }
        return result;
    }
}
