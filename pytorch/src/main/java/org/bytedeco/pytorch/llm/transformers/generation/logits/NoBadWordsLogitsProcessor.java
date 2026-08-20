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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * HF {@code NoBadWordsLogitsProcessor}: forbid sequences of tokens that would
 * complete a known-bad phrase.
 *
 * <p>For each bad phrase id list, we look at the most recent {@code len-1}
 * generated ids; if they match the prefix, we suppress the final id to
 * {@code -inf}.
 *
 * <p>Inputs are pre-filtered: phrases longer than the current generation are
 * skipped (no possible completion yet).
 */
public final class NoBadWordsLogitsProcessor extends LogitsProcessor {

    private static final float FILTER_VALUE = Float.NEGATIVE_INFINITY;
    private final List<int[]> badWordsIds;

    public NoBadWordsLogitsProcessor(List<int[]> badWordsIds) {
        this.badWordsIds = badWordsIds;
    }

    public List<int[]> badWordsIds() { return badWordsIds; }

    @Override
    public Tensor call(Tensor inputIds, Tensor scores) {
        if (badWordsIds == null || badWordsIds.isEmpty() || inputIds == null) return scores;
        long[] shape = inputIds.shape();
        int batchSize = (int) shape[0];
        int seqLen = (int) shape[1];

        Tensor result = scores.clone();
        for (int b = 0; b < batchSize; b++) {
            // Collect last (maxPhraseLen - 1) tokens for prefix matching.
            int maxPhrase = 0;
            for (int[] w : badWordsIds) maxPhrase = Math.max(maxPhrase, w.length);
            if (maxPhrase <= 1) continue;

            int lookback = Math.min(seqLen, maxPhrase - 1);
            long[] recent = new long[lookback];
            for (int s = 0; s < lookback; s++) {
                recent[s] = inputIds.select(0, b).select(0, seqLen - lookback + s).item().toLong();
            }

            for (int[] bad : badWordsIds) {
                if (bad.length > seqLen + 1) continue; // can't have completed
                if (bad.length <= 1) {
                    // Single forbidden token
                    result.select(0, b).select(-1, bad[0]).fill_(new org.bytedeco.pytorch.Scalar(FILTER_VALUE));
                    continue;
                }
                int prefixLen = bad.length - 1;
                boolean match = prefixLen <= lookback;
                for (int k = 0; match && k < prefixLen; k++) {
                    if (recent[lookback - prefixLen + k] != bad[k]) match = false;
                }
                if (match) {
                    result.select(0, b).select(-1, bad[bad.length - 1]).fill_(new org.bytedeco.pytorch.Scalar(FILTER_VALUE));
                }
            }
        }
        return result;
    }
}
