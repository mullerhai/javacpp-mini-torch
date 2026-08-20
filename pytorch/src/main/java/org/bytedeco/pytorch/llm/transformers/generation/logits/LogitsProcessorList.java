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

import java.util.ArrayList;

/**
 * Container that applies a sequence of {@link LogitsProcessor}s left-to-right.
 *
 * <p>Mirrors HF's {@code LogitsProcessorList}. Each {@link #call} clones the
 * input {@code scores} so chained processors don't mutate each other's
 * outputs unintentionally (HF does the same).
 *
 * <p>Provides fluent {@code addTemperature}, {@code addTopK}, {@code addTopP},
 * etc. that are no-ops when their parameter equals the identity (e.g. temp=1,
 * topK=0, topP=1, repetition=1) — matching HF behaviour.
 */
public final class LogitsProcessorList extends ArrayList<LogitsProcessor> {

    private static final long serialVersionUID = 1L;

    public LogitsProcessorList() {}

    public LogitsProcessorList(Iterable<LogitsProcessor> seed) {
        for (LogitsProcessor p : seed) add(p);
    }

    public Tensor call(Tensor inputIds, Tensor scores) {
        Tensor current = scores;
        for (LogitsProcessor p : this) {
            current = p.call(inputIds, current);
        }
        return current;
    }

    // ----- Fluent builders (HF-style, identity-skipping) -----

    public LogitsProcessorList addTemperature(float temperature) {
        if (temperature != 1.0f) {
            add(new TemperatureLogitsWarper(temperature));
        }
        return this;
    }

    public LogitsProcessorList addTopK(int topK) {
        if (topK > 0) {
            add(new TopKLogitsWarper(topK));
        }
        return this;
    }

    public LogitsProcessorList addTopP(float topP) {
        if (topP < 1.0f) {
            add(new TopPLogitsWarper(topP));
        }
        return this;
    }

    public LogitsProcessorList addMinP(float minP) {
        if (minP > 0.0f) {
            add(new MinPLogitsWarper(minP));
        }
        return this;
    }

    public LogitsProcessorList addRepetitionPenalty(float penalty) {
        if (penalty != 1.0f) {
            add(new RepetitionPenaltyLogitsProcessor(penalty));
        }
        return this;
    }

    public LogitsProcessorList addEncoderRepetitionPenalty(float penalty) {
        if (penalty != 1.0f) {
            // encoderInputIds=null is acceptable — degrades to repetition penalty on input only
            add(new EncoderRepetitionPenaltyLogitsProcessor(penalty, null));
        }
        return this;
    }

    public LogitsProcessorList addNoBadWords(java.util.List<int[]> badWordsIds) {
        if (badWordsIds != null && !badWordsIds.isEmpty()) {
            add(new NoBadWordsLogitsProcessor(badWordsIds));
        }
        return this;
    }

    public LogitsProcessorList addMinLength(int minLength, int[] eosTokenIds) {
        if (minLength > 0 && eosTokenIds != null && eosTokenIds.length > 0) {
            add(new MinLengthLogitsProcessor(minLength, eosTokenIds));
        }
        return this;
    }

    public LogitsProcessorList addForceTokens(java.util.Map<Integer, Integer> map) {
        if (map != null && !map.isEmpty()) {
            add(new ForcedBOSTokenLogitsProcessor(map));
        }
        return this;
    }

    public LogitsProcessorList addSuppressionTokens(int[] suppressTokens) {
        if (suppressTokens != null && suppressTokens.length > 0) {
            add(new SuppressTokensLogitsProcessor(suppressTokens));
        }
        return this;
    }
}
