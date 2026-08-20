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
package org.bytedeco.pytorch.llm.transformers.modeling.output;

import org.bytedeco.pytorch.Tensor;

/**
 * Output for extractive question answering.
 * Mimics HuggingFace: transformers.modeling_outputs.QuestionAnsweringOutput
 */
public class QuestionAnsweringOutput extends ModelOutput {
    public final Tensor start_logits;
    public final Tensor end_logits;
    public final Tensor loss;
    public final BaseModelOutput hiddenStates;
    public final Tensor attentions;

    public QuestionAnsweringOutput(Tensor start_logits, Tensor end_logits) {
        this(start_logits, end_logits, null, null, null);
    }

    public QuestionAnsweringOutput(Tensor start_logits, Tensor end_logits, Tensor loss,
                                    BaseModelOutput hiddenStates, Tensor attentions) {
        super();
        this.start_logits = start_logits;
        this.end_logits = end_logits;
        this.loss = loss;
        this.hiddenStates = hiddenStates;
        this.attentions = attentions;
    }

    public boolean hasLoss() { return loss != null; }

    /**
     * Extract answer span from start/end logits.
     * Returns [startChar, endChar] positions in the original text.
     */
    public int[] extractAnswerSpan(int contextOffset) {
        if (start_logits == null || end_logits == null) return new int[]{-1, -1};
        int seqLen = (int) start_logits.size(0);
        double maxScore = Double.NEGATIVE_INFINITY;
        int bestStart = 0, bestEnd = 0;
        for (int s = 0; s < seqLen; s++) {
            for (int e = s; e < Math.min(s + 30, seqLen); e++) {
                double score = start_logits.select(0, s).item().toDouble()
                             + end_logits.select(0, e).item().toDouble();
                if (score > maxScore) {
                    maxScore = score;
                    bestStart = s;
                    bestEnd = e;
                }
            }
        }
        return new int[]{bestStart + contextOffset, bestEnd + contextOffset};
    }
}
