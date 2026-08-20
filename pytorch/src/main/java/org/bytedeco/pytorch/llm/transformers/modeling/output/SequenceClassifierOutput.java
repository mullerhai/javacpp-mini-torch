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
 * Output for sequence classification (text classification, NLI, etc.).
 * Mimics HuggingFace: transformers.modeling_outputs.SequenceClassifierOutput
 */
public class SequenceClassifierOutput extends ModelOutput {
    public final Tensor logits;
    public final Tensor loss;
    public final BaseModelOutput hiddenStates;
    public final Tensor attentions;

    public SequenceClassifierOutput(Tensor logits) {
        this(logits, null, null, null);
    }

    public SequenceClassifierOutput(Tensor logits, Tensor loss,
                                    BaseModelOutput hiddenStates, Tensor attentions) {
        super();
        this.logits = logits;
        this.loss = loss;
        this.hiddenStates = hiddenStates;
        this.attentions = attentions;
    }

    public boolean hasLoss() { return loss != null; }

    /** Get predicted class id (argmax over logits). */
    public int predictedClassId() {
        if (logits == null) return -1;
        return (int) logits.argmax(new org.bytedeco.pytorch.LongOptional((long)(logits.dim() - 1)), false).item().toLong();
    }

    /** Get score for a specific class. */
    public double scoreForClass(int classId) {
        if (logits == null) return Double.NaN;
        return logits.select(0, 0).select(1, (long) classId).item().toDouble();
    }

    /** Softmax probabilities over logits. */
    public Tensor probabilities() {
        if (logits == null) return null;
        return org.bytedeco.pytorch.global.torch.softmax(logits, logits.dim() - 1);
    }
}
