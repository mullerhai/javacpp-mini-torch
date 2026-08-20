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
 * Output for image classification models.
 * Mimics HuggingFace: transformers.modeling_outputs.ImageClassifierOutput
 */
public class ImageClassifierOutput extends ModelOutput {
    public final Tensor logits;
    public final Tensor loss;
    public final BaseModelOutput hiddenStates;
    public final Tensor attentions;

    public ImageClassifierOutput(Tensor logits) {
        this(logits, null, null, null);
    }

    public ImageClassifierOutput(Tensor logits, Tensor loss,
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

    /** Get top-k predicted class ids and their scores. */
    public java.util.List<java.util.Map.Entry<Integer, Double>> topK(int k) {
        if (logits == null) return java.util.Collections.emptyList();
        Tensor flat = logits.reshape(-1);
        org.bytedeco.pytorch.T_TensorTensor_T topk = flat.topk((long)Math.min(k, (int)flat.size(0)), 0, true, true);
        java.util.List<java.util.Map.Entry<Integer, Double>> result = new java.util.ArrayList<>();
        for (int i = 0; i < k && i < topk.get0().numel(); i++) {
            int id = (int) topk.get1().select(0, i).item().toLong();
            double score = topk.get0().select(0, i).item().toDouble();
            result.add(java.util.Map.entry(id, score));
        }
        topk.get0().close();
        topk.get1().close();
        return result;
    }
}
