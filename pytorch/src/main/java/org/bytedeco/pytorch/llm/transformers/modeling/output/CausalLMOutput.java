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
 * Output for causal language modeling (next-token prediction).
 * Mimics HuggingFace: transformers.modeling_outputs.CausalLMOutputWithPast
 */
public class CausalLMOutput extends ModelOutput {
    public final Tensor logits;
    public final Tensor loss;
    public final BaseModelOutput hiddenStates;
    public final Tensor attentions;

    public CausalLMOutput(Tensor logits) {
        this(logits, null, null, null);
    }

    public CausalLMOutput(Tensor logits, Tensor loss, BaseModelOutput hiddenStates, Tensor attentions) {
        super();
        this.logits = logits;
        this.loss = loss;
        this.hiddenStates = hiddenStates;
        this.attentions = attentions;
    }

    public boolean hasLoss() { return loss != null; }
    public boolean hasHiddenStates() { return hiddenStates != null && hiddenStates.hasHiddenStates(); }
    public boolean hasAttentions() { return attentions != null; }

    /** Extract the next-token logits at the last position. */
    public Tensor nextTokenLogits() {
        if (logits == null) return null;
        int dim = (int)logits.size(logits.dim() - 1);
        return logits.select(logits.dim() - 2, logits.size(logits.dim() - 2) - 1);
    }

    /** Top-k token ids for the next token. */
    public long[] topK(int k) {
        if (logits == null) return new long[0];
        Tensor nxt = nextTokenLogits();
        if (nxt == null) return new long[0];
        org.bytedeco.pytorch.T_TensorTensor_T topk = nxt.topk(k, 0, true, true);
        long[] ids = new long[k];
        for (int i = 0; i < k; i++) ids[i] = topk.get1().select(0, i).item().toLong();
        topk.get0().close();
        topk.get1().close();
        return ids;
    }
}
