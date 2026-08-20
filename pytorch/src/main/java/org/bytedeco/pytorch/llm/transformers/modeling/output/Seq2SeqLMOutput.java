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
 * Output for sequence-to-sequence models (encoder-decoder).
 * Mimics HuggingFace: transformers.modeling_outputs.Seq2SeqLMOutput
 */
public class Seq2SeqLMOutput extends ModelOutput {
    public final Tensor logits;
    public final Tensor loss;
    public final BaseModelOutput encoderLastHiddenState;
    public final BaseModelOutput decoderHiddenStates;
    public final Tensor decoderAttentions;
    public final Tensor crossAttentions;
    public final Tensor encoderAttentions;

    public Seq2SeqLMOutput(Tensor logits) {
        this(logits, null, null, null, null, null, null);
    }

    public Seq2SeqLMOutput(Tensor logits, Tensor loss, BaseModelOutput encoderLastHiddenState,
                           BaseModelOutput decoderHiddenStates, Tensor decoderAttentions,
                           Tensor crossAttentions, Tensor encoderAttentions) {
        super();
        this.logits = logits;
        this.loss = loss;
        this.encoderLastHiddenState = encoderLastHiddenState;
        this.decoderHiddenStates = decoderHiddenStates;
        this.decoderAttentions = decoderAttentions;
        this.crossAttentions = crossAttentions;
        this.encoderAttentions = encoderAttentions;
    }

    public boolean hasLoss() { return loss != null; }
    public boolean hasEncoderHiddenStates() { return encoderLastHiddenState != null; }
    public boolean hasDecoderHiddenStates() { return decoderHiddenStates != null && decoderHiddenStates.hasHiddenStates(); }
}
