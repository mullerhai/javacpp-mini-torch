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
package org.bytedeco.pytorch.llm.transformers.modeling.outputs;

/**
 * Output for Vision-Encoder-Decoder models (e.g. ViT-GPT2, Donut).
 * Mimics HuggingFace: transformers.modeling_outputs.VisionEncoderDecoderOutput
 */
public class VisionEncoderDecoderOutput extends org.bytedeco.pytorch.llm.transformers.modeling.output.ModelOutput {
    public final org.bytedeco.pytorch.Tensor loss;
    public final org.bytedeco.pytorch.Tensor logits;
    public final org.bytedeco.pytorch.Tensor decoder_hidden_states;
    public final org.bytedeco.pytorch.Tensor encoder_last_hidden_state;

    public VisionEncoderDecoderOutput(org.bytedeco.pytorch.Tensor logits,
                                     org.bytedeco.pytorch.Tensor decoder_hidden_states) {
        this(null, logits, decoder_hidden_states, null);
    }

    public VisionEncoderDecoderOutput(org.bytedeco.pytorch.Tensor loss,
                                     org.bytedeco.pytorch.Tensor logits,
                                     org.bytedeco.pytorch.Tensor decoder_hidden_states,
                                     org.bytedeco.pytorch.Tensor encoder_last_hidden_state) {
        super();
        this.loss = loss;
        this.logits = logits;
        this.decoder_hidden_states = decoder_hidden_states;
        this.encoder_last_hidden_state = encoder_last_hidden_state;
    }

    public boolean hasLoss() { return loss != null && loss.defined(); }
    public boolean hasDecoderHiddenStates() { return decoder_hidden_states != null && decoder_hidden_states.defined(); }
    public boolean hasEncoderHiddenState() { return encoder_last_hidden_state != null && encoder_last_hidden_state.defined(); }
}
