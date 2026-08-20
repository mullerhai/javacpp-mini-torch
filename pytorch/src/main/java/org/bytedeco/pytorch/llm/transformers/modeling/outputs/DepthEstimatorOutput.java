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
 * Output for depth estimation models.
 * Mimics HuggingFace: transformers.modeling_outputs.DepthEstimatorOutput
 */
public class DepthEstimatorOutput extends org.bytedeco.pytorch.llm.transformers.modeling.output.ModelOutput {
    public final org.bytedeco.pytorch.Tensor predicted_depth;
    public final org.bytedeco.pytorch.Tensor loss;
    public final org.bytedeco.pytorch.Tensor hidden_states;

    public DepthEstimatorOutput(org.bytedeco.pytorch.Tensor predicted_depth) {
        this(null, predicted_depth, null);
    }

    public DepthEstimatorOutput(org.bytedeco.pytorch.Tensor loss,
                                org.bytedeco.pytorch.Tensor predicted_depth,
                                org.bytedeco.pytorch.Tensor hidden_states) {
        super();
        this.loss = loss;
        this.predicted_depth = predicted_depth;
        this.hidden_states = hidden_states;
    }

    public boolean hasLoss() { return loss != null && loss.defined(); }
    public boolean hasHiddenStates() { return hidden_states != null && hidden_states.defined(); }
}
