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
import org.bytedeco.pytorch.llm.transformers.generation.cache.Cache;

import java.util.List;
import java.util.Map;

/**
 * Base output for encoder/decoder models returning last hidden states.
 * Mimics HuggingFace: transformers.modeling_utils.BaseModelOutput
 */
public class BaseModelOutput extends ModelOutput {
    public final Tensor lastHiddenState;
    public final List<Tensor> hiddenStates;
    public final List<Tensor> attentions;
    public final Cache pastKeyValues;
    public final boolean hasPastKeyValues;

    public BaseModelOutput(Tensor lastHiddenState) {
        this(lastHiddenState, null, null, null, false);
    }

    public BaseModelOutput(Tensor lastHiddenState, List<Tensor> hiddenStates,
                           List<Tensor> attentions, Cache pastKeyValues, boolean hasPastKeyValues) {
        super();
        this.lastHiddenState = lastHiddenState;
        this.hiddenStates = hiddenStates != null ? List.copyOf(hiddenStates) : null;
        this.attentions = attentions != null ? List.copyOf(attentions) : null;
        this.pastKeyValues = pastKeyValues;
        this.hasPastKeyValues = hasPastKeyValues;
    }

    public boolean hasHiddenStates() { return hiddenStates != null && !hiddenStates.isEmpty(); }
    public boolean hasAttentions() { return attentions != null && !attentions.isEmpty(); }
    public boolean hasPastKeyValues() { return hasPastKeyValues && pastKeyValues != null; }

    public int batchSize() { return lastHiddenState != null ? (int)lastHiddenState.size(0) : 0; }
    public int seqLen() { return lastHiddenState != null ? (int)lastHiddenState.size(1) : 0; }
    public int hiddenSize() { return lastHiddenState != null ? (int)lastHiddenState.size(lastHiddenState.dim()-1) : 0; }
}
