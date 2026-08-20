/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
package org.bytedeco.pytorch.llm.transformers.modeling_utils;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * 1-D convolution implemented as a Linear layer with transposed weights.
 *
 * <p>Used by GPT-2 which stores QKV projections as Conv1D
 * (equivalent to a Linear with weight shape [out_features, in_features]).
 *
 * <p>Reference: HuggingFace transformers
 * {@code modeling_utils.Conv1D}.
 */
public final class Conv1D extends Module {

    private final LinearImpl linear;

    /**
     * Build a Conv1D mapping {@code nf} inputs to {@code nx} outputs.
     *
     * @param nx number of output features
     * @param nf number of input features
     */
    public Conv1D(int nx, int nf) {
        super("Conv1D");
        // Conv1D is a linear map with weight [nx, nf] and no bias.
        LinearOptions opts = new LinearOptions(nx, nf);
        this.linear = register_module("weight", new LinearImpl(opts));
    }

    @Override
    public Tensor forward(Tensor input) {
        return linear.forward(input);
    }

    public LinearImpl linear() { return linear; }
}
