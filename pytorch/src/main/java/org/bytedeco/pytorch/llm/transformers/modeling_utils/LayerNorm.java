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

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;

/**
 * Layer normalization wrapper.
 *
 * <p>Reference: HuggingFace transformers
 * {@code modeling_utils.LayerNorm}.
 */
public final class LayerNorm extends Module {

    private final LayerNormImpl inner;

    /**
     * Build a LayerNorm with the given normalized shape and epsilon.
     *
     * @param normalizedShape dimension of the features to normalize
     * @param eps             epsilon added to the denominator
     */
    public LayerNorm(long normalizedShape, double eps) {
        super("LayerNorm");
        LayerNormOptions opts = new LayerNormOptions(new long[]{normalizedShape})
                .eps(eps)
                .elementwise_affine(true);
        this.inner = register_module("inner", new LayerNormImpl(opts));
    }

    @Override
    public org.bytedeco.pytorch.Tensor forward(org.bytedeco.pytorch.Tensor input) {
        return inner.forward(input);
    }

    public LayerNormImpl inner() { return inner; }
}
