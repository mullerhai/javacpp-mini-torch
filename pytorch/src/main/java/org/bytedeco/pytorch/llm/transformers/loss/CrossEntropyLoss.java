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
package org.bytedeco.pytorch.llm.transformers.loss;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Standard cross-entropy loss (with optional label smoothing).
 *
 * <p>Wraps {@code torch.nn.functional.cross_entropy}.
 */
public class CrossEntropyLoss implements Loss {

    private final double labelSmoothing;
    private final long ignoreIndex;

    /**
     * Construct with default settings (no label smoothing, ignore_index=-100).
     */
    public CrossEntropyLoss() {
        this(0.0, -100L);
    }

    /**
     * Construct with label smoothing.
     *
     * @param labelSmoothing smoothing factor in [0, 1]
     */
    public CrossEntropyLoss(double labelSmoothing) {
        this(labelSmoothing, -100L);
    }

    /**
     * Full constructor.
     *
     * @param labelSmoothing smoothing factor
     * @param ignoreIndex    label value to ignore in loss (e.g. -100)
     */
    public CrossEntropyLoss(double labelSmoothing, long ignoreIndex) {
        this.labelSmoothing = labelSmoothing;
        this.ignoreIndex = ignoreIndex;
    }

    @Override
    public Tensor compute(Tensor logits, Tensor labels, Map<String, Object> kwargs) {
        double smoothing = kwargs != null && kwargs.containsKey("label_smoothing")
                ? ((Number) kwargs.get("label_smoothing")).doubleValue()
                : labelSmoothing;
        long ignore = kwargs != null && kwargs.containsKey("ignore_index")
                ? ((Number) kwargs.get("ignore_index")).longValue()
                : ignoreIndex;
        return cross_entropy(logits, labels, new org.bytedeco.pytorch.nn.functional.CrossEntropyFuncOptions()
                .ignore_index(ignore)
                .label_smoothing((float) smoothing));
    }
}
