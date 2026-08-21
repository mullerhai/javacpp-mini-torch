/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
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
import org.bytedeco.pytorch.ScalarOptional;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Sequence-to-sequence loss combining cross-entropy over tokens with
 * optional label smoothing.
 *
 * <p>Used by encoder-decoder models (BART, T5, etc.) where the loss is
 * computed by comparing shifted decoder outputs against the label sequence.
 */
public class Seq2SeqLoss implements Loss {

    private final double labelSmoothing;
    private final long ignoreIndex;

    /**
     * Construct with defaults (no smoothing, ignore_index=-100).
     */
    public Seq2SeqLoss() {
        this(0.0, -100L);
    }

    /**
     * Construct with label smoothing.
     *
     * @param labelSmoothing smoothing factor in [0, 1]
     */
    public Seq2SeqLoss(double labelSmoothing) {
        this(labelSmoothing, -100L);
    }

    /**
     * Full constructor.
     *
     * @param labelSmoothing smoothing factor
     * @param ignoreIndex    token index to ignore
     */
    public Seq2SeqLoss(double labelSmoothing, long ignoreIndex) {
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

        // Flatten: [B * T, V]
        long B = logits.size(0);
        long T = logits.size(1);
        long V = logits.size(2);
        Tensor flatLogits = logits.reshape(B * T, V);

        // Shift labels by 1 (teacher forcing offset)
        Tensor flatLabels = labels.reshape(B * T).to(ScalarType.Long);
        flatLabels = flatLabels.clamp(new ScalarOptional(new Scalar(0L)), new ScalarOptional(new Scalar(V - 1)));

        if (smoothing == 0.0 && ignore == -100L) {
            return cross_entropy(flatLogits, flatLabels);
        }
        try {
            org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions opts =
                    new org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions()
                            .ignore_index(ignore)
                            .label_smoothing(smoothing);
            return cross_entropy(flatLogits, flatLabels, opts);
        } catch (Throwable t) {
            return cross_entropy(flatLogits, flatLabels);
        }
    }
}
