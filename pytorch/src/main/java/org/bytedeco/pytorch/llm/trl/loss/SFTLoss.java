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
package org.bytedeco.pytorch.llm.trl.loss;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;

import static org.bytedeco.pytorch.global.torch.Reduction;
import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cross_entropy_loss;

/**
 * Supervised fine-tuning (causal LM) token cross-entropy loss.
 *
 * <p>Standard shift: predict token {@code t+1} from position {@code t}.
 * Mirrors PyTorch {@code nn.CrossEntropyLoss(ignore_index=-100)} used by
 * HuggingFace TRL {@code SFTTrainer}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SFTLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    /** HuggingFace / PyTorch default ignore index for padded / prompt tokens. */
    public static final long DEFAULT_IGNORE_INDEX = -100L;

    private SFTLoss() {}

    /**
     * @param logits {@code [B, T, V]}
     * @param labels {@code [B, T]} (promoted to Long if needed — CE requires Long/Byte targets)
     * @return scalar mean CE over shifted tokens, ignoring {@code -100}
     */
    public static Tensor compute(Tensor logits, Tensor labels) {
        return compute(logits, labels, DEFAULT_IGNORE_INDEX);
    }

    /**
     * Causal-LM CE with an explicit ignore index (Python
     * {@code F.cross_entropy(..., ignore_index=ignoreIndex)}).
     *
     * <p>Positions where {@code labels == ignoreIndex} are excluded from the
     * mean (the denominator is the count of non-ignored tokens), matching
     * {@code nn.CrossEntropyLoss}.
     *
     * @param logits {@code [B, T, V]}
     * @param labels {@code [B, T]}
     * @param ignoreIndex typically {@code -100}
     */
    public static Tensor compute(Tensor logits, Tensor labels, long ignoreIndex) {
        // cross_entropy requires Long/Byte targets. JavaCPP tensor(long[]) without
        // TensorOptions may materialize Float — always promote before CE.
        // scalar_type() returns a non-canonical proxy — intern() before compare/branch.
        ScalarType st = labels.scalar_type().intern();
        if (st != ScalarType.Long && st != ScalarType.Byte && st != ScalarType.Char) {
            labels = labels.to(ScalarType.Long);
        }
        // shift: logits[:, :-1, :] vs labels[:, 1:]
        long t = logits.size(1);
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(t - 1), 1);
        Tensor shiftLabels = labels.slice(1, new LongOptional(1), new LongOptional(labels.size(1)), 1);

        long b = shiftLogits.size(0);
        long tt = shiftLogits.size(1);
        long v = shiftLogits.size(2);
        Tensor flatLogits = shiftLogits.reshape(b * tt, v);
        Tensor flatLabels = shiftLabels.reshape(b * tt).to(ScalarType.Long);
        // Reduction.Mean == 1 in ATen; pass ignore_index so prompt / pad tokens
        // (labels == -100) do not contribute to the mean.
        return cross_entropy_loss(flatLogits, flatLabels, new TensorOptional(),
                Reduction.Mean.value, ignoreIndex, 0.0);
    }
}
