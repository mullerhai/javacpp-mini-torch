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
package org.bytedeco.pytorch.llm.transformers.generation.logits;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;

/**
 * HuggingFace-style LogitsProcessor system (mirrors
 * {@code transformers/generation/logits_process.py}).
 *
 * <p>Each processor transforms a [batch_size, vocab_size] logits tensor
 * conditioned on the input ids. A {@link LogitsProcessorList} chains them
 * left-to-right.
 *
 * <p>All implementations are pure Java and use only the
 * {@link org.bytedeco.pytorch.global.torch} ops namespace; CPU-only paths are
 * fine (logits processing is bandwidth-bound, not compute-bound).
 *
 * <pre>{@code
 * LogitsProcessorList lp = new LogitsProcessorList();
 * lp.addTemperature(0.7f)
 *   .addTopP(0.9f)
 *   .addRepetitionPenalty(1.05f);
 * Tensor filtered = lp.call(inputIds, logits);
 * }</pre>
 */
public abstract class LogitsProcessor {

    /**
     * Apply this processor to {@code scores}.
     *
     * @param inputIds [batch_size, seq_len] running sequence ids
     * @param scores   [batch_size, vocab_size] raw logits (in-place friendly)
     * @return processed logits (may be a new tensor or the same mutated tensor)
     */
    public abstract Tensor call(Tensor inputIds, Tensor scores);

    /** Convenience: {@link #call(Tensor, Tensor)} returning the mutated tensor. */
    public Tensor apply(Tensor inputIds, Tensor scores) {
        return call(inputIds, scores);
    }
}
