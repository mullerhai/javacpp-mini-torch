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
package org.bytedeco.pytorch.llm.transformers.generation.stopping;

import org.bytedeco.pytorch.Tensor;

/**
 * Stop after at most {@code maxNewTokens} tokens are generated past the
 * prompt. Tracks the prompt length internally via {@link #setPromptLength}.
 *
 * <p>Mirrors HF's {@code MaxNewTokensCriteria}.
 */
public final class MaxNewTokensCriteria extends StoppingCriteria {

    private final int maxNewTokens;
    private int promptLength = -1;

    public MaxNewTokensCriteria(int maxNewTokens) {
        if (maxNewTokens <= 0) throw new IllegalArgumentException("maxNewTokens must be > 0");
        this.maxNewTokens = maxNewTokens;
    }

    public int maxNewTokens() { return maxNewTokens; }

    /** Set the prompt length to compute {@code seqLen - promptLength}. */
    public void setPromptLength(int n) { this.promptLength = n; }

    @Override
    public boolean call(Tensor inputIds, Tensor scores) {
        if (inputIds == null) return false;
        long[] shape = inputIds.shape();
        int seqLen = (int) shape[shape.length - 1];
        int generated = promptLength < 0 ? seqLen : (seqLen - promptLength);
        return generated >= maxNewTokens;
    }

    @Override
    public int getMaxLength() {
        // Without prompt length we can't compute this accurately.
        return Integer.MAX_VALUE;
    }
}
