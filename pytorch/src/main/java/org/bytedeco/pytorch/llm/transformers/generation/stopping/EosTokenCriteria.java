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
 * HF-style EOS detection: stop when the last generated token is in the EOS set.
 *
 * <p>Most models set EOS = 1 token id; some (Qwen, Gemma) use a small set —
 * we accept both via the constructor.
 */
public final class EosTokenCriteria extends StoppingCriteria {

    private final java.util.Set<Integer> eosTokenIds;

    public EosTokenCriteria(int eosTokenId) {
        this.eosTokenIds = java.util.Set.of(eosTokenId);
    }

    public EosTokenCriteria(int[] eosTokenIds) {
        java.util.Set<Integer> s = new java.util.HashSet<>();
        for (int t : eosTokenIds) s.add(t);
        this.eosTokenIds = java.util.Collections.unmodifiableSet(s);
    }

    public java.util.Set<Integer> eosTokenIds() { return eosTokenIds; }

    @Override
    public boolean call(Tensor inputIds, Tensor scores) {
        if (inputIds == null) return false;
        long last = inputIds.select(/*dim=*/inputIds.dim() - 1,
                                    /*index=*/inputIds.size(inputIds.dim() - 1) - 1)
                .item_long();
        return eosTokenIds.contains((int) last);
    }
}