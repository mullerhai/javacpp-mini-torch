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
package org.bytedeco.pytorch.llm.transformers.generation.cache;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;

/**
 * Static, pre-allocated KV cache. Each layer is a fixed [batch, num_heads,
 * maxLen, head_dim] tensor that we fill via {@code narrow + copy_} on each
 * update. Avoids repeated {@code torch.cat} allocations — better for
 * production serving workloads where the max length is known in advance.
 *
 * <p>Mirrors HF's {@code StaticCache}.
 */
public final class StaticCache extends Cache {

    private final List<Tensor> keyCache = new ArrayList<>();
    private final List<Tensor> valueCache = new ArrayList<>();
    private final int maxBatchSize;
    private final int maxLen;
    private final long numHeads;
    private final long headDim;
    private final List<Integer> seqLens = new ArrayList<>(); // per-layer running length

    public StaticCache(int numLayers, int maxBatchSize, int maxLen,
                       long numHeads, long headDim) {
        if (maxBatchSize <= 0 || maxLen <= 0 || numHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("cache dims must be positive");
        }
        this.maxBatchSize = maxBatchSize;
        this.maxLen = maxLen;
        this.numHeads = numHeads;
        this.headDim = headDim;
        for (int i = 0; i < numLayers; i++) {
            keyCache.add(torch.zeros(maxBatchSize, numHeads, maxLen, headDim));
            valueCache.add(torch.zeros(maxBatchSize, numHeads, maxLen, headDim));
            seqLens.add(0);
        }
    }

    @Override
    public int numLayers() {
        return keyCache.size();
    }

    @Override
    public int getSeqLength() {
        int max = 0;
        for (Integer s : seqLens) if (s > max) max = s;
        return max;
    }

    @Override
    public List<Tensor> update(Tensor keyStates, Tensor valueStates, int layerIdx) {
        if (layerIdx >= keyCache.size()) {
            throw new IndexOutOfBoundsException("layer " + layerIdx + " >= " + keyCache.size());
        }
        long newLen = keyStates.size(2);
        int current = seqLens.get(layerIdx);
        int total = current + (int) newLen;
        if (total > maxLen) {
            throw new IllegalStateException("StaticCache overflow at layer " + layerIdx
                    + ": " + total + " > " + maxLen);
        }
        Tensor kSlot = keyCache.get(layerIdx);
        Tensor vSlot = valueCache.get(layerIdx);
        kSlot.narrow(2, current, (int) newLen).copy_(keyStates);
        vSlot.narrow(2, current, (int) newLen).copy_(valueStates);
        seqLens.set(layerIdx, total);

        // Returned slice for attention
        Tensor kOut = kSlot.narrow(2, 0, total);
        Tensor vOut = vSlot.narrow(2, 0, total);
        return List.of(kOut, vOut);
    }

    @Override
    public void reset() {
        for (int i = 0; i < seqLens.size(); i++) seqLens.set(i, 0);
    }

    @Override
    public Tensor keyCache(int layerIdx) {
        if (layerIdx >= keyCache.size()) return null;
        return keyCache.get(layerIdx).narrow(2, 0, seqLens.get(layerIdx));
    }

    @Override
    public Tensor valueCache(int layerIdx) {
        if (layerIdx >= valueCache.size()) return null;
        return valueCache.get(layerIdx).narrow(2, 0, seqLens.get(layerIdx));
    }
}
