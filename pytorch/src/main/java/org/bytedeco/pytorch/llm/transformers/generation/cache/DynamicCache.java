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
 * Dynamic (append-only) KV cache. New K/V are concatenated with previous
 * entries along the sequence dimension. Memory usage grows linearly with the
 * generated sequence length — HF's default for non-streaming generation.
 *
 * <p>This mirrors HF's {@code DynamicCache}; we use {@code torch.cat} along
 * dim 2 (the seq dim for a [batch, num_heads, seq, head_dim] layout).
 */
public final class DynamicCache extends Cache {

    private final List<Tensor> keyCache = new ArrayList<>();
    private final List<Tensor> valueCache = new ArrayList<>();

    public DynamicCache() {}

    public DynamicCache(int numLayers) {
        for (int i = 0; i < numLayers; i++) {
            keyCache.add(null);
            valueCache.add(null);
        }
    }

    @Override
    public int numLayers() {
        return Math.max(keyCache.size(), valueCache.size());
    }

    @Override
    public int getSeqLength() {
        int max = 0;
        for (Tensor k : keyCache) {
            if (k != null && k.defined()) {
                int len = (int) k.size(2);
                if (len > max) max = len;
            }
        }
        return max;
    }

    @Override
    public List<Tensor> update(Tensor keyStates, Tensor valueStates, int layerIdx) {
        // Grow per-layer storage if needed.
        while (keyCache.size() <= layerIdx) {
            keyCache.add(null);
            valueCache.add(null);
        }
        Tensor prevK = keyCache.get(layerIdx);
        Tensor prevV = valueCache.get(layerIdx);
        Tensor newK, newV;
        if (prevK == null || !prevK.defined()) {
            newK = keyStates;
            newV = valueStates;
        } else {
            newK = torch.cat(new org.bytedeco.pytorch.TensorVector(prevK, keyStates), 2);
            newV = torch.cat(new org.bytedeco.pytorch.TensorVector(prevV, valueStates), 2);
        }
        keyCache.set(layerIdx, newK);
        valueCache.set(layerIdx, newV);
        return List.of(newK, newV);
    }

    @Override
    public void reset() {
        keyCache.clear();
        valueCache.clear();
    }

    @Override
    public Tensor keyCache(int layerIdx) {
        return layerIdx < keyCache.size() ? keyCache.get(layerIdx) : null;
    }

    @Override
    public Tensor valueCache(int layerIdx) {
        return layerIdx < valueCache.size() ? valueCache.get(layerIdx) : null;
    }
}
