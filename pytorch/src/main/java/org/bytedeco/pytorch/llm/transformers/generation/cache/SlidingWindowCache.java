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
 * Sliding-window KV cache used by Mistral / Gemma-2 and other models with
 * long-context but bounded attention span.
 *
 * <p>After each update, the per-layer cache holds the last {@code windowSize}
 * entries along the sequence dim (oldest are dropped). The cached tensor is
 * truncated in-place via {@code narrow} — no copies.
 *
 * <p>Caller must respect the model's actual window size and combine this with
 * a matching sliding-window mask. The cache itself doesn't enforce that.
 */
public final class SlidingWindowCache extends Cache {

    private final List<Tensor> keyCache = new ArrayList<>();
    private final List<Tensor> valueCache = new ArrayList<>();
    private final int windowSize;

    public SlidingWindowCache(int windowSize) {
        if (windowSize <= 0) throw new IllegalArgumentException("windowSize must be > 0");
        this.windowSize = windowSize;
    }

    public int windowSize() { return windowSize; }

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
        // Truncate to windowSize
        if (newK.size(2) > windowSize) {
            long drop = newK.size(2) - windowSize;
            newK = newK.narrow(2, drop, windowSize).contiguous();
            newV = newV.narrow(2, drop, windowSize).contiguous();
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
