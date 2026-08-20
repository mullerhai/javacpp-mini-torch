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

import java.util.List;

/**
 * HuggingFace-style KV cache base class. Mirrors
 * {@code transformers/cache_utils.py:Cache}.
 *
 * <p>A cache stores per-layer (key, value) tensors that grow as the model
 * generates more tokens. Each layer has its own slot, addressed by
 * {@code layerIdx}.
 *
 * <p>Concrete implementations:
 * <ul>
 *   <li>{@link DynamicCache} — append-only, no eviction</li>
 *   <li>{@link SlidingWindowCache} — keep last {@code window} tokens per layer</li>
 *   <li>{@link StaticCache} — pre-allocated, fixed shape</li>
 * </ul>
 */
public abstract class Cache {

    /** Number of layers this cache tracks. */
    public abstract int numLayers();

    /** Cumulative sequence length stored in the cache (max across layers). */
    public abstract int getSeqLength();

    /**
     * Append new K/V entries for layer {@code layerIdx}.
     *
     * @param keyStates  new K [batch, num_heads, new_len, head_dim]
     * @param valueStates new V [batch, num_heads, new_len, head_dim]
     * @param layerIdx    layer index
     * @return [keyStates, valueStates] — possibly concatenated with the
     *         previously stored entries.
     */
    public abstract List<Tensor> update(Tensor keyStates, Tensor valueStates, int layerIdx);

    /** Reset all layers to empty. */
    public abstract void reset();

    /** Return the stored K tensor for layer {@code layerIdx} (or null). */
    public abstract Tensor keyCache(int layerIdx);

    /** Return the stored V tensor for layer {@code layerIdx} (or null). */
    public abstract Tensor valueCache(int layerIdx);

    /** Return [K, V] tuple for layer {@code layerIdx}. */
    public List<Tensor> layerCache(int layerIdx) {
        return List.of(keyCache(layerIdx), valueCache(layerIdx));
    }
}
