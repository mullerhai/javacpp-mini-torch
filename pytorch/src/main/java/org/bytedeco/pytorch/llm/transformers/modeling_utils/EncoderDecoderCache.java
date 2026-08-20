/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
package org.bytedeco.pytorch.llm.transformers.modeling_utils;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.transformers.generation.cache.Cache;

import java.util.List;

/**
 * Encoder-decoder cache that holds separate caches for self-attention
 * and cross-attention layers.
 *
 * <p>Reference: HuggingFace transformers
 * {@code modeling_utils.EncoderDecoderCache}.
 */
public final class EncoderDecoderCache extends Cache {

    private final Cache selfCache;
    private final Cache crossCache;

    public EncoderDecoderCache(Cache selfCache, Cache crossCache) {
        this.selfCache = selfCache;
        this.crossCache = crossCache;
    }

    @Override
    public int numLayers() {
        return selfCache.numLayers();
    }

    @Override
    public int getSeqLength() {
        return Math.max(selfCache.getSeqLength(), crossCache.getSeqLength());
    }

    @Override
    public List<Tensor> update(Tensor keyStates, Tensor valueStates, int layerIdx) {
        return selfCache.update(keyStates, valueStates, layerIdx);
    }

    @Override
    public void reset() {
        selfCache.reset();
        crossCache.reset();
    }

    @Override
    public Tensor keyCache(int layerIdx) {
        return selfCache.keyCache(layerIdx);
    }

    @Override
    public Tensor valueCache(int layerIdx) {
        return selfCache.valueCache(layerIdx);
    }

    /** Cross-attention key cache for layer {@code layerIdx}. */
    public Tensor crossKeyCache(int layerIdx) {
        return crossCache.keyCache(layerIdx);
    }

    /** Cross-attention value cache for layer {@code layerIdx}. */
    public Tensor crossValueCache(int layerIdx) {
        return crossCache.valueCache(layerIdx);
    }

    public Cache selfCache() { return selfCache; }
    public Cache crossCache() { return crossCache; }
}
