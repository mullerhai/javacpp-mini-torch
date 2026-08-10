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
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.vllm.cache;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.kvcache.PagedKvCache;
import org.bytedeco.pytorch.llm.vllm.EngineConfig;
import org.bytedeco.pytorch.llm.vllm.Sequence;

/**
 * Thin adapter: wires engine sequences ↔ {@link PagedKvCache}.
 *
 * <p>Per-layer K/V shape is {@code [numHeads, headDim]} per token.
 * The cache stores them as {@code [2, blockSize, numHeads, headDim]} blocks.
 */
public final class CacheEngine implements AutoCloseable {

    private final PagedKvCache cache;
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int blockSize;
    private final int maxBlocks;
    private final EngineConfig config;
    private volatile boolean closed;

    // Performance metrics
    private long totalCreateSequenceTimeMs;
    private long totalAppendTimeMs;
    private long totalGatherTimeMs;
    private long totalReleaseTimeMs;
    private int sequencesCreated;
    private int sequencesReleased;
    private int tokensAppended;
    private int tokensGathered;

    public CacheEngine(EngineConfig config) {
        this.config = config;
        this.numLayers = config.numLayers;
        this.numHeads = config.numHeads;
        this.headDim = config.headDim;
        this.blockSize = config.blockSize;
        this.maxBlocks = config.maxBlocks;

        Device dev = "cuda".equalsIgnoreCase(config.device)
                ? new Device("cuda:0") : null;
        this.cache = new PagedKvCache(numLayers, numHeads, headDim, blockSize, maxBlocks, dev);
    }

    public PagedKvCache cache() { return cache; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }
    public int blockSize() { return blockSize; }
    public int freeBlocks() { return cache.freeBlocks(); }
    public int liveSequences() { return cache.liveSequences(); }

    /** Create a cache entry for a newly-scheduled sequence. */
    public long createSequence(Sequence seq) {
        if (closed) throw new IllegalStateException("CacheEngine closed");
        long start = System.currentTimeMillis();
        long id = cache.createSequence();
        seq.setCacheSeqId(id);
        totalCreateSequenceTimeMs += System.currentTimeMillis() - start;
        sequencesCreated++;
        return id;
    }

    /**
     * Append K/V tensors for one new token across all layers.
     * kLayers[i] / vLayers[i] each shape {@code [numHeads, headDim]}.
     */
    public void append(long seqId, int tokenId, Tensor[] kLayers, Tensor[] vLayers) {
        if (closed) throw new IllegalStateException("CacheEngine closed");
        long start = System.currentTimeMillis();
        cache.append(seqId, tokenId, kLayers, vLayers);
        totalAppendTimeMs += System.currentTimeMillis() - start;
        tokensAppended++;
    }

    /** Gather full K/V for a sequence at a given layer. Returns [K, V] each [T, numHeads, headDim]. */
    public Tensor[] gather(long seqId, int layer) {
        if (closed) throw new IllegalStateException("CacheEngine closed");
        long start = System.currentTimeMillis();
        Tensor[] result = cache.gather(seqId, layer);
        totalGatherTimeMs += System.currentTimeMillis() - start;
        tokensGathered++;
        return result;
    }

    /** Release a sequence and return its blocks to the pool. */
    public void releaseSequence(Sequence seq) {
        if (closed) return;
        long start = System.currentTimeMillis();
        if (seq.cacheSeqId() < 0) return;
        cache.releaseSequence(seq.cacheSeqId());
        seq.setCacheSeqId(-1);
        totalReleaseTimeMs += System.currentTimeMillis() - start;
        sequencesReleased++;
    }

    /** Fork a cache entry (prefix sharing). */
    public long fork(long srcSeqId) {
        if (closed) throw new IllegalStateException("CacheEngine closed");
        return cache.fork(srcSeqId);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        cache.close();
        System.out.printf(
                "[CacheEngine] Closed: sequences=%d/%d, tokens=%d/%d, " +
                "times(create=%.2fs, append=%.2fs, gather=%.2fs, release=%.2fs)%n",
                sequencesCreated, sequencesReleased,
                tokensAppended, tokensGathered,
                totalCreateSequenceTimeMs / 1000.0,
                totalAppendTimeMs / 1000.0,
                totalGatherTimeMs / 1000.0,
                totalReleaseTimeMs / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Cache performance statistics.
     */
    public CacheStats getStats() {
        return new CacheStats(
                sequencesCreated, sequencesReleased,
                tokensAppended, tokensGathered,
                totalCreateSequenceTimeMs,
                totalAppendTimeMs,
                totalGatherTimeMs,
                totalReleaseTimeMs,
                cache.allocCount.longValue(),
                cache.evictCount.longValue(),
                cache.cowCount.longValue(),
                cache.prefixHitTokens.longValue()
        );
    }

    /**
     * Cache performance statistics.
     */
    public static final class CacheStats {
        public final int sequencesCreated;
        public final int sequencesReleased;
        public final int tokensAppended;
        public final int tokensGathered;
        public final long createSequenceTimeMs;
        public final long appendTimeMs;
        public final long gatherTimeMs;
        public final long releaseTimeMs;
        public final long blocksAllocated;
        public final long blocksEvicted;
        public final long cowCount;
        public final long prefixHitTokens;

        public CacheStats(int sequencesCreated, int sequencesReleased,
                        int tokensAppended, int tokensGathered,
                        long createSequenceTimeMs, long appendTimeMs,
                        long gatherTimeMs, long releaseTimeMs,
                        long blocksAllocated, long blocksEvicted,
                        long cowCount, long prefixHitTokens) {
            this.sequencesCreated = sequencesCreated;
            this.sequencesReleased = sequencesReleased;
            this.tokensAppended = tokensAppended;
            this.tokensGathered = tokensGathered;
            this.createSequenceTimeMs = createSequenceTimeMs;
            this.appendTimeMs = appendTimeMs;
            this.gatherTimeMs = gatherTimeMs;
            this.releaseTimeMs = releaseTimeMs;
            this.blocksAllocated = blocksAllocated;
            this.blocksEvicted = blocksEvicted;
            this.cowCount = cowCount;
            this.prefixHitTokens = prefixHitTokens;
        }

        public double avgAppendTimeMs() {
            return tokensAppended > 0 ? (double) appendTimeMs / tokensAppended : 0;
        }

        public double avgGatherTimeMs() {
            return tokensGathered > 0 ? (double) gatherTimeMs / tokensGathered : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "CacheStats{seqs=%d/%d, tokens=%d/%d, " +
                    "times(append=%.2fms, gather=%.2fms), " +
                    "alloc=%d, evict=%d, cow=%d, prefixHits=%d",
                    sequencesCreated, sequencesReleased,
                    tokensAppended, tokensGathered,
                    avgAppendTimeMs(), avgGatherTimeMs(),
                    blocksAllocated, blocksEvicted, cowCount, prefixHitTokens);
        }
    }

    /** Number of blocks needed for a given sequence length (rough upper bound). */
    public int blocksForTokens(int tokens) {
        return (tokens + blockSize - 1) / blockSize;
    }

    public EngineConfig config() { return config; }

    public String stats() {
        return String.format("Cache{kv=%d/%d blocks free, live=%d seqs}",
                freeBlocks(), maxBlocks, liveSequences());
    }
}
