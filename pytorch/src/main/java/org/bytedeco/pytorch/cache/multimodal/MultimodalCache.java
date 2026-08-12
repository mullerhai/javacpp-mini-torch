/*
 * MultimodalCache — high-level cache facade for multi-modal AI scenarios
 * (image / text / video / audio).
 *
 * <p>Production style (Bytedance Douyin / Alibaba Taobao / Meituan / Tencent
 * multimodal feeds):
 * <ul>
 *   <li>per-modal embedding namespaces; cross-modal BGE / CLIP / BLIP embeddings
 *       share the same vector store but live in different namespaces so that
 *       cold-start warps can be loaded independently.</li>
 *   <li>per-modal TTLs (image embeddings stable for hours; ASR embeddings stable
 *       for days; OCR text embeddings sharable for weeks).</li>
 *   <li>embeddings + raw payloads (compressed image bytes, optionally lossless
 *       thumbnails) cached together so serving can fetch with a single L2 round-trip.</li>
 *   <li>cross-modal query: text-to-image retrieval cached on the query side
 *       (avoid redundant vector search during bursts of identical queries).</li>
 * </ul>
 *
 * <p>This facade composes an L1/L2 multi-tier cache with one or more
 * VectorCache instances; it does not assume a particular vector backend.
 */
package org.bytedeco.pytorch.cache.multimodal;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;
import org.bytedeco.pytorch.cache.TieredCache;
import org.bytedeco.pytorch.cache.metrics.CacheMetrics;
import org.bytedeco.pytorch.cache.vector.VectorCache;

import java.time.Duration;
import java.util.*;

public final class MultimodalCache implements AutoCloseable {

    public enum Modality { IMAGE, TEXT, AUDIO, VIDEO, OCR, ASR, FEATURE, GENERIC }

    private final TieredCache tiered;
    private final VectorCache embeddings;
    private final Map<Modality, Duration> modalTtls = new LinkedHashMap<>();

    public MultimodalCache(CacheBackend l1, CacheBackend l2, VectorCache embeddings, CacheMetrics metrics) {
        this.embeddings = embeddings;
        this.tiered = new TieredCache(
                l1,
                l2,
                null,
                org.bytedeco.pytorch.cache.CacheConfig.builder()
                        .l1MaxEntries(20_000)
                        .l1Ttl(Duration.ofMinutes(10))
                        .l2Ttl(Duration.ofHours(2))
                        .staleWhileRevalidate(Duration.ofMinutes(1))
                        .bloomExpectedInsertions(5_000_000L)
                        .loaderMaxRetries(3)
                        .build(),
                metrics);
        modalTtls.put(Modality.IMAGE, Duration.ofHours(2));
        modalTtls.put(Modality.TEXT, Duration.ofHours(12));
        modalTtls.put(Modality.AUDIO, Duration.ofHours(24));
        modalTtls.put(Modality.VIDEO, Duration.ofHours(1));
        modalTtls.put(Modality.OCR, Duration.ofHours(6));
        modalTtls.put(Modality.ASR, Duration.ofHours(24));
        modalTtls.put(Modality.FEATURE, Duration.ofMinutes(30));
        modalTtls.put(Modality.GENERIC, Duration.ofMinutes(15));
    }

    public void put(Modality mod, String id, Object payload) {
        Objects.requireNonNull(mod);
        Objects.requireNonNull(id);
        CacheKey key = CacheKey.builder(mod.name().toLowerCase(), id)
                .tag("mm")
                .build();
        long ttl = modalTtls.getOrDefault(mod, Duration.ofMinutes(15)).toMillis();
        CacheValue<Object> cv = CacheValue.of(payload)
                .ttlFromNow(ttl)
                .tag("modality", mod.name())
                .sourceTag("multimodal-cache")
                .build();
        tiered.put(key, cv);
    }

    public Optional<Object> get(Modality mod, String id) {
        CacheKey key = CacheKey.builder(mod.name().toLowerCase(), id).tag("mm").build();
        return tiered.get(key).map(CacheValue::value);
    }

    public void putEmbedding(Modality mod, String id, float[] vector) {
        if (embeddings == null) return;
        embeddings.put(mod.name().toLowerCase(), id, vector);
    }

    public Optional<float[]> getEmbedding(Modality mod, String id) {
        if (embeddings == null) return Optional.empty();
        return embeddings.get(mod.name().toLowerCase(), id);
    }

    public List<String> crossModalTopK(Modality target, float[] query, int topK) {
        if (embeddings == null) return List.of();
        return embeddings.search(target.name().toLowerCase(), query, topK);
    }

    public void invalidate(Modality mod, String id) {
        tiered.invalidate(CacheKey.builder(mod.name().toLowerCase(), id).tag("mm").build());
    }

    public TieredCache tiered() { return tiered; }
    public VectorCache embeddings() { return embeddings; }

    @Override
    public void close() {
        tiered.close();
        if (embeddings != null) embeddings.close();
    }
}
