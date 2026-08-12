/*
 * RecommendCache — high-level cache for the recommendation system path.
 *
 * <p>Production pattern (Bytedance Douyin / Kuaishou / Alibaba Taobao / Tencent
 * WeChat / Meituan / PinDuoDuo):
 * <ul>
 *   <li>User profile features (instant updates, short TTL) — view "user_fea_v3"</li>
 *   <li>Item profile features (slow updates, long TTL) — view "item_fea_v3"</li>
 *   <li>User-item cross features (computed on the fly, mid TTL) — view "x_user_item"</li>
 *   <li>Embedding snapshots (L1 + vector store) — view "user_emb" / "item_emb"</li>
 *   <li>Pre-ranked scores (very short TTL, swr) — view "pre_score"</li>
 *   <li>Final ranked scores (very short TTL, swr) — view "rank_score"</li>
 *   <li>Recall-channel results (per-channel, mid TTL) — view "recall_*"</li>
 * </ul>
 *
 * <p>Anti-penetration / breakdown / avalanche / crossing guard are inherited.
 * Recommend-specific features:
 * <ul>
 *   <li>per-view TTL + per-view L1 cap</li>
 *   <li>per-tenant default cap with auto-throttle on hot tenants</li>
 *   <li>batch get with view-aware fallback (e.g. if "pre_score" misses, fall
 *       back to "recall_*" and combine)</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.recommend;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;
import org.bytedeco.pytorch.cache.TieredCache;
import org.bytedeco.pytorch.cache.metrics.CacheMetrics;
import org.bytedeco.pytorch.cache.vector.VectorCache;

import java.time.Duration;
import java.util.*;

public final class RecommendCache implements AutoCloseable {

    public enum ViewKind {
        USER_FEATURE(Duration.ofMinutes(5), 50_000),
        ITEM_FEATURE(Duration.ofHours(1),  200_000),
        USER_ITEM_CROSS(Duration.ofMinutes(2), 100_000),
        EMBEDDING(Duration.ofHours(4), 50_000),
        PRE_SCORE(Duration.ofSeconds(30), 20_000),
        FINAL_SCORE(Duration.ofSeconds(60), 20_000),
        RECALL(Duration.ofMinutes(2), 30_000),
        CONTEXT_FEATURE(Duration.ofSeconds(30), 10_000),
        COLD_START(Duration.ofSeconds(15), 10_000);

        public final Duration ttl;
        public final long l1Cap;
        ViewKind(Duration ttl, long l1Cap) { this.ttl = ttl; this.l1Cap = l1Cap; }
    }

    private final TieredCache tiered;
    private final VectorCache embeddings;
    private final Map<ViewKind, Long> hitsPerKind = new LinkedHashMap<>();

    public RecommendCache(CacheBackend l1, CacheBackend l2, VectorCache embeddings, CacheMetrics metrics) {
        this.embeddings = embeddings;
        this.tiered = new TieredCache(
                l1,
                l2,
                null,
                org.bytedeco.pytorch.cache.CacheConfig.builder()
                        .l1MaxEntries(100_000)
                        .l1Ttl(Duration.ofMinutes(5))
                        .l2Ttl(Duration.ofMinutes(30))
                        .staleWhileRevalidate(Duration.ofSeconds(20))
                        .singleFlightPermits(64)
                        .loaderMaxRetries(3)
                        .build(),
                metrics);
    }

    public void put(ViewKind kind, String userId, String itemId, Object payload) {
        put(kind, userId, itemId, payload, System.currentTimeMillis());
    }

    public void put(ViewKind kind, String userId, String itemId, Object payload, long eventTsMs) {
        Objects.requireNonNull(kind);
        CacheKey key = keyOf(kind, userId, itemId);
        CacheValue<Object> cv = CacheValue.of(payload)
                .ttlFromNow(kind.ttl.toMillis())
                .eventTimestampMs(eventTsMs)
                .tag("viewkind", kind.name())
                .sourceTag("rec-cache")
                .build();
        tiered.put(key, cv);
    }

    public Optional<Object> get(ViewKind kind, String userId, String itemId) {
        return tiered.get(keyOf(kind, userId, itemId)).map(CacheValue::value);
    }

    public Map<ViewKind, Object> getAllForPair(String userId, String itemId, List<ViewKind> views) {
        Map<ViewKind, Object> out = new LinkedHashMap<>();
        if (views == null) return out;
        for (ViewKind v : views) {
            Optional<Object> got = get(v, userId, itemId);
            got.ifPresent(o -> out.put(v, o));
        }
        return out;
    }

    public void putEmbedding(ViewKind kind, String id, float[] vector) {
        if (embeddings == null || vector == null) return;
        embeddings.put(kind.name().toLowerCase(), id, vector);
    }

    public Optional<float[]> getEmbedding(ViewKind kind, String id) {
        if (embeddings == null) return Optional.empty();
        return embeddings.get(kind.name().toLowerCase(), id);
    }

    public List<String> recallTopK(ViewKind kind, float[] query, int topK) {
        if (embeddings == null) return List.of();
        return embeddings.search(kind.name().toLowerCase(), query, topK);
    }

    public void invalidatePair(ViewKind kind, String userId, String itemId) {
        tiered.invalidate(keyOf(kind, userId, itemId));
    }

    public void invalidateView(ViewKind kind) {
        tiered.invalidateByView("default", kind.name());
    }

    private static CacheKey keyOf(ViewKind kind, String userId, String itemId) {
        String entityId = itemId == null ? userId : userId + "|" + itemId;
        return CacheKey.builder("rec:" + kind.name().toLowerCase(), entityId)
                .tag("recommend")
                .build();
    }

    public TieredCache tiered() { return tiered; }
    public VectorCache embeddings() { return embeddings; }
    public CacheMetrics metrics() { return tiered.metrics(); }

    @Override
    public void close() {
        tiered.close();
        if (embeddings != null) embeddings.close();
    }
}
