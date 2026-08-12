/*
 * CachePlatform — top-level façade wiring the multi-tier, sharded, vector, search,
 * multimodal, and recommend caches into a single object usable by online services.
 *
 * <p>Modeled after enterprise feature-store entry points (Feast getOnlineFeatures,
 * Feast FeatureStore, Databricks FS, Vertex AI FeatureStore, Alibaba Pouch, Tencent
 * TFace). Construct once at startup and reuse across the entire process.
 *
 * <pre>{@code
 * CachePlatform platform = CachePlatform.builder()
 *         .l1(builder -> builder.l1MaxEntries(100_000))
 *         .loader(key -> loadFromSource(key))
 *         .shards(ShardedCache.ShardSpec.of("rs-0", "redis-0", 6379),
 *                 ShardedCache.ShardSpec.of("rs-1", "redis-1", 6379))
 *         .build();
 *
 * Optional<CacheValue<Object>> v = platform.cache().get(
 *         CacheKey.builder("user_features_v3", "u_123").build());
 * }</pre>
 */
package org.bytedeco.pytorch.cache;

import org.bytedeco.pytorch.cache.metrics.CacheMetrics;
import org.bytedeco.pytorch.cache.multimodal.MultimodalCache;
import org.bytedeco.pytorch.cache.recommend.RecommendCache;
import org.bytedeco.pytorch.cache.search.SearchCache;
import org.bytedeco.pytorch.cache.vector.VectorCache;

import java.util.Objects;

public final class CachePlatform implements AutoCloseable {

    private final TieredCache cache;
    private final ShardedCache sharded;
    private final DistributedCache distributed;
    private final VectorCache vector;
    private final SearchCache search;
    private final MultimodalCache multimodal;
    private final RecommendCache recommend;
    private final CacheMetrics metrics;

    private CachePlatform(Builder b) {
        this.metrics = b.metrics != null ? b.metrics : new CacheMetrics();
        this.cache = b.cache;
        this.sharded = b.sharded;
        this.distributed = b.distributed;
        this.vector = b.vector;
        this.search = b.search;
        this.multimodal = b.multimodal;
        this.recommend = b.recommend;
    }

    public TieredCache cache() { return cache; }
    public ShardedCache sharded() { return sharded; }
    public DistributedCache distributed() { return distributed; }
    public VectorCache vector() { return vector; }
    public SearchCache search() { return search; }
    public MultimodalCache multimodal() { return multimodal; }
    public RecommendCache recommend() { return recommend; }
    public CacheMetrics metrics() { return metrics; }

    @Override
    public void close() {
        try { if (cache != null) cache.close(); } catch (Exception ignore) {}
        try { if (sharded != null) sharded.close(); } catch (Exception ignore) {}
        try { if (distributed != null) distributed.close(); } catch (Exception ignore) {}
        try { if (vector != null) vector.close(); } catch (Exception ignore) {}
        try { if (search != null) search.close(); } catch (Exception ignore) {}
        try { if (multimodal != null) multimodal.close(); } catch (Exception ignore) {}
        try { if (recommend != null) recommend.close(); } catch (Exception ignore) {}
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private TieredCache cache;
        private ShardedCache sharded;
        private DistributedCache distributed;
        private VectorCache vector;
        private SearchCache search;
        private MultimodalCache multimodal;
        private RecommendCache recommend;
        private CacheMetrics metrics;

        public Builder cache(TieredCache c) { this.cache = c; return this; }
        public Builder sharded(ShardedCache s) { this.sharded = s; return this; }
        public Builder distributed(DistributedCache d) { this.distributed = d; return this; }
        public Builder vector(VectorCache v) { this.vector = v; return this; }
        public Builder search(SearchCache s) { this.search = s; return this; }
        public Builder multimodal(MultimodalCache m) { this.multimodal = m; return this; }
        public Builder recommend(RecommendCache r) { this.recommend = r; return this; }
        public Builder metrics(CacheMetrics m) { this.metrics = m; return this; }

        public CachePlatform build() {
            Objects.requireNonNull(cache, "cache (TieredCache) is required");
            return new CachePlatform(this);
        }
    }
}
