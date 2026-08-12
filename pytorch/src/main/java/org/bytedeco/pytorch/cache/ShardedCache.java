/*
 * ShardedCache — partitions a {@link TieredCache} pool across multiple
 * hash-ring nodes so heavy-write tenants don't hot-spot a single Redis.
 *
 * <p>In production (Alibaba Pouch / Tair, Tencent Redis Cluster, ByteDance
 * Abase) the per-shard breakdown is typically:
 * <ul>
 *   <li>shard by tenant & view to avoid re-keying the whole ring on rotation</li>
 *   <li>replicate to {@code replicationFactor} primaries so read load is
 *       balanced and SPOFs are avoided</li>
 *   <li>auto-rebalance on add/remove with virtual-node weighted distribution</li>
 * </ul>
 *
 * <p>This class wires the {@link ConsistencyHashRing} to a fixed-sized pool of
 * inner {@link TieredCache} instances. Each call to {@link #get} / {@link #put}
 * first routes the key to the owning tier, then delegates.
 */
package org.bytedeco.pytorch.cache;

import org.bytedeco.pytorch.cache.metrics.CacheMetrics;
import org.bytedeco.pytorch.cache.sharding.ConsistencyHashRing;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public final class ShardedCache implements AutoCloseable {

    private final ConsistencyHashRing<ShardId> ring;
    private final Map<ShardId, TieredCache> tiers;
    private final CacheMetrics metrics;
    private final AtomicLong redirects = new AtomicLong();

    public ShardedCache(List<ShardSpec> specs, ShardFactory factory, CacheMetrics metrics) {
        Objects.requireNonNull(specs);
        Objects.requireNonNull(factory);
        this.metrics = metrics == null ? new CacheMetrics() : metrics;
        Map<ShardId, Integer> weights = new java.util.HashMap<>();
        for (ShardSpec spec : specs) {
            weights.put(new ShardId(spec.name, spec.host, spec.port), spec.weight);
        }
        this.ring = new ConsistencyHashRing<>(id -> Math.max(1, weights.getOrDefault(id, 1)), 200);
        this.tiers = new ConcurrentHashMap<>();
        for (ShardSpec spec : specs) {
            ShardId id = new ShardId(spec.name, spec.host, spec.port);
            tiers.put(id, factory.create(spec));
            ring.addNode(id);
        }
    }

    public Optional<CacheValue<Object>> get(CacheKey key) {
        ShardId id = ring.locate(key.routingKey());
        if (id == null) return Optional.empty();
        TieredCache tier = tiers.get(id);
        if (tier == null) return Optional.empty();
        if (!id.equals(ring.locate(key.routingKey()))) redirects.incrementAndGet();
        metrics.recordShardedHit();
        return tier.get(key);
    }

    public Map<CacheKey, CacheValue<Object>> getBatch(Collection<CacheKey> keys) {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null) return out;
        // group by shard
        Map<ShardId, List<CacheKey>> byShard = new LinkedHashMap<>();
        for (CacheKey k : keys) {
            ShardId id = ring.locate(k.routingKey());
            if (id == null) continue;
            byShard.computeIfAbsent(id, x -> new ArrayList<>()).add(k);
        }
        for (Map.Entry<ShardId, List<CacheKey>> e : byShard.entrySet()) {
            TieredCache tier = tiers.get(e.getKey());
            if (tier == null) continue;
            out.putAll(tier.getBatch(e.getValue()));
        }
        return out;
    }

    public void put(CacheKey key, CacheValue<Object> value) {
        ShardId id = ring.locate(key.routingKey());
        if (id == null) return;
        TieredCache tier = tiers.get(id);
        if (tier != null) tier.put(key, value);
    }

    public void putBatch(Map<CacheKey, CacheValue<Object>> entries) {
        if (entries == null) return;
        Map<ShardId, Map<CacheKey, CacheValue<Object>>> byShard = new LinkedHashMap<>();
        for (Map.Entry<CacheKey, CacheValue<Object>> e : entries.entrySet()) {
            ShardId id = ring.locate(e.getKey().routingKey());
            if (id == null) continue;
            byShard.computeIfAbsent(id, x -> new LinkedHashMap<>()).put(e.getKey(), e.getValue());
        }
        for (Map.Entry<ShardId, Map<CacheKey, CacheValue<Object>>> e : byShard.entrySet()) {
            TieredCache tier = tiers.get(e.getKey());
            if (tier != null) tier.putBatch(e.getValue());
        }
    }

    public void invalidate(CacheKey key) {
        ShardId id = ring.locate(key.routingKey());
        if (id == null) return;
        TieredCache tier = tiers.get(id);
        if (tier != null) tier.invalidate(key);
    }

    public CacheMetrics metrics() { return metrics; }

    public CacheBackend backendForKey(CacheKey key) {
        ShardId id = ring.locate(key.routingKey());
        if (id == null) return null;
        TieredCache tier = tiers.get(id);
        return tier == null ? null : tier.l1();
    }

    public Map<String, Long> shardDistribution() {
        Map<String, Long> out = new LinkedHashMap<>();
        for (Map.Entry<ShardId, TieredCache> e : tiers.entrySet()) {
            out.put(e.getKey().name, e.getValue().l1().size());
        }
        return out;
    }

    @Override
    public void close() {
        for (TieredCache t : tiers.values()) try { t.close(); } catch (Exception ignore) {}
    }

    /** Re-route all keys to a freshly added node. Typically a no-op as items
     * lazily relocate to the new ring; only invalidate if read-after-write is
     * implicitly required for the calling experience. */
    public void rebalance() {
        // virtual-node rotation just changes future routing; existing entries
        // in L1/L2 are untouched and will be re-fetched on next miss.
    }

    public static final class ShardSpec {
        public final String name;
        public final String host;
        public final int port;
        public final int weight;
        public ShardSpec(String name, String host, int port, int weight) {
            this.name = name; this.host = host; this.port = port; this.weight = weight;
        }
        public static ShardSpec of(String name, String host, int port) {
            return new ShardSpec(name, host, port, 1);
        }
    }

    public interface ShardFactory {
        TieredCache create(ShardSpec spec);
    }

    public static final class ShardId {
        public final String name;
        public final String host;
        public final int port;
        public ShardId(String name, String host, int port) {
            this.name = name; this.host = host; this.port = port;
        }
        @Override public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof ShardId)) return false;
            ShardId s = (ShardId) o;
            return port == s.port && name.equals(s.name) && host.equals(s.host);
        }
        @Override public int hashCode() { return name.hashCode() * 31 + host.hashCode() + port; }
        @Override public String toString() { return name + "@" + host + ":" + port; }
    }
}
