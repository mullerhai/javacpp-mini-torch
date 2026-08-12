/*
 * CostAwareEvictionPolicy -- decorator that wraps any {@link EvictionPolicy}
 * and enforces a maximum weighted-size budget in addition to the underlying
 * policy's notion of "capacity".
 *
 * <p>Common case: a backend holds 1M entries averaging 4 KB; the operator
 * wants a 2 GB ceiling rather than a raw count. Decorating LRU with this
 * wrapper yields size-aware LRU.
 *
 * <p>Threading: takes the wrapped policy's monitor; do not call from inside
 * the policy's own callbacks.
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

public final class CostAwareEvictionPolicy implements EvictionPolicy {

    private final EvictionPolicy inner;
    private final CapacityWeigher weigher;
    private final long maxWeight;
    private long currentWeight = 0;
    private final java.util.HashMap<CacheKey, Long> perKeyWeight = new java.util.HashMap<>();

    public CostAwareEvictionPolicy(EvictionPolicy inner, CapacityWeigher weigher, long maxWeight) {
        if (inner == null) throw new IllegalArgumentException("inner==null");
        if (weigher == null) weigher = CapacityWeigher.Singleton.INSTANCE;
        if (maxWeight <= 0) throw new IllegalArgumentException("maxWeight<=0");
        this.inner = inner;
        this.weigher = weigher;
        this.maxWeight = maxWeight;
    }

    @Override public String name() { return "COST:" + inner.name(); }

    @Override
    public synchronized void touch(CacheKey key) {
        inner.touch(key);
    }

    @Override
    public synchronized void admit(CacheKey key) {
        long w = weightOf(key);
        perKeyWeight.put(key, w);
        currentWeight += w;
        inner.admit(key);
        enforce();
    }

    @Override
    public synchronized void remove(CacheKey key) {
        Long w = perKeyWeight.remove(key);
        if (w != null) currentWeight -= w;
        inner.remove(key);
    }

    @Override
    public synchronized long size() { return inner.size(); }

    public synchronized long weightedSize() { return currentWeight; }
    public long maxWeight() { return maxWeight; }

    @Override
    public synchronized CacheKey pickVictim() {
        CacheKey v = inner.pickVictim();
        if (v != null) remove(v);
        return v;
    }

    @Override
    public synchronized void clear() {
        inner.clear();
        perKeyWeight.clear();
        currentWeight = 0;
    }

    @Override
    public EvictionStats stats() { return inner.stats(); }

    private void enforce() {
        while (currentWeight > maxWeight && inner.size() > 0) {
            CacheKey v = inner.pickVictim();
            if (v == null) break;
            Long w = perKeyWeight.remove(v);
            if (w != null) currentWeight -= w;
            inner.remove(v);
        }
    }

    private long weightOf(CacheKey key) {
        long w = weigher.weightOf(key);
        return w <= 0 ? 1 : w;
    }
}
