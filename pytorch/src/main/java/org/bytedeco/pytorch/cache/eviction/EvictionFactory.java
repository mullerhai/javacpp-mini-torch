/*
 * EvictionFactory -- common-policy factory aligned with CacheConfig.EvictionPolicy.
 *
 * <p>Returns a singleton-instance policy for the requested algorithm so the
 * same byte-budget signal can be reused across tiers.
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheConfig;

public final class EvictionFactory {

    private EvictionFactory() {}

    public static EvictionPolicy forConfig(CacheConfig.EvictionPolicy p, long capacity) {
        switch (p) {
            case LRU:         return new LruEvictionPolicy();
            case LFU:         return new LfuEvictionPolicy();
            case FIFO:        return new FifoEvictionPolicy();
            case W_TINY_LFU:  return new WTinyLFUEvictionPolicy(capacity);
            default:          throw new IllegalArgumentException("unknown policy: " + p);
        }
    }

    public static EvictionPolicy weighted(EvictionPolicy inner, CapacityWeigher w, long maxWeight) {
        return new CostAwareEvictionPolicy(inner, w, maxWeight);
    }
}
