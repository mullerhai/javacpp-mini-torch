/*
 * CapacityWeigher -- how an entry contributes to the cost of occupying a cache.
 *
 * <p>Three reference policies:
 * <ul>
 *   <li>{@link Singleton} -- every entry costs 1 unit (default, used for entry-counted)</li>
 *   <li>{@link Fixed} -- every entry costs a configured constant (bytes approximation)</li>
 *   <li>{@link Estimator} -- pluggable per-key cost estimator (e.g. byte size)</li>
 * </ul>
 *
 * <p>Couples with {@link CostAwareEvictionPolicy} to enforce a weighted-size
 * limit rather than a raw count.
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

public interface CapacityWeigher {

    long weightOf(CacheKey key);

    static final class Singleton implements CapacityWeigher {
        public static final Singleton INSTANCE = new Singleton();
        @Override public long weightOf(CacheKey key) { return 1; }
    }

    final class Fixed implements CapacityWeigher {
        private final long fixedWeight;
        public Fixed(long fixedWeight) { this.fixedWeight = Math.max(1, fixedWeight); }
        @Override public long weightOf(CacheKey key) { return fixedWeight; }
    }

    @FunctionalInterface
    interface Estimator {
        long weightOf(CacheKey key);
    }

    static CapacityWeigher estimate(Estimator e) {
        return e::weightOf;
    }
}
