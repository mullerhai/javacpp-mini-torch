/*
 * EvictionPolicy -- eviction SPI for cache tiers that don't ship their own
 * controller (Redis, RocksDB, custom L2). TieredCache delegates to whatever
 * policy the backend installs, so a single tier can switch from LRU to
 * W-TinyLFU without a code change.
 *
 * <p>Each policy operates on (key, metadata) tuples only -- never dereferences
 * the value -- so it composes with encryption / compression layers that sit
 * below the policy.
 *
 * <p>Enterprise contract:
 * <ul>
 *   <li>thread-safe under concurrent access</li>
 *   <li>instrumented via {@link EvictionStats} (offers, accepts, evictions, miss-rate)</li>
 *   <li>cheap: O(1) amortised for hot-path touch(x) on supported policies</li>
 *   <li>deterministic tie-breaking for tests</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

public interface EvictionPolicy {

    /** Stable label for diagnostics / metrics. */
    String name();

    /** Called whenever a key is observed on a get(). */
    void touch(CacheKey key);

    /** Called when a key is admitted (put). */
    void admit(CacheKey key);

    /** Called when a key is explicitly removed (delete, invalidate). */
    void remove(CacheKey key);

    /** Current approximate count of live keys known to this policy. */
    long size();

    /** Pick the next victim; returns null if nothing is eligible. */
    CacheKey pickVictim();

    /** Reset to empty state. */
    void clear();

    /** Snapshot of operational counters. */
    EvictionStats stats();

    /** Cost-aware hook: weight to add to total tracked usage. */
    default void admit(CacheKey key, long weight) { admit(key); }

    /** Cost-aware hook: weight to subtract on remove. */
    default void remove(CacheKey key, long weight) { remove(key); }
}
