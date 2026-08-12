/*
 * LoadFunction — SPI for fetching the source-of-truth value when the cache misses.
 *
 * <p>Mirrors Guava CacheLoader / Caffeine CacheLoader. Implementations must be
 * idempotent and side-effect-free; they will be invoked inside a single-flight
 * guard so concurrent calls collapse to one.
 */
package org.bytedeco.pytorch.cache;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

@FunctionalInterface
public interface LoadFunction {

    /** Load a single value; null indicates "not present in source". */
    CacheValue<Object> load(CacheKey key) throws Exception;

    /** Convenience: batch signature with default sequential implementation. */
    default Map<CacheKey, CacheValue<Object>> loadBatch(Collection<CacheKey> keys) throws Exception {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null) return out;
        for (CacheKey k : keys) {
            CacheValue<Object> v = load(k);
            if (v != null) out.put(k, v);
        }
        return out;
    }
}
