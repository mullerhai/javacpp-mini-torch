/*
 * CacheBackend — Storage SPI for any tier (L1 in-memory, L2 Redis, L3 remote).
 *
 * <p>Design mirrors what big-tech feature stores expose:
 * <ul>
 *   <li>high concurrency (lock-free where possible)</li>
 *   <li>batch hot-path (redis mget / mset)</li>
 *   <li>explicit expiration with event-time awareness</li>
 *   <li>graceful degradation (no exception on missing keys)</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache;
import org.bytedeco.pytorch.jit.*;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public interface CacheBackend extends AutoCloseable {

    /** Stable label for diagnostics / metrics. */
    String name();

    /** Tier label (L1, L2, L3 …). */
    int tier();

    /** Fetch a single value if present and not expired. */
    Optional<CacheValue<Object>> get(CacheKey key);

    /** Batch fetch. Missing keys are simply absent from the returned map. */
    default Map<CacheKey, CacheValue<Object>> getBatch(Collection<CacheKey> keys) {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null) return out;
        for (CacheKey k : keys) {
            get(k).ifPresent(v -> out.put(k, v));
        }
        return out;
    }

    /** Single write — overwrite if present. */
    void put(CacheKey key, CacheValue<Object> value);

    /** Batch write — atomicity is best-effort; partial failures are acceptable. */
    default void putBatch(Map<CacheKey, CacheValue<Object>> entries) {
        if (entries == null) return;
        for (Map.Entry<CacheKey, CacheValue<Object>> e : entries.entrySet()) {
            put(e.getKey(), e.getValue());
        }
    }

    /** Delete one entry. */
    void delete(CacheKey key);

    /** Delete many entries. */
    default void deleteBatch(Collection<CacheKey> keys) {
        if (keys == null) return;
        for (CacheKey k : keys) delete(k);
    }

    /** Approximate cardinality (for monitoring). */
    long size();

    /** Health probe. */
    boolean ping();

    @Override
    default void close() {}
}
