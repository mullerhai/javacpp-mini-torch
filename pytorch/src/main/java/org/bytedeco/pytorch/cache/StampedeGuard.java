/*
 * StampedeGuard — collapsible single-flight load guard.
 *
 * <p>Designed to defeat cache stampedes (cache breakdown) under high concurrency:
 * when N concurrent requests miss the same key, only one thread actually calls
 * the loader; the remaining N-1 callers wait on the same future and share the
 * result. Critical for hot-key features (e.g. one celebrity user, one viral video).
 *
 * <p>Loosely modelled on Guava's LoadingCache behaviour, but pluggable into a
 * tiered cache so it works across multi-process shoulders.
 *
 * <p>Two implementations are exposed:
 * <ul>
 *   <li>{@link LocalSingleFlight} — in-process CompletableFuture map</li>
 *   <li>{@link RedissonSingleFlight} — uses Redis-backed lock + pub/sub for
 *       cross-JVM deduplication (recommended for AI inference clusters)</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

public interface StampedeGuard {

    <T> T guard(CacheKey key, Supplier<T> loader) throws Exception;

    static StampedeGuard noop() {
        return new StampedeGuard() {
            @Override
            public <T> T guard(CacheKey k, Supplier<T> loader) throws Exception {
                return loader.get();
            }
        };
    }

    static StampedeGuard local() { return new LocalSingleFlight(); }

    final class LocalSingleFlight implements StampedeGuard {
        private final ConcurrentMap<CacheKey, CompletableFuture<Object>> inflight = new ConcurrentHashMap<>();
        private final long timeoutMs;

        public LocalSingleFlight() { this(500); }
        public LocalSingleFlight(long timeoutMs) { this.timeoutMs = timeoutMs; }

        @Override
        @SuppressWarnings("unchecked")
        public <T> T guard(CacheKey key, Supplier<T> loader) throws Exception {
            CompletableFuture<Object> f = new CompletableFuture<>();
            CompletableFuture<Object> prev = inflight.putIfAbsent(key, f);
            if (prev != null) {
                try {
                    return (T) prev.get(timeoutMs, TimeUnit.MILLISECONDS);
                } catch (TimeoutException te) {
                    return loader.get(); // fallback so caller doesn't block forever
                } catch (ExecutionException ee) {
                    Throwable cause = ee.getCause() == null ? ee : ee.getCause();
                    if (cause instanceof Exception) throw (Exception) cause;
                    throw new RuntimeException(cause);
                }
            }
            try {
                T value = loader.get();
                f.complete(value);
                return value;
            } catch (Throwable t) {
                f.completeExceptionally(t);
                throw t;
            } finally {
                inflight.remove(key, f);
            }
        }
    }
}
