/*
 * CacheWarmer — startup-time pre-warm for hot keys.
 *
 * <p>Pre-loads the L1 cache with the most-frequent entities from a hot-id
 * stream (typically: previous-day active users, top-K items, top-K queries).
 * Uses the same {@link LoadFunction} SPI as the cache itself, so the warm-up
 * coincides with normal cache reads and benefits from single-flight + bloom.
 */
package org.bytedeco.pytorch.cache;

import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public final class CacheWarmer {

    private final TieredCache tiered;
    private final LoadFunction loader;
    private final int parallelism;
    private final AtomicLong warmed = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();

    public CacheWarmer(TieredCache tiered, LoadFunction loader, int parallelism) {
        this.tiered = Objects.requireNonNull(tiered);
        this.loader = loader;
        this.parallelism = Math.max(1, parallelism);
    }

    /** Warm a collection of keys, blocking until completion. */
    public long warm(Iterable<CacheKey> keys) {
        long n = 0;
        ThreadPoolExecutor exec = new ThreadPoolExecutor(
                parallelism, parallelism,
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(),
                r -> { Thread t = new Thread(r, "cache-warmer"); t.setDaemon(true); return t; });
        try {
            for (CacheKey k : keys) {
                exec.submit(() -> {
                    try {
                        if (loader != null) {
                            CacheValue<Object> v = loader.load(k);
                            if (v != null) {
                                tiered.put(k, v);
                                warmed.incrementAndGet();
                            }
                        } else {
                            tiered.get(k);
                            warmed.incrementAndGet();
                        }
                    } catch (Exception e) {
                        failed.incrementAndGet();
                    }
                });
                n++;
            }
        } finally {
            exec.shutdown();
            try { exec.awaitTermination(5, TimeUnit.MINUTES); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }
        return n;
    }

    public long warmed() { return warmed.get(); }
    public long failed() { return failed.get(); }
}
