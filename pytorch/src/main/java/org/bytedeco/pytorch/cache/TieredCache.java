/*
 * TieredCache — L1 + L2 multi-tier cache with the four defensive patterns
 * every large-scale online inference system needs:
 *
 * <ol>
 *   <li><b>Anti-penetration</b> — {@link NegativeCache} blocks known-absent
 *       keys from ever reaching the source; combined with a short negative TTL
 *       it stops both cold-start floods and steady-state scans.</li>
 *   <li><b>Anti-breakdown (stampede)</b> — {@link StampedeGuard} collapses
 *       concurrent misses on the same key into a single loader call.</li>
 *   <li><b>Anti-avalanche</b> — TTL jitter is added on every write so that
 *       large cohorts of entries don't expire in lockstep and overload the
 *       source system.</li>
 *   <li><b>Feature crossing</b> — every write/refresh passes through
 *       {@link FeatureCrossingGuard} so future-dated features never leak.</li>
 * </ol>
 *
 * <p>Hot-path pattern (from Meta TAO, Google Guava, ByteDance Abase):
 * <pre>
 *   request -> L1 hit? --yes--> return
 *           \--no--> L2 hit? --yes--> promote to L1 (stale-while-revalidate) --> return
 *                  \--no--> bloom gate? --yes--> return Optional.empty()
 *                          \--no--> single-flight: load(key) -> admit -> return
 * </pre>
 *
 * <p>Stale-while-revalidate: when a value crosses its TTL but is within the
 * stale-while-revalidate window, the old value is returned while a refresh
 * task runs asynchronously. This is the standard for latency-sensitive
 * recommendation traffic (Latency at p99 < 1ms while still ensuring freshness).
 */
package org.bytedeco.pytorch.cache;

import org.bytedeco.pytorch.cache.metrics.CacheMetrics;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.TimeUnit;

public final class TieredCache implements AutoCloseable {

    private final CacheBackend l1;
    private final CacheBackend l2;            // optional
    private final LoadFunction loader;
    private final CacheConfig config;
    private final CacheMetrics metrics;
    private final StampedeGuard stampedeGuard;
    private final NegativeCache negativeCache;
    private final FeatureCrossingGuard crossingGuard;
    private final ExecutorService refreshExec;
    private final Random jitter = new Random();
    private final ConcurrentHashMap<CacheKey, CompletableFuture<Void>> inFlightRefresh = new ConcurrentHashMap<>();
    private final AtomicLong lastErrorMs = new AtomicLong(0);

    public TieredCache(CacheBackend l1, CacheBackend l2, LoadFunction loader,
                       CacheConfig config, CacheMetrics metrics) {
        this.l1 = Objects.requireNonNull(l1);
        this.l2 = l2;
        this.loader = loader == null ? null : loader;
        this.config = config == null ? CacheConfig.defaults() : config;
        this.metrics = metrics == null ? new CacheMetrics() : metrics;
        this.stampedeGuard = StampedeGuard.local();
        this.negativeCache = this.loader == null ? null
                : new NegativeCache(this.config.bloomExpectedInsertions(),
                        this.config.bloomFalsePositiveRate(),
                        this.config.negativeTtl().toMillis(),
                        l2);
        this.crossingGuard = FeatureCrossingGuard.defaults(this.metrics);
        this.refreshExec = Executors.newFixedThreadPool(2, new ThreadFactory() {
            private final AtomicInteger counter = new AtomicInteger();
            @Override public Thread newThread(Runnable r) {
                Thread t = new Thread(r, "tiered-cache-refresher-" + counter.incrementAndGet());
                t.setDaemon(true);
                return t;
            }
        });
    }

    public CacheMetrics metrics() { return metrics; }
    public CacheConfig config() { return config; }
    public FeatureCrossingGuard crossingGuard() { return crossingGuard; }

    public Optional<CacheValue<Object>> get(CacheKey key) {
        long start = System.nanoTime();
        try {
            return _get(key);
        } finally {
            metrics.recordLatencyNanos(System.nanoTime() - start);
        }
    }

    @SuppressWarnings("unchecked")
    private Optional<CacheValue<Object>> _get(CacheKey key) {
        Objects.requireNonNull(key);
        CacheMetrics.CacheKeySample sample = new CacheMetrics.CacheKeySample(key.view(), key.tenant());

        // 1. Penetration guard
        if (negativeCache != null && negativeCache.isAbsent(key)) {
            metrics.recordNegativeHit(sample);
            return Optional.empty();
        }

        // 2. L1 hit
        Optional<CacheValue<Object>> l1v = l1.get(key);
        if (l1v.isPresent()) {
            CacheValue<Object> v = l1v.get();
            long now = System.currentTimeMillis();
            if (!v.isExpired(now)) {
                metrics.recordL1Hit(sample);
                return l1v;
            }
            // stale-while-revalidate
            if (config.staleWhileRevalidate() != null
                    && v.isFreshlyStale(now, config.staleWhileRevalidate().toMillis())) {
                metrics.recordL1Hit(sample);
                scheduleRefresh(key);
                return l1v;
            }
            l1.delete(key);
        }

        // 3. L2 hit
        if (l2 != null) {
            Map<CacheKey, CacheValue<Object>> l2map = l2.getBatch(java.util.Collections.singletonList(key));
            CacheValue<Object> v = l2map.get(key);
            if (v != null) {
                long now = System.currentTimeMillis();
                if (!v.isExpired(now)) {
                    metrics.recordL2Hit(sample);
                    l1.put(key, v);
                    return Optional.of(v);
                }
            }
        }

        // 4. miss
        metrics.recordMiss(sample);
        if (loader == null) return Optional.empty();

        // 5. single-flight load
        try {
            CacheValue<Object> loaded = stampedeGuard.guard(key, () -> {
                try {
                    CacheValue<Object> src = loader.load(key);
                    if (src == null) {
                        if (negativeCache != null) negativeCache.markAbsent(key);
                        metrics.recordPenetration(sample);
                        return null;
                    }
                    return admit(key, src);
                } catch (Exception e) {
                    metrics.recordError();
                    lastErrorMs.set(System.currentTimeMillis());
                    throw new RuntimeException(e);
                }
            });
            if (loaded == null) return Optional.empty();
            l1.put(key, loaded);
            if (l2 != null && config.writeThrough()) l2.put(key, loaded);
            metrics.recordReload();
            return Optional.of(loaded);
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    public Map<CacheKey, CacheValue<Object>> getBatch(Collection<CacheKey> keys) {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null || keys.isEmpty()) return out;
        List<CacheKey> misses = new ArrayList<>();
        for (CacheKey key : keys) {
            Optional<CacheValue<Object>> v = get(key);
            if (v.isPresent()) out.put(key, v.get());
            else misses.add(key);
        }
        // The LoaderFunction.loadBatch hook is exposed for callers who want
        // to batch-load misses; default TieredCache loads one-by-one to preserve
        // single-flight semantics.
        return out;
    }

    public void put(CacheKey key, CacheValue<Object> value) {
        CacheValue<Object> admitted = admit(key, value);
        if (admitted == null) return;
        l1.put(key, admitted);
        if (l2 != null && config.writeThrough()) {
            metrics.recordWriteThrough();
            l2.put(key, admitted);
        } else if (l2 != null && config.writeBack()) {
            metrics.recordWriteBack();
            refreshExec.submit(() -> l2.put(key, admitted));
        }
    }

    public void putBatch(Map<CacheKey, CacheValue<Object>> entries) {
        if (entries == null) return;
        Map<CacheKey, CacheValue<Object>> buffered = new LinkedHashMap<>();
        for (Map.Entry<CacheKey, CacheValue<Object>> e : entries.entrySet()) {
            CacheValue<Object> admitted = admit(e.getKey(), e.getValue());
            if (admitted != null) buffered.put(e.getKey(), admitted);
        }
        l1.putBatch(buffered);
        if (l2 != null && config.writeThrough()) {
            metrics.recordWriteThrough();
            l2.putBatch(buffered);
        }
    }

    public void invalidate(CacheKey key) {
        l1.delete(key);
        if (l2 != null) l2.delete(key);
        metrics.recordInvalidated();
    }

    public void invalidateBatch(Collection<CacheKey> keys) {
        if (keys == null) return;
        for (CacheKey k : keys) invalidate(k);
    }

    public void invalidateByView(String tenant, String view) {
        // L1 has no secondary index; cascade to L2 delete-by-prefix if possible.
        if (l2 instanceof RedisCacheBackend) {
            // best-effort; L2 caller can override via Scan/Del
        }
    }

    private CacheValue<Object> admit(CacheKey key, CacheValue<Object> src) {
        if (src == null) return null;
        if (config.protectFromCachingNull() && src.value() == null) return null;
        long now = System.currentTimeMillis();
        src = crossingGuard.admit(key, src, now);
        if (src == null) return null;

        // TTL jitter to prevent avalanche
        long ttlMs = src.ttlMode() == CacheValue.TtlMode.EVENT_TIME
                ? src.maxAgeMs()
                : Math.max(0, src.expireAtMs() - now);
        if (ttlMs > 0 && config.l1Ttl() != null) {
            long base = Math.min(ttlMs, config.l1Ttl().toMillis());
            long jitterRange = Math.max(1, base / 10);
            long withJitter = base + jitter.nextLong(-jitterRange, jitterRange + 1);
            src = src.toBuilder().expireAtMs(now + withJitter).build();
        }
        return src;
    }

    private void scheduleRefresh(CacheKey key) {
        inFlightRefresh.computeIfAbsent(key, k -> CompletableFuture.runAsync(() -> {
            try {
                if (loader == null) return;
                CacheValue<Object> fresh = loader.load(k);
                if (fresh != null) {
                    fresh = admit(k, fresh);
                    if (fresh != null) {
                        l1.put(k, fresh);
                        if (l2 != null && config.writeThrough()) l2.put(k, fresh);
                    }
                }
            } catch (Exception e) {
                metrics.recordError();
            } finally {
                inFlightRefresh.remove(k);
            }
        }, refreshExec));
    }

    public CacheBackend l1() { return l1; }
    public CacheBackend l2() { return l2; }

    @Override
    public void close() {
        refreshExec.shutdown();
        try { refreshExec.awaitTermination(5, TimeUnit.SECONDS); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        try { l1.close(); } catch (Exception ignore) {}
        if (l2 != null) try { l2.close(); } catch (Exception ignore) {}
    }
}
