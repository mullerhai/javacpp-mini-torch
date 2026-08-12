/*
 * CacheConfig — central tunables for the tiered cache.
 *
 * <p>Defaults align with industry-observed inflection points:
 * <ul>
 *   <li>L1 size 50k, ttl 60s — bytedance Abase</li>
 *   <li>L2 size unlimited (Redis), ttl 5 min</li>
 *   <li>negative cache 5–30s — meta TAO pattern</li>
 *   <li>staleWhileRevalidate 30s — meta Dynein / Facebook TAO</li>
 *   <li>singleFlight concurrency 16 — google Guava FastSlow pattern</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

public final class CacheConfig {

    public enum EvictionPolicy { LRU, LFU, W_TINY_LFU, FIFO }

    private final long l1MaxEntries;
    private final EvictionPolicy l1Eviction;
    private final Duration l1Ttl;
    private final Duration l2Ttl;
    private final Duration negativeTtl;
    private final Duration staleWhileRevalidate;
    private final int singleFlightPermits;
    private final Duration loaderTimeout;
    private final int loaderMaxRetries;
    private final long bloomExpectedInsertions;
    private final double bloomFalsePositiveRate;
    private final int maxEntitiesPerBatch;
    private final boolean writeThrough;
    private final boolean writeBack;
    private final boolean refreshAhead;
    private final boolean protectFromCachingNull;

    private CacheConfig(Builder b) {
        this.l1MaxEntries = b.l1MaxEntries;
        this.l1Eviction = b.l1Eviction;
        this.l1Ttl = b.l1Ttl;
        this.l2Ttl = b.l2Ttl;
        this.negativeTtl = b.negativeTtl;
        this.staleWhileRevalidate = b.staleWhileRevalidate;
        this.singleFlightPermits = b.singleFlightPermits;
        this.loaderTimeout = b.loaderTimeout;
        this.loaderMaxRetries = b.loaderMaxRetries;
        this.bloomExpectedInsertions = b.bloomExpectedInsertions;
        this.bloomFalsePositiveRate = b.bloomFalsePositiveRate;
        this.maxEntitiesPerBatch = b.maxEntitiesPerBatch;
        this.writeThrough = b.writeThrough;
        this.writeBack = b.writeBack;
        this.refreshAhead = b.refreshAhead;
        this.protectFromCachingNull = b.protectFromCachingNull;
    }

    public long l1MaxEntries()                         { return l1MaxEntries; }
    public EvictionPolicy l1Eviction()                 { return l1Eviction; }
    public Duration l1Ttl()                            { return l1Ttl; }
    public Duration l2Ttl()                            { return l2Ttl; }
    public Duration negativeTtl()                      { return negativeTtl; }
    public Duration staleWhileRevalidate()             { return staleWhileRevalidate; }
    public int singleFlightPermits()                   { return singleFlightPermits; }
    public Duration loaderTimeout()                    { return loaderTimeout; }
    public int loaderMaxRetries()                      { return loaderMaxRetries; }
    public long bloomExpectedInsertions()              { return bloomExpectedInsertions; }
    public double bloomFalsePositiveRate()             { return bloomFalsePositiveRate; }
    public int maxEntitiesPerBatch()                   { return maxEntitiesPerBatch; }
    public boolean writeThrough()                      { return writeThrough; }
    public boolean writeBack()                         { return writeBack; }
    public boolean refreshAhead()                      { return refreshAhead; }
    public boolean protectFromCachingNull()            { return protectFromCachingNull; }

    public static Builder builder() { return new Builder(); }

    public static CacheConfig defaults() { return builder().build(); }

    public static final class Builder {
        private long l1MaxEntries = 50_000;
        private EvictionPolicy l1Eviction = EvictionPolicy.W_TINY_LFU;
        private Duration l1Ttl = Duration.ofSeconds(60);
        private Duration l2Ttl = Duration.ofMinutes(5);
        private Duration negativeTtl = Duration.ofSeconds(10);
        private Duration staleWhileRevalidate = Duration.ofSeconds(30);
        private int singleFlightPermits = 16;
        private Duration loaderTimeout = Duration.ofMillis(500);
        private int loaderMaxRetries = 2;
        private long bloomExpectedInsertions = 10_000_000L;
        private double bloomFalsePositiveRate = 0.001;
        private int maxEntitiesPerBatch = 512;
        private boolean writeThrough = true;
        private boolean writeBack = false;
        private boolean refreshAhead = false;
        private boolean protectFromCachingNull = true;

        public Builder l1MaxEntries(long n)                         { this.l1MaxEntries = n; return this; }
        public Builder l1Eviction(EvictionPolicy p)                { this.l1Eviction = p; return this; }
        public Builder l1Ttl(Duration d)                            { this.l1Ttl = d; return this; }
        public Builder l2Ttl(Duration d)                            { this.l2Ttl = d; return this; }
        public Builder negativeTtl(Duration d)                      { this.negativeTtl = d; return this; }
        public Builder staleWhileRevalidate(Duration d)             { this.staleWhileRevalidate = d; return this; }
        public Builder singleFlightPermits(int n)                   { this.singleFlightPermits = n; return this; }
        public Builder loaderTimeout(Duration d)                    { this.loaderTimeout = d; return this; }
        public Builder loaderMaxRetries(int n)                      { this.loaderMaxRetries = n; return this; }
        public Builder bloomExpectedInsertions(long n)              { this.bloomExpectedInsertions = n; return this; }
        public Builder bloomFalsePositiveRate(double r)             { this.bloomFalsePositiveRate = r; return this; }
        public Builder maxEntitiesPerBatch(int n)                   { this.maxEntitiesPerBatch = n; return this; }
        public Builder writeThrough(boolean b)                      { this.writeThrough = b; return this; }
        public Builder writeBack(boolean b)                         { this.writeBack = b; return this; }
        public Builder refreshAhead(boolean b)                      { this.refreshAhead = b; return this; }
        public Builder protectFromCachingNull(boolean b)            { this.protectFromCachingNull = b; return this; }

        public CacheConfig build() { return new CacheConfig(this); }
    }

    public static class Helpers {
        public static long ms(Duration d) { return d == null ? 0 : d.toMillis(); }
        public static Duration ttlFromMs(long ms) { return Duration.ofMillis(ms); }
        public static TimeUnit pickUnit(Duration d) {
            long s = d.getSeconds();
            if (s <= 0) return TimeUnit.MILLISECONDS;
            if (s < 60) return TimeUnit.SECONDS;
            if (s < 3600) return TimeUnit.MINUTES;
            return TimeUnit.HOURS;
        }
    }
}
