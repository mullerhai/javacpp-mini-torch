/*
 * NegativeCache — defence against cache penetration (non-existent keys).
 *
 * <p>Combines two industry-standard mechanisms:
 * <ul>
 *   <li><b>Bloom Filter</b> — probabilistic gate in front of the loader so
 *       requests for keys known to be absent never reach the source system.
 *       False positive rate is configurable; used by Alibaba Tair / Lindorm to
 *       differentiate "never seen" from "seen & absent".</li>
 *   <li><b>Negative TTL</b> — short-lived explicit "absent" entries so a
 *       subsequent query for the same key is short-circuited deterministically.
 *       Used by Meta TAO / Dynein / Redis Bloom modules.</li>
 * </ul>
 *
 * <p>The two work together: bloom handles the cold-start flood, and the negative
 * TTL handles the steady-state with bounded false positives.
 */
package org.bytedeco.pytorch.cache;

import java.util.concurrent.atomic.AtomicLong;

public final class NegativeCache {

    private final BloomFilter bloom;
    private final long negativeTtlMs;
    private final CacheBackend backend;        // optional, for storing negative entries
    private final AtomicLong seenNegatives = new AtomicLong();
    private final AtomicLong blocked = new AtomicLong();
    private final AtomicLong falsePositives = new AtomicLong();

    public NegativeCache(long expectedInsertions, double falsePositiveRate,
                         long negativeTtlMs, CacheBackend backend) {
        this.bloom = new BloomFilter(Math.max(1024, expectedInsertions), falsePositiveRate);
        this.negativeTtlMs = negativeTtlMs <= 0 ? 10_000 : negativeTtlMs;
        this.backend = backend;
    }

    /** Confirm the key is known to be absent. */
    public void markAbsent(CacheKey key) {
        bloom.add(key.toStorageKey());
        seenNegatives.incrementAndGet();
        if (backend != null) {
            try {
                CacheValue<Object> nv = CacheValue.of(CacheValue.NEGATIVE_SENTINEL)
                        .ttlFromNow(negativeTtlMs)
                        .sourceTag("negative")
                        .build();
                backend.put(key, nv);
            } catch (Exception ignore) { /* soft-fail */ }
        }
    }

    /** Check whether the key is definitely absent. */
    public boolean isAbsent(CacheKey key) {
        if (backend != null) {
            try {
                CacheValue<Object> nv = backend.get(key).orElse(null);
                if (nv != null && nv.value() == CacheValue.NEGATIVE_SENTINEL) {
                    blocked.incrementAndGet();
                    return true;
                }
            } catch (Exception ignore) { /* soft-fail */ }
        }
        if (bloom.mightContain(key.toStorageKey())) {
            // probability of false positive — only true if backend confirms
            if (backend == null) {
                blocked.incrementAndGet();
                return true;
            }
            try {
                CacheValue<Object> nv = backend.get(key).orElse(null);
                if (nv == null || nv.value() == CacheValue.NEGATIVE_SENTINEL) {
                    blocked.incrementAndGet();
                    return true;
                }
                // present in L2 — false positive
                falsePositives.incrementAndGet();
                return false;
            } catch (Exception e) {
                return false;
            }
        }
        return false;
    }

    public long blockedCount() { return blocked.get(); }
    public long falsePositiveCount() { return falsePositives.get(); }
    public long seenCount() { return seenNegatives.get(); }

    public long approximateCardinality() { return bloom.approximateCount(); }

    /**
     * BloomFilter — 64-bit FNV-1a + multiple hashes (Kirsch-Mitzenmacher).
     * Size rounded to power of two.
     */
    public static final class BloomFilter {
        private final long[] bits;
        private final int sizeMask;
        private final int hashCount;

        public BloomFilter(long expectedInsertions, double falsePositiveRate) {
            if (expectedInsertions <= 0) expectedInsertions = 1024;
            if (falsePositiveRate <= 0 || falsePositiveRate >= 1) falsePositiveRate = 0.001;
            int size = (int) Math.ceil(-expectedInsertions * Math.log(falsePositiveRate)
                                       / (Math.log(2) * Math.log(2)));
            int pow2 = 1;
            while (pow2 < size) pow2 <<= 1;
            if (pow2 < 1024) pow2 = 1024;
            this.bits = new long[pow2 >>> 6];
            this.sizeMask = pow2 - 1;
            int k = Math.max(1, (int) Math.round(Math.log(2) * pow2 / expectedInsertions));
            this.hashCount = Math.min(8, Math.max(1, k));
        }

        public void add(String key) {
            long h1 = fnv1a64(key);
            long h2 = h1 >>> 32;
            for (int i = 0; i < hashCount; i++) {
                int idx = (int) ((h1 + i * h2) & sizeMask);
                bits[idx >>> 6] |= (1L << (idx & 63));
            }
        }

        public boolean mightContain(String key) {
            long h1 = fnv1a64(key);
            long h2 = h1 >>> 32;
            for (int i = 0; i < hashCount; i++) {
                int idx = (int) ((h1 + i * h2) & sizeMask);
                if ((bits[idx >>> 6] & (1L << (idx & 63))) == 0) return false;
            }
            return true;
        }

        public long approximateCount() {
            long set = 0;
            for (long b : bits) set += Long.bitCount(b);
            int m = bits.length * 64;
            int k = hashCount;
            return (long) Math.max(0, Math.round(-m / (double) k
                    * Math.log(1.0 - set / (double) m)));
        }

        private static long fnv1a64(String s) {
            long h = 0xcbf29ce484222325L;
            for (int i = 0; i < s.length(); i++) {
                h ^= s.charAt(i);
                h *= 0x100000001b3L;
            }
            return h;
        }
    }
}
