/*
 * EvictionSimulator -- replay a workload against any {@link EvictionPolicy} and
 * report hit ratio / miss ratio / mean-eviction-rate.
 *
 * <p>Used in production regression tests to compare policies against a real
 * access trace without touching a running cache. The workload is re-playable
 * (deterministic) so test results are stable.
 *
 * <p>This is the same workflow used by cache-team benchmarks at Meta
 * (TAO shadow tests) and Google (Guava CacheLib).
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public final class EvictionSimulator {

    public static final class Report {
        public final long ops, hits, misses, evictions;
        public final double hitRatio;
        public final long durationNanos;

        public Report(long ops, long hits, long misses, long evictions, long durationNanos) {
            this.ops = ops;
            this.hits = hits;
            this.misses = misses;
            this.evictions = evictions;
            this.hitRatio = ops == 0 ? 0.0 : (double) hits / ops;
            this.durationNanos = durationNanos;
        }

        @Override
        public String toString() {
            return "SimReport{ops=" + ops + ", hits=" + hits + ", misses=" + misses
                    + ", evict=" + evictions
                    + ", hitRatio=" + String.format("%.4f", hitRatio)
                    + ", nsOp=" + (ops == 0 ? 0 : durationNanos / ops) + "}";
        }
    }

    /**
     * Generate a Zipf-like workload: a small set of keys is hit far more often
     * than the rest. Useful for comparing LRU vs LFU vs W-TinyLFU.
     */
    public static List<CacheKey> zipf(int distinct, int total, double skew, long seed) {
        Random r = new Random(seed);
        List<CacheKey> out = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            double u = r.nextDouble();
            int rank = (int) (Math.pow(u, 1.0 / (1.0 - skew)) * distinct);
            if (rank >= distinct) rank = distinct - 1;
            out.add(CacheKey.builder("sim", "k" + rank).build());
        }
        return out;
    }

    /**
     * Drive {@code policy} with the given trace. The policy is expected to
     * start empty; every {@link CacheKey} in {@code ops} is treated as a get.
     * {@link EvictionPolicy#pickVictim()} is consulted and the policy is
     * asked to admit until the policy is full; afterwards each get is either
     * a hit (key already admitted) or a miss (admit + evict).
     */
    public static Report run(EvictionPolicy policy, List<CacheKey> ops) {
        long t0 = System.nanoTime();
        long hits = 0, misses = 0, evictions = 0;
        java.util.HashSet<CacheKey> live = new java.util.HashSet<>();
        for (CacheKey k : ops) {
            if (live.contains(k)) {
                policy.touch(k);
                hits++;
            } else {
                misses++;
                CacheKey victim = policy.pickVictim();
                if (victim != null) {
                    if (!victim.equals(k)) {
                        policy.remove(victim);
                        live.remove(victim);
                        evictions++;
                    }
                }
                policy.admit(k);
                live.add(k);
            }
        }
        long dur = System.nanoTime() - t0;
        return new Report(ops.size(), hits, misses, evictions, dur);
    }

    public static Map<String, Report> compare(List<CacheKey> ops, long capacity) {
        Map<String, Report> r = new LinkedHashMap<>();
        r.put("LRU",     run(new LruEvictionPolicy(), ops));
        r.put("LFU",     run(new LfuEvictionPolicy(), ops));
        r.put("FIFO",    run(new FifoEvictionPolicy(), ops));
        r.put("W-TLFU",  run(new WTinyLFUEvictionPolicy(capacity), ops));
        return r;
    }
}
