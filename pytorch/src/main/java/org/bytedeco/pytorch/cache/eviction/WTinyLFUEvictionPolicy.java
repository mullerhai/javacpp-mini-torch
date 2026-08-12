/*
 * WTinyLFUEvictionPolicy -- Window Tiny Least-Frequently-Used.
 *
 * <p>Mainstream pick for high hit-ratio caches (Caffeine, Ristretto, CacheLib).
 * Architecture:
 * <ul>
 *   <li><b>Window cache (1% of capacity)</b> -- admits every new key; uses FIFO
 *       so new contenders can challenge the main cache without immediate
 *       expulsion.</li>
 *   <li><b>Main cache (99% of capacity)</b> -- segmented LRU over keys that
 *       survived the window and the frequency sketch.</li>
 *   <li><b>Frequency sketch (CountMinSketch)</b> -- 4-bit, depth-4, used at
 *       promote/evict time to decide whether a window candidate outranks the
 *       main candidate.</li>
 *   <li><b>Half-life reset</b> -- every {@link #sampleSize} * 10 increments,
 *       counters are halved so old skew doesn't poison hot-set detection.</li>
 * </ul>
 *
 * <p>Locks: all mutators guarded by `this`. This is the same lock discipline
 * as Caffeine's BoundedLocalCache since the lock is held only over a small
 * critical section (linked-list + sketch update).
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

public final class WTinyLFUEvictionPolicy implements EvictionPolicy {

    private final long maxCapacity;
    private final long windowSize;
    private final long mainSize;
    private final double percentMain;
    private final boolean recordStats;

    private final DoubleLinkedList<CacheKey> windowLru = new DoubleLinkedList<>();
    private final DoubleLinkedList<CacheKey> mainLru = new DoubleLinkedList<>();
    private final DoubleLinkedList<CacheKey> mainProtectedLru = new DoubleLinkedList<>();
    private final java.util.HashMap<CacheKey, Boolean> windowVictims = new java.util.HashMap<>();
    private final CountMinSketch sketch;
    private final long sampleSize;
    private final EvictionStats stats = new EvictionStats();

    public WTinyLFUEvictionPolicy(long capacity) {
        this(capacity, 0.99, 16_384, true);
    }

    public WTinyLFUEvictionPolicy(long capacity, double percentMain, int sketchWidth, boolean recordStats) {
        if (capacity < 4) throw new IllegalArgumentException("capacity must be >= 4");
        if (percentMain <= 0.5 || percentMain >= 1.0)
            throw new IllegalArgumentException("percentMain must be in (0.5, 1.0)");
        this.maxCapacity = capacity;
        this.percentMain = percentMain;
        this.recordStats = recordStats;
        this.mainSize = Math.max(1, (long) (capacity * percentMain));
        this.windowSize = Math.max(1, capacity - mainSize);
        this.sketch = new CountMinSketch(4, nextPowerOfTwo(sketchWidth));
        this.sampleSize = 10 * Math.max(1, sketch.width());
    }

    @Override public String name() { return "W-TINY_LFU"; }

    @Override
    public synchronized void touch(CacheKey key) {
        if (key == null) return;
        if (afterAdmit(key)) {
            stats.recordTouch();
        }
    }

    @Override
    public synchronized void admit(CacheKey key) {
        if (key == null) return;
        int hash = hashOf(key);
        sketch.increment(hash);
        if (recordStats && sketch.totalAdds() >= sampleSize) {
            sketch.reset();
        }
        // window candidate
        if (windowLru.size() >= windowSize) {
            CacheKey windowVictim = windowLru.peekFirst();
            if (windowVictim != null) {
                int victimFreq = sketch.estimate(hashOf(windowVictim));
                int candidateFreq = sketch.estimate(hash);
                if (candidateFreq > victimFreq) {
                    windowLru.remove(windowVictim);
                    windowLru.addLast(key);
                    stats.recordAdmit();
                    return;
                }
                if (candidateFreq == 5 && victimFreq != 5) {
                    windowLru.remove(windowVictim);
                    windowLru.addLast(key);
                    stats.recordAdmit();
                    return;
                }
            }
        }
        windowLru.addLast(key);
        stats.recordAdmit();
    }

    @Override
    public synchronized void remove(CacheKey key) {
        if (key == null) return;
        windowLru.remove(key);
        mainLru.remove(key);
        mainProtectedLru.remove(key);
        windowVictims.remove(key);
    }

    @Override
    public synchronized long size() {
        return windowLru.size() + mainLru.size() + mainProtectedLru.size();
    }

    @Override
    public synchronized CacheKey pickVictim() {
        if (windowLru.isEmpty() && mainLru.isEmpty()) return null;
        CacheKey victim = selectVictim();
        if (victim != null) stats.recordEviction();
        return victim;
    }

    @Override
    public synchronized void clear() {
        windowLru.clear();
        mainLru.clear();
        mainProtectedLru.clear();
        windowVictims.clear();
    }

    @Override
    public EvictionStats stats() { return stats; }

    public long mainCapacity() { return mainSize; }
    public long windowCapacity() { return windowSize; }

    private boolean afterAdmit(CacheKey key) {
        if (windowLru.contains(key)) {
            windowLru.moveToBack(key);
            return true;
        }
        if (mainLru.contains(key)) {
            int hash = hashOf(key);
            int freq = sketch.estimate(hash);
            sketch.increment(hash);
            if (recordStats && sketch.totalAdds() >= sampleSize) sketch.reset();
            if (freq > 0) {
                mainLru.remove(key);
                mainProtectedLru.addLast(key);
                maybePromoteProtected();
            } else {
                mainLru.moveToBack(key);
            }
            return true;
        }
        if (mainProtectedLru.contains(key)) {
            int hash = hashOf(key);
            sketch.increment(hash);
            if (recordStats && sketch.totalAdds() >= sampleSize) sketch.reset();
            mainProtectedLru.moveToBack(key);
            return true;
        }
        return false;
    }

    private void maybePromoteProtected() {
        // promote from mainProtected to main tail (keep within 80% of main)
        long limit = (long) (mainSize * 0.8);
        if (mainProtectedLru.size() > limit) {
            CacheKey demoted = mainProtectedLru.popFirst();
            if (demoted != null) mainLru.addLast(demoted);
        }
    }

    private CacheKey selectVictim() {
        // Mirror Caffeine's admit+evict dance: candidate comes from window or
        // main; the loser is evicted.
        CacheKey windowVictim = windowLru.peekFirst();
        CacheKey mainVictim = mainLru.peekFirst();

        if (windowVictim == null) return mainVictim;
        if (mainVictim == null) return windowVictim;

        int wFreq = sketch.estimate(hashOf(windowVictim));
        int mFreq = sketch.estimate(hashOf(mainVictim));

        if (wFreq > mFreq) {
            return windowVictim;
        }
        if (wFreq < mFreq) {
            return mainVictim;
        }
        // tie: prefer evicting from main (older), keep window warm
        if (windowLru.size() >= windowSize) {
            return windowVictim;
        }
        return mainVictim;
    }

    private static int hashOf(CacheKey key) {
        return key.hashCode();
    }

    private static int nextPowerOfTwo(int n) {
        if (n < 16) return 16;
        int p = 16;
        while (p < n) p <<= 1;
        return p;
    }
}
