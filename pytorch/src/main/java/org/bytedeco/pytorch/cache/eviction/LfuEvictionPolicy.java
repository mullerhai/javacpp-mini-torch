/*
 * LfuEvictionPolicy -- pure LFU (least-frequently-used counter).
 *
 * <p>For best production hit-ratio prefer W-TinyLFU. This LFU is useful when
 * the workload is dominated by a small stable hot set (configuration, model
 * metadata) where the cost of the W-TinyLFU sketch is unwarranted.
 *
 * <p>Frequency saturates at {@link #MAX_FREQ} to bound skew from a single
 * runaway key. Eviction picks the key with the lowest frequency; ties broken
 * by oldest admission (not strictly LFU, but avoids pathological O(n) sweeps).
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

public final class LfuEvictionPolicy implements EvictionPolicy {

    static final int MAX_FREQ = 255;

    private final HashMap<CacheKey, Integer> freq = new HashMap<>();
    private final LinkedHashMap<CacheKey, Long> insertion = new LinkedHashMap<>();
    private final Map<Integer, DoubleLinkedList<CacheKey>> byFreq = new HashMap<>();
    private final EvictionStats stats = new EvictionStats();
    private long insertionSerial = 0;

    @Override public String name() { return "LFU"; }

    @Override
    public synchronized void touch(CacheKey key) {
        if (key == null || !freq.containsKey(key)) return;
        int old = freq.get(key);
        int next = Math.min(MAX_FREQ, old + 1);
        freq.put(key, next);
        moveBetweenFreqLists(key, old, next);
        stats.recordTouch();
    }

    @Override
    public synchronized void admit(CacheKey key) {
        if (key == null) return;
        int f = freq.getOrDefault(key, 0);
        if (f == 0) {
            freq.put(key, 1);
            addToFreqList(1, key);
        } else {
            freq.put(key, f + 1);
            moveBetweenFreqLists(key, f, f + 1);
        }
        if (!insertion.containsKey(key)) {
            insertion.put(key, ++insertionSerial);
        }
        stats.recordAdmit();
    }

    @Override
    public synchronized void remove(CacheKey key) {
        if (key == null) return;
        Integer f = freq.remove(key);
        insertion.remove(key);
        if (f != null) {
            DoubleLinkedList<CacheKey> bucket = byFreq.get(f);
            if (bucket != null) bucket.remove(key);
        }
    }

    @Override
    public synchronized long size() { return freq.size(); }

    @Override
    public synchronized CacheKey pickVictim() {
        for (int f = 1; f <= MAX_FREQ; f++) {
            DoubleLinkedList<CacheKey> bucket = byFreq.get(f);
            if (bucket != null && !bucket.isEmpty()) {
                // among lowest-frequency, pick oldest (LRU tie-break)
                CacheKey oldest = null;
                long oldestSerial = Long.MAX_VALUE;
                for (CacheKey k : bucket) {
                    Long s = insertion.get(k);
                    if (s != null && s < oldestSerial) {
                        oldestSerial = s;
                        oldest = k;
                    }
                }
                if (oldest != null) {
                    stats.recordEviction();
                    return oldest;
                }
            }
        }
        return null;
    }

    @Override
    public synchronized void clear() {
        freq.clear();
        insertion.clear();
        byFreq.clear();
        insertionSerial = 0;
    }

    @Override
    public EvictionStats stats() { return stats; }

    private void addToFreqList(int f, CacheKey key) {
        byFreq.computeIfAbsent(f, k -> new DoubleLinkedList<>()).addLast(key);
    }

    private void moveBetweenFreqLists(CacheKey key, int oldF, int newF) {
        DoubleLinkedList<CacheKey> oldBucket = byFreq.get(oldF);
        if (oldBucket != null) oldBucket.remove(key);
        if (newF > 0) addToFreqList(newF, key);
    }
}
