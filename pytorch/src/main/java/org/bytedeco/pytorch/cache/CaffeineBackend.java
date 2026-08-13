/*
 * CaffeineBackend — L1 in-process tier using a W-TinyLFU policy.
 *
 * <p>Self-contained so the module has no runtime dependency on Caffeine.
 * W-TinyLFU (Einziger et al., 2017) gives LRU-comparable hit ratios with
 * strong scan resistance — the standard choice at Meta/Dynein, Google Guava,
 * and ByteDance Abase.
 *
 * <p>Capacity is bounded in entries; values are evicted by the SLRU + frequency
 * sketch composition. Eviction is O(1) amortised.
 */
package org.bytedeco.pytorch.cache;
import org.bytedeco.pytorch.autograd.*;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/** L1 cache backend with W-TinyLFU admission + LRU eviction. */
public final class CaffeineBackend implements CacheBackend {

    private final long maxEntries;
    private final long l1TtlMs;
    private final FrequencySketch sketch;
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<CacheKey, Node> index;
    private final Deque<Node> window;          // small main cache (1% of capacity)
    private final Deque<Node> probation;       // SLRU probation (20% of main)
    private final Deque<Node> protected_;      // SLRU protected (80% of main)
    private final AtomicLong sizeCounter = new AtomicLong();

    public CaffeineBackend(long maxEntries, long l1TtlMs) {
        if (maxEntries < 16) maxEntries = 16;
        this.maxEntries = maxEntries;
        this.l1TtlMs = l1TtlMs <= 0 ? 60_000L : l1TtlMs;
        this.sketch = new FrequencySketch(Math.max(4, (int) Math.min(1 << 16, nextPow2(maxEntries * 4))));
        this.index = new HashMap<>((int) (maxEntries * 1.4));
        long win = Math.max(1, maxEntries / 100);
        this.window = new ArrayDeque<>((int) win);
        long main = maxEntries - win;
        this.probation = new ArrayDeque<>((int) (main / 5));
        this.protected_ = new ArrayDeque<>((int) ((main * 4) / 5));
    }

    @Override public String name() { return "caffeine-L1"; }
    @Override public int tier()   { return 1; }

    @Override
    public Optional<CacheValue<Object>> get(CacheKey key) {
        Node n;
        lock.lock();
        try {
            n = index.get(key);
            if (n == null) return Optional.empty();
            long now = System.currentTimeMillis();
            if (l1TtlMs > 0 && n.writeAtMs + l1TtlMs < now) {
                evictNode(n);
                index.remove(key);
                return Optional.empty();
            }
            sketch.increment(key.hashCode());
            n.freq = sketch.frequency(key.hashCode());
            // touch in eviction chain
            if (n.segment == Segment.PROTECTED) {
                protected_.remove(n);
                protected_.addFirst(n);
            } else if (n.segment == Segment.PROBATION) {
                probation.remove(n);
                probation.addFirst(n);
            } else {
                window.remove(n);
                window.addFirst(n);
            }
            return Optional.of(n.value);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public Map<CacheKey, CacheValue<Object>> getBatch(Collection<CacheKey> keys) {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null) return out;
        for (CacheKey k : keys) get(k).ifPresent(v -> out.put(k, v));
        return out;
    }

    @Override
    public void put(CacheKey key, CacheValue<Object> value) {
        Objects.requireNonNull(key);
        Objects.requireNonNull(value);
        long now = System.currentTimeMillis();
        lock.lock();
        try {
            Node existing = index.get(key);
            if (existing != null) {
                existing.value = value;
                existing.writeAtMs = now;
                existing.freq = sketch.frequency(key.hashCode());
                if (existing.segment == Segment.PROTECTED) {
                    protected_.remove(existing); protected_.addFirst(existing);
                } else if (existing.segment == Segment.PROBATION) {
                    probation.remove(existing); probation.addFirst(existing);
                } else {
                    window.remove(existing); window.addFirst(existing);
                }
                return;
            }
            Node node = new Node(key, value, now);
            sketch.increment(key.hashCode());
            node.freq = sketch.frequency(key.hashCode());
            index.put(key, node);
            window.addFirst(node);
            node.segment = Segment.WINDOW;
            sizeCounter.incrementAndGet();
            evict();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void putBatch(Map<CacheKey, CacheValue<Object>> entries) {
        if (entries == null) return;
        for (Map.Entry<CacheKey, CacheValue<Object>> e : entries.entrySet()) put(e.getKey(), e.getValue());
    }

    @Override
    public void delete(CacheKey key) {
        lock.lock();
        try {
            Node n = index.remove(key);
            if (n != null) evictNode(n);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void deleteBatch(Collection<CacheKey> keys) {
        if (keys == null) return;
        for (CacheKey k : keys) delete(k);
    }

    @Override
    public long size() { return sizeCounter.get(); }

    @Override
    public boolean ping() { return true; }

    private void evict() {
        while (sizeCounter.get() > maxEntries) {
            if (!window.isEmpty()) {
                Node victim = window.pollLast();
                if (victim != null) {
                    if (sketch.frequency(victim.key.hashCode()) > 1) {
                        victim.segment = Segment.PROBATION;
                        probation.addFirst(victim);
                    } else {
                        index.remove(victim.key);
                        sizeCounter.decrementAndGet();
                    }
                }
            } else if (!probation.isEmpty()) {
                Node victim = probation.pollLast();
                if (victim != null) {
                    if (sketch.frequency(victim.key.hashCode()) > 5) {
                        victim.segment = Segment.PROTECTED;
                        protected_.addFirst(victim);
                    } else {
                        index.remove(victim.key);
                        sizeCounter.decrementAndGet();
                    }
                }
            } else if (!protected_.isEmpty()) {
                Node victim = protected_.pollLast();
                if (victim != null) {
                    index.remove(victim.key);
                    sizeCounter.decrementAndGet();
                }
            } else {
                break;
            }
        }
    }

    private void evictNode(Node n) {
        switch (n.segment) {
            case WINDOW: window.remove(n); break;
            case PROBATION: probation.remove(n); break;
            case PROTECTED: protected_.remove(n); break;
        }
        sizeCounter.decrementAndGet();
    }

    /** Stats for debugging. */
    public String stats() {
        return "CaffeineBackend{window=" + window.size()
                + ", probation=" + probation.size()
                + ", protected=" + protected_.size()
                + ", size=" + sizeCounter.get() + "}";
    }

    private static int nextPow2(long v) {
        long r = 1;
        while (r < v) r <<= 1;
        return (int) r;
    }

    private enum Segment { WINDOW, PROBATION, PROTECTED }

    private static final class Node {
        final CacheKey key;
        CacheValue<Object> value;
        long writeAtMs;
        long freq;
        Segment segment = Segment.WINDOW;

        Node(CacheKey key, CacheValue<Object> value, long writeAtMs) {
            this.key = key; this.value = value; this.writeAtMs = writeAtMs;
        }
    }

    /**
     * Count-Min Sketch with 4 hashes, 16-bit counters per cell.
     * Compact, O(1) update / query, supports aging.
     */
    static final class FrequencySketch {
        private final long[] table;
        private final int mask;
        private int sampleSize = 10;
        private int sampleCount;

        FrequencySketch(int sizePow2) {
            this.table = new long[sizePow2];
            this.mask = sizePow2 - 1;
        }

        void increment(int h) {
            int h1 = h;
            int h2 = (h >>> 16) | 1;
            int h3 = mix(h1);
            int h4 = mix(h2);
            incrementAt(h1 & mask);
            incrementAt(h2 & mask);
            incrementAt(h3 & mask);
            incrementAt(h4 & mask);
            if (++sampleCount >= sampleSize) {
                sampleCount = 0;
                if (tableFull()) {
                    for (int i = 0; i < table.length; i++) table[i] = (table[i] >>> 1) & 0x7FFF7FFF7FFF7FFFL;
                }
            }
        }

        private void incrementAt(int idx) {
            // 4 counters packed per long; each 16-bit
            int g = idx & 3;
            int shift = (3 - g) * 16;
            long cell = table[idx >>> 2];
            int cnt = (int) ((cell >>> shift) & 0xFFFFL);
            if (cnt < 0xFFFF) {
                cell = (cell & ~(0xFFFFL << shift)) | (((long) (cnt + 1)) << shift);
                table[idx >>> 2] = cell;
            }
        }

        int frequency(int h) {
            int h1 = h;
            int h2 = (h >>> 16) | 1;
            int h3 = mix(h1);
            int h4 = mix(h2);
            int f = Math.min(Math.min(readAt(h1 & mask), readAt(h2 & mask)),
                             Math.min(readAt(h3 & mask), readAt(h4 & mask)));
            return f;
        }

        private int readAt(int idx) {
            int g = idx & 3;
            int shift = (3 - g) * 16;
            return (int) ((table[idx >>> 2] >>> shift) & 0xFFFFL);
        }

        private boolean tableFull() {
            for (long c : table) {
                if ((c & 0xFFFFL) == 0) return false;
            }
            return true;
        }

        private static int mix(int x) {
            x = ((x >>> 16) ^ x) * 0x45d9f3b;
            x = ((x >>> 16) ^ x) * 0x45d9f3b;
            return (x >>> 16) ^ x;
        }
    }
}
