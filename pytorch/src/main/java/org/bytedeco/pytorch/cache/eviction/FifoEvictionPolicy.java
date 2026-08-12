/*
 * FifoEvictionPolicy -- first-in / first-out. Useful for queues where recency
 * does not indicate value (e.g. write-behind buffers, audit-style caches).
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

public final class FifoEvictionPolicy implements EvictionPolicy {

    private final DoubleLinkedList<CacheKey> list = new DoubleLinkedList<>();
    private final EvictionStats stats = new EvictionStats();

    @Override public String name() { return "FIFO"; }

    @Override
    public synchronized void touch(CacheKey key) {
        // touch is a no-op for FIFO; we record it for observability only
        if (key == null) return;
        stats.recordTouch();
    }

    @Override
    public synchronized void admit(CacheKey key) {
        if (key == null) return;
        list.addLast(key);
        stats.recordAdmit();
    }

    @Override
    public synchronized void remove(CacheKey key) {
        if (key == null) return;
        list.remove(key);
    }

    @Override
    public synchronized long size() { return list.size(); }

    @Override
    public synchronized CacheKey pickVictim() {
        CacheKey v = list.peekFirst();
        if (v != null) stats.recordEviction();
        return v;
    }

    @Override
    public synchronized void clear() { list.clear(); }

    @Override
    public EvictionStats stats() { return stats; }
}
