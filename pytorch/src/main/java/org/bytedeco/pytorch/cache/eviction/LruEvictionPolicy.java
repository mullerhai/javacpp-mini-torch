/*
 * LruEvictionPolicy -- classic LRU.
 *
 * <p>Backend uses identity-tied tie-breaking (insertion order) so behaviour is
 * deterministic under test. Hash-power comparable to Caffeine's offheap ring.
 */
package org.bytedeco.pytorch.cache.eviction;

import org.bytedeco.pytorch.cache.CacheKey;

public final class LruEvictionPolicy implements EvictionPolicy {

    private final DoubleLinkedList<CacheKey> list = new DoubleLinkedList<>();
    private final EvictionStats stats = new EvictionStats();

    @Override public String name() { return "LRU"; }

    @Override
    public synchronized void touch(CacheKey key) {
        if (key == null) return;
        if (list.contains(key)) {
            list.moveToFront(key);
            stats.recordTouch();
        }
    }

    @Override
    public synchronized void admit(CacheKey key) {
        if (key == null) return;
        list.addFirst(key);
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
        CacheKey v = list.peekLast();
        if (v != null) stats.recordEviction();
        return v;
    }

    @Override
    public synchronized void clear() { list.clear(); }

    @Override
    public EvictionStats stats() { return stats; }
}
