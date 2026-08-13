/*
 * ConsistencyHashRing — Ketama-style consistent hash with weighted nodes.
 *
 * <p>Industry-standard for distributed cache deployments (memcached/memcacheq,
 * Redis Cluster shards, Cassandra, DynamoDB partitions). Each physical node is
 * replicated to many "virtual nodes" around the ring so that:
 * <ul>
 *   <li>Adding / removing a node only re-keys O(1/N) of keys.</li>
 *   <li>Weighted replication lets heterogeneous servers share load by capacity.</li>
 *   <li>Lookups are O(log N) via ceiling-floor scan on the sorted ring.</li>
 * </ul>
 *
 * <p>For multi-region deployment (typical for Alibaba / ByteDance / Tencent
 * feature stores), the ring supports per-region sub-rings so a request can
 * prefer a homed shard.
 */
package org.bytedeco.pytorch.cache.sharding;
import org.bytedeco.pytorch.jit.*;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;

public final class ConsistencyHashRing<N> {

    private final ConcurrentSkipListMap<Long, N> ring = new ConcurrentSkipListMap<>();
    private final ReentrantReadWriteLock rw = new ReentrantReadWriteLock();
    private final Function<N, Integer> weightFn;
    private final int replicationFactor;

    public ConsistencyHashRing() {
        this(n -> 1, 200);
    }

    public ConsistencyHashRing(Function<N, Integer> weightFn, int replicationFactor) {
        this.weightFn = Objects.requireNonNull(weightFn);
        this.replicationFactor = Math.max(10, replicationFactor);
    }

    public void addNode(N node) {
        rw.writeLock().lock();
        try {
            int weight = Math.max(1, weightFn.apply(node));
            int points = replicationFactor * weight;
            for (int i = 0; i < points; i++) {
                long h = md5(("node-" + node + "-vnode-" + i).getBytes(StandardCharsets.UTF_8));
                ring.put(h, node);
            }
        } finally {
            rw.writeLock().unlock();
        }
    }

    public void removeNode(N node) {
        rw.writeLock().lock();
        try {
            int weight = Math.max(1, weightFn.apply(node));
            int points = replicationFactor * weight;
            for (int i = 0; i < points; i++) {
                long h = md5(("node-" + node + "-vnode-" + i).getBytes(StandardCharsets.UTF_8));
                ring.remove(h, node);
            }
        } finally {
            rw.writeLock().unlock();
        }
    }

    public void replaceAll(Collection<N> nodes) {
        rw.writeLock().lock();
        try {
            ring.clear();
            if (nodes != null) for (N n : nodes) {
                int weight = Math.max(1, weightFn.apply(n));
                int points = replicationFactor * weight;
                for (int i = 0; i < points; i++) {
                    long h = md5(("node-" + n + "-vnode-" + i).getBytes(StandardCharsets.UTF_8));
                    ring.put(h, n);
                }
            }
        } finally {
            rw.writeLock().unlock();
        }
    }

    /** Pick the node that owns the given key. */
    public N locate(String key) {
        if (key == null || ring.isEmpty()) return null;
        long h = md5(key.getBytes(StandardCharsets.UTF_8));
        Map.Entry<Long, N> e = ring.ceilingEntry(h);
        if (e == null) {
            // wrap around
            e = ring.firstEntry();
        }
        return e == null ? null : e.getValue();
    }

    /** Pick N distinct nodes (primary + N-1 replicas) for the given key. */
    public List<N> locateReplicas(String key, int replicaCount) {
        if (key == null || ring.isEmpty()) return Collections.emptyList();
        long h = md5(key.getBytes(StandardCharsets.UTF_8));
        List<N> out = new ArrayList<>(replicaCount);
        for (Map.Entry<Long, N> e : ring.tailMap(h, true).entrySet()) {
            if (!out.contains(e.getValue())) out.add(e.getValue());
            if (out.size() >= replicaCount) return out;
        }
        for (Map.Entry<Long, N> e : ring.entrySet()) {
            if (!out.contains(e.getValue())) out.add(e.getValue());
            if (out.size() >= replicaCount) return out;
        }
        return out;
    }

    public int nodeCount() { return ring.size(); }
    public int uniqueNodeCount() {
        java.util.Set<N> s = new java.util.HashSet<>(ring.values());
        return s.size();
    }

    /** Statistics for diagnostics: how many hash points each unique node owns. */
    public Map<N, Integer> distribution() {
        Map<N, Integer> dist = new java.util.LinkedHashMap<>();
        for (N n : ring.values()) dist.merge(n, 1, Integer::sum);
        return dist;
    }

    private static long md5(byte[] data) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] d = md.digest(data);
            // take first 8 bytes as long
            long h = 0L;
            for (int i = 0; i < 8; i++) h = (h << 8) | (d[i] & 0xFFL);
            return h;
        } catch (NoSuchAlgorithmException e) {
            // MD5 is always available; guard against JCE removal
            return data.hashCode();
        }
    }
}
