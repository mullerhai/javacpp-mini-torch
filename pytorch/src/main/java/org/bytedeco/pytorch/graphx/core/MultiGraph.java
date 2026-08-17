/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.core;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;

/**
 * Undirected graph allowing parallel edges (multi-edges).
 *
 * <p>Aligned with {@code networkx.MultiGraph}. Each (u, v) pair may have
 * multiple edges distinguished by an integer {@code key}.
 */
public class MultiGraph<N> {
    protected final Map<N, AttrMap> nodeAttr;
    /** _adj: u -> {v -> {key -> EdgeRecord}}. */
    protected final Map<N, Map<N, Map<Integer, Graph.EdgeRecord>>> adj;
    protected long edgeCount;
    /** Auto-increment key for new edges. */
    protected int nextKey;

    public MultiGraph() {
        this(new LinkedHashMap<>(), new LinkedHashMap<>(), 0);
    }

    protected MultiGraph(Map<N, AttrMap> nodeAttr,
                         Map<N, Map<N, Map<Integer, Graph.EdgeRecord>>> adj,
                         int nextKey) {
        this.nodeAttr = nodeAttr;
        this.adj = adj;
        this.nextKey = nextKey;
    }

    public int order() { return nodeAttr.size(); }
    public int numberOfNodes() { return order(); }
    public long size() { return edgeCount; }
    public long numberOfEdges() { return edgeCount; }
    public boolean isEmpty() { return nodeAttr.isEmpty(); }

    public List<N> nodes() {
        return Collections.unmodifiableList(new ArrayList<>(nodeAttr.keySet()));
    }

    public Set<N> nodeSet() {
        return Collections.unmodifiableSet(nodeAttr.keySet());
    }

    public List<Map.Entry<N, N>> edges() {
        List<Map.Entry<N, N>> result = new ArrayList<>((int) edgeCount);
        for (Map.Entry<N, Map<N, Map<Integer, Graph.EdgeRecord>>> e : adj.entrySet()) {
            N u = e.getKey();
            for (N v : e.getValue().keySet()) {
                for (Integer key : e.getValue().get(v).keySet()) {
                    // For undirected, dedupe by (u, v, key) symmetric pair
                    result.add(new java.util.AbstractMap.SimpleImmutableEntry<>(u, v));
                }
            }
        }
        return Collections.unmodifiableList(dedupe(result));
    }

    public List<EdgeKey<N>> edgeKeys() {
        List<EdgeKey<N>> result = new ArrayList<>((int) edgeCount);
        for (Map.Entry<N, Map<N, Map<Integer, Graph.EdgeRecord>>> e : adj.entrySet()) {
            N u = e.getKey();
            for (Map.Entry<N, Map<Integer, Graph.EdgeRecord>> e2 : e.getValue().entrySet()) {
                N v = e2.getKey();
                for (Integer key : e2.getValue().keySet()) {
                    result.add(new EdgeKey<>(u, v, key));
                }
            }
        }
        return Collections.unmodifiableList(dedupeKeys(result));
    }

    private static <N> List<Map.Entry<N, N>> dedupe(List<Map.Entry<N, N>> in) {
        java.util.Set<Long> seen = new java.util.HashSet<>();
        List<Map.Entry<N, N>> out = new ArrayList<>(in.size());
        for (Map.Entry<N, N> e : in) {
            long h = Graph.pairHash(e.getKey(), e.getValue());
            if (seen.add(h)) out.add(e);
        }
        return out;
    }

    private static <N> List<EdgeKey<N>> dedupeKeys(List<EdgeKey<N>> in) {
        java.util.Set<Long> seen = new java.util.HashSet<>();
        List<EdgeKey<N>> out = new ArrayList<>(in.size());
        for (EdgeKey<N> k : in) {
            long h = Objects.hash(k.u(), k.v(), k.key()) ^ Objects.hash(k.v(), k.u(), k.key());
            if (seen.add(h)) out.add(k);
        }
        return out;
    }

    /** Sum of degrees. Self-loop counts 2. */
    public int degree(N node) {
        Map<N, Map<Integer, Graph.EdgeRecord>> nbrs = adj.get(node);
        if (nbrs == null) throw new NoSuchElementException("No node " + node);
        int d = 0;
        for (Map<Integer, Graph.EdgeRecord> m : nbrs.values()) {
            d += m.size();
        }
        Map<Integer, Graph.EdgeRecord> self = nbrs.get(node);
        if (self != null) d += self.size(); // self-loop already counted once above; +1 per copy
        return d;
    }

    public Map<N, Integer> degrees() {
        Map<N, Integer> m = new LinkedHashMap<>();
        for (N n : nodeAttr.keySet()) m.put(n, degree(n));
        return m;
    }

    public double density() {
        int n = order();
        if (n < 2) return 0.0;
        return edgeCount / ((double) n * (n - 1) / 2.0);
    }

    public boolean hasNode(N node) { return nodeAttr.containsKey(node); }
    public boolean hasEdge(N u, N v) {
        Map<N, Map<Integer, Graph.EdgeRecord>> a = adj.get(u);
        return a != null && a.containsKey(v) && !a.get(v).isEmpty();
    }
    public boolean hasEdge(N u, N v, int key) {
        Map<N, Map<Integer, Graph.EdgeRecord>> a = adj.get(u);
        if (a == null) return false;
        Map<Integer, Graph.EdgeRecord> m = a.get(v);
        return m != null && m.containsKey(key);
    }

    public boolean isDirected() { return false; }
    public boolean isMulti() { return true; }

    public boolean addNode(N node) { return addNode(node, AttrMap.empty()); }
    public boolean addNode(N node, AttrMap attr) {
        if (nodeAttr.containsKey(node)) {
            if (attr != null && !attr.isEmpty()) {
                nodeAttr.put(node, nodeAttr.get(node).merged(attr));
            }
            return false;
        }
        nodeAttr.put(node, attr == null ? AttrMap.empty() : attr);
        adj.put(node, new LinkedHashMap<>());
        return true;
    }

    public int addNodesFrom(Collection<? extends N> nodes) {
        int n = 0;
        for (N x : nodes) if (addNode(x)) n++;
        return n;
    }

    public boolean removeNode(N node) {
        if (!nodeAttr.containsKey(node)) return false;
        Map<N, Map<Integer, Graph.EdgeRecord>> nbrs = adj.remove(node);
        if (nbrs != null) {
            for (N nbr : nbrs.keySet()) {
                Map<Integer, Graph.EdgeRecord> rec = nbrs.get(nbr);
                if (!node.equals(nbr)) {
                    Map<N, Map<Integer, Graph.EdgeRecord>> mirror = adj.get(nbr);
                    if (mirror != null) {
                        Map<Integer, Graph.EdgeRecord> mirrorRec = mirror.remove(node);
                        if (mirrorRec != null) edgeCount -= mirrorRec.size();
                    }
                } else {
                    edgeCount -= rec.size();
                }
            }
        }
        nodeAttr.remove(node);
        return true;
    }

    public boolean addEdge(N u, N v) { return addEdge(u, v, 1.0); }
    public boolean addEdge(N u, N v, double weight) {
        return addEdge(u, v, AttrMap.empty().with("weight", weight), null) > 0;
    }
    public boolean addEdge(N u, N v, int key) {
        return addEdge(u, v, AttrMap.empty(), key) > 0;
    }
    public boolean addEdge(N u, N v, Map<String, Object> attr) {
        return addEdge(u, v, AttrMap.of(attr), null) > 0;
    }
    public boolean addEdge(N u, N v, AttrMap attr) {
        return addEdge(u, v, attr, null) > 0;
    }

    /**
     * @param key explicit edge key; if null, a new auto-incremented key is assigned.
     * @return the assigned edge key.
     */
    public int addEdge(N u, N v, AttrMap attr, Integer key) {
        Objects.requireNonNull(u);
        Objects.requireNonNull(v);
        if (!nodeAttr.containsKey(u)) addNode(u);
        if (!nodeAttr.containsKey(v)) addNode(v);
        int assignedKey = key == null ? nextKey++ : key;
        Map<N, Map<Integer, Graph.EdgeRecord>> uAdj = adj.get(u);
        Map<N, Map<Integer, Graph.EdgeRecord>> vAdj = adj.get(v);
        Map<Integer, Graph.EdgeRecord> uEdges = uAdj.computeIfAbsent(v, k -> new LinkedHashMap<>());
        Map<Integer, Graph.EdgeRecord> vEdges = vAdj.computeIfAbsent(u, k -> new LinkedHashMap<>());
        boolean isNew = !uEdges.containsKey(assignedKey);
        Graph.EdgeRecord rec = new Graph.EdgeRecord(attr == null ? AttrMap.empty() : attr);
        uEdges.put(assignedKey, rec);
        vEdges.put(assignedKey, rec);
        if (isNew) edgeCount++;
        return assignedKey;
    }

    public boolean removeEdge(N u, N v, int key) {
        Map<N, Map<Integer, Graph.EdgeRecord>> uAdj = adj.get(u);
        if (uAdj == null) return false;
        Map<Integer, Graph.EdgeRecord> m = uAdj.get(v);
        if (m == null) return false;
        if (m.remove(key) == null) return false;
        edgeCount--;
        if (m.isEmpty()) uAdj.remove(v);
        if (!u.equals(v)) {
            Map<N, Map<Integer, Graph.EdgeRecord>> vAdj = adj.get(v);
            if (vAdj != null) {
                Map<Integer, Graph.EdgeRecord> m2 = vAdj.get(u);
                if (m2 != null) m2.remove(key);
                if (m2 != null && m2.isEmpty()) vAdj.remove(u);
            }
        }
        return true;
    }

    public int removeEdgesFrom(Collection<? extends EdgeKey<N>> edges) {
        int n = 0;
        for (EdgeKey<N> e : edges) if (removeEdge(e.u(), e.v(), e.key())) n++;
        return n;
    }

    public double getEdgeWeight(N u, N v, int key) {
        Graph.EdgeRecord rec = getEdgeRecord(u, v, key);
        if (rec == null) return 1.0;
        Double w = rec.attr.getDouble("weight");
        return w == null ? 1.0 : w;
    }

    public Graph.EdgeRecord getEdgeRecord(N u, N v, int key) {
        Map<N, Map<Integer, Graph.EdgeRecord>> a = adj.get(u);
        if (a == null) return null;
        Map<Integer, Graph.EdgeRecord> m = a.get(v);
        return m == null ? null : m.get(key);
    }

    public Set<N> neighbors(N node) {
        Map<N, Map<Integer, Graph.EdgeRecord>> nbrs = adj.get(node);
        if (nbrs == null) throw new NoSuchElementException("No node " + node);
        return Collections.unmodifiableSet(nbrs.keySet());
    }

    public int neighborCount(N node) {
        return adj.get(node) == null ? 0 : adj.get(node).size();
    }

    /** Returns iterator over all (u, v, key) triples (undirected: deduped by canonical hash). */
    public Iterable<EdgeKey<N>> edgeKeyIter() {
        return edgeKeys();
    }

    @Override
    public String toString() {
        return "MultiGraph(nodes=" + order() + ", edges=" + edgeCount + ")";
    }
}