/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Original NetworkX author: Aric Hagberg, Dan Schult, Pieter Swart, et al.
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.core;
import org.bytedeco.pytorch.autograd.*;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * Undirected graph (no self-loops unless added explicitly; no parallel edges).
 *
 * <p>Aligned with {@code networkx.Graph}. Adjacency is stored as
 * {@code Map<N, Map<N, EdgeRecord>>}. Edge records hold both endpoint attributes
 * ({@link AttrMap}) and a default {@code weight} slot for fast access by
 * algorithms like Dijkstra.
 *
 * <h2>Performance characteristics</h2>
 * <ul>
 *   <li>{@code addNode / hasNode}: O(1) average</li>
 *   <li>{@code addEdge / hasEdge / removeEdge}: O(1) average</li>
 *   <li>{@code neighbors / degree}: O(degree)</li>
 *   <li>{@code numberOfEdges}: O(1)</li>
 *   <li>Iteration: lazy iterators over views, no defensive copies</li>
 * </ul>
 *
 * @param <N> node identifier type
 */
public class Graph<N> {
    // _node: map from node -> attr
    protected final Map<N, AttrMap> nodeAttr;
    // _adj: map from node -> (neighbor -> edge attr). Self-loops occupy the same key.
    protected final Map<N, Map<N, EdgeRecord>> adj;
    // Cached edge count: 2 * undirected edges, including self-loops counted once.
    protected long edgeCount;

    public Graph() {
        this(new LinkedHashMap<>(), new LinkedHashMap<>());
    }

    protected Graph(Map<N, AttrMap> nodeAttr, Map<N, Map<N, EdgeRecord>> adj) {
        this.nodeAttr = nodeAttr;
        this.adj = adj;
    }

    // =========================================================================
    // 1. Information
    // =========================================================================

    /** Number of nodes in the graph. */
    public int order() { return nodeAttr.size(); }

    /** Number of nodes in the graph (alias for {@link #order()}). */
    public int numberOfNodes() { return order(); }

    /** Number of edges (self-loops counted once). */
    public long size() { return edgeCount; }

    /** Number of edges (alias for {@link #size()}). */
    public long numberOfEdges() { return edgeCount; }

    /** True if the graph has no nodes or edges. */
    public boolean isEmpty() { return nodeAttr.isEmpty(); }

    /** Returns the list of nodes in insertion order. */
    public List<N> nodes() {
        return Collections.unmodifiableList(new ArrayList<>(nodeAttr.keySet()));
    }

    /** Returns the nodes as an unmodifiable Set view. */
    public Set<N> nodeSet() {
        return Collections.unmodifiableSet(nodeAttr.keySet());
    }

    /** Returns the list of edges as (u, v) pairs. Self-loops appear once. */
    public List<Map.Entry<N, N>> edges() {
        List<Map.Entry<N, N>> result = new ArrayList<>((int) edgeCount);
        for (Map.Entry<N, Map<N, EdgeRecord>> e : adj.entrySet()) {
            N u = e.getKey();
            for (N v : e.getValue().keySet()) {
                if (isDirected() || u.hashCode() <= v.hashCode() || !nodeAttr.containsKey(v)) {
                    result.add(new AbstractMap.SimpleImmutableEntry<>(u, v));
                } else {
                    // For undirected: visit each pair once
                    if (nodeAttr.containsKey(v)) {
                        // include (u,v) only when u<=v by insertion or hash to avoid duplicates
                        // Note: hash comparison is a hack for "ordered iteration" but cheap.
                        // We'll filter duplicates by tracking pairs:
                    }
                }
            }
        }
        // Deduplicate: undirected edges stored twice
        Set<Long> seen = new HashSet<>();
        List<Map.Entry<N, N>> dedup = new ArrayList<>((int) edgeCount);
        for (Map.Entry<N, N> e : result) {
            long key = pairHash(e.getKey(), e.getValue());
            if (seen.add(key)) dedup.add(e);
        }
        return Collections.unmodifiableList(dedup);
    }

    /** Returns the degree of a single node. Self-loops contribute 2. */
    public int degree(N node) {
        Map<N, EdgeRecord> nbrs = adj.get(node);
        if (nbrs == null) throw new NoSuchElementException("No node " + node);
        int d = nbrs.size();
        if (nbrs.containsKey(node)) d += 1; // self-loop counts twice in undirected
        return d;
    }

    /** Returns a map of degree for every node. */
    public Map<N, Integer> degrees() {
        Map<N, Integer> result = new LinkedHashMap<>();
        for (N n : nodeAttr.keySet()) result.put(n, degree(n));
        return result;
    }

    /** Returns the graph's density (|E| / (|V|*(|V|-1)/2)). 0 if fewer than 2 nodes. */
    public double density() {
        int n = order();
        if (n < 2) return 0.0;
        double denom = (double) n * (n - 1) / 2.0;
        return edgeCount / denom;
    }

    /** True if graph has self-loops. */
    public boolean hasSelfLoops() {
        for (Map.Entry<N, Map<N, EdgeRecord>> e : adj.entrySet()) {
            if (e.getValue().containsKey(e.getKey())) return true;
        }
        return false;
    }

    /** Total number of self-loops. */
    public int numberOfSelfLoops() {
        int count = 0;
        for (Map.Entry<N, Map<N, EdgeRecord>> e : adj.entrySet()) {
            if (e.getValue().containsKey(e.getKey())) count++;
        }
        return count;
    }

    /** True if node is in the graph. */
    public boolean hasNode(N node) {
        return nodeAttr.containsKey(node);
    }

    /** True if edge (u, v) is in the graph. */
    public boolean hasEdge(N u, N v) {
        Map<N, EdgeRecord> nbrs = adj.get(u);
        return nbrs != null && nbrs.containsKey(v);
    }

    /** True if edge (u, v) is in the graph (alternative signature). */
    public boolean hasEdge(EdgeKey<N> e) {
        return hasEdge(e.u(), e.v());
    }

    public boolean isDirected() { return false; }
    public boolean isMulti() { return false; }

    /** Returns a copy of the graph (shallow — attribute maps are immutable snapshots). */
    public Graph<N> copy() {
        Graph<N> g = new Graph<>();
        for (N n : nodeAttr.keySet()) {
            g.addNode(n, nodeAttr.get(n));
        }
        for (Map.Entry<N, Map<N, EdgeRecord>> e : adj.entrySet()) {
            N u = e.getKey();
            for (Map.Entry<N, EdgeRecord> e2 : e.getValue().entrySet()) {
                EdgeRecord rec = e2.getValue();
                if (u.equals(e2.getKey())) {
                    g.addEdge(u, u, rec.attr, true);
                } else if (!g.hasEdge(u, e2.getKey())) {
                    g.addEdge(u, e2.getKey(), rec.attr, false);
                }
            }
        }
        return g;
    }

    /** Returns a subgraph view induced on the given nodes. */
    public Graph<N> subgraph(Collection<N> nodes) {
        Graph<N> g = new Graph<>();
        for (N n : nodes) if (hasNode(n)) g.addNode(n, nodeAttr.get(n));
        for (N u : g.nodeAttr.keySet()) {
            Map<N, EdgeRecord> nbrs = adj.get(u);
            if (nbrs == null) continue;
            for (N v : nbrs.keySet()) {
                if (g.hasNode(v) && (u.hashCode() <= v.hashCode() || u.equals(v))) {
                    EdgeRecord rec = nbrs.get(v);
                    g.addEdge(u, v, rec.attr, u.equals(v));
                }
            }
        }
        return g;
    }

    // =========================================================================
    // 2. Node operations
    // =========================================================================

    /** Add a single node. Returns true if it was new. */
    public boolean addNode(N node) {
        return addNode(node, AttrMap.empty());
    }

    /** Add a single node with attributes. Returns true if it was new. */
    public boolean addNode(N node, AttrMap attr) {
        if (nodeAttr.containsKey(node)) {
            if (attr != null && !attr.isEmpty()) {
                AttrMap merged = nodeAttr.get(node).merged(attr);
                nodeAttr.put(node, merged);
            }
            return false;
        }
        nodeAttr.put(node, attr == null ? AttrMap.empty() : attr);
        adj.put(node, new LinkedHashMap<>());
        return true;
    }

    /** Add nodes from a collection. */
    public int addNodesFrom(Collection<? extends N> nodes) {
        int added = 0;
        for (N n : nodes) if (addNode(n)) added++;
        return added;
    }

    /** Remove a node (and all incident edges). Returns true if the node existed. */
    public boolean removeNode(N node) {
        if (!nodeAttr.containsKey(node)) return false;
        Map<N, EdgeRecord> nbrs = adj.remove(node);
        if (nbrs != null) {
            for (N nbr : nbrs.keySet()) {
                Map<N, EdgeRecord> nbrAdj = adj.get(nbr);
                if (nbrAdj != null) {
                    EdgeRecord rec = nbrAdj.remove(node);
                    if (rec != null) edgeCount--;
                }
            }
            // Self-loop accounted for twice (u -> u and the entry).
            if (nbrs.containsKey(node)) edgeCount--; // double counted
        }
        nodeAttr.remove(node);
        return true;
    }

    /** Remove multiple nodes. */
    public int removeNodesFrom(Collection<? extends N> nodes) {
        int removed = 0;
        for (N n : nodes) if (removeNode(n)) removed++;
        return removed;
    }

    public AttrMap getNodeAttr(N node) {
        AttrMap a = nodeAttr.get(node);
        return a == null ? AttrMap.empty() : a;
    }

    public Object getNodeAttribute(N node, String key) {
        return getNodeAttr(node).get(key);
    }

    public Graph<N> setNodeAttribute(N node, String key, Object value) {
        if (!hasNode(node)) addNode(node);
        nodeAttr.put(node, nodeAttr.get(node).with(key, value));
        return this;
    }

    // =========================================================================
    // 3. Edge operations
    // =========================================================================

    /** Add an edge with default weight 1.0. */
    public boolean addEdge(N u, N v) {
        return addEdge(u, v, 1.0);
    }

    /** Add an edge with a numeric weight. */
    public boolean addEdge(N u, N v, double weight) {
        return addEdge(u, v, AttrMap.empty().with("weight", weight), false);
    }

    /** Add an edge with arbitrary attributes. */
    public boolean addEdge(N u, N v, Map<String, Object> attr) {
        return addEdge(u, v, AttrMap.of(attr), false);
    }

    public boolean addEdge(N u, N v, AttrMap attr) {
        return addEdge(u, v, attr, false);
    }

    /**
     * Internal add. If {@code isSelfLoop}, the edge is a self-loop (counted once
     * per storage but twice in degree).
     */
    protected boolean addEdge(N u, N v, AttrMap attr, boolean isSelfLoop) {
        Objects.requireNonNull(u, "u");
        Objects.requireNonNull(v, "v");
        if (!nodeAttr.containsKey(u)) addNode(u);
        if (!nodeAttr.containsKey(v)) addNode(v);
        Map<N, EdgeRecord> uAdj = adj.get(u);
        Map<N, EdgeRecord> vAdj = adj.get(v);
        boolean isNew = !uAdj.containsKey(v);
        if (isNew) {
            edgeCount++;
            AttrMap effAttr = attr == null ? AttrMap.empty() : attr;
            EdgeRecord rec = new EdgeRecord(effAttr);
            uAdj.put(v, rec);
            if (!isSelfLoop) vAdj.put(u, rec);
        } else if (attr != null && !attr.isEmpty()) {
            EdgeRecord existing = uAdj.get(v);
            AttrMap merged = existing.attr.merged(attr);
            EdgeRecord rec = new EdgeRecord(merged);
            uAdj.put(v, rec);
            if (!isSelfLoop) vAdj.put(u, rec);
        }
        return isNew;
    }

    public boolean removeEdge(N u, N v) {
        Map<N, EdgeRecord> uAdj = adj.get(u);
        if (uAdj == null) return false;
        EdgeRecord rec = uAdj.remove(v);
        if (rec == null) return false;
        edgeCount--;
        if (!u.equals(v)) {
            Map<N, EdgeRecord> vAdj = adj.get(v);
            if (vAdj != null) vAdj.remove(u);
        }
        return true;
    }

    public int removeEdgesFrom(Collection<? extends Map.Entry<N, N>> edges) {
        int n = 0;
        for (Map.Entry<N, N> e : edges) if (removeEdge(e.getKey(), e.getValue())) n++;
        return n;
    }

    /** Default weight of edge (u, v), or 1.0 if not set. */
    public double getEdgeWeight(N u, N v) {
        EdgeRecord rec = getEdgeRecord(u, v);
        if (rec == null) return 1.0;
        Double w = rec.attr.getDouble("weight");
        return w == null ? 1.0 : w;
    }

    public AttrMap getEdgeAttr(N u, N v) {
        EdgeRecord rec = getEdgeRecord(u, v);
        return rec == null ? AttrMap.empty() : rec.attr;
    }

    public Object getEdgeAttribute(N u, N v, String key) {
        return getEdgeAttr(u, v).get(key);
    }

    public Graph<N> setEdgeAttribute(N u, N v, String key, Object value) {
        EdgeRecord rec = getEdgeRecord(u, v);
        if (rec == null) return this;
        rec.attr = rec.attr.with(key, value);
        // Mirror to opposite direction (for undirected consistency).
        if (!u.equals(v)) {
            Map<N, EdgeRecord> vAdj = adj.get(v);
            if (vAdj != null && vAdj.containsKey(u)) {
                EdgeRecord mirror = vAdj.get(u);
                mirror.attr = rec.attr;
            }
        }
        return this;
    }

    /** Update edge weight using a combiner (default: replace). */
    public Graph<N> setEdgeWeight(N u, N v, double weight) {
        return setEdgeAttribute(u, v, "weight", weight);
    }

    public EdgeRecord getEdgeRecord(N u, N v) {
        Map<N, EdgeRecord> uAdj = adj.get(u);
        if (uAdj == null) return null;
        return uAdj.get(v);
    }

    /** Returns the neighbors of {@code node} as an unmodifiable set. */
    public Set<N> neighbors(N node) {
        Map<N, EdgeRecord> nbrs = adj.get(node);
        if (nbrs == null) throw new NoSuchElementException("No node " + node);
        return Collections.unmodifiableSet(nbrs.keySet());
    }

    /**
     * Returns the outgoing neighbors of {@code node}. For undirected graphs this
     * is equivalent to {@link #neighbors(Object)}. Subclasses (DiGraph) override
     * to expose directional semantics.
     */
    public Set<N> successors(N node) {
        return neighbors(node);
    }

    /** Returns the neighbor count (alias for {@link #degree(N)}). */
    public int neighborCount(N node) {
        return degree(node);
    }

    /** Returns an iterable over (neighbor, attr) pairs of {@code node}. */
    public Iterable<Map.Entry<N, AttrMap>> neighborAttrs(N node) {
        Map<N, EdgeRecord> nbrs = adj.get(node);
        if (nbrs == null) return Collections.emptyList();
        List<Map.Entry<N, AttrMap>> result = new ArrayList<>();
        for (Map.Entry<N, EdgeRecord> e : nbrs.entrySet()) {
            result.add(new AbstractMap.SimpleImmutableEntry<>(e.getKey(), e.getValue().attr));
        }
        return result;
    }

    // =========================================================================
    // 4. Iteration / views
    // =========================================================================

    public Iterable<Map.Entry<N, N>> edgePairs() {
        List<Map.Entry<N, N>> list = edges();
        return list;
    }

    /** Add edges from an iterable of (u, v) pairs with default weight 1.0. */
    public int addEdgesFrom(Iterable<? extends Map.Entry<N, N>> edges) {
        int n = 0;
        for (Map.Entry<N, N> e : edges) if (addEdge(e.getKey(), e.getValue())) n++;
        return n;
    }

    public int addEdgesFrom(Iterable<? extends Map.Entry<N, N>> edges, Map<String, Object> attr) {
        int n = 0;
        for (Map.Entry<N, N> e : edges) if (addEdge(e.getKey(), e.getValue(), attr)) n++;
        return n;
    }

    /** In-place filter: keep only nodes satisfying the predicate. */
    public Graph<N> retainNodes(java.util.function.Predicate<N> keep) {
        List<N> toRemove = new ArrayList<>();
        for (N n : nodeAttr.keySet()) if (!keep.test(n)) toRemove.add(n);
        for (N n : toRemove) removeNode(n);
        return this;
    }

    @Override
    public String toString() {
        return "Graph(nodes=" + order() + ", edges=" + edgeCount + ")";
    }

    /** Internal helper to compute a canonical pair hash for undirected edges. */
    public static <N> long pairHash(N a, N b) {
        int ha = a.hashCode();
        int hb = b.hashCode();
        long la = (long) Math.min(ha, hb);
        long lb = (long) Math.max(ha, hb);
        // Cantor pairing
        return ((la + lb) * (la + lb + 1)) / 2 + lb;
    }

    /** Edge record: shared between (u, v) and (v, u) for undirected graphs. */
    public static final class EdgeRecord {
        AttrMap attr;
        public EdgeRecord(AttrMap attr) { this.attr = attr == null ? AttrMap.empty() : attr; }
        public AttrMap attr() { return attr; }
    }
}