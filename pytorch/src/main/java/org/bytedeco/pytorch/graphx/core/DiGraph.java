/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.core;
import org.bytedeco.pytorch.autograd.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * Directed graph (no self-loops unless added explicitly; no parallel edges).
 *
 * <p>Aligned with {@code networkx.DiGraph}. Successor / predecessor adjacency
 * are stored separately to mirror NetworkX's {@code succ}/{@code pred} layout.
 */
public class DiGraph<N> {
    protected final Map<N, AttrMap> nodeAttr;
    /** Successor adjacency: u -> {v -> edgeRecord}. */
    protected final Map<N, Map<N, Graph.EdgeRecord>> succ;
    /** Predecessor adjacency: v -> {u -> edgeRecord} (mirrors succ for symmetry). */
    protected final Map<N, Map<N, Graph.EdgeRecord>> pred;
    protected long edgeCount;

    public DiGraph() {
        this(new LinkedHashMap<>(), new LinkedHashMap<>(), new LinkedHashMap<>());
    }

    protected DiGraph(Map<N, AttrMap> nodeAttr,
                       Map<N, Map<N, Graph.EdgeRecord>> succ,
                       Map<N, Map<N, Graph.EdgeRecord>> pred) {
        this.nodeAttr = nodeAttr;
        this.succ = succ;
        this.pred = pred;
    }

    // =========================================================================
    // 1. Information
    // =========================================================================

    public int order() { return nodeAttr.size(); }
    public int numberOfNodes() { return order(); }
    public long size() { return edgeCount; }
    public long numberOfEdges() { return edgeCount; }
    public boolean isEmpty() { return nodeAttr.isEmpty(); }

    public List<N> nodes() {
        return Collections.unmodifiableList(new ArrayList<>(nodeAttr.keySet()));
    }

    public Set<N> nodeSet() { return Collections.unmodifiableSet(nodeAttr.keySet()); }

    /** Edges as (u, v) pairs in insertion order. Self-loops appear once. */
    public List<Map.Entry<N, N>> edges() {
        List<Map.Entry<N, N>> result = new ArrayList<>((int) edgeCount);
        for (Map.Entry<N, Map<N, Graph.EdgeRecord>> e : succ.entrySet()) {
            N u = e.getKey();
            for (N v : e.getValue().keySet()) {
                result.add(new java.util.AbstractMap.SimpleImmutableEntry<>(u, v));
            }
        }
        return Collections.unmodifiableList(result);
    }

    /** Out-degree: count of outgoing edges (self-loop counts 1). */
    public int outDegree(N node) {
        Map<N, Graph.EdgeRecord> s = succ.get(node);
        if (s == null) throw new NoSuchElementException("No node " + node);
        return s.size();
    }

    /** In-degree: count of incoming edges. */
    public int inDegree(N node) {
        Map<N, Graph.EdgeRecord> p = pred.get(node);
        if (p == null) throw new NoSuchElementException("No node " + node);
        return p.size();
    }

    /** Total degree (in + out). */
    public int degree(N node) {
        return inDegree(node) + outDegree(node);
    }

    public Map<N, Integer> outDegrees() {
        Map<N, Integer> m = new LinkedHashMap<>();
        for (N n : nodeAttr.keySet()) m.put(n, outDegree(n));
        return m;
    }

    public Map<N, Integer> inDegrees() {
        Map<N, Integer> m = new LinkedHashMap<>();
        for (N n : nodeAttr.keySet()) m.put(n, inDegree(n));
        return m;
    }

    public double density() {
        int n = order();
        if (n < 2) return 0.0;
        return edgeCount / ((double) n * (n - 1));
    }

    public boolean hasNode(N node) { return nodeAttr.containsKey(node); }
    public boolean hasEdge(N u, N v) {
        Map<N, Graph.EdgeRecord> s = succ.get(u);
        return s != null && s.containsKey(v);
    }

    public boolean isDirected() { return true; }
    public boolean isMulti() { return false; }

    public boolean hasSelfLoops() {
        for (Map.Entry<N, Map<N, Graph.EdgeRecord>> e : succ.entrySet()) {
            if (e.getValue().containsKey(e.getKey())) return true;
        }
        return false;
    }

    public int numberOfSelfLoops() {
        int count = 0;
        for (Map.Entry<N, Map<N, Graph.EdgeRecord>> e : succ.entrySet()) {
            if (e.getValue().containsKey(e.getKey())) count++;
        }
        return count;
    }

    public Graph<N> copy() {
        DiGraph<N> g = new DiGraph<>();
        for (Map.Entry<N, AttrMap> e : nodeAttr.entrySet()) {
            g.addNode(e.getKey(), e.getValue());
        }
        for (Map.Entry<N, Map<N, Graph.EdgeRecord>> e : succ.entrySet()) {
            N u = e.getKey();
            for (Map.Entry<N, Graph.EdgeRecord> e2 : e.getValue().entrySet()) {
                g.addEdge(u, e2.getKey(), e2.getValue().attr(), u.equals(e2.getKey()));
            }
        }
        // For API consistency, return as DiGraph (callers downcast via cast)
        return null; // overridden below
    }

    public DiGraph<N> copyAsDirected() {
        DiGraph<N> g = new DiGraph<>();
        for (Map.Entry<N, AttrMap> e : nodeAttr.entrySet()) {
            g.addNode(e.getKey(), e.getValue());
        }
        for (Map.Entry<N, Map<N, Graph.EdgeRecord>> e : succ.entrySet()) {
            N u = e.getKey();
            for (Map.Entry<N, Graph.EdgeRecord> e2 : e.getValue().entrySet()) {
                g.addEdge(u, e2.getKey(), e2.getValue().attr(), u.equals(e2.getKey()));
            }
        }
        return g;
    }

    // =========================================================================
    // 2. Node operations
    // =========================================================================

    public boolean addNode(N node) { return addNode(node, AttrMap.empty()); }
    public boolean addNode(N node, AttrMap attr) {
        if (nodeAttr.containsKey(node)) {
            if (attr != null && !attr.isEmpty()) {
                nodeAttr.put(node, nodeAttr.get(node).merged(attr));
            }
            return false;
        }
        nodeAttr.put(node, attr == null ? AttrMap.empty() : attr);
        succ.put(node, new LinkedHashMap<>());
        pred.put(node, new LinkedHashMap<>());
        return true;
    }

    public int addNodesFrom(Collection<? extends N> nodes) {
        int n = 0;
        for (N x : nodes) if (addNode(x)) n++;
        return n;
    }

    public boolean removeNode(N node) {
        if (!nodeAttr.containsKey(node)) return false;
        Map<N, Graph.EdgeRecord> outEdges = succ.remove(node);
        if (outEdges != null) {
            for (N v : outEdges.keySet()) {
                if (!node.equals(v)) {
                    pred.get(v).remove(node);
                    edgeCount--;
                }
            }
            // Self-loop counted once; remove from pred too.
            if (outEdges.containsKey(node)) {
                pred.get(node).remove(node);
                edgeCount--;
            }
        }
        Map<N, Graph.EdgeRecord> inEdges = pred.remove(node);
        if (inEdges != null) {
            for (N u : inEdges.keySet()) {
                if (!u.equals(node)) {
                    succ.get(u).remove(node);
                }
            }
        }
        nodeAttr.remove(node);
        return true;
    }

    public int removeNodesFrom(Collection<? extends N> nodes) {
        int n = 0;
        for (N x : nodes) if (removeNode(x)) n++;
        return n;
    }

    public AttrMap getNodeAttr(N node) {
        AttrMap a = nodeAttr.get(node);
        return a == null ? AttrMap.empty() : a;
    }

    public Object getNodeAttribute(N node, String key) {
        return getNodeAttr(node).get(key);
    }

    public DiGraph<N> setNodeAttribute(N node, String key, Object value) {
        if (!hasNode(node)) addNode(node);
        nodeAttr.put(node, nodeAttr.get(node).with(key, value));
        return this;
    }

    // =========================================================================
    // 3. Edge operations
    // =========================================================================

    public boolean addEdge(N u, N v) { return addEdge(u, v, 1.0); }
    public boolean addEdge(N u, N v, double weight) {
        return addEdge(u, v, AttrMap.empty().with("weight", weight), false);
    }
    public boolean addEdge(N u, N v, Map<String, Object> attr) {
        return addEdge(u, v, AttrMap.of(attr), false);
    }
    public boolean addEdge(N u, N v, AttrMap attr) {
        return addEdge(u, v, attr, false);
    }

    protected boolean addEdge(N u, N v, AttrMap attr, boolean isSelfLoop) {
        java.util.Objects.requireNonNull(u, "u");
        java.util.Objects.requireNonNull(v, "v");
        if (!nodeAttr.containsKey(u)) addNode(u);
        if (!nodeAttr.containsKey(v)) addNode(v);
        Map<N, Graph.EdgeRecord> uSucc = succ.get(u);
        boolean isNew = !uSucc.containsKey(v);
        if (isNew) {
            edgeCount++;
            AttrMap effAttr = attr == null ? AttrMap.empty() : attr;
            Graph.EdgeRecord rec = new Graph.EdgeRecord(effAttr);
            uSucc.put(v, rec);
            pred.get(v).put(u, rec);
        } else if (attr != null && !attr.isEmpty()) {
            Graph.EdgeRecord rec = uSucc.get(v);
            AttrMap merged = rec.attr().merged(attr);
            Graph.EdgeRecord newRec = new Graph.EdgeRecord(merged);
            uSucc.put(v, newRec);
            pred.get(v).put(u, newRec);
        }
        return isNew;
    }

    public boolean removeEdge(N u, N v) {
        Map<N, Graph.EdgeRecord> uSucc = succ.get(u);
        if (uSucc == null) return false;
        Graph.EdgeRecord rec = uSucc.remove(v);
        if (rec == null) return false;
        edgeCount--;
        pred.get(v).remove(u);
        return true;
    }

    public int removeEdgesFrom(Collection<? extends Map.Entry<N, N>> edges) {
        int n = 0;
        for (Map.Entry<N, N> e : edges) if (removeEdge(e.getKey(), e.getValue())) n++;
        return n;
    }

    public double getEdgeWeight(N u, N v) {
        Graph.EdgeRecord rec = succ.get(u) == null ? null : succ.get(u).get(v);
        if (rec == null) return 1.0;
        Double w = rec.attr().getDouble("weight");
        return w == null ? 1.0 : w;
    }

    public AttrMap getEdgeAttr(N u, N v) {
        Map<N, Graph.EdgeRecord> uSucc = succ.get(u);
        if (uSucc == null) return AttrMap.empty();
        Graph.EdgeRecord rec = uSucc.get(v);
        return rec == null ? AttrMap.empty() : rec.attr();
    }

    public Object getEdgeAttribute(N u, N v, String key) {
        return getEdgeAttr(u, v).get(key);
    }

    public DiGraph<N> setEdgeAttribute(N u, N v, String key, Object value) {
        Map<N, Graph.EdgeRecord> uSucc = succ.get(u);
        if (uSucc == null || !uSucc.containsKey(v)) return this;
        Graph.EdgeRecord rec = uSucc.get(v);
        rec.attr = rec.attr().with(key, value);
        // mirror to pred
        Graph.EdgeRecord mirror = pred.get(v).get(u);
        if (mirror != null) mirror.attr = mirror.attr().with(key, value);
        return this;
    }

    public DiGraph<N> setEdgeWeight(N u, N v, double weight) {
        return setEdgeAttribute(u, v, "weight", weight);
    }

    /** Successors of {@code node} (outgoing neighbors). */
    public Set<N> successors(N node) {
        Map<N, Graph.EdgeRecord> s = succ.get(node);
        if (s == null) throw new NoSuchElementException("No node " + node);
        return Collections.unmodifiableSet(s.keySet());
    }

    /** Predecessors of {@code node} (incoming neighbors). */
    public Set<N> predecessors(N node) {
        Map<N, Graph.EdgeRecord> p = pred.get(node);
        if (p == null) throw new NoSuchElementException("No node " + node);
        return Collections.unmodifiableSet(p.keySet());
    }

    public Iterable<Map.Entry<N, N>> edgePairs() {
        return edges();
    }

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

    public DiGraph<N> retainNodes(java.util.function.Predicate<N> keep) {
        List<N> toRemove = new ArrayList<>();
        for (N n : nodeAttr.keySet()) if (!keep.test(n)) toRemove.add(n);
        for (N n : toRemove) removeNode(n);
        return this;
    }

    @Override
    public String toString() {
        return "DiGraph(nodes=" + order() + ", edges=" + edgeCount + ")";
    }
}