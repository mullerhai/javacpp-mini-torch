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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;

/**
 * Directed graph allowing parallel edges (multi-edges).
 *
 * <p>Aligned with {@code networkx.MultiDiGraph}.
 */
public class MultiDiGraph<N> {
    protected final Map<N, AttrMap> nodeAttr;
    protected final Map<N, Map<N, Map<Integer, Graph.EdgeRecord>>> succ;
    protected final Map<N, Map<N, Map<Integer, Graph.EdgeRecord>>> pred;
    protected long edgeCount;
    protected int nextKey;

    public MultiDiGraph() {
        this(new LinkedHashMap<>(), new LinkedHashMap<>(), new LinkedHashMap<>(), 0);
    }

    protected MultiDiGraph(Map<N, AttrMap> nodeAttr,
                            Map<N, Map<N, Map<Integer, Graph.EdgeRecord>>> succ,
                            Map<N, Map<N, Map<Integer, Graph.EdgeRecord>>> pred,
                            int nextKey) {
        this.nodeAttr = nodeAttr;
        this.succ = succ;
        this.pred = pred;
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
        for (Map.Entry<N, Map<N, Map<Integer, Graph.EdgeRecord>>> e : succ.entrySet()) {
            N u = e.getKey();
            for (N v : e.getValue().keySet()) {
                int numKeys = e.getValue().get(v).size();
                for (int i = 0; i < numKeys; i++) {
                    result.add(new java.util.AbstractMap.SimpleImmutableEntry<>(u, v));
                }
            }
        }
        return Collections.unmodifiableList(result);
    }

    public List<EdgeKey<N>> edgeKeys() {
        List<EdgeKey<N>> result = new ArrayList<>((int) edgeCount);
        for (Map.Entry<N, Map<N, Map<Integer, Graph.EdgeRecord>>> e : succ.entrySet()) {
            N u = e.getKey();
            for (Map.Entry<N, Map<Integer, Graph.EdgeRecord>> e2 : e.getValue().entrySet()) {
                N v = e2.getKey();
                for (Integer key : e2.getValue().keySet()) {
                    result.add(new EdgeKey<>(u, v, key));
                }
            }
        }
        return Collections.unmodifiableList(result);
    }

    public int outDegree(N node) {
        Map<N, Map<Integer, Graph.EdgeRecord>> s = succ.get(node);
        if (s == null) throw new NoSuchElementException("No node " + node);
        int d = 0;
        for (Map<Integer, Graph.EdgeRecord> m : s.values()) d += m.size();
        return d;
    }

    public int inDegree(N node) {
        Map<N, Map<Integer, Graph.EdgeRecord>> p = pred.get(node);
        if (p == null) throw new NoSuchElementException("No node " + node);
        int d = 0;
        for (Map<Integer, Graph.EdgeRecord> m : p.values()) d += m.size();
        return d;
    }

    public int degree(N node) { return inDegree(node) + outDegree(node); }

    public double density() {
        int n = order();
        if (n < 2) return 0.0;
        return edgeCount / ((double) n * (n - 1));
    }

    public boolean hasNode(N node) { return nodeAttr.containsKey(node); }
    public boolean hasEdge(N u, N v) {
        Map<N, Map<Integer, Graph.EdgeRecord>> s = succ.get(u);
        return s != null && s.containsKey(v) && !s.get(v).isEmpty();
    }
    public boolean hasEdge(N u, N v, int key) {
        Map<N, Map<Integer, Graph.EdgeRecord>> s = succ.get(u);
        if (s == null) return false;
        Map<Integer, Graph.EdgeRecord> m = s.get(v);
        return m != null && m.containsKey(key);
    }

    public boolean isDirected() { return true; }
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
        Map<N, Map<Integer, Graph.EdgeRecord>> out = succ.remove(node);
        if (out != null) {
            for (N v : out.keySet()) {
                Map<Integer, Graph.EdgeRecord> m = out.get(v);
                if (!node.equals(v)) {
                    pred.get(v).remove(node);
                    edgeCount -= m.size();
                } else {
                    pred.get(node).remove(node);
                    edgeCount -= m.size();
                }
            }
        }
        Map<N, Map<Integer, Graph.EdgeRecord>> in = pred.remove(node);
        if (in != null) {
            for (N u : in.keySet()) {
                if (!u.equals(node)) succ.get(u).remove(node);
            }
        }
        nodeAttr.remove(node);
        return true;
    }

    public int addEdge(N u, N v) { return addEdge(u, v, 1.0); }
    public int addEdge(N u, N v, double weight) {
        return addEdge(u, v, AttrMap.empty().with("weight", weight), null);
    }
    public int addEdge(N u, N v, int key) {
        return addEdge(u, v, AttrMap.empty(), key);
    }
    public int addEdge(N u, N v, Map<String, Object> attr) {
        return addEdge(u, v, AttrMap.of(attr), null);
    }
    public int addEdge(N u, N v, AttrMap attr) {
        return addEdge(u, v, attr, null);
    }

    public int addEdge(N u, N v, AttrMap attr, Integer key) {
        Objects.requireNonNull(u);
        Objects.requireNonNull(v);
        if (!nodeAttr.containsKey(u)) addNode(u);
        if (!nodeAttr.containsKey(v)) addNode(v);
        int assignedKey = key == null ? nextKey++ : key;
        Map<N, Map<Integer, Graph.EdgeRecord>> uSucc = succ.get(u);
        Map<N, Map<Integer, Graph.EdgeRecord>> vPred = pred.get(v);
        Map<Integer, Graph.EdgeRecord> uEdges = uSucc.computeIfAbsent(v, k -> new LinkedHashMap<>());
        Map<Integer, Graph.EdgeRecord> vEdges = vPred.computeIfAbsent(u, k -> new LinkedHashMap<>());
        boolean isNew = !uEdges.containsKey(assignedKey);
        Graph.EdgeRecord rec = new Graph.EdgeRecord(attr == null ? AttrMap.empty() : attr);
        uEdges.put(assignedKey, rec);
        vEdges.put(assignedKey, rec);
        if (isNew) edgeCount++;
        return assignedKey;
    }

    public boolean removeEdge(N u, N v, int key) {
        Map<N, Map<Integer, Graph.EdgeRecord>> uSucc = succ.get(u);
        if (uSucc == null) return false;
        Map<Integer, Graph.EdgeRecord> m = uSucc.get(v);
        if (m == null) return false;
        if (m.remove(key) == null) return false;
        edgeCount--;
        if (m.isEmpty()) uSucc.remove(v);
        pred.get(v).get(u).remove(key);
        return true;
    }

    public Set<N> successors(N node) {
        Map<N, Map<Integer, Graph.EdgeRecord>> s = succ.get(node);
        if (s == null) throw new NoSuchElementException("No node " + node);
        return Collections.unmodifiableSet(s.keySet());
    }

    public Set<N> predecessors(N node) {
        Map<N, Map<Integer, Graph.EdgeRecord>> p = pred.get(node);
        if (p == null) throw new NoSuchElementException("No node " + node);
        return Collections.unmodifiableSet(p.keySet());
    }

    public double getEdgeWeight(N u, N v, int key) {
        Graph.EdgeRecord rec = getEdgeRecord(u, v, key);
        if (rec == null) return 1.0;
        Double w = rec.attr.getDouble("weight");
        return w == null ? 1.0 : w;
    }

    public Graph.EdgeRecord getEdgeRecord(N u, N v, int key) {
        Map<N, Map<Integer, Graph.EdgeRecord>> s = succ.get(u);
        if (s == null) return null;
        Map<Integer, Graph.EdgeRecord> m = s.get(v);
        return m == null ? null : m.get(key);
    }

    @Override
    public String toString() {
        return "MultiDiGraph(nodes=" + order() + ", edges=" + edgeCount + ")";
    }
}