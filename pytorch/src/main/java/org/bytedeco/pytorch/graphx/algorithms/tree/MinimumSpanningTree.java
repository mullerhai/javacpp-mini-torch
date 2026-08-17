/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.algorithms.tree;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Minimum Spanning Tree: Kruskal and Prim algorithms.
 *
 * <p>Aligned with {@code networkx.algorithms.tree.mst}.
 */
public final class MinimumSpanningTree {
    private MinimumSpanningTree() {}

    /**
     * Kruskal's algorithm using Union-Find. Returns list of edges (u, v, weight).
     * Runs in O(E log E).
     */
    public static <N> List<WeightedEdge<N>> kruskal(Graph<N> g) {
        List<WeightedEdge<N>> edges = new ArrayList<>();
        for (Map.Entry<N, N> e : g.edges()) {
            edges.add(new WeightedEdge<>(e.getKey(), e.getValue(), g.getEdgeWeight(e.getKey(), e.getValue())));
        }
        edges.sort(Comparator.comparingDouble(we -> we.weight));

        UnionFind<N> uf = new UnionFind<>();
        for (N n : g.nodes()) uf.add(n);

        List<WeightedEdge<N>> mst = new ArrayList<>();
        for (WeightedEdge<N> e : edges) {
            if (uf.union(e.u, e.v)) mst.add(e);
        }
        return mst;
    }

    /**
     * Prim's algorithm starting from an arbitrary node. O(E log V).
     */
    public static <N> List<WeightedEdge<N>> prim(Graph<N> g) {
        if (g.isEmpty()) return new ArrayList<>();
        N start = g.nodes().get(0);
        return prim(g, start);
    }

    public static <N> List<WeightedEdge<N>> prim(Graph<N> g, N start) {
        Map<N, Boolean> inTree = new LinkedHashMap<>();
        for (N n : g.nodes()) inTree.put(n, false);
        List<WeightedEdge<N>> mst = new ArrayList<>();
        PriorityQueue<WeightedEdge<N>> pq = new PriorityQueue<>(Comparator.comparingDouble(e -> e.weight));
        inTree.put(start, true);
        for (N nbr : g.neighbors(start)) {
            pq.add(new WeightedEdge<>(start, nbr, g.getEdgeWeight(start, nbr)));
        }
        while (!pq.isEmpty() && mst.size() < g.order() - 1) {
            WeightedEdge<N> e = pq.poll();
            if (inTree.get(e.v)) continue;
            inTree.put(e.v, true);
            mst.add(e);
            for (N nbr : g.neighbors(e.v)) {
                if (!inTree.get(nbr)) {
                    pq.add(new WeightedEdge<>(e.v, nbr, g.getEdgeWeight(e.v, nbr)));
                }
            }
        }
        return mst;
    }

    /** Total weight of the MST edges. */
    public static <N> double minimumSpanningTreeWeight(Graph<N> g) {
        List<WeightedEdge<N>> edges = kruskal(g);
        double total = 0;
        for (WeightedEdge<N> e : edges) total += e.weight;
        return total;
    }

    public static final class WeightedEdge<N> {
        public final N u;
        public final N v;
        public final double weight;
        public WeightedEdge(N u, N v, double weight) {
            this.u = u;
            this.v = v;
            this.weight = weight;
        }
        @Override
        public String toString() {
            return "(" + u + "-" + v + ":" + weight + ")";
        }
    }

    /** Simple Union-Find with path compression and union by rank. */
    static final class UnionFind<N> {
        final Map<N, N> parent = new LinkedHashMap<>();
        final Map<N, Integer> rank = new LinkedHashMap<>();

        void add(N x) {
            parent.put(x, x);
            rank.put(x, 0);
        }

        N find(N x) {
            N p = parent.get(x);
            if (p == x) return x;
            N root = find(p);
            parent.put(x, root);
            return root;
        }

        boolean union(N a, N b) {
            N ra = find(a), rb = find(b);
            if (ra.equals(rb)) return false;
            int rankA = rank.get(ra), rankB = rank.get(rb);
            if (rankA < rankB) {
                parent.put(ra, rb);
            } else if (rankA > rankB) {
                parent.put(rb, ra);
            } else {
                parent.put(rb, ra);
                rank.put(ra, rankA + 1);
            }
            return true;
        }
    }
}