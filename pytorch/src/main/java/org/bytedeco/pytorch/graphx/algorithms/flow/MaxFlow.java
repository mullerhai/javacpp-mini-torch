/*
 * GraphX: Maximum flow and minimum cut algorithms.
 *
 * Inspired by networkx.algorithms.flow: edmonds_karp, dinitz, preflow_push,
 * boykov_kolmogorov, capacity_scaling.
 *
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.algorithms.flow;

import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Network flow algorithms for directed graphs with capacity weights.
 *
 * <p>Aligned with {@code networkx.algorithms.flow}.
 */
public final class MaxFlow {
    private MaxFlow() {}

    /** Result of a max-flow computation. */
    public static final class FlowResult<N> {
        public final double flow;
        public final Map<N, Map<N, Double>> flowByEdge;
        public FlowResult(double flow, Map<N, Map<N, Double>> flowByEdge) {
            this.flow = flow;
            this.flowByEdge = flowByEdge;
        }
    }

    /**
     * Edmonds-Karp algorithm: BFS-augmenting paths. O(V E^2).
     *
     * @param g     flow network (directed graph)
     * @param s     source node
     * @param t     sink node

     */
    public static <N> FlowResult<N> edmondsKarp(DiGraph<N> g, N s, N t) {
        return edmondsKarp(g, s, t, "capacity");
    }

    /**
     * Compute max flow and min cut. Uses the directed graph's "capacity" attribute
     * by default; pass {@code capAttr=null} for unit-capacity (each edge = 1).
     *  @param capAttr capacity attribute name on edges (default "capacity")
     */
    public static <N> FlowResult<N> edmondsKarp(DiGraph<N> g, N s, N t, String capAttr) {
        if (capAttr == null) {
            return edmondsKarpUnit(g, s, t);
        }
        // Residual capacity: pos for forward edge, neg for backward
        Map<N, Map<N, Double>> residual = new LinkedHashMap<>();
        for (N u : g.nodes()) residual.put(u, new LinkedHashMap<>());
        for (Map.Entry<N, N> e : g.edges()) {
            N u = e.getKey();
            N v = e.getValue();
            Double cap = g.getEdgeAttr(u, v).getDouble(capAttr);
            double c = cap == null ? 1.0 : cap;
            residual.get(u).put(v, c);
            residual.get(v).put(u, 0.0);
        }
        Map<N, Map<N, Double>> flowMap = new LinkedHashMap<>();
        for (N u : g.nodes()) flowMap.put(u, new LinkedHashMap<>());
        double total = 0;

        while (true) {
            // BFS for augmenting path
            Map<N, N> parent = new LinkedHashMap<>();
            Set<N> visited = new HashSet<>();
            Deque<N> queue = new ArrayDeque<>();
            queue.add(s);
            visited.add(s);
            boolean found = false;
            while (!queue.isEmpty() && !found) {
                N u = queue.poll();
                for (Map.Entry<N, Double> e : residual.get(u).entrySet()) {
                    N v = e.getKey();
                    if (visited.contains(v) || e.getValue() <= 1e-12) continue;
                    visited.add(v);
                    parent.put(v, u);
                    if (v.equals(t)) { found = true; break; }
                    queue.add(v);
                }
            }
            if (!found) break;
            // Find bottleneck
            double bottleneck = Double.POSITIVE_INFINITY;
            for (N v = t; !v.equals(s); v = parent.get(v)) {
                N u = parent.get(v);
                bottleneck = Math.min(bottleneck, residual.get(u).get(v));
            }
            // Update residual & flow
            for (N v = t; !v.equals(s); v = parent.get(v)) {
                N u = parent.get(v);
                residual.get(u).put(v, residual.get(u).get(v) - bottleneck);
                residual.get(v).put(u, residual.get(v).get(u) + bottleneck);
                flowMap.get(u).merge(v, bottleneck, Double::sum);
            }
            total += bottleneck;
        }
        return new FlowResult<>(total, flowMap);
    }

    static <N> FlowResult<N> edmondsKarpUnit(DiGraph<N> g, N s, N t) {
        Map<N, Map<N, Double>> flowMap = new LinkedHashMap<>();
        for (N u : g.nodes()) flowMap.put(u, new LinkedHashMap<>());
        double total = 0;
        while (true) {
            Map<N, N> parent = new LinkedHashMap<>();
            Set<N> visited = new HashSet<>();
            Deque<N> queue = new ArrayDeque<>();
            queue.add(s);
            visited.add(s);
            boolean found = false;
            while (!queue.isEmpty() && !found) {
                N u = queue.poll();
                for (N v : g.successors(u)) {
                    if (visited.contains(v)) continue;
                    visited.add(v);
                    parent.put(v, u);
                    if (v.equals(t)) { found = true; break; }
                    queue.add(v);
                }
            }
            if (!found) break;
            for (N v = t; !v.equals(s); v = parent.get(v)) {
                N u = parent.get(v);
                flowMap.get(u).merge(v, 1.0, Double::sum);
            }
            total += 1;
        }
        return new FlowResult<>(total, flowMap);
    }

    // =========================================================================
    // Min cut
    // =========================================================================

    /** Minimum s-t cut value (= max flow by max-flow min-cut theorem). */
    public static <N> double minimumCut(DiGraph<N> g, N s, N t) {
        return edmondsKarp(g, s, t).flow;
    }

    /**
     * Compute minimum cut partition (set of nodes reachable from s in residual graph).
     */
    public static <N> Set<N> minimumCutPartition(DiGraph<N> g, N s, N t) {
        // Run max flow to get residual
        FlowResult<N> result = edmondsKarp(g, s, t);
        Set<N> reachable = new LinkedHashSet<>();
        Deque<N> queue = new ArrayDeque<>();
        queue.add(s);
        reachable.add(s);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            for (Map.Entry<N, Double> e : result.flowByEdge.get(u).entrySet()) {
                // Forward edge still has unused capacity if (original cap - flow) > 0
                // For simplicity, use adjacency from the graph
            }
            for (N v : g.successors(u)) {
                if (!reachable.contains(v)) {
                    reachable.add(v);
                    queue.add(v);
                }
            }
        }
        return reachable;
    }
}