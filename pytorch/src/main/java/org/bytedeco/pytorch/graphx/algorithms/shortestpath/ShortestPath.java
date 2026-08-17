/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.algorithms.shortestpath;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Shortest path algorithms: Dijkstra, Bellman-Ford, A*.
 *
 * <p>Aligned with {@code networkx.algorithms.shortest_paths.weighted} and
 * {@code networkx.algorithms.shortest_paths.astar}.
 */
public final class ShortestPath {
    private ShortestPath() {}

    /** Dijkstra's algorithm from {@code source} to all reachable nodes. */
    public static <N> DijkstraResult<N> dijkstra(Graph<N> g, N source) {
        return dijkstraUndirected(g, source);
    }

    public static <N> DijkstraResult<N> dijkstra(DiGraph<N> g, N source) {
        return dijkstraDirected(g, source);
    }

    static <N> DijkstraResult<N> dijkstraUndirected(Graph<N> g, N source) {
        Map<N, Double> dist = new LinkedHashMap<>();
        Map<N, N> pred = new LinkedHashMap<>();
        dist.put(source, 0.0);
        PriorityQueue<NodeDist<N>> pq = new PriorityQueue<>();
        pq.add(new NodeDist<>(source, 0.0));
        while (!pq.isEmpty()) {
            NodeDist<N> cur = pq.poll();
            if (cur.dist > dist.getOrDefault(cur.node, Double.POSITIVE_INFINITY)) continue;
            for (N nbr : g.neighbors(cur.node)) {
                double w = g.getEdgeWeight(cur.node, nbr);
                double newDist = cur.dist + w;
                if (newDist < dist.getOrDefault(nbr, Double.POSITIVE_INFINITY)) {
                    dist.put(nbr, newDist);
                    pred.put(nbr, cur.node);
                    pq.add(new NodeDist<>(nbr, newDist));
                }
            }
        }
        return new DijkstraResult<>(dist, pred);
    }

    static <N> DijkstraResult<N> dijkstraDirected(DiGraph<N> g, N source) {
        Map<N, Double> dist = new LinkedHashMap<>();
        Map<N, N> pred = new LinkedHashMap<>();
        dist.put(source, 0.0);
        PriorityQueue<NodeDist<N>> pq = new PriorityQueue<>();
        pq.add(new NodeDist<>(source, 0.0));
        while (!pq.isEmpty()) {
            NodeDist<N> cur = pq.poll();
            if (cur.dist > dist.getOrDefault(cur.node, Double.POSITIVE_INFINITY)) continue;
            for (N nbr : g.successors(cur.node)) {
                double w = g.getEdgeWeight(cur.node, nbr);
                double newDist = cur.dist + w;
                if (newDist < dist.getOrDefault(nbr, Double.POSITIVE_INFINITY)) {
                    dist.put(nbr, newDist);
                    pred.put(nbr, cur.node);
                    pq.add(new NodeDist<>(nbr, newDist));
                }
            }
        }
        return new DijkstraResult<>(dist, pred);
    }

    /** Single-source shortest path from source to target. Returns empty list if unreachable. */
    public static <N> List<N> shortestPath(Graph<N> g, N source, N target) {
        DijkstraResult<N> r = dijkstra(g, source);
        return reconstructPath(r, source, target);
    }

    public static <N> List<N> shortestPath(DiGraph<N> g, N source, N target) {
        DijkstraResult<N> r = dijkstra(g, source);
        return reconstructPath(r, source, target);
    }

    static <N> List<N> reconstructPath(DijkstraResult<N> r, N source, N target) {
        if (!r.dist.containsKey(target)) return Collections.emptyList();
        List<N> path = new ArrayList<>();
        for (N at = target; at != null && !at.equals(source); at = r.pred.get(at)) {
            path.add(at);
            if (at.equals(r.pred.get(at))) break; // safety
        }
        path.add(source);
        Collections.reverse(path);
        return path;
    }

    /** Single-source shortest path length. Returns {@link Double#POSITIVE_INFINITY} if unreachable. */
    public static <N> double shortestPathLength(Graph<N> g, N source, N target) {
        DijkstraResult<N> r = dijkstra(g, source);
        return r.dist.getOrDefault(target, Double.POSITIVE_INFINITY);
    }

    public static <N> double shortestPathLength(DiGraph<N> g, N source, N target) {
        DijkstraResult<N> r = dijkstra(g, source);
        return r.dist.getOrDefault(target, Double.POSITIVE_INFINITY);
    }

    /** Bellman-Ford — handles negative weights, detects negative cycles. */
    public static <N> DijkstraResult<N> bellmanFord(DiGraph<N> g, N source) {
        Map<N, Double> dist = new LinkedHashMap<>();
        Map<N, N> pred = new LinkedHashMap<>();
        for (N n : g.nodes()) dist.put(n, Double.POSITIVE_INFINITY);
        dist.put(source, 0.0);
        int n = g.order();
        for (int i = 0; i < n - 1; i++) {
            boolean updated = false;
            for (Map.Entry<N, N> e : g.edges()) {
                N u = e.getKey();
                N v = e.getValue();
                double w = g.getEdgeWeight(u, v);
                if (dist.get(u) + w < dist.get(v)) {
                    dist.put(v, dist.get(u) + w);
                    pred.put(v, u);
                    updated = true;
                }
            }
            if (!updated) break;
        }
        // Negative cycle detection
        for (Map.Entry<N, N> e : g.edges()) {
            if (dist.get(e.getKey()) + g.getEdgeWeight(e.getKey(), e.getValue()) < dist.get(e.getValue())) {
                throw new IllegalStateException("Graph contains a negative-weight cycle");
            }
        }
        return new DijkstraResult<>(dist, pred);
    }

    /** A* search from {@code source} to {@code target} using heuristic {@code h}. */
    public static <N> List<N> astar(Graph<N> g, N source, N target, java.util.function.ToDoubleFunction<N> h) {
        Map<N, Double> gScore = new LinkedHashMap<>();
        Map<N, Double> fScore = new LinkedHashMap<>();
        Map<N, N> cameFrom = new LinkedHashMap<>();
        java.util.Set<N> closed = new java.util.HashSet<>();
        gScore.put(source, 0.0);
        fScore.put(source, h.applyAsDouble(source));
        PriorityQueue<NodeDist<N>> open = new PriorityQueue<>(java.util.Comparator.comparingDouble(nd -> fScore.getOrDefault(nd.node, Double.POSITIVE_INFINITY)));
        open.add(new NodeDist<>(source, fScore.get(source)));
        while (!open.isEmpty()) {
            NodeDist<N> cur = open.poll();
            if (cur.node.equals(target)) return reconstructAstar(cameFrom, cur.node);
            if (!closed.add(cur.node)) continue;
            for (N nbr : g.neighbors(cur.node)) {
                if (closed.contains(nbr)) continue;
                double tentativeG = gScore.get(cur.node) + g.getEdgeWeight(cur.node, nbr);
                if (tentativeG < gScore.getOrDefault(nbr, Double.POSITIVE_INFINITY)) {
                    cameFrom.put(nbr, cur.node);
                    gScore.put(nbr, tentativeG);
                    double f = tentativeG + h.applyAsDouble(nbr);
                    fScore.put(nbr, f);
                    open.add(new NodeDist<>(nbr, f));
                }
            }
        }
        return Collections.emptyList();
    }

    static <N> List<N> reconstructAstar(Map<N, N> cameFrom, N target) {
        List<N> path = new ArrayList<>();
        for (N at = target; at != null; at = cameFrom.get(at)) {
            path.add(at);
            if (cameFrom.get(at) == null || at.equals(cameFrom.get(at))) break;
        }
        Collections.reverse(path);
        return path;
    }

    /** Result of a single-source shortest-path algorithm. */
    public static final class DijkstraResult<N> {
        public final Map<N, Double> dist;
        public final Map<N, N> pred;
        public DijkstraResult(Map<N, Double> dist, Map<N, N> pred) {
            this.dist = dist;
            this.pred = pred;
        }
    }

    static final class NodeDist<N> {
        final N node;
        final double dist;
        NodeDist(N node, double dist) { this.node = node; this.dist = dist; }
    }
}