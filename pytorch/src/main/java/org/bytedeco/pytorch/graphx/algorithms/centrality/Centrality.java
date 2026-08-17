/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.algorithms.centrality;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.algorithms.shortestpath.ShortestPath;
import org.bytedeco.pytorch.graphx.algorithms.components.ConnectedComponents;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Centrality algorithms: degree, closeness, betweenness, PageRank, harmonic.
 *
 * <p>Aligned with {@code networkx.algorithms.centrality.*}.
 */
public final class Centrality {
    private Centrality() {}

    // ---- Degree centrality ----

    /** Normalized degree centrality: d(v) / (n-1). Undirected uses {@code degree}; directed uses {@code inDegree+outDegree}. */
    public static <N> Map<N, Double> degreeCentrality(Graph<N> g) {
        Map<N, Double> c = new LinkedHashMap<>();
        int n = g.order();
        if (n <= 1) {
            for (N v : g.nodes()) c.put(v, 0.0);
            return c;
        }
        double norm = 1.0 / (n - 1);
        for (N v : g.nodes()) c.put(v, g.degree(v) * norm);
        return c;
    }

    public static <N> Map<N, Double> inDegreeCentrality(DiGraph<N> g) {
        Map<N, Double> c = new LinkedHashMap<>();
        int n = g.order();
        if (n <= 1) {
            for (N v : g.nodes()) c.put(v, 0.0);
            return c;
        }
        double norm = 1.0 / (n - 1);
        for (N v : g.nodes()) c.put(v, g.inDegree(v) * norm);
        return c;
    }

    public static <N> Map<N, Double> outDegreeCentrality(DiGraph<N> g) {
        Map<N, Double> c = new LinkedHashMap<>();
        int n = g.order();
        if (n <= 1) {
            for (N v : g.nodes()) c.put(v, 0.0);
            return c;
        }
        double norm = 1.0 / (n - 1);
        for (N v : g.nodes()) c.put(v, g.outDegree(v) * norm);
        return c;
    }

    // ---- Closeness centrality ----

    /** Closeness = (n-1) / sum of shortest-path distances from v. */
    public static <N> Map<N, Double> closenessCentrality(Graph<N> g) {
        Map<N, Double> c = new LinkedHashMap<>();
        int n = g.order();
        for (N v : g.nodes()) {
            ShortestPath.DijkstraResult<N> r = ShortestPath.dijkstra(g, v);
            double total = 0;
            for (N u : g.nodes()) {
                if (u.equals(v)) continue;
                total += r.dist.getOrDefault(u, 0.0);
            }
            c.put(v, total > 0 ? (n - 1) / total : 0.0);
        }
        return c;
    }

    public static <N> Map<N, Double> closenessCentrality(DiGraph<N> g) {
        Map<N, Double> c = new LinkedHashMap<>();
        int n = g.order();
        for (N v : g.nodes()) {
            ShortestPath.DijkstraResult<N> r = ShortestPath.dijkstra(g, v);
            double total = 0;
            for (N u : g.nodes()) {
                if (u.equals(v)) continue;
                total += r.dist.getOrDefault(u, 0.0);
            }
            c.put(v, total > 0 ? (n - 1) / total : 0.0);
        }
        return c;
    }

    // ---- Harmonic centrality ----

    /** Harmonic = sum of 1/d(v, u) for u != v. Better for disconnected graphs than closeness. */
    public static <N> Map<N, Double> harmonicCentrality(Graph<N> g) {
        Map<N, Double> c = new LinkedHashMap<>();
        int n = g.order();
        for (N v : g.nodes()) {
            ShortestPath.DijkstraResult<N> r = ShortestPath.dijkstra(g, v);
            double sum = 0;
            for (Map.Entry<N, Double> e : r.dist.entrySet()) {
                if (e.getKey().equals(v)) continue;
                if (e.getValue() > 0 && Double.isFinite(e.getValue())) sum += 1.0 / e.getValue();
            }
            c.put(v, sum / (n - 1));
        }
        return c;
    }

    // ---- Betweenness centrality (Brandes' algorithm) ----

    /** Betweenness centrality using Brandes' algorithm: O(VE) for unweighted. */
    public static <N> Map<N, Double> betweennessCentrality(Graph<N> g) {
        return betweennessCentralityUndirected(g, false);
    }

    public static <N> Map<N, Double> betweennessCentrality(Graph<N> g, boolean normalized) {
        return betweennessCentralityUndirected(g, normalized);
    }

    public static <N> Map<N, Double> betweennessCentrality(DiGraph<N> g) {
        return betweennessCentralityDirected(g, false);
    }

    static <N> Map<N, Double> betweennessCentralityUndirected(Graph<N> g, boolean normalized) {
        Map<N, Double> cb = new LinkedHashMap<>();
        for (N v : g.nodes()) cb.put(v, 0.0);
        for (N s : g.nodes()) {
            Map<N, List<N>> pred = new LinkedHashMap<>();
            Map<N, Integer> sigma = new LinkedHashMap<>();
            Map<N, Integer> dist = new LinkedHashMap<>();
            for (N v : g.nodes()) {
                pred.put(v, new ArrayList<>());
                sigma.put(v, 0);
                dist.put(v, -1);
            }
            sigma.put(s, 1);
            dist.put(s, 0);
            Deque<N> stack = new ArrayDeque<>();
            Deque<N> queue = new ArrayDeque<>();
            queue.add(s);
            while (!queue.isEmpty()) {
                N v = queue.poll();
                stack.push(v);
                for (N w : g.neighbors(v)) {
                    if (dist.get(w) < 0) {
                        queue.add(w);
                        dist.put(w, dist.get(v) + 1);
                    }
                    if (dist.get(w) == dist.get(v) + 1) {
                        sigma.put(w, sigma.get(w) + sigma.get(v));
                        pred.get(w).add(v);
                    }
                }
            }
            Map<N, Double> delta = new LinkedHashMap<>();
            for (N v : g.nodes()) delta.put(v, 0.0);
            while (!stack.isEmpty()) {
                N w = stack.pop();
                for (N v : pred.get(w)) {
                    delta.put(v, delta.get(v) + (sigma.get(v) * 1.0 / sigma.get(w)) * (1.0 + delta.get(w)));
                }
                if (!w.equals(s)) {
                    cb.put(w, cb.get(w) + delta.get(w));
                }
            }
        }
        if (normalized && g.order() > 2) {
            double scale = 1.0 / ((g.order() - 1) * (g.order() - 2));
            Map<N, Double> norm = new LinkedHashMap<>();
            for (N v : cb.keySet()) norm.put(v, cb.get(v) * 2.0 * scale);
            return norm;
        }
        return cb;
    }

    static <N> Map<N, Double> betweennessCentralityDirected(DiGraph<N> g, boolean normalized) {
        Map<N, Double> cb = new LinkedHashMap<>();
        for (N v : g.nodes()) cb.put(v, 0.0);
        for (N s : g.nodes()) {
            Map<N, List<N>> pred = new LinkedHashMap<>();
            Map<N, Integer> sigma = new LinkedHashMap<>();
            Map<N, Integer> dist = new LinkedHashMap<>();
            for (N v : g.nodes()) {
                pred.put(v, new ArrayList<>());
                sigma.put(v, 0);
                dist.put(v, -1);
            }
            sigma.put(s, 1);
            dist.put(s, 0);
            Deque<N> stack = new ArrayDeque<>();
            Deque<N> queue = new ArrayDeque<>();
            queue.add(s);
            while (!queue.isEmpty()) {
                N v = queue.poll();
                stack.push(v);
                for (N w : g.successors(v)) {
                    if (dist.get(w) < 0) {
                        queue.add(w);
                        dist.put(w, dist.get(v) + 1);
                    }
                    if (dist.get(w) == dist.get(v) + 1) {
                        sigma.put(w, sigma.get(w) + sigma.get(v));
                        pred.get(w).add(v);
                    }
                }
            }
            Map<N, Double> delta = new LinkedHashMap<>();
            for (N v : g.nodes()) delta.put(v, 0.0);
            while (!stack.isEmpty()) {
                N w = stack.pop();
                for (N v : pred.get(w)) {
                    delta.put(v, delta.get(v) + (sigma.get(v) * 1.0 / sigma.get(w)) * (1.0 + delta.get(w)));
                }
                if (!w.equals(s)) cb.put(w, cb.get(w) + delta.get(w));
            }
        }
        if (normalized && g.order() > 2) {
            double scale = 1.0 / ((g.order() - 1) * (g.order() - 2));
            Map<N, Double> norm = new LinkedHashMap<>();
            for (N v : cb.keySet()) norm.put(v, cb.get(v) * scale);
            return norm;
        }
        return cb;
    }

    // ---- PageRank ----

    /** PageRank (networkx default: alpha=0.85, no personalization). */
    public static <N> Map<N, Double> pagerank(DiGraph<N> g) {
        return pagerank(g, 0.85, 1e-6, 100, null);
    }

    public static <N> Map<N, Double> pagerank(DiGraph<N> g, double alpha, double tol, int maxIter, Map<N, Double> personalization) {
        int n = g.order();
        if (n == 0) return new LinkedHashMap<>();
        Map<N, Double> pr = new LinkedHashMap<>();
        double init = 1.0 / n;
        for (N v : g.nodes()) pr.put(v, init);
        // Personalization: default to uniform if null.
        Map<N, Double> p = personalization;
        if (p == null) {
            p = new LinkedHashMap<>();
            for (N v : g.nodes()) p.put(v, init);
        }
        double teleport = 1.0 - alpha;
        for (int iter = 0; iter < maxIter; iter++) {
            Map<N, Double> next = new LinkedHashMap<>();
            double danglingSum = 0;
            for (N v : g.nodes()) {
                if (g.outDegree(v) == 0) {
                    danglingSum += pr.get(v);
                }
            }
            for (N v : g.nodes()) {
                double val = teleport * p.get(v);
                for (N u : g.predecessors(v)) {
                    int outDeg = g.outDegree(u);
                    if (outDeg > 0) val += alpha * pr.get(u) / outDeg;
                }
                val += alpha * danglingSum * p.get(v);
                next.put(v, val);
            }
            double diff = 0;
            for (N v : g.nodes()) diff += Math.abs(next.get(v) - pr.get(v));
            pr = next;
            if (diff < tol) break;
        }
        return pr;
    }

    public static <N> Map<N, Double> pagerank(Graph<N> g) {
        // Convert undirected to DiGraph (both directions)
        DiGraph<N> dg = new DiGraph<>();
        for (N n : g.nodes()) dg.addNode(n);
        for (Map.Entry<N, N> e : g.edges()) {
            dg.addEdge(e.getKey(), e.getValue());
            dg.addEdge(e.getValue(), e.getKey());
        }
        return pagerank(dg);
    }

    // ---- Eigenvector centrality (power iteration, undirected) ----

    /**
     * Eigenvector centrality via power iteration. Stops when L1 delta is below {@code tol}
     * or {@code maxIter} iterations. For directed graphs use the in-edge variant.
     */
    public static <N> Map<N, Double> eigenvectorCentrality(Graph<N> g) {
        return eigenvectorCentrality(g, 1e-6, 100);
    }

    public static <N> Map<N, Double> eigenvectorCentrality(Graph<N> g, double tol, int maxIter) {
        Map<N, Double> x = new LinkedHashMap<>();
        for (N n : g.nodes()) x.put(n, 1.0 / Math.sqrt(g.order()));
        for (int iter = 0; iter < maxIter; iter++) {
            Map<N, Double> xNext = new LinkedHashMap<>();
            for (N n : g.nodes()) xNext.put(n, 0.0);
            for (Map.Entry<N, N> e : g.edges()) {
                xNext.put(e.getKey(), xNext.get(e.getKey()) + x.get(e.getValue()));
                xNext.put(e.getValue(), xNext.get(e.getValue()) + x.get(e.getKey()));
            }
            double norm = 0;
            for (double v : xNext.values()) norm += v * v;
            norm = Math.sqrt(norm);
            if (norm < 1e-12) break;
            for (N n : g.nodes()) xNext.put(n, xNext.get(n) / norm);
            double diff = 0;
            for (N n : g.nodes()) diff += Math.abs(xNext.get(n) - x.get(n));
            x = xNext;
            if (diff < tol) break;
        }
        return x;
    }

    /** Katz centrality. Defaults match NetworkX: alpha=0.1, beta=1.0. */
    public static <N> Map<N, Double> katzCentrality(Graph<N> g) {
        return katzCentrality(g, 0.1, 1.0, 1e-6, 100);
    }

    public static <N> Map<N, Double> katzCentrality(Graph<N> g, double alpha, double beta, double tol, int maxIter) {
        Map<N, Double> x = new LinkedHashMap<>();
        for (N n : g.nodes()) x.put(n, beta);
        for (int iter = 0; iter < maxIter; iter++) {
            Map<N, Double> xNext = new LinkedHashMap<>();
            for (N n : g.nodes()) xNext.put(n, 0.0);
            for (Map.Entry<N, N> e : g.edges()) {
                xNext.put(e.getKey(), xNext.get(e.getKey()) + alpha * x.get(e.getValue()));
                xNext.put(e.getValue(), xNext.get(e.getValue()) + alpha * x.get(e.getKey()));
            }
            for (N n : g.nodes()) xNext.put(n, xNext.get(n) + beta);
            double diff = 0;
            for (N n : g.nodes()) diff += Math.abs(xNext.get(n) - x.get(n));
            x = xNext;
            if (diff < tol) break;
        }
        return x;
    }

    // ---- HITS (Hyperlink-Induced Topic Search) ----

    /** HITS hubs and authorities for directed graphs. */
    public static <N> HITSResult<N> hits(DiGraph<N> g) {
        return hits(g, 1e-6, 100);
    }

    public static <N> HITSResult<N> hits(DiGraph<N> g, double tol, int maxIter) {
        Map<N, Double> hubs = new LinkedHashMap<>();
        Map<N, Double> auths = new LinkedHashMap<>();
        for (N n : g.nodes()) {
            hubs.put(n, 1.0);
            auths.put(n, 1.0);
        }
        for (int iter = 0; iter < maxIter; iter++) {
            // Update authorities: sum of hubs pointing to v
            Map<N, Double> newAuth = new LinkedHashMap<>();
            for (N v : g.nodes()) newAuth.put(v, 0.0);
            for (Map.Entry<N, N> e : g.edges()) {
                newAuth.put(e.getValue(), newAuth.get(e.getValue()) + hubs.get(e.getKey()));
            }
            double authNorm = Math.sqrt(newAuth.values().stream().mapToDouble(d -> d * d).sum());
            if (authNorm > 0) for (N v : g.nodes()) newAuth.put(v, newAuth.get(v) / authNorm);
            // Update hubs: sum of auths v points to
            Map<N, Double> newHub = new LinkedHashMap<>();
            for (N v : g.nodes()) newHub.put(v, 0.0);
            for (Map.Entry<N, N> e : g.edges()) {
                newHub.put(e.getKey(), newHub.get(e.getKey()) + newAuth.get(e.getValue()));
            }
            double hubNorm = Math.sqrt(newHub.values().stream().mapToDouble(d -> d * d).sum());
            if (hubNorm > 0) for (N v : g.nodes()) newHub.put(v, newHub.get(v) / hubNorm);
            double diff = 0;
            for (N v : g.nodes()) diff += Math.abs(newHub.get(v) - hubs.get(v)) + Math.abs(newAuth.get(v) - auths.get(v));
            hubs = newHub;
            auths = newAuth;
            if (diff < tol) break;
        }
        return new HITSResult<>(hubs, auths);
    }

    public static final class HITSResult<N> {
        public final Map<N, Double> hubs;
        public final Map<N, Double> authorities;
        public HITSResult(Map<N, Double> hubs, Map<N, Double> authorities) {
            this.hubs = hubs;
            this.authorities = authorities;
        }
    }
}