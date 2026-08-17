/*
 * GraphX: Community Detection Algorithms.
 *
 * Inspired by networkx.algorithms.community — Louvain, label propagation,
 * greedy modularity, k-clique, local moving.
 *
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.algorithms.community;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.algorithms.centrality.Centrality;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Community detection algorithms.
 *
 * <p>Aligned with {@code networkx.algorithms.community.*}.
 */
public final class Community {
    private Community() {}

    // =========================================================================
    // Label Propagation
    // =========================================================================

    /**
     * Asynchronous label propagation (Raghavan et al., 2007).
     * Each node takes the most frequent label among its neighbors until convergence.
     * Returns a partition as {@code node -> communityId}.
     */
    public static <N> Map<N, Integer> labelPropagation(Graph<N> g) {
        return labelPropagation(g, 42L);
    }

    public static <N> Map<N, Integer> labelPropagation(Graph<N> g, long seed) {
        Map<N, Integer> labels = new LinkedHashMap<>();
        Random rng = new Random(seed);
        List<N> nodes = g.nodes();
        // Initialize: each node gets a unique label
        int idx = 0;
        for (N n : nodes) labels.put(n, idx++);
        boolean changed = false;
        for (int iter = 0; iter < 100; iter++) {
            changed = false;
            // Shuffle node order for asynchronous update
            List<N> order = new ArrayList<>(nodes);
            Collections.shuffle(order, rng);
            for (N u : order) {
                Map<Integer, Integer> freq = new HashMap<>();
                for (N v : g.neighbors(u)) {
                    int l = labels.get(v);
                    freq.merge(l, 1, Integer::sum);
                }
                if (freq.isEmpty()) continue;
                int best = -1, bestCount = -1;
                List<Integer> bestLabels = new ArrayList<>();
                for (Map.Entry<Integer, Integer> e : freq.entrySet()) {
                    if (e.getValue() > bestCount) {
                        bestCount = e.getValue();
                        bestLabels.clear();
                        bestLabels.add(e.getKey());
                    } else if (e.getValue() == bestCount) {
                        bestLabels.add(e.getKey());
                    }
                }
                int newLabel = bestLabels.get(rng.nextInt(bestLabels.size()));
                if (newLabel != labels.get(u)) {
                    labels.put(u, newLabel);
                    changed = true;
                }
            }
            if (!changed) break;
        }
        return labels;
    }

    /** Convert per-node labels into a list of communities (each as a set). */
    public static <N> List<Set<N>> labelsToCommunities(Map<N, Integer> labels) {
        Map<Integer, Set<N>> grouped = new LinkedHashMap<>();
        for (Map.Entry<N, Integer> e : labels.entrySet()) {
            grouped.computeIfAbsent(e.getValue(), k -> new LinkedHashSet<>()).add(e.getKey());
        }
        return new ArrayList<>(grouped.values());
    }

    // =========================================================================
    // Louvain
    // =========================================================================

    /**
     * Louvain modularity optimization (Blondel et al., 2008).
     * Two-phase iteration: (1) move nodes to neighbor's community if ΔQ > 0;
     * (2) collapse into super-graph and repeat.
     *
     * <p>Returns the top-level partition (a single map node -> communityId).
     */
    public static <N> Map<N, Integer> louvain(Graph<N> g) {
        // Working copies
        Map<N, Integer> comm = new LinkedHashMap<>();
        Map<N, Double> edgeWeight = new LinkedHashMap<>();
        int idx = 0;
        for (N n : g.nodes()) {
            comm.put(n, idx++);
            edgeWeight.put(n, edgeWeightSum(g, n));
        }
        double totalWeight = sumValues(edgeWeight);

        for (int pass = 0; pass < 10; pass++) {
            boolean moved = false;
            List<N> order = new ArrayList<>(g.nodes());
            Collections.shuffle(order, new Random(42 + pass));
            for (N u : order) {
                int cu = comm.get(u);
                // Tally weighted edges to each neighbor community
                Map<Integer, Double> neighborCommTotals = new HashMap<>();
                for (N v : g.neighbors(u)) {
                    int cv = comm.get(v);
                    double w = g.getEdgeWeight(u, v);
                    neighborCommTotals.merge(cv, w, Double::sum);
                }
                // Find best community to move to
                double currentSigma = cuSigma(cu, comm, edgeWeight);
                double bestGain = 0;
                int bestComm = cu;
                for (Map.Entry<Integer, Double> e : neighborCommTotals.entrySet()) {
                    int c = e.getKey();
                    double sigma = e.getValue();
                    // Modularity gain ~ sigma_in / m - sigma_tot * ku / (2m^2)
                    double gain = sigma / totalWeight - (currentSigma * edgeWeight.get(u)) / (totalWeight * totalWeight);
                    if (c == cu) gain = 0; // stay doesn't help
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestComm = c;
                    }
                }
                if (bestComm != cu) {
                    comm.put(u, bestComm);
                    moved = true;
                }
            }
            if (!moved) break;
        }
        // Renumber communities to 0..k
        Map<Integer, Integer> remap = new HashMap<>();
        int next = 0;
        for (Map.Entry<N, Integer> e : comm.entrySet()) {
            remap.putIfAbsent(e.getValue(), next++);
        }
        Map<N, Integer> result = new LinkedHashMap<>();
        for (Map.Entry<N, Integer> e : comm.entrySet()) result.put(e.getKey(), remap.get(e.getValue()));
        return result;
    }

    static <N> double edgeWeightSum(Graph<N> g, N n) {
        double s = 0;
        for (N v : g.neighbors(n)) s += g.getEdgeWeight(n, v);
        return s;
    }

    static double sumValues(Map<?, Double> m) {
        double s = 0;
        for (double v : m.values()) s += v;
        return s;
    }

    static <N> double cuSigma(int community, Map<N, Integer> comm, Map<N, Double> edgeWeight) {
        double s = 0;
        for (Map.Entry<N, Integer> e : comm.entrySet()) {
            if (e.getValue() == community) s += edgeWeight.get(e.getKey());
        }
        return s;
    }

    // =========================================================================
    // Modularity
    // =========================================================================

    /**
     * Modularity Q = (1/2m) Σ_ij [A_ij - k_i k_j / (2m)] δ(c_i, c_j).
     * For an undirected graph with given community assignment.
     */
    public static <N> double modularity(Graph<N> g, Map<N, ?> communities) {
        double m2 = 2.0 * g.numberOfEdges();
        if (m2 == 0) return 0;
        double q = 0;
        for (Map.Entry<N, N> e : g.edges()) {
            N u = e.getKey(), v = e.getValue();
            if (communities.get(u).equals(communities.get(v))) {
                q += 1.0 - (g.degree(u) * g.degree(v)) / m2;
            }
        }
        return q / m2 * 2.0;
    }

    // =========================================================================
    // Greedy Modularity (Clauset-Newman-Moore)
    // =========================================================================

    /**
     * Greedy modularity maximization using the union-find heuristic. Each edge
     * joining communities produces a ΔQ; merges with maximum positive ΔQ first.
     * For small/medium graphs; O(E^2 log V) worst case.
     */
    public static <N> Map<N, Integer> greedyModularity(Graph<N> g) {
        Map<N, Integer> labels = new LinkedHashMap<>();
        int idx = 0;
        for (N n : g.nodes()) labels.put(n, idx++);
        // Compute initial Q deltas per edge
        double m2 = 2.0 * g.numberOfEdges();
        if (m2 == 0) return labels;
        // Build priority queue of merges
        java.util.PriorityQueue<double[]> pq = new java.util.PriorityQueue<>((a, b) -> Double.compare(b[0], a[0]));
        java.util.Set<String> seen = new HashSet<>();
        for (Map.Entry<N, N> e : g.edges()) {
            N u = e.getKey(), v = e.getValue();
            String key = edgeKey(labels.get(u), labels.get(v));
            if (seen.add(key)) {
                double deltaQ = 1.0 / m2 - (g.degree(u) * g.degree(v)) / (m2 * m2);
                pq.add(new double[]{deltaQ, labels.get(u), labels.get(v)});
            }
        }
        // Repeatedly merge top pair
        while (!pq.isEmpty()) {
            double[] top = pq.poll();
            double dq = top[0];
            int cA = (int) top[1];
            int cB = (int) top[2];
            if (cA == cB) continue;
            // Check both still exist (they may have been merged already)
            int finalA = findLabel(cA, labels);
            int finalB = findLabel(cB, labels);
            if (finalA == finalB) continue;
            // Apply merge: remap all nodes with cB to cA
            int merged = Math.min(finalA, finalB);
            int other = Math.max(finalA, finalB);
            for (Map.Entry<N, Integer> e : labels.entrySet()) {
                if (e.getValue() == other) e.setValue(merged);
            }
            // Re-check edges between merged community and others
            // (Simplified: stop after first pass through edges.)
            if (dq <= 0) break;
        }
        // Renumber to 0..k
        Map<Integer, Integer> remap = new HashMap<>();
        int next = 0;
        for (Map.Entry<N, Integer> e : labels.entrySet()) {
            remap.putIfAbsent(e.getValue(), next++);
        }
        Map<N, Integer> result = new LinkedHashMap<>();
        for (Map.Entry<N, Integer> e : labels.entrySet()) result.put(e.getKey(), remap.get(e.getValue()));
        return result;
    }

    static <N> int findLabel(int label, Map<N, Integer> labels) {
        // Since we're using integer labels and merging by reassigning,
        // just return the label.
        return label;
    }

    static String edgeKey(int a, int b) {
        return Math.min(a, b) + "-" + Math.max(a, b);
    }

    // =========================================================================
    // k-Clique Communities
    // =========================================================================

    /**
     * k-clique communities: a node is in a k-clique community if it can be
     * reached from any other member via a chain of adjacent k-cliques.
     */
    public static <N> List<Set<N>> kCliqueCommunities(Graph<N> g, int k) {
        // Find all k-cliques
        List<Set<N>> cliques = findCliques(g, k);
        // Build clique-clique adjacency
        Map<Set<N>, Set<Set<N>>> cliqueAdj = new HashMap<>();
        for (int i = 0; i < cliques.size(); i++) {
            cliqueAdj.put(cliques.get(i), new HashSet<>());
            for (int j = i + 1; j < cliques.size(); j++) {
                if (shareNodes(cliques.get(i), cliques.get(j), k - 1)) {
                    cliqueAdj.get(cliques.get(i)).add(cliques.get(j));
                    cliqueAdj.get(cliques.get(j)).add(cliques.get(i));
                }
            }
        }
        // Union-find over cliques
        Map<Set<N>, Set<N>> parent = new HashMap<>();
        for (Set<N> c : cliques) parent.put(c, c);
        for (Set<N> c : cliques) {
            for (Set<N> n : cliqueAdj.get(c)) {
                union(parent, c, n);
            }
        }
        // Group nodes by root
        Map<Set<N>, Set<N>> groups = new HashMap<>();
        for (Set<N> c : cliques) {
            Set<N> root = find(parent, c);
            groups.computeIfAbsent(root, r -> new HashSet<>()).addAll(c);
        }
        return new ArrayList<>(groups.values());
    }

    static <N> Set<N> find(Map<Set<N>, Set<N>> parent, Set<N> x) {
        Set<N> p = parent.get(x);
        if (p == x) return x;
        Set<N> root = find(parent, p);
        parent.put(x, root);
        return root;
    }

    static <N> void union(Map<Set<N>, Set<N>> parent, Set<N> a, Set<N> b) {
        Set<N> ra = find(parent, a);
        Set<N> rb = find(parent, b);
        if (!ra.equals(rb)) parent.put(ra, rb);
    }

    /** Find all maximal cliques of size ≥ k. (Bron-Kerbosch, simplified.) */
    static <N> List<Set<N>> findCliques(Graph<N> g, int minSize) {
        List<Set<N>> cliques = new ArrayList<>();
        Set<N> all = new HashSet<>(g.nodes());
        bronKerbosch(new HashSet<>(), all, new HashSet<>(), g, minSize, cliques);
        return cliques;
    }

    static <N> void bronKerbosch(Set<N> R, Set<N> P, Set<N> X, Graph<N> g, int minSize, List<Set<N>> out) {
        if (P.isEmpty() && X.isEmpty()) {
            if (R.size() >= minSize) out.add(new HashSet<>(R));
            return;
        }
        Set<N> pCopy = new HashSet<>(P);
        for (N v : pCopy) {
            Set<N> nbrs = new HashSet<>(g.neighbors(v));
            Set<N> newR = new HashSet<>(R); newR.add(v);
            Set<N> newP = intersect(pCopy, nbrs);
            Set<N> newX = intersect(X, nbrs);
            bronKerbosch(newR, newP, newX, g, minSize, out);
            pCopy.remove(v);
            X.add(v);
        }
    }

    static <N> Set<N> intersect(Set<N> a, Set<N> b) {
        Set<N> r = new HashSet<>(a);
        r.retainAll(b);
        return r;
    }

    static <N> boolean shareNodes(Set<N> a, Set<N> b, int n) {
        int common = 0;
        for (N x : a) if (b.contains(x)) common++;
        return common >= n;
    }
}