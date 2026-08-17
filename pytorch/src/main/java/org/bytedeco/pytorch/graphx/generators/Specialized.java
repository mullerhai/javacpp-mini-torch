/*
 * GraphX: Specialized graph generators — social, ego, internet, geographic, tree.
 *
 * Inspired by networkx.generators.social, ego, internet_as_graphs, geographic, trees.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.generators;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Specialized generators covering social, ego, geographic, and tree graph families.
 */
public final class Specialized {
    private Specialized() {}

    // =========================================================================
    // Geographic
    // =========================================================================

    /**
     * Random geometric graph: n nodes randomly placed in a unit cube, edges
     * added between nodes within Euclidean distance {@code radius}.
     */
    public static Graph<Integer> randomGeometricGraph(int n, double radius, long seed) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rng.nextDouble();
            y[i] = rng.nextDouble();
        }
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        double r2 = radius * radius;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                if (dx * dx + dy * dy <= r2) g.addEdge(i, j);
            }
        }
        return g;
    }

    /** Waxman graph: geographic with probability ∝ exp(-d / (β L)). */
    public static Graph<Integer> waxmanGraph(int n, double beta, double alpha, long seed) {
        Random rng = new Random(seed);
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rng.nextDouble();
            y[i] = rng.nextDouble();
        }
        // L = max distance
        double L = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = x[i] - x[j], dy = y[i] - y[j];
                L = Math.max(L, Math.sqrt(dx * dx + dy * dy));
            }
        }
        if (L == 0) L = 1;
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = x[i] - x[j], dy = y[i] - y[j];
                double d = Math.sqrt(dx * dx + dy * dy);
                double p = alpha * Math.exp(-d / (beta * L));
                if (rng.nextDouble() < p) g.addEdge(i, j);
            }
        }
        return g;
    }

    // =========================================================================
    // Trees
    // =========================================================================

    /** Random labelled tree on {@code n} nodes (Cayley formula: n^(n-2) trees). */
    public static Graph<Integer> randomLabeledTree(int n, long seed) {
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        if (n <= 1) return g;
        // Prüfer sequence: n-2 labels from {0, ..., n-1}
        int[] prufer = new int[n - 2];
        for (int i = 0; i < n - 2; i++) prufer[i] = rng.nextInt(n);
        int[] degree = new int[n];
        for (int i = 0; i < n; i++) degree[i] = 1;
        for (int v : prufer) degree[v]++;
        // Decode
        java.util.PriorityQueue<Integer> pq = new java.util.PriorityQueue<>();
        for (int i = 0; i < n; i++) if (degree[i] == 1) pq.add(i);
        for (int v : prufer) {
            int u = pq.poll();
            g.addEdge(u, v);
            degree[u]--;
            degree[v]--;
            if (degree[v] == 1) pq.add(v);
        }
        // Last two nodes
        int[] last = new int[2];
        int idx = 0;
        for (int i = 0; i < n; i++) if (degree[i] == 1) last[idx++] = i;
        if (idx == 2) g.addEdge(last[0], last[1]);
        return g;
    }

    /** Random rooted tree: BFS tree from a random root with random branching. */
    public static Graph<Integer> randomLabeledRootedTree(int n, long seed) {
        if (n == 0) return new Graph<>();
        Graph<Integer> g = newGraph();
        Random rng = new Random(seed);
        java.util.Deque<Integer> stack = new java.util.ArrayDeque<>();
        g.addNode(0);
        stack.push(0);
        int nextId = 1;
        while (!stack.isEmpty() && nextId < n) {
            int u = stack.pop();
            int children = 1 + rng.nextInt(Math.max(1, n - nextId));
            for (int i = 0; i < children && nextId < n; i++) {
                int v = nextId++;
                g.addNode(v);
                g.addEdge(u, v);
                stack.push(v);
            }
        }
        return g;
    }

    private static Graph<Integer> newGraph() { return new Graph<>(); }

    // =========================================================================
    // Social / ego
    // =========================================================================

    /**
     * Ego graph: induced subgraph of {@code g} on the neighbors of {@code center},
     * plus the center node itself.
     */
    public static <N> Graph<N> egoGraph(Graph<N> g, N center) {
        Set<N> alters = new LinkedHashSet<>();
        alters.add(center);
        for (N n : g.neighbors(center)) alters.add(n);
        return g.subgraph(alters);
    }

    /** Friendship graph: n nodes, each pair of friends with probability p. */
    public static Graph<Integer> friendshipGraph(int n, double p, long seed) {
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (rng.nextDouble() < p) g.addEdge(i, j);
            }
        }
        return g;
    }

    // =========================================================================
    // Social / Internet
    // =========================================================================

    /** Internet AS-level graph: preferential attachment with constraints. */
    public static DiGraph<Integer> randomInternetASGraph(int n, long seed) {
        Random rng = new Random(seed);
        DiGraph<Integer> g = new DiGraph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        if (n <= 1) return g;
        // Initial seed of 3 nodes fully connected
        for (int i = 0; i < Math.min(3, n); i++) {
            for (int j = i + 1; j < Math.min(3, n); j++) {
                g.addEdge(i, j);
                g.addEdge(j, i);
            }
        }
        for (int v = 3; v < n; v++) {
            int m = 2 + rng.nextInt(3);
            Set<Integer> picked = new LinkedHashSet<>();
            while (picked.size() < m) {
                int target = rng.nextInt(v);
                if (!picked.contains(target)) picked.add(target);
            }
            for (int t : picked) {
                g.addEdge(v, t);
                g.addEdge(t, v);
            }
        }
        return g;
    }

    // =========================================================================
    // Stochastic Block Model
    // =========================================================================

    /**
     * Stochastic block model: nodes in {@code blockSizes} groups, edge probability
     * between groups is {@code pMatrix[i][j]}.
     */
    public static Graph<Integer> stochasticBlockModel(int[] blockSizes, double[][] pMatrix, long seed) {
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        // Assign node IDs to blocks
        List<Integer> nodeIds = new ArrayList<>();
        int idx = 0;
        int[] blockOfNode = new int[sum(blockSizes)];
        for (int b = 0; b < blockSizes.length; b++) {
            for (int k = 0; k < blockSizes[b]; k++) {
                nodeIds.add(idx);
                blockOfNode[idx++] = b;
            }
        }
        for (int id : nodeIds) g.addNode(id);
        for (int i = 0; i < nodeIds.size(); i++) {
            for (int j = i + 1; j < nodeIds.size(); j++) {
                int bi = blockOfNode[i], bj = blockOfNode[j];
                double p = pMatrix[bi][bj];
                if (rng.nextDouble() < p) g.addEdge(nodeIds.get(i), nodeIds.get(j));
            }
        }
        return g;
    }

    static int sum(int[] a) {
        int s = 0;
        for (int v : a) s += v;
        return s;
    }
}