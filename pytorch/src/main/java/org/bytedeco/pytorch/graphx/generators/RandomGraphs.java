/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.generators;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Random graph generators: Erdős-Rényi, Barabási-Albert, Watts-Strogatz, etc.
 *
 * <p>Aligned with {@code networkx.generators.random_graphs}.
 */
public final class RandomGraphs {
    private RandomGraphs() {}

    /** Binomial / Erdős-Rényi G(n, p). Each edge included with probability p. */
    public static Graph<Integer> gnpRandomGraph(int n, double p, long seed) {
        Random rng = new Random(seed);
        return gnpRandomGraph(n, p, rng);
    }

    public static Graph<Integer> gnpRandomGraph(int n, double p, Random rng) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (rng.nextDouble() < p) g.addEdge(i, j);
            }
        }
        return g;
    }

    /** Erdős-Rényi G(n, m) — random graph with exactly m edges. */
    public static Graph<Integer> gnmRandomGraph(int n, int m, long seed) {
        Random rng = new Random(seed);
        return gnmRandomGraph(n, m, rng);
    }

    public static Graph<Integer> gnmRandomGraph(int n, int m, Random rng) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        long maxEdges = (long) n * (n - 1) / 2;
        if (m > maxEdges) m = (int) maxEdges;
        Set<Long> seen = new HashSet<>();
        int added = 0;
        while (added < m) {
            int u = rng.nextInt(n);
            int v = rng.nextInt(n);
            if (u == v) continue;
            if (u > v) { int tmp = u; u = v; v = tmp; }
            long key = ((long) u << 32) | v;
            if (seen.add(key)) {
                g.addEdge(u, v);
                added++;
            }
        }
        return g;
    }

    /** Alias for gnp_random_graph with default seed. */
    public static Graph<Integer> erdosRenyiGraph(int n, double p, long seed) {
        return gnpRandomGraph(n, p, seed);
    }

    /**
     * Watts-Strogatz small-world model: start with ring lattice, rewire edges with
     * probability p. Each node has k neighbors (k must be even).
     */
    public static Graph<Integer> wattsStrogatzGraph(int n, int k, double p, long seed) {
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        // Ring lattice: each node i connects to k/2 neighbors on each side
        int halfK = k / 2;
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= halfK; j++) {
                int target = (i + j) % n;
                if (i < target) g.addEdge(i, target);
                else g.addEdge(target, i);
            }
        }
        // Rewire
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= halfK; j++) {
                int target = (i + j) % n;
                if (rng.nextDouble() < p) {
                    // Remove old edge
                    g.removeEdge(i, target);
                    // Pick new target different from i and current neighbors
                    Set<Integer> existing = new LinkedHashSet<>(g.neighbors(i));
                    existing.add(i);
                    int newTarget = rng.nextInt(n);
                    while (existing.contains(newTarget)) newTarget = rng.nextInt(n);
                    g.addEdge(i, newTarget);
                }
            }
        }
        return g;
    }

    /**
     * Barabási-Albert preferential attachment model. Each new node adds m edges
     * to existing nodes with probability proportional to degree.
     */
    public static Graph<Integer> barabasiAlbertGraph(int n, int m, long seed) {
        if (m < 1 || m >= n) throw new IllegalArgumentException("m must be 1 <= m < n");
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        // Initial connected seed of m+1 nodes
        for (int i = 0; i <= m; i++) g.addNode(i);
        for (int i = 1; i <= m; i++) g.addEdge(0, i);

        // Cumulative degree sum for preferential sampling
        int[] degs = new int[n];
        for (int i = 0; i <= m; i++) degs[i] = g.degree(i);
        int totalDeg = 0;
        for (int i = 0; i <= m; i++) totalDeg += degs[i];

        for (int newNode = m + 1; newNode < n; newNode++) {
            g.addNode(newNode);
            Set<Integer> picked = new LinkedHashSet<>();
            int attempts = 0;
            while (picked.size() < m && attempts < 1000) {
                int target = samplePreferential(degs, totalDeg, rng);
                attempts++;
                if (picked.contains(target)) continue;
                picked.add(target);
                g.addEdge(newNode, target);
                degs[target]++;
                degs[newNode]++;
                totalDeg += 2;
            }
        }
        return g;
    }

    private static int samplePreferential(int[] degs, int totalDeg, Random rng) {
        double r = rng.nextDouble() * totalDeg;
        double cum = 0;
        for (int i = 0; i < degs.length; i++) {
            cum += degs[i];
            if (cum >= r) return i;
        }
        return degs.length - 1;
    }

    /**
     * Newman-Watts-Strogatz: ring lattice + additional random edges (no rewiring).
     */
    public static Graph<Integer> newmanWattsStrogatzGraph(int n, int k, double p, long seed) {
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        int halfK = k / 2;
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= halfK; j++) {
                int target = (i + j) % n;
                if (i < target) g.addEdge(i, target);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= halfK; j++) {
                int target = (i + j) % n;
                if (rng.nextDouble() < p && !g.hasEdge(i, target)) {
                    g.addEdge(i, target);
                }
            }
        }
        return g;
    }

    /** Random regular graph: every node has degree {@code d}. */
    public static Graph<Integer> randomRegularGraph(int d, int n, long seed) {
        if (d * n % 2 != 0) throw new IllegalArgumentException("d*n must be even");
        if (d >= n) throw new IllegalArgumentException("d < n required");
        Random rng = new Random(seed);
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        // Use the configuration model approach
        int[] degs = new int[n];
        java.util.Arrays.fill(degs, d);
        boolean success = false;
        int tries = 0;
        while (!success && tries < 100) {
            int[] stubs = new int[n * d];
            int idx = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) stubs[idx++] = i;
            }
            // Shuffle
            for (int i = stubs.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int t = stubs[i]; stubs[i] = stubs[j]; stubs[j] = t;
            }
            // Try to pair
            Graph<Integer> candidate = new Graph<>();
            for (int i = 0; i < n; i++) candidate.addNode(i);
            boolean ok = true;
            for (int i = 0; i < stubs.length; i += 2) {
                int a = stubs[i], b = stubs[i + 1];
                if (a == b || candidate.hasEdge(a, b)) { ok = false; break; }
                candidate.addEdge(a, b);
            }
            if (ok) {
                g = candidate;
                success = true;
            }
            tries++;
        }
        if (!success) throw new RuntimeException("Failed to construct regular graph");
        return g;
    }
}