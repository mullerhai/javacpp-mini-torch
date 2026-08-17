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
import java.util.List;
import java.util.Random;

/**
 * Classic graph generators (complete, cycle, path, star, wheel, etc.).
 *
 * <p>Aligned with {@code networkx.generators.classic}.
 */
public final class Classic {
    private Classic() {}

    public static Graph<Integer> completeGraph(int n) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                g.addEdge(i, j);
            }
        }
        return g;
    }

    public static DiGraph<Integer> completeGraphDirected(int n) {
        DiGraph<Integer> g = new DiGraph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) g.addEdge(i, j);
            }
        }
        return g;
    }

    public static Graph<Integer> cycleGraph(int n) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            if (i < j) g.addEdge(i, j);
        }
        return g;
    }

    public static Graph<Integer> pathGraph(int n) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n - 1; i++) g.addEdge(i, i + 1);
        return g;
    }

    public static <N> Graph<N> starGraph(N center, List<? extends N> leaves) {
        Graph<N> g = new Graph<>();
        g.addNode(center);
        for (N leaf : leaves) {
            g.addNode(leaf);
            g.addEdge(center, leaf);
        }
        return g;
    }

    public static Graph<Integer> starGraph(int n) {
        Graph<Integer> g = new Graph<>();
        if (n <= 0) return g;
        g.addNode(0);
        for (int i = 1; i < n; i++) {
            g.addNode(i);
            g.addEdge(0, i);
        }
        return g;
    }

    public static Graph<Integer> wheelGraph(int n) {
        Graph<Integer> g = cycleGraph(n - 1);
        g.addNode(n - 1);
        for (int i = 0; i < n - 1; i++) g.addEdge(i, n - 1);
        return g;
    }

    public static Graph<Integer> ladderGraph(int n) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < 2 * n; i++) g.addNode(i);
        for (int i = 0; i < n - 1; i++) {
            g.addEdge(i, i + 1);
            g.addEdge(n + i, n + i + 1);
            g.addEdge(i, n + i);
        }
        return g;
    }

    public static Graph<Integer> circularLadderGraph(int n) {
        Graph<Integer> g = ladderGraph(n);
        g.addEdge(0, n - 1);
        g.addEdge(n, 2 * n - 1);
        return g;
    }

    public static Graph<Integer> barbellGraph(int m1, int m2) {
        Graph<Integer> g = completeGraph(m1);
        int base = m1;
        g.addNode(base);
        Graph<Integer> right = completeGraph(m2);
        for (int i = 0; i < m2; i++) {
            g.addNode(base + 1 + i);
            if (right.hasEdge(i, j(m2, i, 0))) {
                // copy edges from right
            }
        }
        // Manually connect two complete graphs via a bridge
        // Re-build the second complete graph on top
        for (int i = 0; i < m2; i++) {
            for (int j = i + 1; j < m2; j++) {
                g.addEdge(base + 1 + i, base + 1 + j);
            }
        }
        // Bridge
        g.addEdge(m1 - 1, base);
        g.addEdge(base, base + 1);
        return g;
    }

    private static int j(int n, int i, int def) {
        return def;
    }

    public static Graph<Integer> lollipopGraph(int m, int n) {
        Graph<Integer> g = completeGraph(m);
        int base = m;
        for (int i = 0; i < n; i++) g.addNode(base + i);
        for (int i = 0; i < n - 1; i++) g.addEdge(base + i, base + i + 1);
        g.addEdge(m - 1, base);
        return g;
    }

    /** Turán graph: complete multipartite graph. */
    public static Graph<Integer> turanGraph(int n, int r) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int partI = i % r;
                int partJ = j % r;
                if (partI != partJ) g.addEdge(i, j);
            }
        }
        return g;
    }

    /** Null graph: n isolated nodes. */
    public static Graph<Integer> nullGraph(int n) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        return g;
    }

    /** Empty graph (alias for null_graph). */
    public static Graph<Integer> emptyGraph(int n) { return nullGraph(n); }

    /** Balanced r-ary tree of height h. Number of nodes = (r^(h+1)-1)/(r-1). */
    public static Graph<Integer> balancedTree(int r, int h) {
        Graph<Integer> g = new Graph<>();
        if (r <= 0 || h < 0) return g;
        g.addNode(0);
        java.util.Deque<int[]> stack = new java.util.ArrayDeque<>();
        stack.push(new int[]{0, 0});
        int nextId = 1;
        while (!stack.isEmpty()) {
            int[] top = stack.pop();
            int parent = top[0];
            int depth = top[1];
            if (depth == h) continue;
            for (int i = 0; i < r; i++) {
                int child = nextId++;
                g.addNode(child);
                g.addEdge(parent, child);
                stack.push(new int[]{child, depth + 1});
            }
        }
        return g;
    }

    /** Full r-ary tree of height h (all leaves at the same level). */
    public static Graph<Integer> fullRaryTree(int r, int h) { return balancedTree(r, h); }

    public static Graph<Integer> binomialTree(int n) { return balancedTree(2, n - 1); }

    /** Circulant graph: nodes 0..n-1; each i connects to i±offsets[j] mod n. */
    public static Graph<Integer> circulantGraph(int n, int[] offsets) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int off : offsets) {
                int j = ((i + off) % n + n) % n;
                if (i < j) g.addEdge(i, j);
            }
        }
        return g;
    }

    public static DiGraph<Integer> circulantGraphDirected(int n, int[] offsets) {
        DiGraph<Integer> g = new DiGraph<>();
        for (int i = 0; i < n; i++) g.addNode(i);
        for (int i = 0; i < n; i++) {
            for (int off : offsets) {
                int j = ((i + off) % n + n) % n;
                g.addEdge(i, j);
            }
        }
        return g;
    }

    public static Graph<Integer> completeMultipartiteGraph(int[] sizes) {
        Graph<Integer> g = new Graph<>();
        List<Integer> nodes = new ArrayList<>();
        int idx = 0;
        for (int sz : sizes) {
            for (int i = 0; i < sz; i++) nodes.add(idx++);
        }
        for (int i = 0; i < nodes.size(); i++) g.addNode(nodes.get(i));
        int[] starts = new int[sizes.length];
        for (int i = 1; i < sizes.length; i++) starts[i] = starts[i - 1] + sizes[i - 1];
        for (int p = 0; p < sizes.length; p++) {
            for (int q = p + 1; q < sizes.length; q++) {
                for (int a = starts[p]; a < starts[p] + sizes[p]; a++) {
                    for (int b = starts[q]; b < starts[q] + sizes[q]; b++) {
                        g.addEdge(nodes.get(a), nodes.get(b));
                    }
                }
            }
        }
        return g;
    }
}