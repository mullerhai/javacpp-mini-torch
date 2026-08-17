/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.algorithms.traversal;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.*;

/**
 * Graph traversal: BFS and DFS for undirected/directed graphs.
 *
 * <p>Mirrors {@code networkx.algorithms.traversal.breadth_first_search} and
 * {@code networkx.algorithms.traversal.depth_first_search}.
 */
public final class Traversal {
    private Traversal() {}

    // ---- BFS (unweighted) ----

    /**
     * Compute shortest-path distances from {@code source} using BFS.
     * Returns {@code null} for unreachable nodes.
     */
    public static <N> Map<N, Integer> bfsDistances(Graph<N> g, N source) {
        return bfsDistancesUndirected(g, source);
    }

    static <N> Map<N, Integer> bfsDistancesUndirected(Graph<N> g, N source) {
        Map<N, Integer> dist = new LinkedHashMap<>();
        dist.put(source, 0);
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            int d = dist.get(u);
            for (N v : g.neighbors(u)) {
                if (!dist.containsKey(v)) {
                    dist.put(v, d + 1);
                    queue.add(v);
                }
            }
        }
        return dist;
    }

    public static <N> Map<N, Integer> bfsDistances(DiGraph<N> g, N source) {
        Map<N, Integer> dist = new LinkedHashMap<>();
        dist.put(source, 0);
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            int d = dist.get(u);
            for (N v : g.successors(u)) {
                if (!dist.containsKey(v)) {
                    dist.put(v, d + 1);
                    queue.add(v);
                }
            }
        }
        return dist;
    }

    /** Compute predecessor map from BFS tree (NetworkX {@code bfs_predecessors}). */
    public static <N> Map<N, N> bfsPredecessors(Graph<N> g, N source) {
        Map<N, N> pred = new LinkedHashMap<>();
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            for (N v : g.neighbors(u)) {
                if (!pred.containsKey(v) && !v.equals(source)) {
                    pred.put(v, u);
                    queue.add(v);
                }
            }
        }
        return pred;
    }

    public static <N> Map<N, N> bfsPredecessors(DiGraph<N> g, N source) {
        Map<N, N> pred = new LinkedHashMap<>();
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            for (N v : g.successors(u)) {
                if (!pred.containsKey(v) && !v.equals(source)) {
                    pred.put(v, u);
                    queue.add(v);
                }
            }
        }
        return pred;
    }

    /** Compute BFS tree edges (NetworkX {@code bfs_edges}). */
    public static <N> List<Map.Entry<N, N>> bfsEdges(Graph<N> g, N source) {
        List<Map.Entry<N, N>> edges = new ArrayList<>();
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        Set<N> visited = new LinkedHashSet<>();
        visited.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            for (N v : g.neighbors(u)) {
                if (visited.add(v)) {
                    edges.add(new java.util.AbstractMap.SimpleImmutableEntry<>(u, v));
                    queue.add(v);
                }
            }
        }
        return edges;
    }

    public static <N> List<Map.Entry<N, N>> bfsEdges(DiGraph<N> g, N source) {
        List<Map.Entry<N, N>> edges = new ArrayList<>();
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        Set<N> visited = new LinkedHashSet<>();
        visited.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            for (N v : g.successors(u)) {
                if (visited.add(v)) {
                    edges.add(new java.util.AbstractMap.SimpleImmutableEntry<>(u, v));
                    queue.add(v);
                }
            }
        }
        return edges;
    }

    /** Generic BFS traversal yielding nodes in BFS order. */
    public static <N> List<N> bfsOrder(Graph<N> g, N source) {
        List<N> order = new ArrayList<>();
        Deque<N> queue = new ArrayDeque<>();
        queue.add(source);
        Set<N> visited = new LinkedHashSet<>();
        visited.add(source);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            order.add(u);
            for (N v : g.neighbors(u)) {
                if (visited.add(v)) queue.add(v);
            }
        }
        return order;
    }

    /** DFS preorder traversal (NetworkX {@code dfs_preorder}). */
    public static <N> List<N> dfsPreorder(Graph<N> g, N source) {
        List<N> order = new ArrayList<>();
        Set<N> visited = new LinkedHashSet<>();
        Deque<N> stack = new ArrayDeque<>();
        stack.push(source);
        while (!stack.isEmpty()) {
            N u = stack.pop();
            if (visited.add(u)) {
                order.add(u);
                for (N v : g.neighbors(u)) {
                    if (!visited.contains(v)) stack.push(v);
                }
            }
        }
        return order;
    }

    public static <N> List<N> dfsPreorder(DiGraph<N> g, N source) {
        List<N> order = new ArrayList<>();
        Set<N> visited = new LinkedHashSet<>();
        Deque<N> stack = new ArrayDeque<>();
        stack.push(source);
        while (!stack.isEmpty()) {
            N u = stack.pop();
            if (visited.add(u)) {
                order.add(u);
                for (N v : g.successors(u)) {
                    if (!visited.contains(v)) stack.push(v);
                }
            }
        }
        return order;
    }

    /** DFS postorder traversal (NetworkX {@code dfs_postorder}). */
    public static <N> List<N> dfsPostorder(Graph<N> g, N source) {
        List<N> order = new ArrayList<>();
        Set<N> visited = new LinkedHashSet<>();
        Deque<Map.Entry<N, Iterator<N>>> stack = new ArrayDeque<>();
        stack.push(new java.util.AbstractMap.SimpleImmutableEntry<>(source, g.neighbors(source).iterator()));
        while (!stack.isEmpty()) {
            Map.Entry<N, Iterator<N>> frame = stack.peek();
            N u = frame.getKey();
            Iterator<N> it = frame.getValue();
            if (it.hasNext()) {
                N v = it.next();
                if (visited.add(v)) {
                    stack.push(new java.util.AbstractMap.SimpleImmutableEntry<>(v, g.neighbors(v).iterator()));
                }
            } else {
                stack.pop();
                order.add(u);
            }
        }
        return order;
    }
}