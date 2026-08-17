/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.algorithms.traversal.Traversal;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * BFS tree layout — root at top, children arranged below.
 *
 * <p>Aligned with {@code networkx.drawing.layout.bfs_layout} (the v3.x BFS-tree layout).
 */
public final class BFSLayout {
    private BFSLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g, N start) {
        Map<N, Integer> depth = new LinkedHashMap<>();
        java.util.Map<Integer, java.util.List<N>> levels = new LinkedHashMap<>();
        java.util.Deque<N> queue = new java.util.ArrayDeque<>();
        queue.add(start);
        depth.put(start, 0);
        levels.computeIfAbsent(0, k -> new java.util.ArrayList<>()).add(start);
        while (!queue.isEmpty()) {
            N u = queue.poll();
            int d = depth.get(u);
            for (N v : g.neighbors(u)) {
                if (!depth.containsKey(v)) {
                    depth.put(v, d + 1);
                    levels.computeIfAbsent(d + 1, k -> new java.util.ArrayList<>()).add(v);
                    queue.add(v);
                }
            }
        }
        Map<N, double[]> pos = new LinkedHashMap<>();
        for (Map.Entry<Integer, java.util.List<N>> e : levels.entrySet()) {
            int d = e.getKey();
            java.util.List<N> level = e.getValue();
            int m = level.size();
            for (int i = 0; i < m; i++) {
                double x = (m == 1) ? 0.5 : (double) i / (m - 1);
                pos.put(level.get(i), new double[]{x, -d});
            }
        }
        return pos;
    }
}