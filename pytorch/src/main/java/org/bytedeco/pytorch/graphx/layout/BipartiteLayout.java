/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Bipartite layout — places top nodes along x-axis and bottom nodes along y-axis.
 *
 * <p>Aligned with {@code networkx.drawing.layout.bipartite_layout}.
 */
public final class BipartiteLayout {
    private BipartiteLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g, java.util.Set<N> topNodes) {
        Map<N, double[]> pos = new LinkedHashMap<>();
        java.util.List<N> topList = new java.util.ArrayList<>(topNodes);
        java.util.List<N> bottomList = new java.util.ArrayList<>();
        for (N n : g.nodes()) {
            if (!topNodes.contains(n)) bottomList.add(n);
        }
        for (int i = 0; i < topList.size(); i++) {
            double x = (topList.size() == 1) ? 0.5 : (double) i / (topList.size() - 1);
            pos.put(topList.get(i), new double[]{x, 1.0});
        }
        for (int i = 0; i < bottomList.size(); i++) {
            double x = (bottomList.size() == 1) ? 0.5 : (double) i / (bottomList.size() - 1);
            pos.put(bottomList.get(i), new double[]{x, 0.0});
        }
        return pos;
    }
}