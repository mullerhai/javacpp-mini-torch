/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * Circular layout — places nodes uniformly around a circle.
 *
 * <p>Aligned with {@code networkx.drawing.layout.circular_layout}.
 */
public final class CircularLayout {
    private CircularLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, null);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, Double scale) {
        double s = scale == null ? 1.0 : scale;
        Map<N, double[]> pos = new LinkedHashMap<>();
        java.util.List<N> nodes = g.nodes();
        int n = nodes.size();
        if (n == 0) return pos;
        for (int i = 0; i < n; i++) {
            double theta = 2.0 * Math.PI * i / n;
            pos.put(nodes.get(i), new double[]{s * Math.cos(theta), s * Math.sin(theta)});
        }
        return pos;
    }
}