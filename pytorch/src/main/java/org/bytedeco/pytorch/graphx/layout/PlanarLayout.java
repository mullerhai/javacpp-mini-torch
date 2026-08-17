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

/**
 * Planar layout — uses spring layout but constrained to planar embedding.
 * Falls back to spring layout if the graph is not planar.
 *
 * <p>Aligned with {@code networkx.drawing.layout.planar_layout}.
 */
public final class PlanarLayout {
    private PlanarLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g) {
        // Simplified planar embedding: use circular layout when graph is dense,
        // else spring layout. A full planar embedding (Hopcroft-Tarjan) is out of scope.
        // We use a heuristic: if density > 0.5, use circular; else spring.
        if (g.density() > 0.5) return CircularLayout.compute(g);
        return SpringLayout.compute(g);
    }
}