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
import java.util.Random;

/**
 * Random layout — uniformly random positions in a unit square.
 *
 * <p>Aligned with {@code networkx.drawing.layout.random_layout}.
 */
public final class RandomLayout {
    private RandomLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, 42L);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, long seed) {
        List<N> nodes = g.nodes();
        Map<N, double[]> pos = new LinkedHashMap<>();
        Random rng = new Random(seed);
        for (N n : nodes) pos.put(n, new double[]{rng.nextDouble(), rng.nextDouble()});
        return pos;
    }
}