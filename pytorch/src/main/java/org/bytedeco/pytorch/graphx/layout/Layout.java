/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.Map;
import java.util.Set;

/**
 * Unified layout façade — convenience methods matching
 * {@code networkx.drawing.layout.*_layout} naming.
 */
public final class Layout {
    private Layout() {}

    public enum Kind {
        SPRING, CIRCULAR, RANDOM, SHELL, KAMADA_KAWAI, SPECTRAL,
        BIPARTITE, PLANAR, BFS, FORCE_ATLAS2, ARF
    }

    public static <N> Map<N, double[]> spring(Graph<N> g) {
        return SpringLayout.compute(g);
    }

    public static <N> Map<N, double[]> spring(Graph<N> g, Map<N, double[]> posInit, double k, int iterations) {
        return SpringLayout.compute(g, posInit, k, iterations, 1e-4, false, 42L);
    }

    public static <N> Map<N, double[]> forceAtlas2(Graph<N> g) {
        return ForceAtlas2Layout.compute(g);
    }

    public static <N> Map<N, double[]> arf(Graph<N> g) {
        return ARFLayout.compute(g);
    }

    public static <N> Map<N, double[]> circular(Graph<N> g) {
        return CircularLayout.compute(g);
    }

    public static <N> Map<N, double[]> random(Graph<N> g) {
        return RandomLayout.compute(g);
    }

    public static <N> Map<N, double[]> random(Graph<N> g, long seed) {
        return RandomLayout.compute(g, seed);
    }

    public static <N> Map<N, double[]> shell(Graph<N> g, java.util.List<java.util.List<N>> shells) {
        return ShellLayout.compute(g, shells);
    }

    public static <N> Map<N, double[]> kamadaKawai(Graph<N> g) {
        return KamadaKawaiLayout.compute(g);
    }

    public static <N> Map<N, double[]> spectral(Graph<N> g) {
        return SpectralLayout.compute(g);
    }

    public static <N> Map<N, double[]> bipartite(Graph<N> g, Set<N> topNodes) {
        return BipartiteLayout.compute(g, topNodes);
    }

    public static <N> Map<N, double[]> planar(Graph<N> g) {
        return PlanarLayout.compute(g);
    }

    public static <N> Map<N, double[]> bfs(Graph<N> g, N source) {
        return BFSLayout.compute(g, source);
    }

    public static <N> Map<N, double[]> compute(Kind kind, Graph<N> g, Object... params) {
        switch (kind) {
            case SPRING: return spring(g);
            case CIRCULAR: return circular(g);
            case RANDOM: return random(g);
            case SHELL: return shell(g, (java.util.List<java.util.List<N>>) params[0]);
            case KAMADA_KAWAI: return kamadaKawai(g);
            case SPECTRAL: return spectral(g);
            case BIPARTITE: return bipartite(g, (Set<N>) params[0]);
            case PLANAR: return planar(g);
            case BFS: return bfs(g, (N) params[0]);
            case FORCE_ATLAS2: return forceAtlas2(g);
            case ARF: return arf(g);
            default: throw new IllegalArgumentException("Unknown layout: " + kind);
        }
    }

    /** NetworkX-compatible rescale to [-1, 1] range. */
    public static <N> Map<N, double[]> rescale(Map<N, double[]> pos) {
        if (pos.isEmpty()) return pos;
        double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
        for (double[] p : pos.values()) {
            if (p[0] < minX) minX = p[0];
            if (p[0] > maxX) maxX = p[0];
            if (p[1] < minY) minY = p[1];
            if (p[1] > maxY) maxY = p[1];
        }
        double m = Math.max(Math.max(maxX - minX, 1e-9), Math.max(maxY - minY, 1e-9));
        double cx = (minX + maxX) / 2;
        double cy = (minY + maxY) / 2;
        Map<N, double[]> result = new java.util.LinkedHashMap<>();
        for (Map.Entry<N, double[]> e : pos.entrySet()) {
            result.put(e.getKey(), new double[]{(e.getValue()[0] - cx) / m, (e.getValue()[1] - cy) / m});
        }
        return result;
    }
}