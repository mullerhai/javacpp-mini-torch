/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.algorithms.shortestpath.ShortestPath;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Kamada-Kawai layout — places nodes to minimize an energy function based on
 * graph-theoretic distances (faster variant than the full spring layout).
 *
 * <p>Aligned with {@code networkx.drawing.layout.kamada_kawai_layout}.
 */
public final class KamadaKawaiLayout {
    private KamadaKawaiLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, null, 1.0, 0.1, 50);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, Map<N, double[]> posInit, double scale, double k, int maxIter) {
        int n = g.order();
        if (n == 0) return new LinkedHashMap<>();
        Map<N, Integer> idx = new LinkedHashMap<>();
        List<N> nodes = g.nodes();
        for (int i = 0; i < n; i++) idx.put(nodes.get(i), i);

        // Compute all-pairs shortest paths
        Map<N, double[]> distMatrix = new LinkedHashMap<>();
        for (N u : nodes) {
            ShortestPath.DijkstraResult<N> r = ShortestPath.dijkstra(g, u);
            double[] row = new double[n];
            for (int j = 0; j < n; j++) {
                row[j] = r.dist.getOrDefault(nodes.get(j), Double.POSITIVE_INFINITY);
            }
            distMatrix.put(u, row);
        }

        // Initialize positions in a circle
        double[][] pos = new double[n][2];
        for (int i = 0; i < n; i++) {
            double theta = 2 * Math.PI * i / n;
            pos[i][0] = Math.cos(theta);
            pos[i][1] = Math.sin(theta);
        }
        if (posInit != null) {
            for (Map.Entry<N, double[]> e : posInit.entrySet()) {
                Integer ix = idx.get(e.getKey());
                if (ix != null) {
                    pos[ix][0] = e.getValue()[0];
                    pos[ix][1] = e.getValue()[1];
                }
            }
        }

        // Spring constants K[i][j] = 1 / d[i][j]^2
        double[][] springK = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double d = distMatrix.get(nodes.get(i))[j];
                if (d == 0 || !Double.isFinite(d)) springK[i][j] = 0;
                else springK[i][j] = k / (d * d);
            }
        }
        double[] idealLen = new double[n];
        for (int i = 0; i < n; i++) {
            double maxD = 0;
            for (int j = 0; j < n; j++) if (distMatrix.get(nodes.get(i))[j] > maxD) maxD = distMatrix.get(nodes.get(i))[j];
            idealLen[i] = (Double.isFinite(maxD) && maxD > 0) ? scale / maxD : 1.0;
        }

        // Iterative optimization (simplified)
        for (int iter = 0; iter < maxIter; iter++) {
            double totalEnergy = 0;
            for (int i = 0; i < n; i++) {
                double fx = 0, fy = 0;
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double d = distMatrix.get(nodes.get(i))[j];
                    if (!Double.isFinite(d) || d == 0) continue;
                    double dx = pos[i][0] - pos[j][0];
                    double dy = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1e-9) continue;
                    double diff = dist - d * idealLen[i];
                    fx += dx / dist * diff * springK[i][j];
                    fy += dy / dist * diff * springK[i][j];
                }
                pos[i][0] -= 0.01 * fx;
                pos[i][1] -= 0.01 * fy;
                totalEnergy += fx * fx + fy * fy;
            }
            if (totalEnergy < 1e-6) break;
        }

        Map<N, double[]> result = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) result.put(nodes.get(i), new double[]{pos[i][0], pos[i][1]});
        return result;
    }
}