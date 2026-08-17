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
 * Spring / Fruchterman-Reingold layout.
 *
 * <p>Aligned with {@code networkx.drawing.layout.spring_layout}.
 */
public final class SpringLayout {
    private SpringLayout() {}

    /** NetworkX default parameters: k=null (auto), iterations=50, threshold=1e-4. */
    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, null, null, 50, 1e-4, false, 42L);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, Map<N, double[]> posInit,
                                                 Double k, Integer iterations, Double threshold,
                                                 boolean scale, long seed) {
        List<N> nodes = g.nodes();
        int n = nodes.size();
        if (n == 0) return new LinkedHashMap<>();
        int iter = iterations == null ? 50 : iterations;
        double tol = threshold == null ? 1e-4 : threshold;
        Random rng = new Random(seed);

        double kOptimal = k != null ? k : 1.0 / Math.sqrt(n);

        double[][] pos = new double[n][2];
        if (posInit != null) {
            for (int i = 0; i < n; i++) {
                double[] p = posInit.get(nodes.get(i));
                if (p != null) { pos[i][0] = p[0]; pos[i][1] = p[1]; }
            }
        } else {
            for (int i = 0; i < n; i++) {
                pos[i][0] = rng.nextDouble();
                pos[i][1] = rng.nextDouble();
            }
        }

        Map<N, Integer> idx = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) idx.put(nodes.get(i), i);

        double t = 1.0;
        for (int step = 0; step < iter; step++) {
            double[] dx = new double[n];
            double[] dy = new double[n];

            // Repulsive forces (O(n^2))
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double xDelta = pos[i][0] - pos[j][0];
                    double yDelta = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(xDelta * xDelta + yDelta * yDelta);
                    if (dist < 1e-9) continue;
                    double rep = (kOptimal * kOptimal) / dist;
                    dx[i] += (xDelta / dist) * rep;
                    dy[i] += (yDelta / dist) * rep;
                }
            }

            // Attractive forces along edges (each edge once via dedup)
            java.util.Set<Long> seen = new java.util.HashSet<>();
            for (int i = 0; i < n; i++) {
                N u = nodes.get(i);
                for (N v : g.neighbors(u)) {
                    Integer j = idx.get(v);
                    if (j == null) continue;
                    int a = Math.min(i, j);
                    int b = Math.max(i, j);
                    long key = ((long) a << 32) | b;
                    if (!seen.add(key)) continue;
                    double xDelta = pos[i][0] - pos[j][0];
                    double yDelta = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(xDelta * xDelta + yDelta * yDelta);
                    if (dist < 1e-9) continue;
                    double attr = (dist * dist) / kOptimal;
                    double f = attr / dist;
                    dx[i] -= xDelta * f;
                    dy[i] -= yDelta * f;
                    dx[j] += xDelta * f;
                    dy[j] += yDelta * f;
                }
            }

            double maxDelta = 0;
            for (int i = 0; i < n; i++) {
                double dist = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
                if (dist < 1e-9) continue;
                double limit = Math.min(dist, t);
                pos[i][0] += (dx[i] / dist) * limit;
                pos[i][1] += (dy[i] / dist) * limit;
                if (limit > maxDelta) maxDelta = limit;
            }
            t = Math.max(t * 0.95, tol);
            if (maxDelta < tol) break;
        }

        if (scale) {
            double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
            double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (pos[i][0] < minX) minX = pos[i][0];
                if (pos[i][0] > maxX) maxX = pos[i][0];
                if (pos[i][1] < minY) minY = pos[i][1];
                if (pos[i][1] > maxY) maxY = pos[i][1];
            }
            double m = Math.max(Math.max(maxX - minX, 1e-9), Math.max(maxY - minY, 1e-9));
            double cx = (minX + maxX) / 2;
            double cy = (minY + maxY) / 2;
            for (int i = 0; i < n; i++) {
                pos[i][0] = (pos[i][0] - cx) / m;
                pos[i][1] = (pos[i][1] - cy) / m;
            }
        }

        Map<N, double[]> result = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) result.put(nodes.get(i), new double[]{pos[i][0], pos[i][1]});
        return result;
    }
}