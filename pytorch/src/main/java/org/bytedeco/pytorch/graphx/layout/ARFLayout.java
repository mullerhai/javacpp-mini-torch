/*
 * GraphX: ARF (Attractive-Repulsive Forces) Layout.
 *
 * Inspired by networkx.drawing.layout.arf_layout. A simple force-directed
 * layout using attractive forces on edges and repulsive forces between all
 * node pairs. BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * ARF layout — attractive + repulsive force-directed.
 */
public final class ARFLayout {
    private ARFLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, 1000, 0.1, 0.5, 42L);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, int iterations) {
        return compute(g, iterations, 0.1, 0.5, 42L);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, int iterations,
                                                  double a, double r, long seed) {
        List<N> nodes = g.nodes();
        int n = nodes.size();
        if (n == 0) return new LinkedHashMap<>();
        double[][] pos = new double[n][2];
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            pos[i][0] = rng.nextDouble() - 0.5;
            pos[i][1] = rng.nextDouble() - 0.5;
        }
        for (int iter = 0; iter < iterations; iter++) {
            double[] fx = new double[n];
            double[] fy = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double dx = pos[i][0] - pos[j][0];
                    double dy = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1e-6) continue;
                    fx[i] += -dx * r / (dist * dist * dist);
                    fy[i] += -dy * r / (dist * dist * dist);
                }
            }
            java.util.Set<Long> seen = new java.util.HashSet<>();
            for (int i = 0; i < n; i++) {
                N u = nodes.get(i);
                for (N v : g.neighbors(u)) {
                    int j = nodes.indexOf(v);
                    int a0 = Math.min(i, j);
                    int b = Math.max(i, j);
                    long key = ((long) a0 << 32) | b;
                    if (!seen.add(key)) continue;
                    double dx = pos[i][0] - pos[j][0];
                    double dy = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    fx[i] += dx * a * dist;
                    fy[i] += dy * a * dist;
                    fx[j] -= dx * a * dist;
                    fy[j] -= dy * a * dist;
                }
            }
            double t = 1.0 / (1 + iter / 100.0);
            for (int i = 0; i < n; i++) {
                pos[i][0] += fx[i] * t;
                pos[i][1] += fy[i] * t;
            }
        }
        Map<N, double[]> result = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) result.put(nodes.get(i), new double[]{pos[i][0], pos[i][1]});
        return result;
    }
}