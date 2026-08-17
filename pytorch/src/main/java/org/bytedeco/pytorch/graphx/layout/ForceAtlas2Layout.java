/*
 * GraphX: ForceAtlas2 Layout.
 *
 * Continuous force-directed layout from Gephi, designed for large networks.
 * Uses linear attraction + degree-dependent repulsion with simulated gravity.
 *
 * Inspired by networkx.drawing.layout.forceatlas2_layout and the original
 * Jacomy et al. (2014) paper. BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.layout;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * ForceAtlas2-style force-directed layout.
 */
public final class ForceAtlas2Layout {
    private ForceAtlas2Layout() {}

    /**
     * Compute ForceAtlas2 layout positions.
     *
     * @param g          graph to lay out
     * @param iterations number of force iterations (default 100)
     * @param seed       PRNG seed for initial positions
     */
    public static <N> Map<N, double[]> compute(Graph<N> g, int iterations, long seed) {
        return compute(g, iterations, 1.0, 1.0, 1.0, 1.0, seed);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, 100, 42L);
    }

    /**
     * @param g           graph
     * @param iterations  number of iterations
     * @param kGravity    gravity strength (attracts nodes to center)
     * @param kRepulsion  repulsion strength
     * @param kAttraction attraction strength along edges
     * @param scale       overall scale factor
     * @param seed        PRNG seed
     */
    public static <N> Map<N, double[]> compute(Graph<N> g, int iterations,
                                                double kGravity, double kRepulsion,
                                                double kAttraction, double scale, long seed) {
        List<N> nodes = g.nodes();
        int n = nodes.size();
        if (n == 0) return new LinkedHashMap<>();
        double[][] pos = new double[n][2];
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            pos[i][0] = rng.nextDouble() - 0.5;
            pos[i][1] = rng.nextDouble() - 0.5;
        }
        // Degrees for repulsion weight
        int[] deg = new int[n];
        for (int i = 0; i < n; i++) deg[i] = g.degree(nodes.get(i));

        double[] swinging = new double[n];
        double[] traction = new double[n];
        double globalSw = 0;

        for (int iter = 0; iter < iterations; iter++) {
            double[] fx = new double[n];
            double[] fy = new double[n];
            // Repulsion: O(n^2) Barnes-Hut approximation is omitted; full pair
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double dx = pos[i][0] - pos[j][0];
                    double dy = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1e-6) continue;
                    double force = kRepulsion * (deg[i] + 1) * (deg[j] + 1) / dist;
                    fx[i] += dx / dist * force;
                    fy[i] += dy / dist * force;
                }
            }
            // Attraction (along edges)
            java.util.Set<Long> seen = new java.util.HashSet<>();
            for (int i = 0; i < n; i++) {
                N u = nodes.get(i);
                for (N v : g.neighbors(u)) {
                    int j = nodes.indexOf(v);
                    int a = Math.min(i, j);
                    int b = Math.max(i, j);
                    long key = ((long) a << 32) | b;
                    if (!seen.add(key)) continue;
                    double dx = pos[i][0] - pos[j][0];
                    double dy = pos[i][1] - pos[j][1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1e-6) continue;
                    double force = kAttraction * dist;
                    fx[i] -= dx / dist * force;
                    fy[i] -= dy / dist * force;
                    fx[j] += dx / dist * force;
                    fy[j] += dy / dist * force;
                }
            }
            // Gravity to center
            for (int i = 0; i < n; i++) {
                fx[i] -= kGravity * pos[i][0];
                fy[i] -= kGravity * pos[i][1];
            }
            // Apply forces with adaptive cooling (swinging/traction)
            double totalSwing = 0;
            double totalTraction = 0;
            for (int i = 0; i < n; i++) {
                double fMag = Math.sqrt(fx[i] * fx[i] + fy[i] * fy[i]);
                double prevSw = swinging[i];
                swinging[i] = 0.8 * prevSw + 0.2 * fMag;
                traction[i] = 0.8 * traction[i] + 0.2 * fMag;
                totalSwing += swinging[i];
                totalTraction += traction[i];
                if (fMag > 0) {
                    double factor = Math.min(1.0, scale * traction[i] / fMag);
                    pos[i][0] += fx[i] * factor;
                    pos[i][1] += fy[i] * factor;
                }
            }
            if (totalTraction > 0) {
                globalSw = 0.8 * globalSw + 0.2 * totalSwing / totalTraction;
            }
            if (globalSw < 0.01) break;
        }

        Map<N, double[]> result = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) result.put(nodes.get(i), new double[]{pos[i][0], pos[i][1]});
        return result;
    }
}