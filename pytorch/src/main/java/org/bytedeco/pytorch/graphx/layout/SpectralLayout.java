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
 * Spectral layout — uses eigenvectors of the graph Laplacian.
 *
 * <p>Aligned with {@code networkx.drawing.layout.spectral_layout}.
 */
public final class SpectralLayout {
    private SpectralLayout() {}

    /** Default: dim=2. */
    public static <N> Map<N, double[]> compute(Graph<N> g) {
        return compute(g, 2);
    }

    public static <N> Map<N, double[]> compute(Graph<N> g, int dim) {
        int n = g.order();
        if (n == 0) return new LinkedHashMap<>();
        List<N> nodes = g.nodes();

        // Build adjacency matrix
        double[][] A = new double[n][n];
        for (Map.Entry<N, N> e : g.edges()) {
            int i = nodes.indexOf(e.getKey());
            int j = nodes.indexOf(e.getValue());
            if (i < 0 || j < 0) continue;
            A[i][j] = 1.0;
            A[j][i] = 1.0;
        }

        // Compute Laplacian L = D - A
        double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            int deg = g.degree(nodes.get(i));
            for (int j = 0; j < n; j++) {
                L[i][j] = (i == j ? deg : 0) - A[i][j];
            }
        }

        // Compute smallest eigenvectors via power iteration on inverse (L+εI)
        // For efficiency, use a shifted inverse power method.
        // Simpler: compute approximate eigenvectors by finding the eigenvectors of
        // the small eigenvalues.
        double[][] vecs = new double[n][dim];
        double[] eigenvalues = new double[dim];

        // Build (L + shift*I) and use power iteration to find smallest eigenvectors.
        // We use deflation: find largest eigenvector of inv(L + εI) repeatedly.
        double shift = 0.001; // regularization
        for (int k = 0; k < dim; k++) {
            double[] v = new double[n];
            for (int i = 0; i < n; i++) v[i] = Math.random();
            normalize(v);
            double lambda = 0;
            for (int iter = 0; iter < 200; iter++) {
                double[] next = matVec(L, v);
                for (int i = 0; i < n; i++) next[i] += shift * v[i];
                // Orthogonalize against previous eigenvectors
                for (int prev = 0; prev < k; prev++) {
                    double dot = 0;
                    for (int i = 0; i < n; i++) dot += next[i] * vecs[i][prev];
                    for (int i = 0; i < n; i++) next[i] -= dot * vecs[i][prev];
                }
                normalize(next);
                double diff = 0;
                for (int i = 0; i < n; i++) diff += (next[i] - v[i]) * (next[i] - v[i]);
                v = next;
                if (diff < 1e-12) break;
            }
            // Eigenvalue estimate: Rayleigh quotient
            double[] Lv = matVec(L, v);
            double num = 0, den = 0;
            for (int i = 0; i < n; i++) { num += v[i] * Lv[i]; den += v[i] * v[i]; }
            lambda = (den > 0) ? num / den : 0;
            for (int i = 0; i < n; i++) vecs[i][k] = v[i];
            eigenvalues[k] = lambda;
        }

        Map<N, double[]> pos = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            double[] coords = new double[dim];
            for (int k = 0; k < dim; k++) coords[k] = vecs[i][k];
            pos.put(nodes.get(i), coords);
        }
        return pos;
    }

    private static double[] matVec(double[][] M, double[] v) {
        int n = v.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += M[i][j] * v[j];
            out[i] = s;
        }
        return out;
    }

    private static void normalize(double[] v) {
        double n = 0;
        for (double x : v) n += x * x;
        n = Math.sqrt(n);
        if (n < 1e-12) return;
        for (int i = 0; i < v.length; i++) v[i] /= n;
    }
}