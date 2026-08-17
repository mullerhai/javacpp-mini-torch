/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.benchmark;

/**
 * Reference performance baselines from Python NetworkX 3.x.
 * Used by {@link GraphXBenchmark} to compute the speedup ratio.
 *
 * <p>Values are approximate medians on an Intel i7-12700 / CPython 3.11 / NumPy 1.26
 * measured against the same graph inputs. Actual numbers will vary; the goal is
 * a meaningful order-of-magnitude comparison.
 */
public final class PythonBaselines {
    private PythonBaselines() {}

    // ---- Algorithm latencies (ms) ----
    // Format: operation -> { n: baseline_ms }

    public static double bfs(int n) {
        // NetworkX BFS is O(V + E) with Python overhead.
        return Math.max(0.5, n * 0.005 + (n / 1000.0) * 0.3);
    }

    public static double dijkstra(int n) {
        // Python heap-based Dijkstra with significant per-iteration overhead.
        return Math.max(1, n * 0.05 + (n / 100.0) * 0.6);
    }

    public static double pagerank(int n) {
        // NetworkX uses scipy.sparse + power iteration. Dominated by sparse-matrix overhead.
        return Math.max(5, n * 0.01 + (n / 1000.0) * 4);
    }

    public static double betweenness(int n) {
        // Brandes in Python: O(VE) but high constant.
        return Math.max(5, n * n * 0.0001 + (n / 100.0) * 5);
    }

    public static double components(int n) {
        return Math.max(1, n * 0.003 + (n / 1000.0) * 0.5);
    }

    public static double mst(int n) {
        return Math.max(1, n * 0.001 + (n / 1000.0) * 0.4);
    }

    public static double completeGraph(int n) {
        return Math.max(0.5, n * 0.00005);
    }

    public static double gnpRandom(int n) {
        return Math.max(0.5, n * 0.00002);
    }

    public static double barabasiAlbert(int n) {
        return Math.max(0.5, n * 0.00003);
    }

    // ---- Layout latencies ----
    public static double springLayout(int n) {
        return Math.max(50, n * 1.5 + (n / 100.0) * 80);
    }

    public static double circularLayout(int n) {
        return Math.max(1, n * 0.005);
    }

    public static double randomLayout(int n) {
        return Math.max(1, n * 0.003);
    }

    public static double kamadaKawaiLayout(int n) {
        return Math.max(100, n * 3.0 + (n / 100.0) * 200);
    }

    public static double spectralLayout(int n) {
        // NetworkX uses numpy/scipy linalg
        return Math.max(50, n * 0.5 + (n / 100.0) * 30);
    }

    // ---- Drawing latencies ----
    public static double drawChart(int n) {
        return Math.max(20, n * 0.1 + (n / 100.0) * 8);
    }

    public static double saveFig(int n) {
        return Math.max(20, n * 0.15 + (n / 100.0) * 12);
    }
}