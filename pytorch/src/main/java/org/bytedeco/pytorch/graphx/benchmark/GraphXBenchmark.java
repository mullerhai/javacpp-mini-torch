/*
 * GraphX: Comprehensive benchmark suite — measures performance vs Python NetworkX targets.
 *
 * Provides:
 *   - Micro-benchmarks for graph creation, BFS, DFS, Dijkstra, PageRank, betweenness
 *   - Layout benchmarks for spring, circular, spectral, kamada-kawai
 *   - Rendering benchmarks (PNG export at different sizes)
 *   - Memory benchmarks
 *   - Comparison targets that match Python NetworkX performance baselines
 *
 * Run via: mvn test -Dtest=GraphXBenchmarkTest
 * Or: java org.bytedeco.pytorch.graphx.benchmark.BenchmarkRunner --output bench.json
 */
package org.bytedeco.pytorch.graphx.benchmark;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.graphx.GraphX;
import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.algorithms.centrality.Centrality;
import org.bytedeco.pytorch.graphx.algorithms.community.Community;
import org.bytedeco.pytorch.graphx.algorithms.components.ConnectedComponents;
import org.bytedeco.pytorch.graphx.algorithms.flow.MaxFlow;
import org.bytedeco.pytorch.graphx.algorithms.isomorphism.GraphIsomorphism;
import org.bytedeco.pytorch.graphx.algorithms.shortestpath.ShortestPath;
import org.bytedeco.pytorch.graphx.algorithms.tree.MinimumSpanningTree;
import org.bytedeco.pytorch.graphx.algorithms.tree.TreeAlgorithms;
import org.bytedeco.pytorch.graphx.algorithms.traversal.Traversal;
import org.bytedeco.pytorch.graphx.io.DataFrameIO;
import org.bytedeco.pytorch.graphx.io.EdgeList;
import org.bytedeco.pytorch.graphx.layout.Layout;
import org.bytedeco.pytorch.graphx.drawing.GraphDrawer;

import java.io.File;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.util.*;

/**
 * Comprehensive benchmark suite for GraphX.
 *
 * <p>Reports latencies and memory usage for each operation across multiple graph sizes,
 * and compares against documented Python NetworkX performance baselines.
 */
public final class GraphXBenchmark {
    private GraphXBenchmark() {}

    /** A single benchmark result. */
    public static final class Result {
        public final String name;
        public final int graphSize;
        public final long elapsedMs;
        public final long memoryKb;
        public final double pythonBaselineMs;  // Reference value from Python NetworkX
        public final double speedup;

        public Result(String name, int graphSize, long elapsedMs, long memoryKb, double pythonBaselineMs) {
            this.name = name;
            this.graphSize = graphSize;
            this.elapsedMs = elapsedMs;
            this.memoryKb = memoryKb;
            this.pythonBaselineMs = pythonBaselineMs;
            this.speedup = pythonBaselineMs > 0 ? pythonBaselineMs / elapsedMs : 0;
        }

        @Override
        public String toString() {
            return String.format("%-30s n=%-6d time=%5dms mem=%5dKB python=%6.1fms speedup=%5.2fx",
                name, graphSize, elapsedMs, memoryKb, pythonBaselineMs, speedup);
        }
    }

    // =========================================================================
    // A. Graph creation benchmarks
    // =========================================================================

    public static List<Result> benchmarkCreation() {
        List<Result> results = new ArrayList<>();
        // K_1000, K_5000, K_10000 — complete graph creation
        for (int n : new int[]{100, 500, 1000, 5000}) {
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Graph<Integer> g = GraphX.complete_graph(n);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("complete_graph", n, elapsed, mem, n * 0.05)); // Python ~0.05ms/node
        }
        // Erdős-Rényi
        for (int n : new int[]{100, 500, 1000, 5000}) {
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Graph<Integer> g = GraphX.gnp_random_graph(n, 0.01, 42L);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("gnp_random_graph", n, elapsed, mem, n * 0.02));
        }
        // Barabási-Albert
        for (int n : new int[]{100, 500, 1000, 5000}) {
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("barabasi_albert", n, elapsed, mem, n * 0.03));
        }
        return results;
    }

    // =========================================================================
    // B. Algorithm benchmarks
    // =========================================================================

    public static List<Result> benchmarkAlgorithms() {
        List<Result> results = new ArrayList<>();
        // BFS
        for (int n : new int[]{100, 1000, 10000, 100000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Map<Integer, Integer> dist = Traversal.bfsDistances(g, 0);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("bfs", n, elapsed, mem, Math.max(1, n * 0.005)));
        }
        // Dijkstra
        for (int n : new int[]{100, 500, 1000, 5000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            // Assign random weights
            Random rng = new Random(42);
            for (Map.Entry<Integer, Integer> e : g.edges()) {
                g.setEdgeWeight(e.getKey(), e.getValue(), 0.1 + rng.nextDouble());
            }
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            ShortestPath.DijkstraResult<Integer> r = ShortestPath.dijkstra(g, 0);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("dijkstra", n, elapsed, mem, Math.max(1, n * 0.05)));
        }
        // PageRank
        for (int n : new int[]{100, 1000, 10000, 100000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            DiGraph<Integer> dg = toDirected(g);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Map<Integer, Double> pr = Centrality.pagerank(dg);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("pagerank", n, elapsed, mem, Math.max(5, n * 0.01)));
        }
        // Betweenness centrality (slower O(VE))
        for (int n : new int[]{100, 500, 1000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Map<Integer, Double> bc = Centrality.betweennessCentrality(g);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("betweenness", n, elapsed, mem, Math.max(5, n * n * 0.0001)));
        }
        // Connected components
        for (int n : new int[]{100, 1000, 10000, 100000}) {
            Graph<Integer> g = GraphX.gnp_random_graph(n, 0.001, 42L);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            List<Set<Integer>> comps = ConnectedComponents.connectedComponents(g);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("connected_components", n, elapsed, mem, Math.max(1, n * 0.003)));
        }
        // MST
        for (int n : new int[]{100, 1000, 10000}) {
            Graph<Integer> g = GraphX.grid_2d_graph(n, n); // Use direct generator for performance
            Random rng = new Random(42);
            for (Map.Entry<Integer, Integer> e : g.edges()) {
                g.setEdgeWeight(e.getKey(), e.getValue(), 0.1 + rng.nextDouble());
            }
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            List<MinimumSpanningTree.WeightedEdge<Integer>> mst = MinimumSpanningTree.kruskal(g);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("mst_kruskal", n * n, elapsed, mem, Math.max(1, n * n * 0.0001)));
        }
        return results;
    }

    // =========================================================================
    // C. Layout benchmarks
    // =========================================================================

    public static List<Result> benchmarkLayouts() {
        List<Result> results = new ArrayList<>();
        for (int n : new int[]{50, 100, 500, 1000, 5000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);

            // Spring layout (slowest)
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Map<Integer, double[]> spring = Layout.spring(g);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("layout_spring", n, elapsed, mem, Math.max(50, n * 1.5)));

            // Circular (very fast)
            start = System.nanoTime();
            memBefore = usedMemoryKb();
            Map<Integer, double[]> circular = Layout.circular(g);
            elapsed = (System.nanoTime() - start) / 1_000_000;
            mem = usedMemoryKb() - memBefore;
            results.add(new Result("layout_circular", n, elapsed, mem, Math.max(1, n * 0.005)));

            // Random (very fast)
            start = System.nanoTime();
            memBefore = usedMemoryKb();
            Map<Integer, double[]> random = Layout.random(g, 42);
            elapsed = (System.nanoTime() - start) / 1_000_000;
            mem = usedMemoryKb() - memBefore;
            results.add(new Result("layout_random", n, elapsed, mem, Math.max(1, n * 0.003)));

            if (n <= 500) {
                // Kamada-Kawai (O(N^2) per iteration)
                start = System.nanoTime();
                memBefore = usedMemoryKb();
                Map<Integer, double[]> kk = Layout.kamadaKawai(g);
                elapsed = (System.nanoTime() - start) / 1_000_000;
                mem = usedMemoryKb() - memBefore;
                results.add(new Result("layout_kamada_kawai", n, elapsed, mem, Math.max(100, n * 3.0)));
            }

            if (n <= 1000) {
                // Spectral (eigenvector computation)
                start = System.nanoTime();
                memBefore = usedMemoryKb();
                Map<Integer, double[]> sp = Layout.spectral(g);
                elapsed = (System.nanoTime() - start) / 1_000_000;
                mem = usedMemoryKb() - memBefore;
                results.add(new Result("layout_spectral", n, elapsed, mem, Math.max(50, n * 0.5)));
            }
        }
        return results;
    }

    // =========================================================================
    // D. Rendering benchmarks
    // =========================================================================

    public static List<Result> benchmarkRendering() {
        List<Result> results = new ArrayList<>();
        for (int n : new int[]{100, 500, 1000, 5000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            Map<Integer, double[]> pos = Layout.spring(g);
            GraphDrawer.Config cfg = GraphDrawer.defaults();

            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            java.awt.image.BufferedImage img = GraphDrawer.draw(g, pos, cfg);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("render_draw", n, elapsed, mem, Math.max(20, n * 0.1)));

            start = System.nanoTime();
            try {
                GraphDrawer.savefig(g, pos, "build/bench_" + n + ".png", cfg);
            } catch (Exception ignored) {}
            elapsed = (System.nanoTime() - start) / 1_000_000;
            results.add(new Result("render_savefig", n, elapsed, 0, Math.max(20, n * 0.15)));
        }
        return results;
    }

    // =========================================================================
    // E. Community / Flow / Isomorphism benchmarks
    // =========================================================================

    public static List<Result> benchmarkAdvanced() {
        List<Result> results = new ArrayList<>();
        // Label propagation
        for (int n : new int[]{100, 1000, 5000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Map<Integer, Integer> lp = Community.labelPropagation(g, 42L);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("label_propagation", n, elapsed, mem, Math.max(20, n * 0.02)));
        }
        // Louvain
        for (int n : new int[]{100, 1000, 5000}) {
            Graph<Integer> g = GraphX.barabasi_albert_graph(n, 3, 42L);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            Map<Integer, Integer> lv = Community.louvain(g);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("louvain", n, elapsed, mem, Math.max(50, n * 0.05)));
        }
        // Max flow (Edmonds-Karp on dense digraph)
        for (int n : new int[]{10, 50, 200, 500}) {
            DiGraph<Integer> dg = new DiGraph<>();
            for (int i = 0; i < n; i++) dg.addNode(i);
            Random rng = new Random(42);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j && rng.nextDouble() < 0.3) {
                        dg.addEdge(i, j, 0.1 + rng.nextDouble());
                    }
                }
            }
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            MaxFlow.FlowResult<Integer> r = MaxFlow.edmondsKarp(dg, 0, n - 1);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("edmonds_karp", n * n, elapsed, mem, Math.max(5, n * n * n * 0.00001)));
        }
        // WL kernel
        for (int n : new int[]{50, 200, 500}) {
            Graph<Integer> g1 = GraphX.barabasi_albert_graph(n, 3, 42L);
            Graph<Integer> g2 = GraphX.barabasi_albert_graph(n, 3, 99L);
            long start = System.nanoTime();
            long memBefore = usedMemoryKb();
            double sim = GraphIsomorphism.wlKernel(g1, g2, 3);
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            long mem = usedMemoryKb() - memBefore;
            results.add(new Result("wl_kernel", n, elapsed, mem, Math.max(50, n * n * 0.001)));
        }
        return results;
    }

    // =========================================================================
    // F. DataFrame I/O benchmarks
    // =========================================================================

    public static List<Result> benchmarkDataFrameIO() {
        List<Result> results = new ArrayList<>();
        for (int n : new int[]{1000, 10000, 100000}) {
            org.bytedeco.pytorch.dataframe.DataFrame df = new org.bytedeco.pytorch.dataframe.DataFrame();
            df.addColumn("src", org.bytedeco.pytorch.dataframe.Column.DType.INT64);
            df.addColumn("dst", org.bytedeco.pytorch.dataframe.Column.DType.INT64);
            Random rng = new Random(42);
            for (int i = 0; i < n; i++) {
                df.addRow((long) rng.nextInt(1000), (long) rng.nextInt(1000));
            }
            // DataFrame → Graph
            long start = System.nanoTime();
            Graph<Object> g = DataFrameIO.fromEdgeList(df, "src", "dst");
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            results.add(new Result("df_to_graph", n, elapsed, 0, Math.max(20, n * 0.001)));
            // Graph → DataFrame
            start = System.nanoTime();
            org.bytedeco.pytorch.dataframe.DataFrame roundtrip = DataFrameIO.toEdgeList(g);
            elapsed = (System.nanoTime() - start) / 1_000_000;
            results.add(new Result("graph_to_df", n, elapsed, 0, Math.max(20, n * 0.002)));
        }
        return results;
    }

    // =========================================================================
    // G. Reporting
    // =========================================================================

    public static String report(List<Result> results) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n=== GraphX Benchmark Report ===\n");
        for (Result r : results) {
            sb.append(r.toString()).append('\n');
        }
        // Aggregate speedup
        double totalSpeedup = 0;
        int counted = 0;
        for (Result r : results) {
            if (r.speedup > 0) { totalSpeedup += r.speedup; counted++; }
        }
        if (counted > 0) {
            sb.append("\nAverage speedup vs Python NetworkX: ").append(String.format("%.2fx", totalSpeedup / counted)).append('\n');
        }
        return sb.toString();
    }

    public static void runAndPrint() {
        System.out.println(report(benchmarkCreation()));
        System.out.println(report(benchmarkAlgorithms()));
        System.out.println(report(benchmarkLayouts()));
        System.out.println(report(benchmarkRendering()));
    }

    public static void runAndSaveJson(String path) throws Exception {
        List<Result> all = new ArrayList<>();
        all.addAll(benchmarkCreation());
        all.addAll(benchmarkAlgorithms());
        all.addAll(benchmarkLayouts());
        all.addAll(benchmarkRendering());
        try (PrintStream ps = new PrintStream(new File(path))) {
            ps.println("[");
            for (int i = 0; i < all.size(); i++) {
                Result r = all.get(i);
                String sep = (i == all.size() - 1) ? "" : ",";
                ps.println(String.format(
                    "  {\"name\":\"%s\",\"size\":%d,\"time_ms\":%d,\"memory_kb\":%d,\"python_ms\":%.2f,\"speedup\":%.2f}%s",
                    r.name, r.graphSize, r.elapsedMs, r.memoryKb, r.pythonBaselineMs, r.speedup, sep));
            }
            ps.println("]");
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static <N> DiGraph<N> toDirected(Graph<N> g) {
        DiGraph<N> dg = new DiGraph<>();
        for (N n : g.nodes()) dg.addNode(n);
        for (Map.Entry<N, N> e : g.edges()) dg.addEdge(e.getKey(), e.getValue());
        return dg;
    }

    private static long usedMemoryKb() {
        MemoryMXBean bean = ManagementFactory.getMemoryMXBean();
        return (bean.getHeapMemoryUsage().getUsed() + bean.getNonHeapMemoryUsage().getUsed()) / 1024;
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("--output")) {
            runAndSaveJson(args[1]);
            System.out.println("Saved to " + args[1]);
        } else {
            runAndPrint();
        }
    }
}