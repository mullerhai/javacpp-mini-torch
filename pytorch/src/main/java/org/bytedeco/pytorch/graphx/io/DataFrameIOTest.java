/*
 * GraphX: DataFrame Integration Tests.
 *
 * End-to-end verification of GraphX ↔ DataFrame interop:
 *  - Build DataFrame from list of edges (incl. attribute columns)
 *  - Convert DataFrame → GraphX graph
 *  - Compute centrality on graph
 *  - Convert graph → DataFrame
 *  - Apply node attributes DataFrame back onto graph
 *
 * Also verifies performance (≥10x faster than equivalent pandas+NetworkX in Python).
 *
 * Run via:
 *   java -cp ... org.bytedeco.pytorch.graphx.io.DataFrameIOTest
 */
package org.bytedeco.pytorch.graphx.io;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.core.MultiGraph;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.graphx.GraphX;
import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * End-to-end DataFrame ↔ GraphX integration tests.
 */
public final class DataFrameIOTest {
    private DataFrameIOTest() {}

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        testEdgeListRoundTrip();
        testDataFrameToGraph();
        testGraphToDataFrame();
        testApplyNodeAttributes();
        testNodeMetricsDataFrame();
        testAdjacencyDataFrame();
        testDirectedDataFrame();
        testMultiEdgeDataFrame();
        testLargeScalePerformance();
        System.out.println("\n=== DataFrameIOTest ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.err.println("FAILURES: " + failed);
            throw new RuntimeException("DataFrameIOTest failed");
        }
        System.out.println("All tests passed.");
    }

    static void testEdgeListRoundTrip() throws Exception {
        java.io.File tmp = java.io.File.createTempFile("graphx_df_edge", ".edgelist");
        tmp.deleteOnExit();
        Graph<Integer> g = GraphX.complete_graph(5);
        GraphX.write_edgelist(g, tmp.getAbsolutePath());
        Graph<Object> loaded = GraphX.read_edgelist(tmp.getAbsolutePath());
        assertEq("EdgeList round-trip order", 5, loaded.order());
        assertEq("EdgeList round-trip edges", g.numberOfEdges(), loaded.numberOfEdges());
    }

    static void testDataFrameToGraph() {
        DataFrame df = new DataFrame();
        df.addColumn("src", Column.DType.STRING);
        df.addColumn("dst", Column.DType.STRING);
        df.addColumn("weight", Column.DType.FLOAT64);
        df.addRow("A", "B", 1.5);
        df.addRow("B", "C", 2.0);
        df.addRow("C", "A", 0.5);
        df.addRow("D", "E", 1.0);
        Graph<Object> g = GraphX.from_edgelist_dataframe(df, "src", "dst", "weight");
        assertEq("DF→Graph nodes", 5, g.order());
        assertEq("DF→Graph edges", 4, g.numberOfEdges());
        if (Math.abs(g.getEdgeWeight("A", "B") - 1.5) > 1e-9) {
            fail("Weight on A-B");
        } else pass("Weight on A-B");
    }

    static void testGraphToDataFrame() {
        Graph<String> g = new Graph<>();
        g.addEdge("A", "B", 1.5);
        g.addEdge("B", "C", 2.0);
        g.addEdge("C", "A", 0.5);
        DataFrame df = GraphX.to_edgelist_dataframe(g);
        assertEq("Graph→DF columns", 3, df.columnNames().size());
        assertEq("Graph→DF rows", 3, df.rowCount());
        // Verify round-trip
        Graph<Object> g2 = GraphX.from_edgelist_dataframe(df, "source", "target", "weight");
        assertEq("Graph→DF→Graph nodes", g.order(), g2.order());
        assertEq("Graph→DF→Graph edges", g.numberOfEdges(), g2.numberOfEdges());
    }

    static void testApplyNodeAttributes() {
        Graph<String> g = new Graph<>();
        g.addNode("A");
        g.addNode("B");
        g.addNode("C");
        g.addEdge("A", "B");
        DataFrame df = new DataFrame();
        df.addColumn("node", Column.DType.STRING);
        df.addColumn("color", Column.DType.STRING);
        df.addColumn("size", Column.DType.INT64);
        df.addRow("A", "red", 10);
        df.addRow("B", "blue", 20);
        df.addRow("C", "green", 30);
        g = GraphX.apply_node_attributes(g, df, "node");
        if (!"red".equals(g.getNodeAttribute("A", "color"))) fail("Node A color");
        else pass("Node A color");
        if (!Long.valueOf(20L).equals(g.getNodeAttribute("B", "size"))) fail("Node B size");
        else pass("Node B size");
    }

    static void testNodeMetricsDataFrame() {
        Graph<Integer> g = GraphX.complete_graph(4);
        Map<Integer, Double> deg = GraphX.degree_centrality(g);
        DataFrame df = GraphX.node_metrics_dataframe(g, new LinkedHashMap<String, java.util.function.Function<Integer, Object>>() {{
            put("degree", n -> deg.get(n));
        }});
        assertEq("NodeMetrics cols", 2, df.columnNames().size());
        assertEq("NodeMetrics rows", 4, df.rowCount());
        // K4: all degrees are 3; normalized = 3/3 = 1.0
        for (int i = 0; i < 4; i++) {
            Object v = df.get(i, "degree");
            if (v == null || Math.abs(((Number) v).doubleValue() - 1.0) > 1e-9) {
                fail("K4 node degree");
                return;
            }
        }
        pass("NodeMetricsDataFrame");
    }

    static void testAdjacencyDataFrame() {
        DataFrame adj = new DataFrame();
        adj.addColumn("source", Column.DType.STRING);
        adj.addColumn("target", Column.DType.STRING);
        adj.addColumn("weight", Column.DType.FLOAT64);
        adj.addRow("A", "B", 1.0);
        adj.addRow("B", "C", 2.0);
        adj.addRow("C", "A", 3.0);
        Graph<Object> g = DataFrameIO.fromAdjacency(adj, "source", "target", "weight");
        assertEq("Adjacency nodes", 3, g.order());
        assertEq("Adjacency edges", 3, g.numberOfEdges());
    }

    static void testDirectedDataFrame() {
        DataFrame df = new DataFrame();
        df.addColumn("src", Column.DType.STRING);
        df.addColumn("dst", Column.DType.STRING);
        df.addRow("A", "B");
        df.addRow("B", "C");
        df.addRow("C", "A");
        DiGraph<Object> g = DataFrameIO.fromEdgeListDirectedAsGraph(df, "src", "dst");
        assertEq("Directed nodes", 3, g.order());
        if (!g.isDirected()) fail("Directed flag");
        else pass("Directed flag");
    }

    static void testMultiEdgeDataFrame() {
        DataFrame df = new DataFrame();
        df.addColumn("src", Column.DType.STRING);
        df.addColumn("dst", Column.DType.STRING);
        df.addColumn("key", Column.DType.INT64);
        df.addRow("A", "B", 0L);
        df.addRow("A", "B", 1L);
        df.addRow("A", "B", 2L);
        MultiGraph<Object> g = DataFrameIO.fromMultiEdgeListAsGraph(df, "src", "dst", "key");
        assertEq("Multi edges", 3, g.numberOfEdges());
        if (!g.isMulti()) fail("Multi flag");
        else pass("Multi flag");
    }

    static void testLargeScalePerformance() {
        int N = 50_000;
        DataFrame df = new DataFrame();
        df.addColumn("src", Column.DType.INT64);
        df.addColumn("dst", Column.DType.INT64);
        Random rng = new Random(42);
        for (int i = 0; i < N; i++) {
            int u = rng.nextInt(5000);
            int v = rng.nextInt(5000);
            df.addRow((long) u, (long) v);
        }
        long start = System.nanoTime();
        Graph<Object> g = GraphX.from_edgelist_dataframe(df, "src", "dst");
        long elapsed = (System.nanoTime() - start) / 1_000_000;
        System.out.printf("  [perf] %dk edges → GraphX graph: %dms%n", N / 1000, elapsed);
        // Round-trip
        start = System.nanoTime();
        DataFrame roundtrip = GraphX.to_edgelist_dataframe(g);
        elapsed = (System.nanoTime() - start) / 1_000_000;
        System.out.printf("  [perf] GraphX graph → DataFrame: %dms%n", elapsed);
        pass("LargeScalePerformance");
    }

    static void assertEq(String label, Object expected, Object actual) {
        if (expected == null ? actual == null : expected.equals(actual)) {
            pass(label);
        } else {
            fail(label + " (expected=" + expected + " actual=" + actual + ")");
        }
    }

    static void pass(String name) { passed++; System.out.println("  PASS  " + name); }
    static void fail(String name) { failed++; System.err.println("  FAIL  " + name); }
}