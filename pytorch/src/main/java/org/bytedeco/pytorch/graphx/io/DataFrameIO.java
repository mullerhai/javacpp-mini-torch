/*
 * GraphX ↔ DataFrame Conversion.
 *
 * Bridging layer between the DataFrame module and GraphX graph data structures.
 * Three conversion modes:
 *
 *  1. **Edge-list DataFrame** → GraphX
 *     Two-column DataFrame (source, target, [weight, attrs...]) → GraphX.Graph
 *
 *  2. **GraphX.Graph → Edge-list DataFrame**
 *     Each (u, v, attrs) edge becomes one DataFrame row, with attributes
 *     flattened into additional columns. Special handling for nested maps
 *     via JSON serialization.
 *
 *  3. **Adjacency-list DataFrame** → GraphX
 *     Adjacency DataFrame is a sparse two-column structure where each row
 *     records one (node, neighbor) edge. Used to load adjacency tensors
 *     (e.g., from NumPy, sparse matrices) into a graph.
 *
 *  4. **Centrality DataFrame** → graph annotation
 *     Apply a per-node centrality DataFrame as node attributes on a graph.
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * interaction with pandas DataFrames.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.io;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.core.MultiGraph;
import org.bytedeco.pytorch.graphx.core.AttrMap;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Convert between {@link DataFrame} and GraphX graph structures.
 *
 * <p>Common use cases:
 * <pre>{@code
 * // Load edges from CSV/Parquet → Graph
 * DataFrame edges = DataFrame.readCsv("edges.csv");
 * Graph<String> g = DataFrameIO.fromEdgeList(edges, "src", "dst");
 *
 * // Graph → DataFrame for SQL/Pandas-style queries
 * DataFrame df = DataFrameIO.toEdgeList(g);
 *
 * // Apply a per-node centrality DataFrame as graph attributes
 * Graph<String> g2 = DataFrameIO.applyNodeAttributes(graph, centralityDf, "node");
 * }</pre>
 */
public final class DataFrameIO {
    private DataFrameIO() {}

    // =========================================================================
    // Edge-list DataFrame ↔ Graph
    // =========================================================================

    /**
     * Build a {@link Graph} from a DataFrame whose rows are (source, target)
     * pairs, optionally with edge attributes in extra columns.
     *
     * @param edges   edge DataFrame
     * @param srcCol  source-node column name
     * @param dstCol  target-node column name
     * @param attrCols optional additional attribute column names
     */
    public static Graph<Object> fromEdgeList(DataFrame edges, String srcCol, String dstCol, String... attrCols) {
        Graph<Object> g = new Graph<>();
        int n = edges.rowCount();
        for (int i = 0; i < n; i++) {
            Object u = edges.get(i, srcCol);
            Object v = edges.get(i, dstCol);
            if (attrCols.length == 0) {
                g.addEdge(u, v);
            } else {
                Map<String, Object> attrs = new LinkedHashMap<>();
                for (String a : attrCols) {
                    attrs.put(a, edges.get(i, a));
                }
                g.addEdge(u, v, attrs);
            }
        }
        return g;
    }

    /**
     * Same as {@link #fromEdgeList} but returns a {@link DiGraph}.
     */
    public static DiGraph<Object> fromEdgeListDirected(DataFrame edges, String srcCol, String dstCol, String... attrCols) {
        DiGraph<Object> g = new DiGraph<>();
        int n = edges.rowCount();
        for (int i = 0; i < n; i++) {
            Object u = edges.get(i, srcCol);
            Object v = edges.get(i, dstCol);
            if (attrCols.length == 0) {
                g.addEdge(u, v);
            } else {
                Map<String, Object> attrs = new LinkedHashMap<>();
                for (String a : attrCols) {
                    attrs.put(a, edges.get(i, a));
                }
                g.addEdge(u, v, attrs);
            }
        }
        return g;
    }

    /** Variant returning a generic Graph reference (DiGraph is a Graph). */
    public static DiGraph<Object> fromEdgeListDirectedAsGraph(DataFrame edges, String srcCol, String dstCol) {
        return fromEdgeListDirected(edges, srcCol, dstCol);
    }

    /**
     * Build a {@link MultiGraph} from a DataFrame of edges with integer 'key' column.
     */
    public static MultiGraph<Object> fromMultiEdgeList(DataFrame edges, String srcCol, String dstCol,
                                                        String keyCol, String... attrCols) {
        MultiGraph<Object> g = new MultiGraph<>();
        int n = edges.rowCount();
        for (int i = 0; i < n; i++) {
            Object u = edges.get(i, srcCol);
            Object v = edges.get(i, dstCol);
            Object kObj = edges.get(i, keyCol);
            int key = (kObj instanceof Number) ? ((Number) kObj).intValue() : 0;
            if (attrCols.length == 0) {
                g.addEdge(u, v, key);
            } else {
                Map<String, Object> attrs = new LinkedHashMap<>();
                for (String a : attrCols) attrs.put(a, edges.get(i, a));
                g.addEdge(u, v, AttrMap.of(attrs), key);
            }
        }
        return g;
    }

    /** Variant returning a generic Graph reference. */
    public static MultiGraph<Object> fromMultiEdgeListAsGraph(DataFrame edges, String srcCol, String dstCol, String keyCol) {
        return fromMultiEdgeList(edges, srcCol, dstCol, keyCol);
    }

    /**
     * Convert an undirected graph to an edge-list DataFrame.
     *
     * <p>The resulting DataFrame has columns {@code source, target} plus one column
     * per edge attribute name (deduped across all edges).
     */
    public static <N> DataFrame toEdgeList(Graph<N> g) {
        return toEdgeListGeneric(g, false);
    }

    public static <N> DataFrame toEdgeList(DiGraph<N> g) {
        return toEdgeListGeneric(g, true);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <N> DataFrame toEdgeListGeneric(Object graph, boolean directed) {
        // First pass: collect attribute keys
        java.util.Set<String> attrKeys = new java.util.LinkedHashSet<>();
        java.util.List<java.util.Map.Entry<Object, Object>> edges = new ArrayList<>();
        if (directed) {
            DiGraph dg = (DiGraph) graph;
            for (Object e : dg.edges()) {
                java.util.Map.Entry<Object, Object> entry = (java.util.Map.Entry<Object, Object>) e;
                edges.add(entry);
                AttrMap am = dg.getEdgeAttr(entry.getKey(), entry.getValue());
                for (java.util.Map.Entry<String, Object> a : am.asMap().entrySet()) {
                    attrKeys.add(a.getKey());
                }
            }
        } else {
            Graph g = (Graph) graph;
            for (Object e : g.edges()) {
                java.util.Map.Entry<Object, Object> entry = (java.util.Map.Entry<Object, Object>) e;
                edges.add(entry);
                AttrMap am = g.getEdgeAttr(entry.getKey(), entry.getValue());
                for (java.util.Map.Entry<String, Object> a : am.asMap().entrySet()) {
                    attrKeys.add(a.getKey());
                }
            }
        }
        DataFrame df = new DataFrame();
        df.addColumn("source", Column.DType.STRING);
        df.addColumn("target", Column.DType.STRING);
        for (String key : attrKeys) {
            df.addColumn(key, inferType(key, edges, graph, directed));
        }
        // Fill rows
        for (java.util.Map.Entry<Object, Object> e : edges) {
            df.addRow(formatNode(e.getKey()), formatNode(e.getValue()));
            int row = df.rowCount() - 1;
            Object u = e.getKey(), v = e.getValue();
            AttrMap attr = directed
                ? ((DiGraph) graph).getEdgeAttr(u, v)
                : ((Graph) graph).getEdgeAttr(u, v);
            for (String key : attrKeys) {
                Object val = attr.get(key);
                int colIdx = df.columnIndex(key);
                if (val == null) val = defaultForType(df.column(colIdx).dtype());
                df.set(row, colIdx, val);
            }
        }
        return df;
    }

    // =========================================================================
    // Adjacency DataFrame ↔ Graph
    // =========================================================================

    /**
     * Build a graph from an adjacency DataFrame where each row is
     * (node, neighbor, [weight]). Used for converting adjacency tensors/matrices
     * to graphs.
     *
     * @param adj       adjacency DataFrame
     * @param nodeCol   column holding the source node
     * @param nbrCol    column holding the neighbor node
     * @param weightCol optional weight column (if null, weight defaults to 1.0)
     */
    public static Graph<Object> fromAdjacency(DataFrame adj, String nodeCol, String nbrCol, String weightCol) {
        Graph<Object> g = new Graph<>();
        int n = adj.rowCount();
        for (int i = 0; i < n; i++) {
            Object u = adj.get(i, nodeCol);
            Object v = adj.get(i, nbrCol);
            if (u.equals(v)) continue; // Skip self-loops by default
            if (weightCol != null) {
                Object w = adj.get(i, weightCol);
                if (w instanceof Number) {
                    g.addEdge(u, v, ((Number) w).doubleValue());
                } else {
                    g.addEdge(u, v);
                }
            } else {
                g.addEdge(u, v);
            }
        }
        return g;
    }

    /**
     * Convert an undirected graph to an adjacency-list DataFrame.
     *
     * <p>Each row records one (source, neighbor, weight) tuple. For unweighted graphs,
     * the {@code weight} column is filled with 1.0.
     */
    public static <N> DataFrame toAdjacency(Graph<N> g) {
        DataFrame df = new DataFrame();
        df.addColumn("source", Column.DType.STRING);
        df.addColumn("target", Column.DType.STRING);
        df.addColumn("weight", Column.DType.FLOAT64);
        for (java.util.Map.Entry<N, N> e : g.edges()) {
            df.addRow(formatNode(e.getKey()), formatNode(e.getValue()), g.getEdgeWeight(e.getKey(), e.getValue()));
        }
        return df;
    }

    // =========================================================================
    // Node-attribute DataFrame integration
    // =========================================================================

    /**
     * Apply a per-node attribute DataFrame as node attributes on a graph.
     *
     * <p>The DataFrame must contain a column whose name matches {@code nodeCol}
     * identifying the node. Other columns become node attributes.
     *
     * @return the same graph, mutated.
     */
    public static <N> Graph<N> applyNodeAttributes(Graph<N> g, DataFrame attrs, String nodeCol) {
        int n = attrs.rowCount();
        for (int i = 0; i < n; i++) {
            Object nodeObj = attrs.get(i, nodeCol);
            @SuppressWarnings("unchecked")
            N node = (N) nodeObj;
            if (!g.hasNode(node)) g.addNode(node);
            for (String colName : attrs.columnNames()) {
                if (colName.equals(nodeCol)) continue;
                Object val = attrs.get(i, colName);
                g.setNodeAttribute(node, colName, val);
            }
        }
        return g;
    }

    /**
     * Compute a per-node DataFrame (e.g., degree, betweenness) from a graph.
     *
     * <p>The resulting DataFrame has columns {@code node} and one column per
     * supplied metric.
     */
    public static <N> DataFrame nodeMetricsToDataFrame(Graph<N> g,
                                                        java.util.Map<String, java.util.function.Function<N, Object>> metrics) {
        DataFrame df = new DataFrame();
        df.addColumn("node", Column.DType.STRING);
        for (String name : metrics.keySet()) df.addColumn(name, Column.DType.FLOAT64);
        for (N n : g.nodes()) {
            df.addRow(formatNode(n));
            int row = df.rowCount() - 1;
            for (Map.Entry<String, java.util.function.Function<N, Object>> m : metrics.entrySet()) {
                int colIdx = df.columnIndex(m.getKey());
                Object val = m.getValue().apply(n);
                df.set(row, colIdx, val);
            }
        }
        return df;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static String formatNode(Object n) {
        if (n == null) return "";
        return n.toString();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Column.DType inferType(String key, java.util.List<java.util.Map.Entry<Object, Object>> edges,
                                           Object graph, boolean directed) {
        for (java.util.Map.Entry<Object, Object> e : edges) {
            AttrMap attr = directed
                ? ((DiGraph) graph).getEdgeAttr(e.getKey(), e.getValue())
                : ((Graph) graph).getEdgeAttr(e.getKey(), e.getValue());
            Object val = attr.get(key);
            if (val != null) return inferDType(val);
        }
        return Column.DType.STRING;
    }

    private static Column.DType inferDType(Object val) {
        if (val instanceof Integer || val instanceof Long || val instanceof Short || val instanceof Byte) {
            return Column.DType.INT64;
        }
        if (val instanceof Float || val instanceof Double) return Column.DType.FLOAT64;
        if (val instanceof Boolean) return Column.DType.BOOLEAN;
        return Column.DType.STRING;
    }

    private static Object defaultForType(Column.DType dtype) {
        switch (dtype) {
            case INT8: case INT16: case INT32: case INT64: return 0L;
            case FLOAT16: case FLOAT32: case FLOAT64: return 0.0;
            case BOOLEAN: return false;
            default: return "";
        }
    }
}