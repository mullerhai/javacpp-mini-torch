/*
 * GraphX: draw_networkx_* — NetworkX-compatible graph visualization API
 * that renders to a GraphChart (a BaseChart subclass), natively integrated
 * with the project's matplotlib/seaborn Plot module.
 *
 * Mirrors Python NetworkX draw_networkx_nodes, draw_networkx_edges,
 * draw_networkx_labels, draw_networkx_edge_labels, and the convenience
 * composite draw_networkx.
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx).
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.plot;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.layout.Layout;

import java.awt.Color;
import java.awt.Font;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

/**
 * NetworkX-compatible graph drawing functions.
 *
 * <p>Each function returns a {@link GraphChart} (a {@link org.bytedeco.pytorch.plot.chart.BaseChart}
 * subclass) which can be combined with any other plot via {@link org.bytedeco.pytorch.plot.chart.Figure}.
 *
 * <h2>API parity with NetworkX</h2>
 * <pre>{@code
 * // Mirrors nx.draw(g, pos=...)
 * GraphChart chart = draw_networkx(g, pos, with_labels=true);
 * chart.savefig("out.png");
 *
 * // Or split into pieces
 * GraphChart chart = new GraphChart(800, 600);
 * draw_networkx_nodes(chart, g, pos, node_color, node_size);
 * draw_networkx_edges(chart, g, pos, edge_color, width);
 * draw_networkx_labels(chart, g, pos, labels);
 * }</pre>
 */
public final class drawNetworkx {
    private drawNetworkx() {}
    private static final String FONT_SANS = java.awt.Font.SANS_SERIF;

    // =========================================================================
    // draw_networkx — composite
    // =========================================================================

    /**
     * Mirror of {@code nx.draw_networkx(g, pos=...)} — single-call drawing.
     */
    public static <N> GraphChart draw_networkx(Graph<N> g, Map<N, double[]> pos) {
        return draw_networkx(g, pos, defaults());
    }

    public static <N> GraphChart draw_networkx(Graph<N> g, Map<N, double[]> pos, DrawConfig cfg) {
        GraphChart chart = new GraphChart(800, 600);
        applyStyle(chart, cfg);
        if (pos == null) pos = Layout.spring(g);
        draw_networkx_edges(chart, g, pos, cfg);
        draw_networkx_nodes(chart, g, pos, cfg);
        if (cfg.withLabels) {
            draw_networkx_labels(chart, g, pos, cfg);
        }
        if (cfg.withEdgeLabels) {
            draw_networkx_edge_labels(chart, g, pos, cfg);
        }
        chart.autoAxis(0.08);
        return chart;
    }

    /**
     * Default config (mirrors matplotlib's rcParams for nx.draw).
     */
    public static DrawConfig defaults() {
        return new DrawConfig();
    }

    /** Configuration for {@code draw_networkx}. */
    public static final class DrawConfig {
        public boolean withLabels = false;
        public boolean withEdgeLabels = false;
        public Color nodeColor = new Color(0x4878d0);
        public Color edgeColor = new Color(0x666666);
        public int nodeSize = 300;
        public double edgeWidth = 1.0;
        public Color labelColor = Color.BLACK;
        public Color arrowColor = new Color(0x333333);
        public Font labelFont = new Font(FONT_SANS, Font.PLAIN, 12);
        // Per-node overrides
        public Function<Object, Color> nodeColorFn;
        public Function<Object, Integer> nodeSizeFn;
        public Function<Object, String> labelFn;
        // Per-edge overrides
        public Function<Object, Color> edgeColorFn;
        public ToDoubleFunction<Object> edgeWidthFn;
        public Function<Object, String> edgeLabelFn;

        public DrawConfig withLabels(boolean v) { this.withLabels = v; return this; }
        public DrawConfig withEdgeLabels(boolean v) { this.withEdgeLabels = v; return this; }
        public DrawConfig nodeColor(Color c) { this.nodeColor = c; return this; }
        public DrawConfig edgeColor(Color c) { this.edgeColor = c; return this; }
        public DrawConfig nodeSize(int s) { this.nodeSize = s; return this; }
        public DrawConfig edgeWidth(double w) { this.edgeWidth = w; return this; }
        public DrawConfig labelColor(Color c) { this.labelColor = c; return this; }
        public DrawConfig nodeColorMap(Function<Object, Color> fn) { this.nodeColorFn = fn; return this; }
        public DrawConfig nodeSizeMap(Function<Object, Integer> fn) { this.nodeSizeFn = fn; return this; }
        public DrawConfig labels(Function<Object, String> fn) { this.labelFn = fn; return this; }
        public DrawConfig edgeColorMap(Function<Object, Color> fn) { this.edgeColorFn = fn; return this; }
        public DrawConfig edgeWidthMap(ToDoubleFunction<Object> fn) { this.edgeWidthFn = fn; return this; }
        public DrawConfig edgeLabels(Function<Object, String> fn) { this.edgeLabelFn = fn; return this; }
    }

    private static void applyStyle(GraphChart chart, DrawConfig cfg) {
        chart.defaultNodeColor(cfg.nodeColor);
        chart.defaultEdgeColor(cfg.edgeColor);
        chart.defaultEdgeWidth((float) cfg.edgeWidth);
        chart.defaultLabelColor(cfg.labelColor);
        chart.arrowColor(cfg.arrowColor);
        chart.fontSize(cfg.labelFont.getSize());
    }

    // =========================================================================
    // draw_networkx_nodes
    // =========================================================================

    /** Mirror of {@code nx.draw_networkx_nodes}. */
    public static <N> void draw_networkx_nodes(GraphChart chart, Graph<N> g, Map<N, double[]> pos) {
        draw_networkx_nodes(chart, g, pos, defaults());
    }

    public static <N> void draw_networkx_nodes(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                 DrawConfig cfg) {
        java.util.Set<N> nodeSet = new java.util.LinkedHashSet<>(g.nodes());
        // Per-node coloring
        if (cfg.nodeColorFn != null) {
            for (N n : nodeSet) {
                double[] p = pos.get(n);
                if (p == null) continue;
                Color c = cfg.nodeColorFn.apply(n);
                int size = cfg.nodeSizeFn != null ? cfg.nodeSizeFn.apply(n) : cfg.nodeSize;
                chart.addNode(n, p[0], p[1], c, Math.max(2, (int) Math.sqrt(size)),
                        cfg.labelFn != null ? cfg.labelFn.apply(n) : null);
            }
        } else {
            int r = Math.max(2, (int) Math.sqrt(cfg.nodeSize));
            for (N n : nodeSet) {
                double[] p = pos.get(n);
                if (p == null) continue;
                chart.addNode(n, p[0], p[1], cfg.nodeColor, r,
                        cfg.labelFn != null ? cfg.labelFn.apply(n) : null);
            }
        }
    }

    /** Convenience: node color from a value map using a colormap. */
    public static <N> void draw_networkx_nodes(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                 Map<N, Double> values, Colormap cmap) {
        java.util.Set<N> nodeSet = new java.util.LinkedHashSet<>(g.nodes());
        double vMin = Double.POSITIVE_INFINITY, vMax = Double.NEGATIVE_INFINITY;
        for (double v : values.values()) {
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
        }
        int r = Math.max(2, (int) Math.sqrt(defaults().nodeSize));
        for (N n : nodeSet) {
            double[] p = pos.get(n);
            if (p == null) continue;
            Double v = values.get(n);
            Color c = v == null ? Color.GRAY : cmap.color(v, vMin, vMax);
            chart.addNode(n, p[0], p[1], c, r, null);
        }
    }

    /** Convenience: node color from a category map (community id, etc.). */
    public static <N> void draw_networkx_nodes(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                 Map<N, Integer> categories) {
        java.util.Set<N> nodeSet = new java.util.LinkedHashSet<>(g.nodes());
        int r = Math.max(2, (int) Math.sqrt(defaults().nodeSize));
        for (N n : nodeSet) {
            double[] p = pos.get(n);
            if (p == null) continue;
            Integer cat = categories.get(n);
            Color c = cat == null ? Color.GRAY : GraphChart.category(cat);
            chart.addNode(n, p[0], p[1], c, r, null);
        }
    }

    // =========================================================================
    // draw_networkx_edges
    // =========================================================================

    /** Mirror of {@code nx.draw_networkx_edges}. */
    public static <N> void draw_networkx_edges(GraphChart chart, Graph<N> g, Map<N, double[]> pos) {
        draw_networkx_edges(chart, g, pos, defaults());
    }

    public static <N> void draw_networkx_edges(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                 DrawConfig cfg) {
        // Both Graph and DiGraph expose successors(); for undirected Graph this
        // yields neighbors. DiGraph instances are detected at runtime through
        // Class.isAssignableFrom to avoid an instanceof check across unrelated
        // generic class hierarchies.
        boolean directed = DiGraph.class.isAssignableFrom(g.getClass());
        chart.directed(directed);
        // Build edge set (avoid double-drawing in undirected)
        java.util.Set<String> seen = new java.util.HashSet<>();
        java.util.Set<N> nodeSet = new java.util.LinkedHashSet<>(g.nodes());
        for (N u : nodeSet) {
            double[] pu = pos.get(u);
            if (pu == null) continue;
            for (N v : g.successors(u)) {
                double[] pv = pos.get(v);
                if (pv == null) continue;
                String key = directed ? u + "→" + v : pairKey(u, v);
                if (!seen.add(key)) continue;
                Object edgeId = directed ? (u + "→" + v) : pairKey(u, v);
                Color c = cfg.edgeColorFn != null ? cfg.edgeColorFn.apply(edgeId) : cfg.edgeColor;
                float w = (float) (cfg.edgeWidthFn != null ? cfg.edgeWidthFn.applyAsDouble(edgeId) : cfg.edgeWidth);
                String label = cfg.edgeLabelFn != null ? cfg.edgeLabelFn.apply(edgeId) : null;
                chart.addEdge(edgeId, edgeId, pu[0], pu[1], pv[0], pv[1], c, w, label);
            }
        }
    }

    /** Convenience: edge width from a weight map. */
    public static <N> void draw_networkx_edges(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                 Map<Object, Double> weights, double scale) {
        DrawConfig cfg = defaults();
        cfg.edgeWidthMap(e -> {
            Double w = weights.get(e);
            return w == null ? 1.0 : w * scale;
        });
        draw_networkx_edges(chart, g, pos, cfg);
    }

    /** Convenience: edge color from a value map using a colormap. */
    public static <N> void draw_networkx_edges(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                 Map<Object, Double> values, Colormap cmap) {
        double vMin = Double.POSITIVE_INFINITY, vMax = Double.NEGATIVE_INFINITY;
        for (double v : values.values()) {
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
        }
        final double vMinF = vMin;
        final double vMaxF = vMax;
        DrawConfig cfg = defaults();
        cfg.edgeColorMap(e -> {
            Double v = values.get(e);
            return v == null ? Color.GRAY : cmap.color(v, vMinF, vMaxF);
        });
        draw_networkx_edges(chart, g, pos, cfg);
    }

    // =========================================================================
    // draw_networkx_labels
    // =========================================================================

    /** Mirror of {@code nx.draw_networkx_labels}. */
    public static <N> void draw_networkx_labels(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                  DrawConfig cfg) {
        chart.showLabels(true);
        chart.defaultLabelColor(cfg.labelColor);
        Function<Object, String> labelFn = cfg.labelFn != null ? cfg.labelFn : Object::toString;
        // Build label lookup
        java.util.Map<Object, String> labels = new LinkedHashMap<>();
        for (N n : g.nodes()) labels.put(n, labelFn.apply(n));
        // Add labels to existing nodes (NodeGlyph already holds them via addNode)
        // If nodes were drawn separately, walk nodeGlyphs and attach labels
        for (GraphChart.NodeGlyph ng : chart.nodeGlyphs()) {
            if (ng.label == null) {
                ng.label = labels.get(ng.id);
            }
        }
    }

    public static <N> void draw_networkx_labels(GraphChart chart, Graph<N> g, Map<N, double[]> pos) {
        draw_networkx_labels(chart, g, pos, defaults());
    }

    public static <N> void draw_networkx_labels(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                  Map<N, String> labels) {
        DrawConfig cfg = defaults();
        cfg.labelFn = labels::get;
        draw_networkx_labels(chart, g, pos, cfg);
    }

    // =========================================================================
    // draw_networkx_edge_labels
    // =========================================================================

    /** Mirror of {@code nx.draw_networkx_edge_labels}. */
    public static <N> void draw_networkx_edge_labels(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                       DrawConfig cfg) {
        chart.showEdgeLabels(true);
        Function<Object, String> labelFn = cfg.edgeLabelFn != null ? cfg.edgeLabelFn : Object::toString;
        // Labels were already assigned via EdgeSegment(label) in draw_networkx_edges;
        // if not, attach now
        for (GraphChart.EdgeSegment es : chart.edgeSegments()) {
            if (es.label == null) es.label = labelFn.apply(es.u);
        }
    }

    public static <N> void draw_networkx_edge_labels(GraphChart chart, Graph<N> g, Map<N, double[]> pos) {
        draw_networkx_edge_labels(chart, g, pos, defaults());
    }

    public static <N> void draw_networkx_edge_labels(GraphChart chart, Graph<N> g, Map<N, double[]> pos,
                                                       Map<Object, String> edgeLabels) {
        DrawConfig cfg = defaults();
        cfg.edgeLabelFn = edgeLabels::get;
        draw_networkx_edge_labels(chart, g, pos, cfg);
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static String pairKey(Object u, Object v) {
        int h1 = System.identityHashCode(u), h2 = System.identityHashCode(v);
        return Math.min(h1, h2) + "-" + Math.max(h1, h2);
    }

    // =========================================================================
    // Colormap enum
    // =========================================================================

    /** Available colormaps for node/edge value-to-color mapping. */
    public enum Colormap {
        VIRIDIS, MAGMA, COOLWARM;

        public Color color(double v, double vMin, double vMax) {
            switch (this) {
                case VIRIDIS: return GraphChart.viridis(v, vMin, vMax);
                case MAGMA:   return GraphChart.magma(v, vMin, vMax);
                case COOLWARM: return GraphChart.coolwarm(v, vMin, vMax);
                default: return Color.GRAY;
            }
        }
    }

    // =========================================================================
    // High-level composite visualizations
    // =========================================================================

    /**
     * {@code nx.draw(g, node_color=communities)} — color nodes by community id.
     */
    public static <N> GraphChart draw_communities(Graph<N> g, Map<N, double[]> pos,
                                                    Map<N, Integer> community) {
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Louvain Communities (" + community.values().stream().distinct().count() + ")");
        chart.setXAxisLabel("x");
        chart.setYAxisLabel("y");
        draw_networkx_edges(chart, g, pos, defaults());
        draw_networkx_nodes(chart, g, pos, community);
        draw_networkx_labels(chart, g, pos, defaults());
        chart.autoAxis(0.08);
        return chart;
    }

    /** {@code nx.draw(g, node_color=centrality)} — color nodes by a continuous centrality. */
    public static <N> GraphChart draw_centrality(Graph<N> g, Map<N, double[]> pos,
                                                    Map<N, Double> centrality, Colormap cmap) {
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Centrality Heatmap");
        draw_networkx_edges(chart, g, pos, defaults());
        draw_networkx_nodes(chart, g, pos, centrality, cmap);
        draw_networkx_labels(chart, g, pos, defaults());
        chart.autoAxis(0.08);
        return chart;
    }

    /**
     * {@code nx.draw(g, edge_color=weights, width=weights)} — color/width by edge weight.
     */
    public static <N> GraphChart draw_weighted(Graph<N> g, Map<N, double[]> pos,
                                                 Map<Object, Double> weights) {
        double wMin = Double.POSITIVE_INFINITY;
        double wMax = Double.NEGATIVE_INFINITY;
        for (double v : weights.values()) {
            if (v < wMin) wMin = v;
            if (v > wMax) wMax = v;
        }
        final double wMinF = wMin;
        final double wMaxF = wMax;
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Weighted Graph");
        // Edges: width by weight, color by coolwarm
        java.util.function.Function<Object, Color> colorFn = e -> {
            Double w = weights.get(e);
            return w == null ? Color.GRAY : Colormap.COOLWARM.color(w, wMinF, wMaxF);
        };
        java.util.function.ToDoubleFunction<Object> widthFn = e -> {
            Double w = weights.get(e);
            if (w == null || wMaxF - wMinF <= 0) return 1.0;
            return 1.0 + 5.0 * (w - wMinF) / (wMaxF - wMinF);
        };
        DrawConfig cfg = defaults();
        cfg.edgeColorMap(colorFn);
        cfg.edgeWidthMap(widthFn);
        draw_networkx_edges(chart, g, pos, cfg);
        draw_networkx_nodes(chart, g, pos, defaults());
        draw_networkx_labels(chart, g, pos, defaults());
        chart.autoAxis(0.08);
        return chart;
    }

    /**
     * {@code nx.draw_shell(g, nlist)} — concentric shells layout & draw.
     */
    public static <N> GraphChart draw_shells(Graph<N> g, java.util.List<? extends java.util.Set<N>> shells) {
        java.util.List<java.util.List<N>> shellList = new java.util.ArrayList<>();
        for (java.util.Set<N> shell : shells) shellList.add(new java.util.ArrayList<>(shell));
        Map<N, double[]> pos = Layout.shell(g, shellList);
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Concentric Shells (" + shells.size() + " layers)");
        draw_networkx_edges(chart, g, pos, defaults());
        // Color each shell with a category color
        java.util.Map<N, Integer> shellId = new LinkedHashMap<>();
        for (int i = 0; i < shells.size(); i++) {
            for (N n : shells.get(i)) shellId.put(n, i);
        }
        draw_networkx_nodes(chart, g, pos, shellId);
        draw_networkx_labels(chart, g, pos, defaults());
        chart.autoAxis(0.05);
        return chart;
    }

    /**
     * {@code nx.draw_bipartite(g, top_nodes)} — bipartite layout & draw.
     */
    public static <N> GraphChart draw_bipartite(Graph<N> g, Set<N> topNodes) {
        Map<N, double[]> pos = Layout.bipartite(g, topNodes);
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Bipartite Graph");
        // Color top vs bottom
        java.util.Map<N, Integer> cat = new LinkedHashMap<>();
        for (N n : g.nodes()) cat.put(n, topNodes.contains(n) ? 0 : 1);
        draw_networkx_edges(chart, g, pos, defaults());
        draw_networkx_nodes(chart, g, pos, cat);
        draw_networkx_labels(chart, g, pos, defaults());
        chart.autoAxis(0.05);
        return chart;
    }

    /**
     * {@code nx.draw_kamada_kawai(g)} — Kamada-Kawai layout.
     */
    public static <N> GraphChart draw_kamada_kawai(Graph<N> g) {
        return draw_networkx(g, Layout.kamadaKawai(g), defaults());
    }

    /** {@code nx.draw_spring(g)} — spring layout. */
    public static <N> GraphChart draw_spring(Graph<N> g) {
        return draw_networkx(g, Layout.spring(g), defaults());
    }

    /** {@code nx.draw_circular(g)} — circular layout. */
    public static <N> GraphChart draw_circular(Graph<N> g) {
        return draw_networkx(g, Layout.circular(g), defaults());
    }

    /** {@code nx.draw_random(g)} — random layout. */
    public static <N> GraphChart draw_random(Graph<N> g) {
        return draw_networkx(g, Layout.random(g), defaults());
    }
}