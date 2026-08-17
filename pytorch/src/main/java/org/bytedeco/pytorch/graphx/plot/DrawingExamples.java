/*
 * GraphX: Drawing Examples — NetworkX drawing API integration with the
 * project's matplotlib/seaborn Plot module.
 *
 * Demonstrates that draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels,
 * draw_networkx_edge_labels all cooperate with the existing BaseChart/Figure
 * matplotlib API.
 *
 * Inspired by Python NetworkX drawing examples.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.plot;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.graphx.GraphX;
import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.layout.Layout;
import org.bytedeco.pytorch.graphx.plot.drawNetworkx.Colormap;

import java.awt.Color;
import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * NetworkX-style graph drawing examples, ported to Java + our Plot module.
 */
public final class DrawingExamples {
    private DrawingExamples() {}
    public static final String OUT_DIR = "build/graphx-drawing";

    /**
     * Composite: draw nodes, edges, labels in one call.
     * Mirrors nx.draw_networkx(g, with_labels=True).
     */
    public static File plotDrawNetworkx() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);
        GraphChart chart = drawNetworkx.draw_networkx(g, pos,
                drawNetworkx.defaults().withLabels(true));
        chart.setTitle("Zachary's Karate Club — Spring Layout");
        File f = new File(OUT_DIR, "draw_networkx.png");
        chart.savefig(f.getAbsolutePath());
        if (!f.exists()) throw new AssertionError("File not produced");
        return f;
    }

    /**
     * Split: draw nodes / edges / labels separately with different styles.
     * Mirrors the canonical "labels and colors" example.
     */
    public static File plotLabelsAndColors() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);

        // Color nodes by Louvain community
        Map<String, Integer> community = GraphX.louvain_communities(g);

        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Community-Colored Nodes");
        // Step 1: edges
        drawNetworkx.draw_networkx_edges(chart, g, pos, drawNetworkx.defaults());
        // Step 2: nodes by community
        drawNetworkx.draw_networkx_nodes(chart, g, pos, community);
        // Step 3: labels
        drawNetworkx.draw_networkx_labels(chart, g, pos);
        chart.autoAxis(0.08);
        File f = new File(OUT_DIR, "labels_and_colors.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Continuous colormap: degree centrality → viridis.
     * Mirrors nx.draw(g, node_color=degree, cmap=plt.cm.viridis).
     */
    public static File plotNodeColormap() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);
        Map<String, Double> cent = GraphX.degree_centrality(g);
        GraphChart chart = drawNetworkx.draw_centrality(g, pos, cent, Colormap.VIRIDIS);
        chart.setTitle("GraphX — Degree Centrality Heatmap (Viridis)");
        File f = new File(OUT_DIR, "node_colormap_viridis.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Edge colormap + width by weight.
     * Mirrors nx.draw(g, edge_color=weights, width=weights).
     */
    public static File plotEdgeColormap() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);
        // Use degree of source node as "weight"
        Map<Object, Double> weights = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : g.edges()) {
            double w = 0.5 + 0.5 * (double) g.degree(e.getKey()) / g.order();
            weights.put(e.getKey() + "-" + e.getValue(), w);
        }
        GraphChart chart = drawNetworkx.draw_weighted(g, pos, weights);
        chart.setTitle("GraphX — Weighted Edges (Coolwarm)");
        File f = new File(OUT_DIR, "edge_colormap_coolwarm.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Edge labels: annotate each edge with a label.
     */
    public static File plotEdgeLabels() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.circular(g);
        Map<Object, String> edgeLabels = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : g.edges()) {
            edgeLabels.put(e.getKey() + "-" + e.getValue(),
                e.getKey().substring(0, Math.min(3, e.getKey().length())) + "→" +
                e.getValue().substring(0, Math.min(3, e.getValue().length())));
        }
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Edge Labels");
        drawNetworkx.draw_networkx_edges(chart, g, pos, drawNetworkx.defaults());
        drawNetworkx.draw_networkx_nodes(chart, g, pos, drawNetworkx.defaults());
        drawNetworkx.draw_networkx_labels(chart, g, pos);
        drawNetworkx.draw_networkx_edge_labels(chart, g, pos, edgeLabels);
        chart.autoAxis(0.1);
        File f = new File(OUT_DIR, "edge_labels.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Concentric shells layout.
     */
    public static File plotShells() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, Integer> louv = GraphX.louvain_communities(g);
        // Build shells list — one per community
        Map<Integer, Set<String>> byId = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> e : louv.entrySet()) {
            byId.computeIfAbsent(e.getValue(), k -> new java.util.LinkedHashSet<>()).add(e.getKey());
        }
        List<Set<String>> shells = new java.util.ArrayList<>(byId.values());
        GraphChart chart = drawNetworkx.draw_shells(g, shells);
        File f = new File(OUT_DIR, "shells.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Bipartite layout with categorical colors.
     */
    public static File plotBipartite() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = new Graph<>();
        String[] A = {"u1", "u2", "u3", "u4"};
        String[] B = {"v1", "v2", "v3", "v4", "v5"};
        for (String u : A) for (String v : B) g.addEdge(u, v);
        java.util.Set<String> top = new java.util.LinkedHashSet<>(Arrays.asList(A));
        GraphChart chart = drawNetworkx.draw_bipartite(g, top);
        File f = new File(OUT_DIR, "bipartite.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Composite: all major layouts side-by-side using {@link org.bytedeco.pytorch.plot.chart.Figure}.
     * Mirrors matplotlib's {@code plt.subplots(2, 2)}.
     */
    public static File plotLayoutGallery() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        org.bytedeco.pytorch.plot.chart.Figure fig = new org.bytedeco.pytorch.plot.chart.Figure(2, 2);
        fig.setSize(1200, 900);
        fig.set(0, 0, drawNetworkx.draw_circular(g));
        fig.set(0, 1, drawNetworkx.draw_random(g));
        fig.set(1, 0, drawNetworkx.draw_spring(g));
        fig.set(1, 1, drawNetworkx.draw_kamada_kawai(g));
        File f = new File(OUT_DIR, "layout_gallery.png");
        fig.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Centrality-over-time: community + centrality colormap combined.
     */
    public static File plotCentralityAndCommunity() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);
        GraphChart chart = new GraphChart(900, 700);
        chart.setTitle("GraphX — Community + Betweenness Centrality");
        // Edges
        drawNetworkx.draw_networkx_edges(chart, g, pos, drawNetworkx.defaults());
        // Nodes colored by Louvain community, sized by betweenness
        Map<String, Integer> community = GraphX.louvain_communities(g);
        Map<String, Double> betweenness = GraphX.betweenness_centrality(g);
        double maxB = Double.NEGATIVE_INFINITY;
        for (double v : betweenness.values()) maxB = Math.max(maxB, v);
        int base = 80;
        int maxSize = 800;
        for (String n : g.nodes()) {
            double[] p = pos.get(n);
            if (p == null) continue;
            int sz = base + (int) (maxSize * (betweenness.get(n) / (maxB + 1e-9)));
            Color c = GraphChart.category(community.get(n));
            chart.addNode(n, p[0], p[1], c, Math.max(3, (int) Math.sqrt(sz)),
                    String.valueOf(community.get(n)));
        }
        chart.autoAxis(0.08);
        File f = new File(OUT_DIR, "centrality_community.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Apply seaborn darkgrid theme — mirrors {@code sns.set_style('darkgrid')}
     * combined with nx.draw. Demonstrates cross-module integration.
     */
    public static File plotSeabornStyled() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);
        Map<String, Double> cent = GraphX.degree_centrality(g);
        GraphChart chart = drawNetworkx.draw_centrality(g, pos, cent, Colormap.MAGMA);
        chart.applyStyle("seaborn-darkgrid");
        chart.setTitle("GraphX — Seaborn Darkgrid + Magma Centrality");
        File f = new File(OUT_DIR, "seaborn_darkgrid_magma.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /**
     * Apply ggplot theme — another popular matplotlib style.
     */
    public static File plotGgplotStyled() throws Exception {
        new File(OUT_DIR).mkdirs();
        Graph<String> g = GraphX.karate_club_graph();
        Map<String, double[]> pos = Layout.spring(g);
        Map<String, Integer> comm = GraphX.louvain_communities(g);
        GraphChart chart = drawNetworkx.draw_communities(g, pos, comm);
        chart.applyStyle("ggplot");
        File f = new File(OUT_DIR, "ggplot_louvain.png");
        chart.savefig(f.getAbsolutePath());
        return f;
    }

    /** Run all drawing examples. */
    public static int runAll() throws Exception {
        int n = 0;
        File[] out = {
            plotDrawNetworkx(),
            plotLabelsAndColors(),
            plotNodeColormap(),
            plotEdgeColormap(),
            plotEdgeLabels(),
            plotShells(),
            plotBipartite(),
            plotLayoutGallery(),
            plotCentralityAndCommunity(),
            plotSeabornStyled(),
            plotGgplotStyled(),
        };
        for (File f : out) {
            if (f.exists() && f.length() > 0) n++;
            else System.err.println("  FAIL: " + f);
        }
        return n;
    }
}