/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.core.MultiGraph;
import org.bytedeco.pytorch.graphx.core.MultiDiGraph;
import org.bytedeco.pytorch.graphx.generators.Classic;
import org.bytedeco.pytorch.graphx.generators.RandomGraphs;
import org.bytedeco.pytorch.graphx.generators.Lattice;
import org.bytedeco.pytorch.graphx.generators.Small;
import org.bytedeco.pytorch.graphx.algorithms.traversal.Traversal;
import org.bytedeco.pytorch.graphx.algorithms.components.ConnectedComponents;
import org.bytedeco.pytorch.graphx.algorithms.shortestpath.ShortestPath;
import org.bytedeco.pytorch.graphx.algorithms.tree.MinimumSpanningTree;
import org.bytedeco.pytorch.graphx.algorithms.centrality.Centrality;
import org.bytedeco.pytorch.graphx.layout.Layout;
import org.bytedeco.pytorch.graphx.drawing.GraphDrawer;

/**
 * GraphX main façade — direct API parity with Python's
 * {@code import networkx as nx} style usage.
 *
 * <p>Example:
 * <pre>{@code
 * Graph<String> g = GraphX.karate_club_graph();
 * Map<N, Double> pr = GraphX.pagerank(g, 0.85);
 * BufferedImage img = GraphX.draw(g);
 * }</pre>
 */
public final class GraphX {
    private GraphX() {}

    // ---- Generators (NetworkX naming) ----

    public static Graph<Integer> complete_graph(int n) { return Classic.completeGraph(n); }
    public static Graph<Integer> cycle_graph(int n) { return Classic.cycleGraph(n); }
    public static Graph<Integer> path_graph(int n) { return Classic.pathGraph(n); }
    public static Graph<Integer> star_graph(int n) { return Classic.starGraph(n); }
    public static Graph<Integer> wheel_graph(int n) { return Classic.wheelGraph(n); }
    public static Graph<Integer> ladder_graph(int n) { return Classic.ladderGraph(n); }
    public static Graph<Integer> circular_ladder_graph(int n) { return Classic.circularLadderGraph(n); }
    public static Graph<Integer> balanced_tree(int r, int h) { return Classic.balancedTree(r, h); }
    public static Graph<Integer> full_rary_tree(int r, int h) { return Classic.fullRaryTree(r, h); }
    public static Graph<Integer> binomial_tree(int n) { return Classic.binomialTree(n); }
    public static Graph<Integer> null_graph(int n) { return Classic.nullGraph(n); }
    public static Graph<Integer> empty_graph(int n) { return Classic.emptyGraph(n); }
    public static Graph<Integer> circulant_graph(int n, int[] offsets) { return Classic.circulantGraph(n, offsets); }
    public static Graph<Integer> turan_graph(int n, int r) { return Classic.turanGraph(n, r); }
    public static Graph<Integer> barbell_graph(int m1, int m2) { return Classic.barbellGraph(m1, m2); }
    public static Graph<Integer> lollipop_graph(int m, int n) { return Classic.lollipopGraph(m, n); }

    // ---- Random ----

    public static Graph<Integer> gnp_random_graph(int n, double p, long seed) { return RandomGraphs.gnpRandomGraph(n, p, seed); }
    public static Graph<Integer> gnm_random_graph(int n, int m, long seed) { return RandomGraphs.gnmRandomGraph(n, m, seed); }
    public static Graph<Integer> erdos_renyi_graph(int n, double p, long seed) { return RandomGraphs.erdosRenyiGraph(n, p, seed); }
    public static Graph<Integer> watts_strogatz_graph(int n, int k, double p, long seed) { return RandomGraphs.wattsStrogatzGraph(n, k, p, seed); }
    public static Graph<Integer> barabasi_albert_graph(int n, int m, long seed) { return RandomGraphs.barabasiAlbertGraph(n, m, seed); }
    public static Graph<Integer> newman_watts_strogatz_graph(int n, int k, double p, long seed) { return RandomGraphs.newmanWattsStrogatzGraph(n, k, p, seed); }
    public static Graph<Integer> random_regular_graph(int d, int n, long seed) { return RandomGraphs.randomRegularGraph(d, n, seed); }

    // ---- Lattice ----

    public static Graph<Integer> grid_2d_graph(int m, int n) { return Lattice.grid2dGraph(m, n); }
    public static Graph<Integer> grid_2d_graph(int m, int n, boolean periodic, boolean diagonal) { return Lattice.grid2dGraph(m, n, periodic, diagonal); }
    public static Graph<Integer> grid_graph(int[] dims) { return Lattice.gridGraph(dims); }
    public static Graph<Integer> hypercube_graph(int n) { return Lattice.hypercubeGraph(n); }

    // ---- Small graphs ----

    public static Graph<String> karate_club_graph() { return Small.karateClubGraph(); }
    public static Graph<String> davis_southern_women_graph() { return Small.davisSouthernWomenGraph(); }
    public static Graph<String> florentine_families_graph() { return Small.florentineFamiliesGraph(); }
    public static Graph<String> les_miserables_graph() { return Small.lesMiserablesGraph(); }

    // ---- Traversal ----

    public static <N> java.util.List<N> bfs_order(Graph<N> g, N source) { return Traversal.bfsOrder(g, source); }
    public static <N> java.util.List<N> dfs_preorder(Graph<N> g, N source) { return Traversal.dfsPreorder(g, source); }
    public static <N> java.util.List<N> dfs_postorder(Graph<N> g, N source) { return Traversal.dfsPostorder(g, source); }
    public static <N> java.util.List<N> bfs_predecessors(Graph<N> g, N source) {
        java.util.List<N> result = new java.util.ArrayList<>();
        for (N n : Traversal.bfsPredecessors(g, source).keySet()) result.add(n);
        return result;
    }

    // ---- Components ----

    public static <N> java.util.List<java.util.Set<N>> connected_components(Graph<N> g) { return ConnectedComponents.connectedComponents(g); }
    public static <N> boolean is_connected(Graph<N> g) { return ConnectedComponents.isConnected(g); }
    public static <N> java.util.List<java.util.Set<N>> strongly_connected_components(DiGraph<N> g) { return ConnectedComponents.stronglyConnectedComponents(g); }
    public static <N> boolean is_strongly_connected(DiGraph<N> g) { return ConnectedComponents.isStronglyConnected(g); }

    // ---- Shortest paths ----

    public static <N> java.util.List<N> shortest_path(Graph<N> g, N source, N target) { return ShortestPath.shortestPath(g, source, target); }
    public static <N> double shortest_path_length(Graph<N> g, N source, N target) { return ShortestPath.shortestPathLength(g, source, target); }
    public static <N> ShortestPath.DijkstraResult<N> dijkstra(Graph<N> g, N source) { return ShortestPath.dijkstra(g, source); }

    // ---- MST ----

    public static <N> java.util.List<MinimumSpanningTree.WeightedEdge<N>> minimum_spanning_tree(Graph<N> g) { return MinimumSpanningTree.kruskal(g); }
    public static <N> java.util.List<MinimumSpanningTree.WeightedEdge<N>> minimum_spanning_edges(Graph<N> g) { return MinimumSpanningTree.kruskal(g); }

    // ---- Centrality ----

    public static <N> java.util.Map<N, Double> degree_centrality(Graph<N> g) { return Centrality.degreeCentrality(g); }
    public static <N> java.util.Map<N, Double> closeness_centrality(Graph<N> g) { return Centrality.closenessCentrality(g); }
    public static <N> java.util.Map<N, Double> harmonic_centrality(Graph<N> g) { return Centrality.harmonicCentrality(g); }
    public static <N> java.util.Map<N, Double> betweenness_centrality(Graph<N> g) { return Centrality.betweennessCentrality(g); }
    public static <N> java.util.Map<N, Double> eigenvector_centrality(Graph<N> g) { return Centrality.eigenvectorCentrality(g); }
    public static <N> java.util.Map<N, Double> katz_centrality(Graph<N> g) { return Centrality.katzCentrality(g); }
    public static <N> java.util.Map<N, Double> pagerank(DiGraph<N> g, double alpha) {
        return Centrality.pagerank(g, alpha, 1e-6, 100, null);
    }
    public static <N> java.util.Map<N, Double> pagerank(DiGraph<N> g) { return Centrality.pagerank(g); }
    public static <N> Centrality.HITSResult<N> hits(DiGraph<N> g) { return Centrality.hits(g); }

    // ---- Layout ----

    public static <N> java.util.Map<N, double[]> spring_layout(Graph<N> g) { return Layout.spring(g); }
    public static <N> java.util.Map<N, double[]> circular_layout(Graph<N> g) { return Layout.circular(g); }
    public static <N> java.util.Map<N, double[]> random_layout(Graph<N> g, long seed) { return Layout.random(g, seed); }
    public static <N> java.util.Map<N, double[]> shell_layout(Graph<N> g, java.util.List<java.util.List<N>> shells) { return Layout.shell(g, shells); }
    public static <N> java.util.Map<N, double[]> kamada_kawai_layout(Graph<N> g) { return Layout.kamadaKawai(g); }
    public static <N> java.util.Map<N, double[]> spectral_layout(Graph<N> g) { return Layout.spectral(g); }
    public static <N> java.util.Map<N, double[]> planar_layout(Graph<N> g) { return Layout.planar(g); }

    // ---- I/O ----

    public static <N> java.awt.image.BufferedImage draw(Graph<N> g) { return GraphDrawer.draw(g); }
    public static <N> java.awt.image.BufferedImage draw(Graph<N> g, java.util.Map<N, double[]> pos) { return GraphDrawer.draw(g, pos); }
    public static <N> void savefig(Graph<N> g, String path) throws Exception { GraphDrawer.savefig(g, path); }
    public static <N> void savefig(Graph<N> g, java.util.Map<N, double[]> pos, String path) throws Exception {
        GraphDrawer.savefig(g, pos, path, GraphDrawer.defaults());
    }

    // Edge-list / Adj-list
    public static Graph<Object> read_edgelist(String path) throws java.io.IOException {
        return org.bytedeco.pytorch.graphx.io.EdgeList.read(path);
    }
    public static Graph<Object> read_edgelist(String path, boolean directed) throws java.io.IOException {
        return org.bytedeco.pytorch.graphx.io.EdgeList.read(path, directed);
    }
    public static <N> void write_edgelist(Graph<N> g, String path) throws java.io.IOException {
        org.bytedeco.pytorch.graphx.io.EdgeList.write(g, path);
    }
    public static Graph<Object> read_adjlist(String path) throws java.io.IOException {
        return org.bytedeco.pytorch.graphx.io.AdjList.read(path);
    }
    public static <N> void write_adjlist(Graph<N> g, String path) throws java.io.IOException {
        org.bytedeco.pytorch.graphx.io.AdjList.write(g, path);
    }
    // GraphML / GEXF / JSON
    public static Graph<Object> read_graphml(String path) throws java.io.IOException {
        return org.bytedeco.pytorch.graphx.io.GraphML.read(path);
    }
    public static <N> void write_graphml(Graph<N> g, String path) throws java.io.IOException {
        org.bytedeco.pytorch.graphx.io.GraphML.write(g, path);
    }
    public static Graph<Object> read_gexf(String path) throws java.io.IOException {
        return org.bytedeco.pytorch.graphx.io.GEXF.read(path);
    }
    public static <N> void write_gexf(Graph<N> g, String path) throws java.io.IOException {
        org.bytedeco.pytorch.graphx.io.GEXF.write(g, path);
    }
    public static Graph<Object> read_json(String path) throws java.io.IOException {
        return org.bytedeco.pytorch.graphx.io.JSONGraph.readNodeLink(path);
    }
    public static <N> void write_json(Graph<N> g, String path) throws java.io.IOException {
        org.bytedeco.pytorch.graphx.io.JSONGraph.writeNodeLink(g, path);
    }

    // ---- DataFrame I/O ----

    public static Graph<Object> from_edgelist_dataframe(org.bytedeco.pytorch.dataframe.DataFrame df,
                                                        String srcCol, String dstCol, String... attrCols) {
        return org.bytedeco.pytorch.graphx.io.DataFrameIO.fromEdgeList(df, srcCol, dstCol, attrCols);
    }

    public static org.bytedeco.pytorch.dataframe.DataFrame to_edgelist_dataframe(Graph<?> g) {
        return org.bytedeco.pytorch.graphx.io.DataFrameIO.toEdgeList(g);
    }

    public static <N> Graph<N> apply_node_attributes(Graph<N> g, org.bytedeco.pytorch.dataframe.DataFrame df,
                                                      String nodeCol) {
        return org.bytedeco.pytorch.graphx.io.DataFrameIO.applyNodeAttributes(g, df, nodeCol);
    }

    public static <N> org.bytedeco.pytorch.dataframe.DataFrame node_metrics_dataframe(
            Graph<N> g, java.util.Map<String, java.util.function.Function<N, Object>> metrics) {
        return org.bytedeco.pytorch.graphx.io.DataFrameIO.nodeMetricsToDataFrame(g, metrics);
    }

    // ---- Community / Matching ----

    public static <N> java.util.Map<N, Integer> label_propagation_communities(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.algorithms.community.Community.labelPropagation(g);
    }
    public static <N> java.util.Map<N, Integer> louvain_communities(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.algorithms.community.Community.louvain(g);
    }
    public static <N> java.util.List<java.util.Set<N>> greedy_modularity_communities(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.algorithms.community.Community.labelsToCommunities(
            org.bytedeco.pytorch.graphx.algorithms.community.Community.greedyModularity(g));
    }
    public static <N> double modularity(Graph<N> g, java.util.Map<N, ?> communities) {
        return org.bytedeco.pytorch.graphx.algorithms.community.Community.modularity(g, communities);
    }

    // ---- Flow ----

    public static <N> org.bytedeco.pytorch.graphx.algorithms.flow.MaxFlow.FlowResult<N> maximum_flow(
            org.bytedeco.pytorch.graphx.core.DiGraph<N> g, N s, N t) {
        return org.bytedeco.pytorch.graphx.algorithms.flow.MaxFlow.edmondsKarp(g, s, t);
    }
    public static <N> double minimum_cut(org.bytedeco.pytorch.graphx.core.DiGraph<N> g, N s, N t) {
        return org.bytedeco.pytorch.graphx.algorithms.flow.MaxFlow.minimumCut(g, s, t);
    }

    // ---- Isomorphism / WL Kernel ----

    public static <N1, N2> java.util.Map<N1, N2> vf2_subgraph_isomorphism(
            Graph<N1> pattern, Graph<N2> target) {
        return org.bytedeco.pytorch.graphx.algorithms.isomorphism.GraphIsomorphism.vf2Subgraph(pattern, target);
    }
    public static <N> double wl_kernel(Graph<N> g1, Graph<N> g2, int iterations) {
        return org.bytedeco.pytorch.graphx.algorithms.isomorphism.GraphIsomorphism.wlKernel(g1, g2, iterations);
    }

    // ---- Specialized generators ----
    public static Graph<Integer> random_geometric_graph(int n, double radius, long seed) {
        return org.bytedeco.pytorch.graphx.generators.Specialized.randomGeometricGraph(n, radius, seed);
    }
    public static Graph<Integer> waxman_graph(int n, double beta, double alpha, long seed) {
        return org.bytedeco.pytorch.graphx.generators.Specialized.waxmanGraph(n, beta, alpha, seed);
    }
    public static Graph<Integer> random_labeled_tree(int n, long seed) {
        return org.bytedeco.pytorch.graphx.generators.Specialized.randomLabeledTree(n, seed);
    }
    public static Graph<Integer> stochastic_block_model(int[] blockSizes, double[][] pMatrix, long seed) {
        return org.bytedeco.pytorch.graphx.generators.Specialized.stochasticBlockModel(blockSizes, pMatrix, seed);
    }

    // ---- Force-directed layouts ----
    public static <N> java.util.Map<N, double[]> force_atlas2_layout(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.layout.Layout.forceAtlas2(g);
    }
    public static <N> java.util.Map<N, double[]> arf_layout(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.layout.Layout.arf(g);
    }

    // ---- NetworkX-style drawing (renders to our BaseChart/seaborn stack) ----
    // Mirror nx.draw / nx.draw_networkx_*. Each returns a GraphChart (BaseChart subclass)
    // that can be saved, displayed, or composed with other charts via Figure.
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_nullpos(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx(g, null);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_pos(Graph<N> g,
                                                                       java.util.Map<N, double[]> pos) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx(g, pos);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw(Graph<N> g,
                                                                       java.util.Map<N, double[]> pos,
                                                                       org.bytedeco.pytorch.graphx.plot.drawNetworkx.DrawConfig cfg) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx(g, pos, cfg);
    }
    public static <N> void draw_networkx_nodes(org.bytedeco.pytorch.graphx.plot.GraphChart chart,
                                                Graph<N> g, java.util.Map<N, double[]> pos) {
        org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx_nodes(chart, g, pos);
    }
    public static <N> void draw_networkx_edges(org.bytedeco.pytorch.graphx.plot.GraphChart chart,
                                                Graph<N> g, java.util.Map<N, double[]> pos) {
        org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx_edges(chart, g, pos);
    }
    public static <N> void draw_networkx_labels(org.bytedeco.pytorch.graphx.plot.GraphChart chart,
                                                 Graph<N> g, java.util.Map<N, double[]> pos) {
        org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx_labels(chart, g, pos);
    }
    public static <N> void draw_networkx_edge_labels(org.bytedeco.pytorch.graphx.plot.GraphChart chart,
                                                      Graph<N> g, java.util.Map<N, double[]> pos) {
        org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_networkx_edge_labels(chart, g, pos);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_communities(
            Graph<N> g, java.util.Map<N, double[]> pos, java.util.Map<N, Integer> community) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_communities(g, pos, community);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_centrality(
            Graph<N> g, java.util.Map<N, double[]> pos, java.util.Map<N, Double> centrality,
            org.bytedeco.pytorch.graphx.plot.drawNetworkx.Colormap cmap) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_centrality(g, pos, centrality, cmap);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_weighted(
            Graph<N> g, java.util.Map<N, double[]> pos, java.util.Map<Object, Double> weights) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_weighted(g, pos, weights);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_shells(
            Graph<N> g, java.util.List<? extends java.util.Set<N>> shells) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_shells(g, shells);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_bipartite(
            Graph<N> g, java.util.Set<N> topNodes) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_bipartite(g, topNodes);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_spring(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_spring(g);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_circular(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_circular(g);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_kamada_kawai(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_kamada_kawai(g);
    }
    public static <N> org.bytedeco.pytorch.graphx.plot.GraphChart draw_random(Graph<N> g) {
        return org.bytedeco.pytorch.graphx.plot.drawNetworkx.draw_random(g);
    }
}