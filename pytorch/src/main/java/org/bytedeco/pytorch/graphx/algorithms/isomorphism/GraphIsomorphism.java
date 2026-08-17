/*
 * GraphX: Graph isomorphism and structural metrics.
 *
 * Inspired by networkx.algorithms.isomorphism and networkx.algorithms.distance_regular.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.algorithms.isomorphism;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Graph isomorphism algorithms and structural fingerprints.
 *
 * <p>Aligned with {@code networkx.algorithms.isomorphism}.
 */
public final class GraphIsomorphism {
    private GraphIsomorphism() {}

    // =========================================================================
    // Weisfeiler-Lehman 1-D color refinement (graph kernel)
    // =========================================================================

    /**
     * WL 1-d color refinement: iteratively hash each node's neighbor-color multiset.
     * Two isomorphic graphs will produce identical color histograms after enough iterations.
     *
     * @return node → color (Long), plus final color histogram
     */
    public static <N> WLResult<N> weisfeilerLehman(Graph<N> g, int iterations) {
        Map<N, Long> colors = new LinkedHashMap<>();
        Map<Long, Long> colorHistogram = new LinkedHashMap<>();
        // Initial colors: hash of degree
        for (N n : g.nodes()) {
            long c = hashLong(31, g.degree(n));
            colors.put(n, c);
            colorHistogram.merge(c, 1L, Long::sum);
        }
        for (int iter = 0; iter < iterations; iter++) {
            Map<N, Long> next = new LinkedHashMap<>();
            Map<Long, Long> histNext = new LinkedHashMap<>();
            for (N u : g.nodes()) {
                List<Long> nbrColors = new ArrayList<>();
                for (N v : g.neighbors(u)) nbrColors.add(colors.get(v));
                Collections.sort(nbrColors);
                long newColor = hashList(colors.get(u), nbrColors);
                next.put(u, newColor);
                histNext.merge(newColor, 1L, Long::sum);
            }
            colors = next;
            colorHistogram = histNext;
        }
        return new WLResult<>(colors, colorHistogram);
    }

    public static final class WLResult<N> {
        public final Map<N, Long> colors;
        public final Map<Long, Long> histogram;
        public WLResult(Map<N, Long> colors, Map<Long, Long> histogram) {
            this.colors = colors;
            this.histogram = histogram;
        }
    }

    /** Compute WL kernel similarity between two graphs. */
    public static <N> double wlKernel(Graph<N> g1, Graph<N> g2, int iterations) {
        WLResult<N> r1 = weisfeilerLehman(g1, iterations);
        WLResult<N> r2 = weisfeilerLehman(g2, iterations);
        double dot = 0;
        Set<Long> allKeys = new HashSet<>();
        allKeys.addAll(r1.histogram.keySet());
        allKeys.addAll(r2.histogram.keySet());
        for (Long k : allKeys) {
            dot += r1.histogram.getOrDefault(k, 0L) * r2.histogram.getOrDefault(k, 0L);
        }
        double norm1 = norm(r1.histogram);
        double norm2 = norm(r2.histogram);
        return dot / (norm1 * norm2 + 1e-12);
    }

    static double norm(Map<Long, Long> hist) {
        double s = 0;
        for (long v : hist.values()) s += v * v;
        return Math.sqrt(s);
    }

    static long hashLong(long seed, long v) {
        long h = seed;
        h = h * 31 + v;
        h = h * 31 ^ (h >>> 13);
        return h;
    }

    static long hashList(long seed, List<Long> list) {
        long h = seed;
        for (long v : list) h = h * 31 + v;
        return h;
    }

    // =========================================================================
    // Degree sequence matching
    // =========================================================================

    /** Returns the degree sequence of {@code g} sorted descending. */
    public static <N> List<Integer> degreeSequence(Graph<N> g) {
        List<Integer> deg = new ArrayList<>();
        for (N n : g.nodes()) deg.add(g.degree(n));
        Collections.sort(deg, Collections.reverseOrder());
        return deg;
    }

    /** True if two graphs have the same degree sequence (necessary but not sufficient for isomorphism). */
    public static <N> boolean sameDegreeSequence(Graph<N> g1, Graph<N> g2) {
        return degreeSequence(g1).equals(degreeSequence(g2));
    }

    // =========================================================================
    // VF2 subgraph isomorphism (simplified)
    // =========================================================================

    /**
     * Check whether {@code pattern} appears as a subgraph of {@code target}.
     * Returns one mapping (pattern → target node) if found, else empty.
     *
     * <p>This is a simplified VF2 without semantic feasibility rules.
     */
    public static <N1, N2> Map<N1, N2> vf2Subgraph(Graph<N1> pattern, Graph<N2> target) {
        Map<N1, N2> mapping = new LinkedHashMap<>();
        if (pattern.order() > target.order()) return mapping;
        List<N1> patNodes = new ArrayList<>(pattern.nodes());
        Collections.sort(patNodes, (a, b) -> Integer.compare(pattern.degree(b), pattern.degree(a)));
        if (vf2Search(pattern, target, patNodes, 0, mapping, new HashSet<>())) return mapping;
        return Collections.emptyMap();
    }

    static <N1, N2> boolean vf2Search(Graph<N1> pattern, Graph<N2> target,
                                       List<N1> patNodes, int depth,
                                       Map<N1, N2> mapping, Set<N2> used) {
        if (depth == patNodes.size()) return true;
        N1 pNode = patNodes.get(depth);
        List<N2> tgtNodes = new ArrayList<>(target.nodes());
        Collections.sort(tgtNodes, (a, b) -> Integer.compare(target.degree(b), target.degree(a)));
        for (N2 tNode : tgtNodes) {
            if (used.contains(tNode)) continue;
            // Quick degree filter
            if (target.degree(tNode) < pattern.degree(pNode)) continue;
            mapping.put(pNode, tNode);
            used.add(tNode);
            if (consistent(pattern, target, mapping, pNode, tNode)) {
                if (vf2Search(pattern, target, patNodes, depth + 1, mapping, used)) return true;
            }
            mapping.remove(pNode);
            used.remove(tNode);
        }
        return false;
    }

    static <N1, N2> boolean consistent(Graph<N1> pattern, Graph<N2> target,
                                        Map<N1, N2> mapping, N1 pNode, N2 tNode) {
        // Check that mapped neighbors in pattern correspond to neighbors in target
        for (Map.Entry<N1, N2> e : mapping.entrySet()) {
            N1 pOther = e.getKey();
            N2 tOther = e.getValue();
            if (pOther.equals(pNode) || tOther.equals(tNode)) continue;
            boolean patAdj = pattern.hasEdge(pNode, pOther);
            boolean tgtAdj = target.hasEdge(tNode, tOther);
            if (patAdj != tgtAdj) return false;
        }
        return true;
    }
}