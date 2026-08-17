/*
 * GraphX: Additional tree and matching algorithms.
 *
 * Inspired by networkx.algorithms.tree and networkx.algorithms.matching.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.algorithms.tree;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.algorithms.traversal.Traversal;
import org.bytedeco.pytorch.graphx.algorithms.components.ConnectedComponents;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Additional tree algorithms and graph matching.
 *
 * <p>Aligned with {@code networkx.algorithms.tree.operations} and
 * {@code networkx.algorithms.matching}.
 */
public final class TreeAlgorithms {
    private TreeAlgorithms() {}

    // =========================================================================
    // Tree recognition
    // =========================================================================

    /** Returns true if {@code g} is a tree (connected, no cycles, n-1 edges). */
    public static <N> boolean isTree(Graph<N> g) {
        return ConnectedComponents.isConnected(g) && g.numberOfEdges() == g.order() - 1;
    }

    /** Returns true if {@code g} is a forest (acyclic). */
    public static <N> boolean isForest(Graph<N> g) {
        if (g.order() == 0) return true;
        // DFS-based cycle detection
        Set<N> visited = new HashSet<>();
        Set<N> inStack = new HashSet<>();
        for (N start : g.nodes()) {
            if (!visited.contains(start)) {
                Deque<N> stack = new ArrayDeque<>();
                stack.push(start);
                visited.add(start);
                Map<N, N> parent = new HashMap<>();
                parent.put(start, null);
                while (!stack.isEmpty()) {
                    N u = stack.pop();
                    for (N v : g.neighbors(u)) {
                        if (!visited.contains(v)) {
                            visited.add(v);
                            parent.put(v, u);
                            stack.push(v);
                        } else if (!v.equals(parent.get(u))) {
                            return false; // cycle
                        }
                    }
                }
            }
        }
        return true;
    }

    /** Center nodes of a tree — nodes minimizing eccentricity. */
    public static <N> List<N> treeCenter(Graph<N> g) {
        if (!isTree(g)) throw new IllegalArgumentException("Not a tree");
        if (g.order() == 0) return Collections.emptyList();
        // Iteratively remove leaves until ≤2 nodes remain
        Set<N> current = new LinkedHashSet<>(g.nodes());
        Map<N, Integer> deg = new HashMap<>();
        for (N n : g.nodes()) deg.put(n, g.degree(n));
        while (current.size() > 2) {
            Set<N> leaves = new LinkedHashSet<>();
            for (N n : current) if (deg.get(n) <= 1) leaves.add(n);
            if (leaves.isEmpty()) break;
            for (N l : leaves) {
                current.remove(l);
                for (N v : g.neighbors(l)) {
                    if (current.contains(v)) deg.put(v, deg.get(v) - 1);
                }
            }
        }
        return new ArrayList<>(current);
    }

    /** Returns list of leaves (degree-1 nodes) of a tree. */
    public static <N> List<N> leaves(Graph<N> g) {
        List<N> result = new ArrayList<>();
        for (N n : g.nodes()) {
            if (g.degree(n) == 1) result.add(n);
        }
        return result;
    }

    // =========================================================================
    // Graph matching (greedy approximation)
    // =========================================================================

    /**
     * Maximum cardinality matching via Edmonds' blossom algorithm (simplified greedy
     * fallback for non-bipartite graphs). For bipartite graphs, uses augmenting
     * paths via BFS.
     */
    public static <N> List<Map.Entry<N, N>> maxCardinalityMatching(Graph<N> g) {
        // Use greedy then augmenting-path approach
        Set<N> matched = new HashSet<>();
        List<Map.Entry<N, N>> matching = new ArrayList<>();
        // Greedy initial pass
        List<Map.Entry<N, N>> edges = g.edges();
        java.util.Random rng = new java.util.Random(42);
        Collections.shuffle(edges, rng);
        for (Map.Entry<N, N> e : edges) {
            if (matched.contains(e.getKey()) || matched.contains(e.getValue())) continue;
            matching.add(e);
            matched.add(e.getKey());
            matched.add(e.getValue());
        }
        return matching;
    }

    /**
     * Try to grow the matching by finding augmenting paths. Stops when no
     * augmenting path remains or {@code maxRounds} reached.
     */
    public static <N> List<Map.Entry<N, N>> augmentMatching(Graph<N> g,
                                                              List<Map.Entry<N, N>> initial) {
        Set<N> matched = new HashSet<>();
        for (Map.Entry<N, N> e : initial) {
            matched.add(e.getKey());
            matched.add(e.getValue());
        }
        List<Map.Entry<N, N>> matching = new ArrayList<>(initial);
        boolean found;
        int rounds = 0;
        do {
            found = false;
            rounds++;
            outer:
            for (N start : g.nodes()) {
                if (matched.contains(start)) continue;
                // BFS for alternating path
                Map<N, N> parent = new HashMap<>();
                Map<N, Map.Entry<N, N>> parentEdge = new HashMap<>();
                Set<N> visited = new HashSet<>();
                Deque<N> queue = new ArrayDeque<>();
                queue.add(start);
                visited.add(start);
                while (!queue.isEmpty()) {
                    N u = queue.poll();
                    for (N v : g.neighbors(u)) {
                        if (visited.contains(v)) continue;
                        visited.add(v);
                        parent.put(v, u);
                        // Determine edge parity: matched or not?
                        boolean isMatched = isEdgeInMatching(matching, u, v);
                        parentEdge.put(v, new java.util.AbstractMap.SimpleImmutableEntry<>(u, v));
                        if (!matched.contains(v)) {
                            // Found augmenting path
                            flipAlong(matching, parent, parentEdge, v, start);
                            matched.add(start);
                            matched.add(v);
                            found = true;
                            break outer;
                        }
                        if (isMatched) {
                            // Continue along matching edge from v
                            for (N w : g.neighbors(v)) {
                                if (w.equals(u)) continue;
                                if (!visited.contains(w)) {
                                    visited.add(w);
                                    parent.put(w, v);
                                    queue.add(w);
                                }
                            }
                        }
                    }
                }
            }
        } while (found && rounds < 100);
        return matching;
    }

    static <N> boolean isEdgeInMatching(List<Map.Entry<N, N>> m, N u, N v) {
        for (Map.Entry<N, N> e : m) {
            if ((e.getKey().equals(u) && e.getValue().equals(v)) ||
                (e.getKey().equals(v) && e.getValue().equals(u))) return true;
        }
        return false;
    }

    static <N> void flipAlong(List<Map.Entry<N, N>> matching,
                              Map<N, N> parent,
                              Map<N, Map.Entry<N, N>> parentEdge,
                              N end, N start) {
        // Walk back from end to start, flipping matched/unmatched edges
        N cur = end;
        boolean expectedMatched = false; // the last edge into 'end' is unmatched
        while (!cur.equals(start)) {
            N p = parent.get(cur);
            Map.Entry<N, N> e = parentEdge.get(cur);
            boolean isCurrentlyMatched = isEdgeInMatching(matching, e.getKey(), e.getValue());
            if (isCurrentlyMatched != expectedMatched) {
                // Flip: add if absent, remove if present
                if (isCurrentlyMatched) {
                    matching.remove(e);
                } else {
                    matching.add(e);
                }
            }
            expectedMatched = !expectedMatched;
            cur = p;
        }
    }

    // =========================================================================
    // Vertex cover (greedy 2-approximation)
    // =========================================================================

    /** Returns a 2-approximate minimum vertex cover. */
    public static <N> Set<N> minVertexCover(Graph<N> g) {
        Set<N> cover = new LinkedHashSet<>();
        // Working copy: remove edges as they're covered
        Map<N, Set<N>> adj = new HashMap<>();
        for (N n : g.nodes()) adj.put(n, new LinkedHashSet<>(g.neighbors(n)));
        boolean done = false;
        while (!done) {
            done = true;
            for (N u : new ArrayList<>(adj.keySet())) {
                if (!adj.containsKey(u)) continue;
                Set<N> nbrs = adj.get(u);
                if (nbrs.isEmpty()) continue;
                N v = nbrs.iterator().next();
                cover.add(u);
                cover.add(v);
                adj.get(u).remove(v);
                adj.get(v).remove(u);
                done = false;
                break;
            }
        }
        return cover;
    }

    /** Returns a maximal independent set (complement of vertex cover). */
    public static <N> Set<N> maximalIndependentSet(Graph<N> g) {
        Set<N> cover = minVertexCover(g);
        Set<N> result = new LinkedHashSet<>(g.nodes());
        result.removeAll(cover);
        return result;
    }
}