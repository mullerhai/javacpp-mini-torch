/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.algorithms.components;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.algorithms.traversal.Traversal;

import java.util.*;

/**
 * Connected components for undirected and directed graphs.
 *
 * <p>Aligned with {@code networkx.algorithms.components.connected},
 * {@code strongly_connected}, {@code weakly_connected}.
 */
public final class ConnectedComponents {
    private ConnectedComponents() {}

    /** Connected components (undirected). Returns list of sets, in BFS-discovery order. */
    public static <N> List<Set<N>> connectedComponents(Graph<N> g) {
        List<Set<N>> result = new ArrayList<>();
        Set<N> visited = new LinkedHashSet<>();
        for (N start : g.nodes()) {
            if (visited.contains(start)) continue;
            Set<N> comp = new LinkedHashSet<>();
            List<N> order = Traversal.bfsOrder(g, start);
            comp.addAll(order);
            visited.addAll(order);
            result.add(comp);
        }
        return result;
    }

    /** Returns a map node -> component label (incrementing from 0). */
    public static <N> Map<N, Integer> connectedComponentLabels(Graph<N> g) {
        Map<N, Integer> labels = new LinkedHashMap<>();
        int label = 0;
        Set<N> visited = new LinkedHashSet<>();
        for (N start : g.nodes()) {
            if (visited.contains(start)) continue;
            List<N> order = Traversal.bfsOrder(g, start);
            for (N n : order) {
                labels.put(n, label);
            }
            visited.addAll(order);
            label++;
        }
        return labels;
    }

    /** Returns whether graph is connected. */
    public static <N> boolean isConnected(Graph<N> g) {
        if (g.isEmpty()) return true;
        List<Set<N>> comps = connectedComponents(g);
        return comps.size() == 1;
    }

    /** Returns number of connected components. */
    public static <N> int numberConnectedComponents(Graph<N> g) {
        return connectedComponents(g).size();
    }

    /** Returns the largest component (by node count). */
    public static <N> Set<N> largestComponent(Graph<N> g) {
        Set<N> largest = new LinkedHashSet<>();
        for (Set<N> c : connectedComponents(g)) {
            if (c.size() > largest.size()) largest = c;
        }
        return largest;
    }

    // ---- Directed ----

    /** Weakly-connected components (treat edges as undirected). */
    public static <N> List<Set<N>> weaklyConnectedComponents(DiGraph<N> g) {
        // Build a temporary undirected shadow and run BFS.
        Graph<N> shadow = new Graph<>();
        for (N n : g.nodes()) shadow.addNode(n);
        for (Map.Entry<N, N> e : g.edges()) shadow.addEdge(e.getKey(), e.getValue());
        return connectedComponents(shadow);
    }

    /** Strongly-connected components via Tarjan's algorithm (single pass, O(V+E)). */
    public static <N> List<Set<N>> stronglyConnectedComponents(DiGraph<N> g) {
        return new TarjanSCC<>(g).compute();
    }

    public static <N> boolean isStronglyConnected(DiGraph<N> g) {
        if (g.isEmpty()) return true;
        return stronglyConnectedComponents(g).size() == 1;
    }

    /** Tarjan's SCC algorithm — iterative, no recursion limit issues. */
    static final class TarjanSCC<N> {
        final DiGraph<N> g;
        final Map<N, Integer> index = new LinkedHashMap<>();
        final Map<N, Integer> lowlink = new LinkedHashMap<>();
        final java.util.Deque<N> stack = new ArrayDeque<>();
        final Set<N> onStack = new LinkedHashSet<>();
        final List<Set<N>> result = new ArrayList<>();
        int counter = 0;

        TarjanSCC(DiGraph<N> g) { this.g = g; }

        List<Set<N>> compute() {
            for (N v : g.nodes()) if (!index.containsKey(v)) strongconnect(v);
            return result;
        }

        void strongconnect(N start) {
            java.util.Deque<Frame<N>> callStack = new ArrayDeque<>();
            callStack.push(new Frame<>(start, g.successors(start).iterator()));
            index.put(start, counter);
            lowlink.put(start, counter);
            counter++;
            stack.push(start);
            onStack.add(start);

            while (!callStack.isEmpty()) {
                Frame<N> f = callStack.peek();
                if (f.it.hasNext()) {
                    N w = f.it.next();
                    if (!index.containsKey(w)) {
                        // Successor w not visited: recurse
                        index.put(w, counter);
                        lowlink.put(w, counter);
                        counter++;
                        stack.push(w);
                        onStack.add(w);
                        callStack.push(new Frame<>(w, g.successors(w).iterator()));
                    } else if (onStack.contains(w)) {
                        // Back edge
                        int newLow = Math.min(lowlink.get(f.node), index.get(w));
                        lowlink.put(f.node, newLow);
                    }
                } else {
                    // All successors processed: pop and possibly form SCC
                    if (lowlink.get(f.node).equals(index.get(f.node))) {
                        Set<N> scc = new LinkedHashSet<>();
                        N x;
                        do {
                            x = stack.pop();
                            onStack.remove(x);
                            scc.add(x);
                        } while (!x.equals(f.node));
                        result.add(scc);
                    }
                    callStack.pop();
                    if (!callStack.isEmpty()) {
                        Frame<N> parent = callStack.peek();
                        int newLow = Math.min(lowlink.get(parent.node), lowlink.get(f.node));
                        lowlink.put(parent.node, newLow);
                    }
                }
            }
        }

        static final class Frame<N> {
            final N node;
            final java.util.Iterator<N> it;
            Frame(N node, java.util.Iterator<N> it) { this.node = node; this.it = it; }
        }
    }
}