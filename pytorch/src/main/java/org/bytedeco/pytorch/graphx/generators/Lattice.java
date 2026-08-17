/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.generators;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Lattice / grid graph generators.
 *
 * <p>Aligned with {@code networkx.generators.lattice}.
 */
public final class Lattice {
    private Lattice() {}

    /** 2D grid with {@code m} rows and {@code n} columns. Node id = r*n + c. */
    public static Graph<Integer> grid2dGraph(int m, int n) {
        return grid2dGraph(m, n, false, false);
    }

    public static Graph<Integer> grid2dGraph(int m, int n, boolean periodic, boolean diagonal) {
        Graph<Integer> g = new Graph<>();
        for (int i = 0; i < m * n; i++) g.addNode(i);
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                int id = r * n + c;
                int right = r * n + ((c + 1) % n);
                int down = ((r + 1) % m) * n + c;
                if (c + 1 < n || periodic) g.addEdge(id, right);
                if (r + 1 < m || periodic) g.addEdge(id, down);
                if (diagonal && r + 1 < m && c + 1 < n) {
                    int dr = (r + 1) * n + (c + 1);
                    g.addEdge(id, dr);
                }
                if (diagonal && r + 1 < m && c - 1 >= 0) {
                    int dl = (r + 1) * n + (c - 1);
                    g.addEdge(id, dl);
                }
            }
        }
        return g;
    }

    /** N-dimensional grid graph. Node id = linear index. */
    public static Graph<Integer> gridGraph(int[] dims) {
        return gridGraph(dims, false);
    }

    public static Graph<Integer> gridGraph(int[] dims, boolean periodic) {
        Graph<Integer> g = new Graph<>();
        int total = 1;
        for (int d : dims) total *= d;
        for (int i = 0; i < total; i++) g.addNode(i);
        int[] dimsCopy = dims.clone();
        for (int idx = 0; idx < total; idx++) {
            int[] coords = unravelIndex(idx, dims);
            for (int axis = 0; axis < dims.length; axis++) {
                if (coords[axis] + 1 < dimsCopy[axis] || periodic) {
                    int next = (int) (idx + stride(dims, axis));
                    if (next > idx) g.addEdge(idx, next);
                }
            }
        }
        return g;
    }

    static int stride(int[] dims, int axis) {
        int s = 1;
        for (int i = 0; i < axis; i++) s *= dims[i];
        return s;
    }

    static int[] unravelIndex(int idx, int[] dims) {
        int[] coords = new int[dims.length];
        int rem = idx;
        for (int i = dims.length - 1; i >= 0; i--) {
            coords[i] = rem % dims[i];
            rem /= dims[i];
        }
        return coords;
    }

    /** N-dimensional hypercube graph. */
    public static Graph<Integer> hypercubeGraph(int n) {
        Graph<Integer> g = new Graph<>();
        int total = 1 << n;
        for (int i = 0; i < total; i++) g.addNode(i);
        for (int i = 0; i < total; i++) {
            for (int bit = 0; bit < n; bit++) {
                int j = i ^ (1 << bit);
                if (i < j) g.addEdge(i, j);
            }
        }
        return g;
    }

    /** Hexagonal lattice graph (default: m=2, n=3 — 6 nodes, 7 edges). */
    public static Graph<int[]> hexagonalLatticeGraph(int m, int n) {
        Graph<int[]> g = new Graph<>();
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                int[] node = {r, c};
                g.addNode(node);
                if (c > 0) g.addEdge(node, new int[]{r, c - 1});
                if (r > 0) g.addEdge(node, new int[]{r - 1, c});
            }
        }
        return g;
    }

    /** Triangular lattice graph. */
    public static Graph<Integer> triangularLatticeGraph(int m, int n) {
        Graph<Integer> g = grid2dGraph(m, n, false, true);
        // Already has diagonals
        return g;
    }
}