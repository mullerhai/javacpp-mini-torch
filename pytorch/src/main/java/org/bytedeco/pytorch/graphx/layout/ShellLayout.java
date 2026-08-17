/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.layout;

import org.bytedeco.pytorch.graphx.core.Graph;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Shell layout — concentric circles.
 *
 * <p>Aligned with {@code networkx.drawing.layout.shell_layout}.
 */
public final class ShellLayout {
    private ShellLayout() {}

    public static <N> Map<N, double[]> compute(Graph<N> g, List<List<N>> shells) {
        Map<N, double[]> pos = new LinkedHashMap<>();
        int nShells = shells.size();
        for (int s = 0; s < nShells; s++) {
            List<N> shell = shells.get(s);
            double radius = s + 1; // shells indexed from 0: innermost first
            int shellSize = shell.size();
            for (int i = 0; i < shellSize; i++) {
                double theta = 2 * Math.PI * i / shellSize;
                pos.put(shell.get(i), new double[]{radius * Math.cos(theta), radius * Math.sin(theta)});
            }
        }
        return pos;
    }

    /** Default: split nodes evenly into {@code nShells} concentric circles. */
    public static <N> Map<N, double[]> compute(Graph<N> g, int nShells) {
        List<N> nodes = g.nodes();
        int n = nodes.size();
        List<List<N>> shells = new ArrayList<>();
        for (int s = 0; s < nShells; s++) {
            List<N> shell = new ArrayList<>();
            int start = (int) Math.round((double) s * n / nShells);
            int end = (int) Math.round((double) (s + 1) * n / nShells);
            for (int i = start; i < end; i++) shell.add(nodes.get(i));
            shells.add(shell);
        }
        return compute(g, shells);
    }
}