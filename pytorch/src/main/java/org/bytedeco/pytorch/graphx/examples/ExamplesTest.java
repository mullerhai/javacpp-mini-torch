/*
 * GraphX: End-to-end Example Test Harness.
 *
 * Runs every ported NetworkX example + DataFrame integration + I/O round-trip.
 * Used as the validation entry point. Each example must call assertions and
 * produce a visualization file.
 *
 * Run via:
 *   java -cp ... org.bytedeco.pytorch.graphx.examples.ExamplesTest
 */
package org.bytedeco.pytorch.graphx.examples;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.graphx.examples.Examples;
import org.bytedeco.pytorch.graphx.GraphX;
import org.bytedeco.pytorch.graphx.algorithms.flow.MaxFlow;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.core.Graph;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class ExamplesTest {
    private ExamplesTest() {}
    private static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        new File(Examples.OUT_DIR).mkdirs();
        System.out.println("=== Running GraphX examples test suite ===\n");

        // ---- Original 15 ----
        safe("simple_graph", () -> Examples.plotSimpleGraph().getName());
        safe("properties", () -> Examples.plotProperties().length());
        safe("read_write", () -> Examples.plotReadWrite().order());
        safe("dijkstra", () -> Examples.plotDijkstra().length);
        safe("betweenness", () -> Examples.plotBetweennessCentrality().entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey());
        safe("scc", () -> Examples.plotStronglyConnected().size());
        safe("spring_layout", () -> Examples.plotSpringLayout().getName());
        safe("circular_layout", () -> Examples.plotCircular().getName());
        safe("node_colormap", () -> Examples.plotNodeColormap().size());
        safe("edge_colormap", () -> Examples.plotEdgeColormap().getName());
        safe("karate", () -> Examples.plotKarateClub());
        safe("erdos_renyi", () -> Examples.plotErdosRenyi().getName());
        safe("mst", () -> Examples.plotMST().getName());
        safe("astar", () -> Examples.plotAStar().length);
        safe("find_path", () -> Examples.plotFindShortestPath().size());

        // ---- Community detection ----
        safe("girvan_newman", () -> Examples.plotGirvanNewman().size());
        safe("label_prop", () -> Examples.plotLabelPropagation().size());
        safe("louvain", () -> Examples.plotLouvain().size());

        // ---- Force-directed layouts ----
        safe("forceatlas2", () -> Examples.plotForceAtlas2().getName());
        safe("arf", () -> Examples.plotARFLayout().getName());

        // ---- I/O round-trips ----
        safe("edgelist_rt", () -> Examples.plotEdgeListRoundTrip().order());
        safe("graphml_rt", () -> Examples.plotGraphMLRoundTrip().order());
        safe("json_rt", () -> Examples.plotJsonRoundTrip().order());

        System.out.println("\n=== ExamplesTest ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.err.println("FAILURES: " + failed);
            throw new RuntimeException("ExamplesTest failed");
        }
        System.out.println("All example tests passed.");
    }

    interface ThrowingSupplier<T> { T get() throws Exception; }

    static void safe(String name, ThrowingSupplier<?> fn) {
        try {
            Object r = fn.get();
            passed++;
            System.out.println("  PASS  " + name + " -> " + (r == null ? "null" : r.toString()));
        } catch (Throwable t) {
            failed++;
            System.err.println("  FAIL  " + name + " -> " + t.getMessage());
            t.printStackTrace(System.err);
        }
    }
}