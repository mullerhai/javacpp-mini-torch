/*
 * GraphX: Drawing Examples Test Harness.
 *
 * Runs each NetworkX-style drawing example, verifies a PNG file is produced
 * with non-zero size, and reports counts.
 *
 * Run via:
 *   java -cp ... org.bytedeco.pytorch.graphx.plot.DrawingExamplesTest
 */
package org.bytedeco.pytorch.graphx.plot;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

public final class DrawingExamplesTest {
    private DrawingExamplesTest() {}
    private static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=== Running GraphX drawing examples test suite ===\n");

        safe("draw_networkx", () -> DrawingExamples.plotDrawNetworkx().getName());
        safe("labels_and_colors", () -> DrawingExamples.plotLabelsAndColors().getName());
        safe("node_colormap", () -> DrawingExamples.plotNodeColormap().getName());
        safe("edge_colormap", () -> DrawingExamples.plotEdgeColormap().getName());
        safe("edge_labels", () -> DrawingExamples.plotEdgeLabels().getName());
        safe("shells", () -> DrawingExamples.plotShells().getName());
        safe("bipartite", () -> DrawingExamples.plotBipartite().getName());
        safe("layout_gallery", () -> DrawingExamples.plotLayoutGallery().getName());
        safe("centrality_community", () -> DrawingExamples.plotCentralityAndCommunity().getName());
        safe("seaborn_darkgrid", () -> DrawingExamples.plotSeabornStyled().getName());
        safe("ggplot_louvain", () -> DrawingExamples.plotGgplotStyled().getName());

        System.out.println("\n=== DrawingExamplesTest ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.err.println("FAILURES: " + failed);
            throw new RuntimeException("DrawingExamplesTest failed");
        }
        System.out.println("All drawing example tests passed.");
    }

    interface ThrowingSupplier<T> { T get() throws Exception; }

    static void safe(String name, ThrowingSupplier<?> fn) {
        try {
            Object r = fn.get();
            if (r == null) { failed++; System.err.println("  FAIL  " + name + " -> null"); return; }
            File f = new File(DrawingExamples.OUT_DIR, r.toString());
            if (!f.exists() || f.length() == 0) {
                failed++; System.err.println("  FAIL  " + name + " -> " + f);
                return;
            }
            passed++;
            System.out.println("  PASS  " + name + " -> " + f.getName() + " (" + f.length() + " bytes)");
        } catch (Throwable t) {
            failed++;
            System.err.println("  FAIL  " + name + " -> " + t.getMessage());
            t.printStackTrace(System.err);
        }
    }
}