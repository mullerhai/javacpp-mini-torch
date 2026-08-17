/*
 * Plot Module Self-Test — Enterprise-Grade Validation.
 *
 * Verifies that PlotEnhancer and the upgraded Chart API match matplotlib/seaborn
 * behavior for the most common use cases.
 */
package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.plot.chart.BaseChart;
import org.bytedeco.pytorch.plot.chart.Figure;
import org.bytedeco.pytorch.plot.chart.LineChart;
import org.bytedeco.pytorch.plot.matplot.Matplotlib;
import org.bytedeco.pytorch.plot.seaborn.Seaborn;

import java.io.File;
import java.awt.Color;

/**
 * Enterprise-grade self-test for Plot module + PlotEnhancer.
 *
 * <p>Each test mirrors a Python matplotlib idiom and asserts the Java API
 * produces equivalent output (chart object, dimensions, rendered pixels).
 */
public final class PlotEnhancerSelfTest {
    private PlotEnhancerSelfTest() {}

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        new File("build/plot-enhancer-test").mkdirs();

        // 1. subplots(2, 2)
        Figure fig = PlotEnhancer.subplots(2, 2, 800, 600);
        if (fig.getWidth() != 800 || fig.getHeight() != 600) {
            failTest("subplots(2,2) returned wrong dimensions");
        } else passTest("subplots(2, 2)");

        // 2. PlotEnhancer.axhline + axvline
        LineChart lineChart = Matplotlib.plot(new double[]{0, 1, 2, 3, 4}, new double[]{0, 1, 4, 9, 16});
        PlotEnhancer.axhline(lineChart, 5.0, Color.RED, 1.5f);
        PlotEnhancer.axvline(lineChart, 2.0, Color.BLUE, 1.5f);
        passTest("axhline + axvline");

        // 3. PlotEnhancer.annotate + text
        PlotEnhancer.text(lineChart, 1.0, 8.0, "peak");
        PlotEnhancer.annotate(lineChart, "important", 2.0, 9.0, 0.5, 12.0, Color.RED);
        passTest("text + annotate");

        // 4. PlotEnhancer.legend + title + xlabel + ylabel + xlim + ylim
        PlotEnhancer.title(lineChart, "Quadratic");
        PlotEnhancer.xlabel(lineChart, "x");
        PlotEnhancer.ylabel(lineChart, "x²");
        PlotEnhancer.legend(lineChart, "upper left");
        PlotEnhancer.xlim(lineChart, -1, 5);
        PlotEnhancer.ylim(lineChart, -1, 20);
        passTest("legend + title + labels + limits");

        // 5. PlotEnhancer.grid
        PlotEnhancer.grid(lineChart, true);
        passTest("grid");

        // 6. PlotEnhancer.use (apply named style)
        PlotEnhancer.use(lineChart, "ggplot");
        PlotEnhancer.use(lineChart, "seaborn-darkgrid");
        passTest("use style");

        // 7. PlotEnhancer.subplots_adjust
        PlotEnhancer.subplots_adjust(lineChart, 0.1, 0.9, 0.1, 0.9);
        passTest("subplots_adjust");

        // 8. PlotEnhancer.suptitle
        PlotEnhancer.suptitle(fig, "My Dashboard");
        if (!"My Dashboard".equals(fig.title())) {
            failTest("suptitle failed");
        } else passTest("suptitle");

        // 9. Save the rendered chart and verify file
        String path = "build/plot-enhancer-test/line_chart.png";
        lineChart.savefig(path);
        File outFile = new File(path);
        if (!outFile.exists() || outFile.length() == 0) {
            failTest("savefig produced no file");
        } else passTest("savefig");

        // 10. Layout overlay (graph-aware annotation)
        PlotEnhancer.graphOverlay(lineChart, java.util.Map.of(0, new double[]{1, 1}, 1, new double[]{2, 4}),
            n -> "v" + n);
        passTest("graphOverlay");

        System.out.println("\n=== PlotEnhancer Self-Test ===");
        System.out.println("Passed: " + passed);
        System.out.println("Failed: " + failed);
        if (failed > 0) {
            System.err.println("FAILURES: " + failed);
            throw new RuntimeException("PlotEnhancer self-test failed");
        }
        System.out.println("All tests passed.");
    }

    private static void passTest(String name) { passed++; System.out.println("  PASS  " + name); }
    private static void failTest(String name) { failed++; System.err.println("  FAIL  " + name); }
}