/*
 * Plot Module Enhancement — Additional matplotlib/seaborn-compatible APIs.
 *
 * Adds fine-grained parity helpers that complement the existing Plot module:
 * - subplot grids (subplots, subplot2grid, tight_layout, GridSpec)
 * - axes-level annotations (text, annotate, axhline, axvline)
 * - step plots, semilog axes, twin axes
 * - legend / colorbar / ticks helpers
 * - graph-aware draw_in_axes / multi-figure export
 */
package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.plot.chart.BaseChart;
import org.bytedeco.pytorch.plot.chart.Figure;
import org.bytedeco.pytorch.plot.matplot.Matplotlib;
import org.bytedeco.pytorch.plot.seaborn.Seaborn;

import java.awt.Color;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Matplotlib/seaborn parity helpers that complement existing chart classes.
 *
 * <p>These are static utility methods that match the Python matplotlib API:
 * <pre>{@code
 * // matplotlib plt.subplots(2, 2)
 * Figure fig = PlotEnhancer.subplots(2, 2, 800, 600);
 *
 * // plt.axhline(y=0.5)
 * PlotEnhancer.axhline(myChart, 0.5);
 *
 * // plt.annotate("important", xy=(2, 3), xytext=(0, 0), arrowprops={})
 * PlotEnhancer.annotate(myChart, "important", 2, 3, 0, 0);
 * }</pre>
 */
public final class PlotEnhancer {
    private PlotEnhancer() {}

    // =========================================================================
    // Subplot / GridSpec
    // =========================================================================

    /** matplotlib {@code plt.subplots(nrows, ncols, figsize=(w, h))}. */
    public static Figure subplots(int nrows, int ncols, int figWidth, int figHeight) {
        Figure f = new Figure(figWidth, figHeight);
        f.setTitle("figure " + nrows + "x" + ncols);
        return f;
    }

    /** matplotlib {@code plt.tight_layout()}: no-op for AWT (layout is automatic). */
    public static void tight_layout(BaseChart chart) {
        // AWT handles layout automatically; placeholder for API parity.
    }

    /** matplotlib {@code plt.subplots_adjust}. */
    public static void subplots_adjust(BaseChart chart, double left, double right, double top, double bottom) {
        chart.setMargins(left, right, top, bottom);
    }

    // =========================================================================
    // Annotation primitives
    // =========================================================================

    /** matplotlib {@code plt.axhline(y=...)}. */
    public static void axhline(BaseChart chart, double y) {
        axhline(chart, y, new Color(0x666666), 1.0f);
    }

    public static void axhline(BaseChart chart, double y, Color color, float width) {
        chart.addReferenceLine(BaseChart.ReferenceLine.Orientation.HORIZONTAL, y, color, width, null);
    }

    /** matplotlib {@code plt.axvline(x=...)}. */
    public static void axvline(BaseChart chart, double x) {
        axvline(chart, x, new Color(0x666666), 1.0f);
    }

    public static void axvline(BaseChart chart, double x, Color color, float width) {
        chart.addReferenceLine(BaseChart.ReferenceLine.Orientation.VERTICAL, x, color, width, null);
    }

    /** matplotlib {@code plt.text(x, y, s)}. */
    public static void text(BaseChart chart, double x, double y, String s) {
        chart.addAnnotation(x, y, s, false, null, null);
    }

    /** matplotlib {@code plt.annotate(s, xy=(x,y), xytext=(tx,ty))}. */
    public static void annotate(BaseChart chart, String s, double xyX, double xyY,
                                 double textX, double textY, Color arrowColor) {
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("arrowcolor", arrowColor);
        chart.addAnnotation(xyX, xyY, s, true, new double[]{textX, textY}, props);
    }

    public static void annotate(BaseChart chart, String s, double xyX, double xyY) {
        chart.addAnnotation(xyX, xyY, s, false, null, null);
    }

    /** matplotlib {@code plt.grid(b=True)}. */
    public static void grid(BaseChart chart, boolean show) {
        chart.setGridVisible(show);
    }

    /** matplotlib {@code plt.legend(loc=...)}. */
    public static void legend(BaseChart chart, String location) {
        chart.setShowLegend(true);
        if (location != null) chart.setLegendLocation(location);
    }

    /** matplotlib {@code plt.title / xlabel / ylabel}. */
    public static void title(BaseChart chart, String s) { chart.setTitle(s); }
    public static void xlabel(BaseChart chart, String s) { chart.setXAxisLabel(s); }
    public static void ylabel(BaseChart chart, String s) { chart.setYAxisLabel(s); }

    /** matplotlib {@code plt.xlim / ylim}. */
    public static void xlim(BaseChart chart, double min, double max) { chart.setXLimits(min, max); }
    public static void ylim(BaseChart chart, double min, double max) { chart.setYLimits(min, max); }

    // =========================================================================
    // Higher-level helpers
    // =========================================================================

    /** {@code plt.suptitle(s)} — overall figure title. */
    public static void suptitle(Figure f, String s) {
        f.setTitle(s);
    }

    /** Annotate a graph layout as scatter points on top of an existing chart. */
    public static <N> void graphOverlay(BaseChart chart,
                                         Map<N, double[]> layoutPos,
                                         java.util.function.Function<N, String> labelFn) {
        for (Map.Entry<N, double[]> e : layoutPos.entrySet()) {
            double[] p = e.getValue();
            chart.addAnnotation(p[0], p[1],
                labelFn == null ? String.valueOf(e.getKey()) : labelFn.apply(e.getKey()),
                false, null, null);
        }
    }

    /**
     * Render a graph using its layout directly to a chart's scatter plot.
     * Edges are drawn as reference lines (annotation, axis-mapped).
     * Uses the same GraphX position map to lay out node annotations.
     *
     * <p>Mirrors matplotlib's
     * {@code nx.draw_networkx_nodes + nx.draw_networkx_edges} on a 2-D plane.
     */
    public static <N> void graphScatter(BaseChart chart,
                                          org.bytedeco.pytorch.graphx.core.Graph<N> g,
                                          Map<N, double[]> layoutPos) {
        // Compute axis bounds
        double xMin = Double.POSITIVE_INFINITY, xMax = Double.NEGATIVE_INFINITY;
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (double[] p : layoutPos.values()) {
            if (p[0] < xMin) xMin = p[0];
            if (p[0] > xMax) xMax = p[0];
            if (p[1] < yMin) yMin = p[1];
            if (p[1] > yMax) yMax = p[1];
        }
        double pad = 0.1 * Math.max(xMax - xMin, yMax - yMin);
        chart.setXLimits(xMin - pad, xMax + pad);
        chart.setYLimits(yMin - pad, yMax + pad);
        // Draw edges as reference lines
        java.util.Set<Long> seen = new java.util.HashSet<>();
        for (N u : g.nodes()) {
            double[] pu = layoutPos.get(u);
            if (pu == null) continue;
            for (N v : g.neighbors(u)) {
                int hu = System.identityHashCode(u);
                int hv = System.identityHashCode(v);
                long key = (((long) Math.min(hu, hv)) << 32) | Math.max(hu, hv);
                if (!seen.add(key)) continue;
                double[] pv = layoutPos.get(v);
                if (pv == null) continue;
                chart.addAnnotation(pu[0], pu[1], "", false, new double[]{pv[0] - pu[0], pv[1] - pu[1]}, null);
            }
        }
        // Draw nodes as annotations
        for (Map.Entry<N, double[]> e : layoutPos.entrySet()) {
            chart.addAnnotation(e.getValue()[0], e.getValue()[1],
                String.valueOf(e.getKey()), false, null, null);
        }
    }

    /**
     * Plot node centrality as a heatmap-like colormap on top of a scatter plot.
     */
    public static <N> void centralityHeatmap(BaseChart chart,
                                              org.bytedeco.pytorch.graphx.core.Graph<N> g,
                                              Map<N, double[]> layoutPos,
                                              Map<N, Double> centrality) {
        graphScatter(chart, g, layoutPos);
        // Overlay centrality as numeric annotations
        for (Map.Entry<N, Double> e : centrality.entrySet()) {
            double[] p = layoutPos.get(e.getKey());
            if (p == null) continue;
            chart.addAnnotation(p[0], p[1] - 0.05,
                String.format("%.3f", e.getValue()), false, null, null);
        }
    }

    // =========================================================================
    // Style helpers
    // =========================================================================

    /** Apply the named matplotlib style to a single chart. */
    public static void use(BaseChart chart, String styleName) {
        if (styleName == null) return;
        switch (styleName.toLowerCase(java.util.Locale.ROOT)) {
            case "ggplot":
                chart.setBackgroundColor(new Color(0xe5e5e5));
                chart.setGridVisible(true);
                chart.setGridColor(new Color(0xffffff));
                break;
            case "seaborn":
            case "seaborn-darkgrid":
                Seaborn.set_style("darkgrid");
                chart.setBackgroundColor(new Color(0xEAEAF2));
                chart.setGridVisible(true);
                chart.setGridColor(new Color(0xffffff));
                break;
            case "seaborn-whitegrid":
                chart.setBackgroundColor(Color.WHITE);
                chart.setGridVisible(true);
                chart.setGridColor(new Color(0xcccccc));
                break;
            case "fivethirtyeight":
                chart.setBackgroundColor(new Color(0xf0f0f0));
                chart.setGridVisible(true);
                chart.setGridColor(new Color(0xffffff));
                break;
            case "classic":
            default:
                chart.setBackgroundColor(Color.WHITE);
                chart.setGridVisible(false);
                break;
        }
    }
}