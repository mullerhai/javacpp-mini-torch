/*
 * GraphX: GraphChart — a BaseChart subclass that natively understands graphs.
 *
 * Renders a graph on a 2-D canvas via AWT, while remaining fully interoperable
 * with the matplotlib/seaborn pipeline:
 *  - holds reference lines, annotations, axis settings, grid, background,
 *    legend, title — just like any BaseChart
 *  - additionally holds node glyphs and edge segments as first-class objects
 *    so we can apply colormaps / centrality / community colors per node/edge
 *  - exports to PNG/JPEG via {@code savefig()}
 *  - composes with other charts via Figure (subplot grids)
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx) matplotlib
 * backend. BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.plot;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.plot.chart.BaseChart;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A chart that holds a graph layout as a first-class object.
 *
 * <p>Used by GraphX's draw_networkx_* family of functions; can also be embedded
 * inside a {@link org.bytedeco.pytorch.plot.chart.Figure} subplot matrix.
 */
public final class GraphChart extends BaseChart {
    private static final long serialVersionUID = 1L;

    private final List<NodeGlyph> nodes = new ArrayList<>();
    private final List<EdgeSegment> edges = new ArrayList<>();
    private boolean directed = false;
    private Color defaultEdgeColor = new Color(0x666666);
    private float defaultEdgeWidth = 1.0f;
    private int defaultNodeRadius = 8;
    private Color defaultNodeColor = new Color(0x4878d0);
    private Color defaultLabelColor = Color.BLACK;
    private boolean showLabels = false;
    private boolean showEdgeLabels = false;
    private Color arrowColor = new Color(0x333333);
    private int labelFontSize = 12;

    /** A drawn node — circle at (x, y) with optional label. */
    public static final class NodeGlyph {
        public final Object id;
        public double x, y;
        public Color color;
        public int radius;
        public String label;
        public Double labelOffsetY; // null = default (-radius-3)
        public NodeGlyph(Object id, double x, double y, Color color, int radius, String label) {
            this.id = id; this.x = x; this.y = y; this.color = color; this.radius = radius;
            this.label = label;
        }
    }

    /** A drawn edge — segment between two node positions. */
    public static final class EdgeSegment {
        public final Object u;
        public final Object v;
        public double x1, y1, x2, y2;
        public Color color;
        public float width;
        public String label;
        public EdgeSegment(Object u, Object v, double x1, double y1, double x2, double y2,
                           Color color, float width, String label) {
            this.u = u; this.v = v;
            this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2;
            this.color = color; this.width = width; this.label = label;
        }
    }

    public GraphChart() { super("Graph Chart"); }
    public GraphChart(int w, int h) { super("Graph Chart"); setSize(w, h); }

    public GraphChart directed(boolean d) { this.directed = d; return this; }
    public GraphChart defaultEdgeColor(Color c) { this.defaultEdgeColor = c; return this; }
    public GraphChart defaultEdgeWidth(float w) { this.defaultEdgeWidth = w; return this; }
    public GraphChart defaultNodeColor(Color c) { this.defaultNodeColor = c; return this; }
    public GraphChart defaultNodeRadius(int r) { this.defaultNodeRadius = r; return this; }
    public GraphChart defaultLabelColor(Color c) { this.defaultLabelColor = c; return this; }
    public GraphChart showLabels(boolean b) { this.showLabels = b; return this; }
    public GraphChart showEdgeLabels(boolean b) { this.showEdgeLabels = b; return this; }
    public GraphChart arrowColor(Color c) { this.arrowColor = c; return this; }
    public GraphChart fontSize(int s) { this.labelFontSize = s; return this; }

    /**
     * Apply matplotlib/seaborn style preset.
     *
     * <p>Mirrors {@code plt.style.use('seaborn-darkgrid')} etc.
     */
    public GraphChart applyStyle(String name) {
        if (name == null) return this;
        switch (name.toLowerCase(java.util.Locale.ROOT)) {
            case "ggplot":
                background = new Color(0xe5e5e5);
                gridColor = new Color(0xffffff);
                showGrid = true;
                break;
            case "seaborn":
            case "seaborn-darkgrid":
                background = new Color(0xEAEAF2);
                gridColor = new Color(0xffffff);
                showGrid = true;
                break;
            case "seaborn-whitegrid":
                background = Color.WHITE;
                gridColor = new Color(0xcccccc);
                showGrid = true;
                break;
            case "seaborn-dark":
                background = new Color(0x22272e);
                gridColor = new Color(0x3d4754);
                defaultLabelColor = new Color(0xeeeeee);
                showGrid = false;
                break;
            case "default":
                background = Color.WHITE;
                gridColor = new Color(0xdddddd);
                showGrid = true;
                break;
            default:
                break;
        }
        return this;
    }

    public List<NodeGlyph> nodeGlyphs() { return nodes; }
    public List<EdgeSegment> edgeSegments() { return edges; }

    /** Add a node glyph. */
    public GraphChart addNode(Object id, double x, double y, Color color, int radius, String label) {
        nodes.add(new NodeGlyph(id, x, y, color, radius, label));
        return this;
    }

    /** Add an edge segment. */
    public GraphChart addEdge(Object u, Object v, double x1, double y1, double x2, double y2,
                                Color color, float width, String label) {
        edges.add(new EdgeSegment(u, v, x1, y1, x2, y2, color, width, label));
        return this;
    }

    /** Compute axis bounds from node positions; auto-pads. */
    public void autoAxis(double padFrac) {
        if (nodes.isEmpty()) return;
        double xMin = Double.POSITIVE_INFINITY, xMax = Double.NEGATIVE_INFINITY;
        double yMin = Double.POSITIVE_INFINITY, yMax = Double.NEGATIVE_INFINITY;
        for (NodeGlyph n : nodes) {
            if (n.x < xMin) xMin = n.x;
            if (n.x > xMax) xMax = n.x;
            if (n.y < yMin) yMin = n.y;
            if (n.y > yMax) yMax = n.y;
        }
        double pad = padFrac * Math.max(xMax - xMin, yMax - yMin);
        if (pad <= 0) pad = 0.5;
        setXLimits(xMin - pad, xMax + pad);
        setYLimits(yMin - pad, yMax + pad);
    }

    @Override
    public BufferedImage render() {
        int w = width, h = height;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = img.createGraphics();
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            // Background
            Color bg = getBackgroundColor();
            g2.setColor(bg);
            g2.fillRect(0, 0, w, h);
            int ml = (int) (marginLeft * w);
            int mr = (int) (marginRight * w);
            int mt = (int) (marginTop * h);
            int mb = (int) (marginBottom * h);
            int plotW = Math.max(10, w - ml - mr);
            int plotH = Math.max(10, h - mt - mb);
            // Determine axis bounds — auto-fit if NaN
            double xMinL = Double.isNaN(xMin) ? 0 : xMin;
            double xMaxL = Double.isNaN(xMax) ? 1 : xMax;
            double yMinL = Double.isNaN(yMin) ? 0 : yMin;
            double yMaxL = Double.isNaN(yMax) ? 1 : yMax;
            double xRange = xMaxL - xMinL, yRange = yMaxL - yMinL;
            if (xRange <= 0) xRange = 1;
            if (yRange <= 0) yRange = 1;
            final double fxMin = xMinL, fxRange = xRange, fyMin = yMinL, fyRange = yRange;
            final int fml = ml, fmt = mt, fplotW = plotW, fplotH = plotH;
            java.util.function.DoubleUnaryOperator toPx = xv -> fml + (xv - fxMin) / fxRange * fplotW;
            java.util.function.DoubleUnaryOperator toPy = yv -> fmt + (1.0 - (yv - fyMin) / fyRange) * fplotH;
            // Grid
            if (showGrid) drawGrid(g2, ml, mt, plotW, plotH);
            // Edges first (so nodes overlay)
            for (EdgeSegment e : edges) {
                g2.setColor(e.color == null ? defaultEdgeColor : e.color);
                g2.setStroke(new BasicStroke(e.width <= 0 ? defaultEdgeWidth : e.width,
                        BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                int x1 = (int) Math.round(toPx.applyAsDouble(e.x1));
                int y1 = (int) Math.round(toPy.applyAsDouble(e.y1));
                int x2 = (int) Math.round(toPx.applyAsDouble(e.x2));
                int y2 = (int) Math.round(toPy.applyAsDouble(e.y2));
                g2.drawLine(x1, y1, x2, y2);
                if (directed) drawArrowHead(g2, x1, y1, x2, y2, arrowColor);
            }
            // Edge labels
            if (showEdgeLabels) {
                g2.setColor(defaultLabelColor);
                g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, labelFontSize));
                for (EdgeSegment e : edges) {
                    if (e.label == null) continue;
                    int mx = (int) ((toPx.applyAsDouble(e.x1) + toPx.applyAsDouble(e.x2)) / 2);
                    int my = (int) ((toPy.applyAsDouble(e.y1) + toPy.applyAsDouble(e.y2)) / 2);
                    FontMetrics fm = g2.getFontMetrics();
                    g2.drawString(e.label, mx - fm.stringWidth(e.label) / 2, my + fm.getAscent() / 2);
                }
            }
            // Nodes
            for (NodeGlyph n : nodes) {
                g2.setColor(n.color == null ? defaultNodeColor : n.color);
                int px = (int) toPx.applyAsDouble(n.x);
                int py = (int) toPy.applyAsDouble(n.y);
                int r = n.radius > 0 ? n.radius : defaultNodeRadius;
                g2.fill(new Ellipse2D.Double(px - r, py - r, r * 2.0, r * 2.0));
                if (n.label != null && showLabels) {
                    g2.setColor(defaultLabelColor);
                    g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, labelFontSize));
                    FontMetrics fm = g2.getFontMetrics();
                    double offY = n.labelOffsetY != null ? n.labelOffsetY : -r - 3;
                    g2.drawString(n.label, px - fm.stringWidth(n.label) / 2,
                            py + (int) offY + fm.getAscent());
                }
            }
            // Title
            String title = title();
            if (title != null && !title.isEmpty()) {
                g2.setColor(Color.BLACK);
                g2.setFont(new Font(Font.SANS_SERIF, Font.BOLD, labelFontSize + 4));
                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(title, (w - fm.stringWidth(title)) / 2, 18);
            }
            // Axis labels
            String xLbl = xAxisLabel(), yLbl = yAxisLabel();
            if (xLbl != null && !xLbl.isEmpty()) {
                g2.setColor(Color.DARK_GRAY);
                g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, labelFontSize));
                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(xLbl, (w - fm.stringWidth(xLbl)) / 2, h - 6);
            }
            if (yLbl != null && !yLbl.isEmpty()) {
                g2.setColor(Color.DARK_GRAY);
                g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, labelFontSize));
                FontMetrics fm = g2.getFontMetrics();
                java.awt.geom.AffineTransform orig = g2.getTransform();
                g2.rotate(-Math.PI / 2);
                g2.drawString(yLbl, -(h + fm.stringWidth(yLbl)) / 2, 12);
                g2.setTransform(orig);
            }
        } finally {
            g2.dispose();
        }
        return img;
    }

    private void drawGrid(Graphics2D g2, int ml, int mt, int plotW, int plotH) {
        Color gridC = gridColor == null ? new Color(0xdddddd) : gridColor;
        g2.setColor(gridC);
        g2.setStroke(new BasicStroke(0.5f));
        int nX = 10, nY = 8;
        for (int i = 1; i < nX; i++) {
            int x = ml + plotW * i / nX;
            g2.drawLine(x, mt, x, mt + plotH);
        }
        for (int i = 1; i < nY; i++) {
            int y = mt + plotH * i / nY;
            g2.drawLine(ml, y, ml + plotW, y);
        }
    }

    private void drawArrowHead(Graphics2D g2, int x1, int y1, int x2, int y2, Color color) {
        int dx = x2 - x1;
        int dy = y2 - y1;
        double len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1) return;
        double ux = dx / len, uy = dy / len;
        int size = 8;
        int baseX = (int) (x2 - ux * size);
        int baseY = (int) (y2 - uy * size);
        double px = -uy, py = ux;
        int lx1 = (int) (baseX + px * size / 2.0);
        int ly1 = (int) (baseY + py * size / 2.0);
        int lx2 = (int) (baseX - px * size / 2.0);
        int ly2 = (int) (baseY - py * size / 2.0);
        g2.setColor(color);
        int[] xs = {x2, lx1, lx2};
        int[] ys = {y2, ly1, ly2};
        g2.fillPolygon(xs, ys, 3);
    }

    // === Color helpers for colormap-style mappings ===

    /** Map a value in [vMin, vMax] to a color using a viridis-like colormap. */
    public static Color viridis(double v, double vMin, double vMax) {
        if (vMax - vMin <= 0) return new Color(0x440154);
        double t = Math.max(0.0, Math.min(1.0, (v - vMin) / (vMax - vMin)));
        // 9-stop viridis approximation
        int[][] stops = {
            {0x44, 0x01, 0x54}, {0x48, 0x1d, 0x6a}, {0x47, 0x37, 0x73},
            {0x3e, 0x4a, 0x89}, {0x31, 0x67, 0x88}, {0x21, 0x85, 0x82},
            {0x29, 0xaf, 0x7f}, {0x86, 0xd3, 0x49}, {0xfd, 0xe7, 0x24}
        };
        double idxF = t * (stops.length - 1);
        int i0 = (int) Math.floor(idxF);
        int i1 = Math.min(stops.length - 1, i0 + 1);
        double f = idxF - i0;
        int r = (int) (stops[i0][0] * (1 - f) + stops[i1][0] * f);
        int gg = (int) (stops[i0][1] * (1 - f) + stops[i1][1] * f);
        int b = (int) (stops[i0][2] * (1 - f) + stops[i1][2] * f);
        return new Color(r, gg, b);
    }

    /** Map a value in [vMin, vMax] to a color using a magma-like colormap. */
    public static Color magma(double v, double vMin, double vMax) {
        if (vMax - vMin <= 0) return new Color(0x00, false);
        double t = Math.max(0.0, Math.min(1.0, (v - vMin) / (vMax - vMin)));
        int[][] stops = {
            {0x00, 0x00, 0x04}, {0x21, 0x0e, 0x4e}, {0x55, 0x1e, 0x7b},
            {0x86, 0x36, 0x8c}, {0xb5, 0x3a, 0x82}, {0xe6, 0x5a, 0x63},
            {0xf8, 0x86, 0x3c}, {0xfc, 0xb5, 0x16}, {0xfc, 0xff, 0xa0}
        };
        double idxF = t * (stops.length - 1);
        int i0 = (int) Math.floor(idxF);
        int i1 = Math.min(stops.length - 1, i0 + 1);
        double f = idxF - i0;
        int r = (int) (stops[i0][0] * (1 - f) + stops[i1][0] * f);
        int gg = (int) (stops[i0][1] * (1 - f) + stops[i1][1] * f);
        int b = (int) (stops[i0][2] * (1 - f) + stops[i1][2] * f);
        return new Color(r, gg, b);
    }

    /** Map a value in [vMin, vMax] to a color using a cool-warm diverging colormap. */
    public static Color coolwarm(double v, double vMin, double vMax) {
        if (vMax - vMin <= 0) return new Color(0x3b, 0x4e, 0xf0);
        double t = Math.max(0.0, Math.min(1.0, (v - vMin) / (vMax - vMin)));
        // blue→white→red
        if (t < 0.5) {
            double f = t * 2;
            int r = (int) (0x3b * (1 - f) + 0xff * f);
            int gg = (int) (0x4e * (1 - f) + 0xff * f);
            int b = (int) (0xf0 * (1 - f) + 0xff * f);
            return new Color(r, gg, b);
        } else {
            double f = (t - 0.5) * 2;
            int r = 0xff;
            int gg = (int) (0xff * (1 - f) + 0x70 * f);
            int b = (int) (0xff * (1 - f) + 0x6b * f);
            return new Color(r, gg, b);
        }
    }

    /** Get a categorical color for a community id (qualitative palette). */
    public static Color category(int i) {
        // Tableau-10 inspired
        Color[] palette = {
            new Color(0x4c, 0x78, 0xa8), new Color(0xf5, 0x85, 0x29),
            new Color(0xe4, 0x57, 0x56), new Color(0x72, 0xb7, 0xb2),
            new Color(0x54, 0xa2, 0x4b), new Color(0xe7, 0x9a, 0x52),
            new Color(0xb2, 0x79, 0xa2), new Color(0xd6, 0x82, 0xb3),
            new Color(0x9e, 0xc7, 0x64), new Color(0xc1, 0xc1, 0xc1)
        };
        return palette[((i % palette.length) + palette.length) % palette.length];
    }
}