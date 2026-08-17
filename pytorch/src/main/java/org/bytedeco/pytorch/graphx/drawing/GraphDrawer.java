/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.drawing;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.layout.Layout;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

/**
 * Graph drawing utility — renders a graph to a {@link BufferedImage}.
 *
 * <p>Mirrors {@code networkx.drawing.nx_pylab.draw} / {@code draw_networkx_*}.
 */
public final class GraphDrawer {
    private GraphDrawer() {}

    /** Default drawing config. */
    public static class Config {
        public int width = 800;
        public int height = 600;
        public Color background = Color.WHITE;
        public Color nodeColor = new Color(0x4878d0);
        public Color edgeColor = new Color(0x666666);
        public Color labelColor = Color.BLACK;
        public Color arrowColor = new Color(0x333333);
        public int nodeSize = 300;            // pixels (area-ish)
        public int nodeRadius = 10;           // direct pixel radius
        public int edgeWidth = 1;
        public int fontSize = 12;
        public boolean withLabels = false;
        public boolean withEdgeLabels = false;
        public boolean directed = false;
        public double margin = 0.05;          // padding around content

        public Config directed(boolean d) { this.directed = d; return this; }
        public Config width(int w) { this.width = w; return this; }
        public Config height(int h) { this.height = h; return this; }
        public Config nodeColor(Color c) { this.nodeColor = c; return this; }
        public Config edgeColor(Color c) { this.edgeColor = c; return this; }
        public Config nodeSize(int s) { this.nodeRadius = Math.max(2, (int) Math.sqrt(s)); return this; }
        public Config edgeWidth(int w) { this.edgeWidth = w; return this; }
        public Config withLabels(boolean b) { this.withLabels = b; return this; }
        public Config withEdgeLabels(boolean b) { this.withEdgeLabels = b; return this; }
        public Config background(Color c) { this.background = c; return this; }
        public Config fontSize(int s) { this.fontSize = s; return this; }
        public Config margin(double m) { this.margin = m; return this; }
    }

    /** Default configuration. */
    public static Config defaults() {
        return new Config();
    }

    /** Draw {@code g} with spring layout. */
    public static <N> BufferedImage draw(Graph<N> g) {
        return draw(g, Layout.spring(g), defaults());
    }

    public static <N> BufferedImage draw(Graph<N> g, Map<N, double[]> pos) {
        return draw(g, pos, defaults());
    }

    public static <N> BufferedImage draw(Graph<N> g, Map<N, double[]> pos, Config cfg) {
        BufferedImage img = new BufferedImage(cfg.width, cfg.height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = img.createGraphics();
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g2.setColor(cfg.background);
            g2.fillRect(0, 0, cfg.width, cfg.height);
            // Compute scale to map positions to canvas
            double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
            double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
            for (double[] p : pos.values()) {
                if (p[0] < minX) minX = p[0];
                if (p[0] > maxX) maxX = p[0];
                if (p[1] < minY) minY = p[1];
                if (p[1] > maxY) maxY = p[1];
            }
            double sx = maxX - minX, sy = maxY - minY;
            if (sx < 1e-9) sx = 1;
            if (sy < 1e-9) sy = 1;
            double margin = cfg.margin;
            double availW = cfg.width * (1.0 - 2 * margin);
            double availH = cfg.height * (1.0 - 2 * margin);
            double scale = Math.min(availW / sx, availH / sy);
            double offsetX = cfg.width * margin - minX * scale;
            double offsetY = cfg.height * margin - minY * scale;

            // Draw edges first
            g2.setStroke(new BasicStroke(cfg.edgeWidth));
            g2.setColor(cfg.edgeColor);
            java.util.Set<Long> seen = new java.util.HashSet<>();
            for (Map.Entry<N, N> e : g.edges()) {
                N u = e.getKey(), v = e.getValue();
                long key = org.bytedeco.pytorch.graphx.core.Graph.<N>pairHash(u, v);
                if (!seen.add(key)) continue;
                double[] pu = pos.get(u);
                double[] pv = pos.get(v);
                if (pu == null || pv == null) continue;
                int x1 = (int) (offsetX + pu[0] * scale);
                int y1 = (int) (offsetY + pu[1] * scale);
                int x2 = (int) (offsetX + pv[0] * scale);
                int y2 = (int) (offsetY + pv[1] * scale);
                // Trim line so it doesn't overlap the node circle
                double dx = x2 - x1, dy = y2 - y1;
                double len = Math.sqrt(dx * dx + dy * dy);
                if (len < 1e-6) continue;
                int tx1 = x1 + (int) (dx * cfg.nodeRadius / len);
                int ty1 = y1 + (int) (dy * cfg.nodeRadius / len);
                int tx2 = x2 - (int) (dx * cfg.nodeRadius / len);
                int ty2 = y2 - (int) (dy * cfg.nodeRadius / len);
                g2.drawLine(tx1, ty1, tx2, ty2);
                if (cfg.directed) {
                    // Draw arrowhead at (tx2, ty2)
                    double angle = Math.atan2(dy, dx);
                    int ah = 8;
                    int ax1 = (int) (tx2 - ah * Math.cos(angle - Math.PI / 6));
                    int ay1 = (int) (ty2 - ah * Math.sin(angle - Math.PI / 6));
                    int ax2 = (int) (tx2 - ah * Math.cos(angle + Math.PI / 6));
                    int ay2 = (int) (ty2 - ah * Math.sin(angle + Math.PI / 6));
                    g2.setColor(cfg.arrowColor);
                    int[] xPoints = {tx2, ax1, ax2};
                    int[] yPoints = {ty2, ay1, ay2};
                    g2.fillPolygon(xPoints, yPoints, 3);
                    g2.setColor(cfg.edgeColor);
                }
            }

            // Draw nodes
            g2.setColor(cfg.nodeColor);
            for (Map.Entry<N, double[]> e : pos.entrySet()) {
                double[] p = e.getValue();
                int cx = (int) (offsetX + p[0] * scale);
                int cy = (int) (offsetY + p[1] * scale);
                g2.fillOval(cx - cfg.nodeRadius, cy - cfg.nodeRadius, 2 * cfg.nodeRadius, 2 * cfg.nodeRadius);
                if (cfg.withLabels) {
                    g2.setColor(cfg.labelColor);
                    g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, cfg.fontSize));
                    String label = String.valueOf(e.getKey());
                    int tx = cx + cfg.nodeRadius + 2;
                    int ty = cy + cfg.fontSize / 2;
                    g2.drawString(label, tx, ty);
                    g2.setColor(cfg.nodeColor);
                }
            }

            // Edge labels
            if (cfg.withEdgeLabels) {
                g2.setColor(cfg.labelColor);
                g2.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, cfg.fontSize));
                for (Map.Entry<N, N> e : g.edges()) {
                    double[] pu = pos.get(e.getKey());
                    double[] pv = pos.get(e.getValue());
                    if (pu == null || pv == null) continue;
                    int mx = (int) (offsetX + (pu[0] + pv[0]) / 2 * scale);
                    int my = (int) (offsetY + (pu[1] + pv[1]) / 2 * scale);
                    String w = String.valueOf(g.getEdgeWeight(e.getKey(), e.getValue()));
                    g2.drawString(w, mx, my);
                }
            }
        } finally {
            g2.dispose();
        }
        return img;
    }

    /** Save the rendered graph to {@code path} (PNG/JPG based on extension). */
    public static <N> void savefig(Graph<N> g, String path) throws Exception {
        savefig(g, Layout.spring(g), path, defaults());
    }

    public static <N> void savefig(Graph<N> g, Map<N, double[]> pos, String path, Config cfg) throws Exception {
        BufferedImage img = draw(g, pos, cfg);
        File f = new File(path);
        if (f.getParentFile() != null) f.getParentFile().mkdirs();
        String lower = path.toLowerCase();
        String fmt = lower.endsWith(".jpg") || lower.endsWith(".jpeg") ? "jpg" : "png";
        ImageIO.write(img, fmt, f);
    }

    /** Draw with a specific layout kind. */
    public static <N> BufferedImage drawWith(Graph<N> g, Layout.Kind kind, Object... params) {
        Map<N, double[]> pos = Layout.compute(kind, g, params);
        Config cfg = defaults();
        if (g.isDirected()) cfg.directed = true;
        return draw(g, pos, cfg);
    }
}