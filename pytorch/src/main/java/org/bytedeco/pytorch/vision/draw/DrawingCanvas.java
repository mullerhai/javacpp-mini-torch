/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 */
package org.bytedeco.pytorch.vision.draw;

import java.awt.AlphaComposite;
import java.awt.BasicStroke;
import java.awt.Composite;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.font.FontRenderContext;
import java.awt.font.LineBreakMeasurer;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Enterprise drawing surface for the {@link org.bytedeco.pytorch.vision.draw}
 * module.
 *
 * <p>This is the single class every higher-level helper (2D primitives, text,
 * 3D, charts, overlays) delegates to. It maintains a complete state machine
 * (pen, brush, font, transform, clipping, opacity, composition rules) and
 * emits AWT {@link Graphics2D} calls on the underlying {@link BufferedImage}.
 *
 * <p>Typical usage:
 *
 * <pre>{@code
 * BufferedImage img = new BufferedImage(800, 600, BufferedImage.TYPE_INT_ARGB);
 * try (DrawingCanvas c = DrawingCanvas.on(img)) {
 *     c.clear(DColor.of("#202428"));
 *     c.setPen(DPen.solid(2, DColor.of("#ffcc00")));
 *     c.setBrush(DBrush.linearGradient(0, 0, 800, 600, DColor.of("#5f9"), DColor.of("#39c")));
 *     c.fillRoundedRect(40, 40, 720, 520, 18);
 *     c.setFont(DFont.sans(28).bold());
 *     c.text("Enterprise Drawing", 60, 110, DColor.of("white"));
 * }
 * ImageIO.write(img, "png", new File("out.png"));
 * }</pre>
 *
 * <p>The class is not thread-safe — one {@code DrawingCanvas} per rendering
 * thread, the way AWT / Swing want it.
 */
public final class DrawingCanvas implements AutoCloseable {

    private final BufferedImage image;
    private final Graphics2D g2d;
    private final FontRenderContext frc;

    // ── State ────────────────────────────────────────────────────────────
    private DPen pen = DPen.solid(1f, DColor.of(0, 0, 0));
    private DBrush brush = DBrush.solid(DColor.of(0, 0, 0));
    private DFont font = DFont.sans(14f);
    private DColor fillColor = DColor.of(0, 0, 0);
    private DColor strokeColor = DColor.of(0, 0, 0);
    private Composite composite = AlphaComposite.SrcOver;
    private boolean antiAlias = true;
    private boolean closed;

    // ── Construction ─────────────────────────────────────────────────────

    /** Wrap an existing image; modifications are written immediately. */
    public static DrawingCanvas on(BufferedImage img) {
        Objects.requireNonNull(img, "image");
        return new DrawingCanvas(img);
    }

    /** Create a fresh ARGB image and return its drawing canvas. */
    public static DrawingCanvas create(int width, int height) {
        return create(width, height, BufferedImage.TYPE_INT_ARGB);
    }

    public static DrawingCanvas create(int width, int height, int imageType) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("width/height must be > 0");
        return new DrawingCanvas(new BufferedImage(width, height, imageType));
    }

    private DrawingCanvas(BufferedImage img) {
        this.image = img;
        this.g2d = img.createGraphics();
        this.frc = g2d.getFontRenderContext();
        applyHints();
        applyTransform(DTransform.identity());
    }

    // ── Lifecycle ────────────────────────────────────────────────────────

    public BufferedImage image() { return image; }

    public Graphics2D raw() { return g2d; }

    public int width() { return image.getWidth(); }

    public int height() { return image.getHeight(); }

    public DPoint center() { return DPoint.of(width() * 0.5f, height() * 0.5f); }

    @Override public void close() {
        if (closed) return;
        closed = true;
        g2d.dispose();
    }

    /** Force flush of any pending ops (no-op in AWT but useful for testing). */
    public void flush() {
        g2d.dispose();
        // Reacquire? AWT doesn't really need this; included for API symmetry with
        // DirectX/Cairo-like renderers a future implementation may target.
    }

    // ── State accessors / mutators ───────────────────────────────────────

    public DPen pen() { return pen; }
    public DrawingCanvas setPen(DPen pen) { this.pen = pen; return this; }

    public DBrush brush() { return brush; }
    public DrawingCanvas setBrush(DBrush brush) { this.brush = brush; return this; }

    public DFont font() { return font; }
    public DrawingCanvas setFont(DFont f) { this.font = f; return this; }

    public DColor fillColor() { return fillColor; }
    public DrawingCanvas setFillColor(DColor c) { this.fillColor = c; this.brush = DBrush.solid(c); return this; }

    public DColor strokeColor() { return strokeColor; }
    public DrawingCanvas setStrokeColor(DColor c) { this.strokeColor = c; return this; }

    public boolean antiAlias() { return antiAlias; }
    public DrawingCanvas setAntiAlias(boolean on) {
        this.antiAlias = on;
        applyHints();
        return this;
    }

    public Composite composite() { return composite; }
    public DrawingCanvas setComposite(Composite c) {
        this.composite = c;
        g2d.setComposite(c);
        return this;
    }

    /** Shortcut for alpha-only composites — commonly used for overlays. */
    public DrawingCanvas setOpacity(float alpha) {
        if (alpha < 0f) alpha = 0f;
        if (alpha > 1f) alpha = 1f;
        return setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
    }

    public DTransform transform() { return DTransform.fromAwt(g2d.getTransform()); }

    public DrawingCanvas setTransform(DTransform t) {
        applyTransform(t);
        return this;
    }

    public DrawingCanvas concatTransform(DTransform t) {
        return setTransform(transform().compose(t));
    }

    public DrawingCanvas translate(double tx, double ty) { return concatTransform(DTransform.translate(tx, ty)); }
    public DrawingCanvas scale(double sx, double sy) { return concatTransform(DTransform.scale(sx, sy)); }
    public DrawingCanvas rotate(double radians) { return concatTransform(DTransform.rotate(radians)); }

    private void applyTransform(DTransform t) {
        g2d.setTransform(t.toAwt());
    }

    private void applyHints() {
        if (antiAlias) {
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        } else {
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
            g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_OFF);
        }
        g2d.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
    }

    // ── Clearing ─────────────────────────────────────────────────────────

    public DrawingCanvas clear() { return clear(DColor.of(0, 0, 0, 0)); }
    public DrawingCanvas clear(DColor color) {
        Composite prev = g2d.getComposite();
        g2d.setComposite(AlphaComposite.Src);
        g2d.setPaint(color.toAwt());
        g2d.fillRect(0, 0, width(), height());
        g2d.setComposite(prev);
        return this;
    }

    // ── Clipping ─────────────────────────────────────────────────────────

    public DrawingCanvas clipRect(DRect r) {
        g2d.clipRect(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height));
        return this;
    }

    public DrawingCanvas clipShape(Shape s) {
        g2d.clip(s);
        return this;
    }

    public DrawingCanvas resetClip() {
        g2d.setClip(null);
        return this;
    }

    // ── Primitives ───────────────────────────────────────────────────────

    public DrawingCanvas drawLine(float x1, float y1, float x2, float y2) {
        applyPenStroke();
        g2d.drawLine(Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2));
        return this;
    }

    public DrawingCanvas drawLine(DPoint a, DPoint b) { return drawLine(a.x, a.y, b.x, b.y); }

    public DrawingCanvas drawRect(DRect r) {
        applyPenStroke();
        g2d.drawRect(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height));
        return this;
    }

    public DrawingCanvas fillRect(DRect r) {
        applyFillPaint();
        g2d.fillRect(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height));
        return this;
    }

    public DrawingCanvas drawRoundedRect(DRect r, float radius) {
        applyPenStroke();
        g2d.drawRoundRect(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height),
                Math.round(radius * 2), Math.round(radius * 2));
        return this;
    }

    public DrawingCanvas fillRoundedRect(DRect r, float radius) {
        applyFillPaint();
        g2d.fillRoundRect(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height),
                Math.round(radius * 2), Math.round(radius * 2));
        return this;
    }

    public DrawingCanvas drawEllipse(DEllipse e) {
        applyPenStroke();
        float x = e.center.x - e.rx, y = e.center.y - e.ry, w = 2 * e.rx, h = 2 * e.ry;
        g2d.draw(new java.awt.geom.Ellipse2D.Float(x, y, w, h));
        return this;
    }

    public DrawingCanvas fillEllipse(DEllipse e) {
        applyFillPaint();
        float x = e.center.x - e.rx, y = e.center.y - e.ry, w = 2 * e.rx, h = 2 * e.ry;
        g2d.fill(new java.awt.geom.Ellipse2D.Float(x, y, w, h));
        return this;
    }

    public DrawingCanvas drawArc(DRect r, double startDeg, double sweepDeg) {
        applyPenStroke();
        g2d.drawArc(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height),
                (int) Math.round(startDeg), (int) Math.round(sweepDeg));
        return this;
    }

    public DrawingCanvas fillArc(DRect r, double startDeg, double sweepDeg) {
        applyFillPaint();
        g2d.fillArc(Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height),
                (int) Math.round(startDeg), (int) Math.round(sweepDeg));
        return this;
    }

    /** Draw a closed polygon path. */
    public DrawingCanvas drawPolygon(DPoint... pts) {
        applyPenStroke();
        int n = pts.length;
        int[] xs = new int[n]; int[] ys = new int[n];
        for (int i = 0; i < n; i++) { xs[i] = Math.round(pts[i].x); ys[i] = Math.round(pts[i].y); }
        g2d.drawPolygon(xs, ys, n);
        return this;
    }

    public DrawingCanvas fillPolygon(DPoint... pts) {
        applyFillPaint();
        int n = pts.length;
        int[] xs = new int[n]; int[] ys = new int[n];
        for (int i = 0; i < n; i++) { xs[i] = Math.round(pts[i].x); ys[i] = Math.round(pts[i].y); }
        g2d.fillPolygon(xs, ys, n);
        return this;
    }

    public DrawingCanvas drawPath(DPath p) {
        applyPenStroke();
        g2d.draw(p.toAwt());
        return this;
    }

    public DrawingCanvas fillPath(DPath p) {
        applyFillPaint();
        g2d.fill(p.toAwt());
        return this;
    }

    public DrawingCanvas stroke(Shape s) { applyPenStroke(); g2d.draw(s); return this; }
    public DrawingCanvas fill(Shape s) { applyFillPaint(); g2d.fill(s); return this; }

    private void applyPenStroke() {
        g2d.setStroke(pen.toAwt());
        g2d.setPaint(strokeColor.toAwt());
    }

    private void applyFillPaint() {
        g2d.setPaint(brush.toAwt());
    }

    // ── Images / textures ────────────────────────────────────────────────

    public DrawingCanvas drawImage(BufferedImage src, float x, float y) {
        g2d.drawImage(src, Math.round(x), Math.round(y), null);
        return this;
    }

    public DrawingCanvas drawImage(BufferedImage src, DRect dst) {
        g2d.drawImage(src, Math.round(dst.x), Math.round(dst.y), Math.round(dst.width), Math.round(dst.height), null);
        return this;
    }

    public DrawingCanvas drawImage(BufferedImage src, DRect dst, DRect srcRect) {
        java.awt.geom.Rectangle2D.Float s = srcRect.toAwt();
        g2d.drawImage(src,
                Math.round(dst.x), Math.round(dst.y), Math.round(dst.x2()), Math.round(dst.y2()),
                Math.round(s.x), Math.round(s.y), Math.round(s.x + s.width), Math.round(s.y + s.height),
                null);
        return this;
    }

    // ── Text ─────────────────────────────────────────────────────────────

    /** Draw a single-line text at baseline (x, y). */
    public DrawingCanvas text(String s, float x, float y) { return text(s, x, y, strokeColor); }
    public DrawingCanvas text(String s, float x, float y, DColor color) {
        if (s == null || s.isEmpty()) return this;
        java.awt.Font awt = font.toAwt();
        g2d.setFont(awt);
        g2d.setPaint(color.toAwt());
        g2d.drawString(s, x, y);
        return this;
    }

    /** Draw text along a polyline baseline (one drawString per segment using rotated glyphs). */
    public DrawingCanvas textAlongPath(String s, DPoint a, DPoint b) {
        if (s == null || s.isEmpty()) return this;
        double angle = Math.atan2(b.y - a.y, b.x - a.x);
        AffineTransform saved = g2d.getTransform();
        g2d.translate(a.x, a.y);
        g2d.rotate(angle);
        g2d.setFont(font.toAwt());
        g2d.setPaint(strokeColor.toAwt());
        g2d.drawString(s, 0, 0);
        g2d.setTransform(saved);
        return this;
    }

    /** Word-wrapped text block inside {@code rect}; returns total height drawn. */
    public float textBlock(String text, DRect rect, int align) {
        if (text == null || text.isEmpty()) return 0f;
        java.awt.Font awt = font.toAwt();
        g2d.setFont(awt);
        g2d.setPaint(strokeColor.toAwt());
        AttributedString as = new AttributedString(text);
        as.addAttribute(TextAttribute.FONT, awt);
        AttributedCharacterIterator it = as.getIterator();
        LineBreakMeasurer lbm = new LineBreakMeasurer(it, frc);
        float y = rect.y + font.ascent(frc);
        float maxWidth = rect.width;
        int paraStart = it.getBeginIndex();
        int paraEnd = it.getEndIndex();
        lbm.setPosition(paraStart);
        while (lbm.getPosition() < paraEnd) {
            int next = lbm.nextOffset(maxWidth);
            int limit = next;
            if (limit < paraEnd) {
                int lastSpace = text.lastIndexOf(' ', limit - 1);
                if (lastSpace >= paraStart) limit = lastSpace + 1;
            }
            TextLayout layout = lbm.nextLayout(maxWidth, limit, false);
            float x = rect.x;
            if (align == 1) x += (maxWidth - layout.getAdvance()) / 2;
            else if (align == 2) x += (maxWidth - layout.getAdvance());
            layout.draw(g2d, x, y);
            y += layout.getAscent() + layout.getDescent() + layout.getLeading();
        }
        return y - rect.y;
    }

    /** Returns the measured size of {@code text} in the current font. */
    public DRect measure(String text) {
        Rectangle2D b = font.toAwt().getStringBounds(text, frc);
        return new DRect((float) b.getX(), (float) b.getY(), (float) b.getWidth(), (float) b.getHeight());
    }

    /** Returns the layout rectangle for a wrapped block. */
    public DRect measureBlock(String text, float maxWidth) {
        AttributedString as = new AttributedString(text);
        as.addAttribute(TextAttribute.FONT, font.toAwt());
        AttributedCharacterIterator it = as.getIterator();
        LineBreakMeasurer lbm = new LineBreakMeasurer(it, frc);
        float y = font.ascent(frc);
        int pos = it.getBeginIndex();
        int end = it.getEndIndex();
        lbm.setPosition(pos);
        float maxX = 0f;
        while (lbm.getPosition() < end) {
            int next = lbm.nextOffset(maxWidth);
            int limit = next < end ? next : end;
            TextLayout layout = lbm.nextLayout(maxWidth, limit, false);
            if (layout.getAdvance() > maxX) maxX = layout.getAdvance();
            y += layout.getAscent() + layout.getDescent() + layout.getLeading();
            pos = limit;
        }
        return new DRect(0, 0, maxX, y);
    }

    // ---- pixel access -----------------------------------------------------

    /** Read ARGB at {@code (x, y)}. */
    public int getRGB(int x, int y) { return image.getRGB(x, y); }

    /** Set ARGB at {@code (x, y)}. */
    public DrawingCanvas setRGB(int x, int y, int argb) {
        image.setRGB(x, y, argb);
        return this;
    }

    /** Bulk write an ARGB row range. */
    public DrawingCanvas setRGB(int x, int y, int w, int h, int[] rgb) {
        if (rgb.length < w * h) throw new IllegalArgumentException("rgb too small");
        image.setRGB(x, y, w, h, rgb, 0, w);
        return this;
    }

    public int[] getRGB(int x, int y, int w, int h, int[] rgb) {
        return image.getRGB(x, y, w, h, rgb, 0, w);
    }

    // ---- region save/restore ---------------------------------------------

    /**
     * Save current canvas state (pen, brush, font, transform, clip) into a stack.
     * {@link #restore()} pops the last saved state.
     */
    public DrawingCanvas save() {
        if (stack == null) stack = new ArrayList<>();
        stack.add(new Snapshot(pen, brush, font, strokeColor, fillColor, DTransform.fromAwt(g2d.getTransform()),
                g2d.getClip(), g2d.getComposite()));
        return this;
    }

    public DrawingCanvas restore() {
        if (stack == null || stack.isEmpty()) throw new IllegalStateException("no saved state");
        Snapshot s = stack.remove(stack.size() - 1);
        this.pen = s.pen;
        this.brush = s.brush;
        this.font = s.font;
        this.strokeColor = s.stroke;
        this.fillColor = s.fill;
        this.composite = s.composite;
        applyTransform(s.transform);
        g2d.setClip(s.clip);
        g2d.setComposite(s.composite);
        return this;
    }

    private List<Snapshot> stack;
    private static final class Snapshot {
        final DPen pen; final DBrush brush; final DFont font;
        final DColor stroke; final DColor fill;
        final DTransform transform; final Shape clip; final Composite composite;
        Snapshot(DPen p, DBrush b, DFont f, DColor st, DColor fi, DTransform t, Shape c, Composite co) {
            pen = p; brush = b; font = f; stroke = st; fill = fi; transform = t; clip = c; composite = co;
        }
    }
}