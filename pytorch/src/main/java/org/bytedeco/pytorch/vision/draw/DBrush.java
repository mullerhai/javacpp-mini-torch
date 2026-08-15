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

import java.awt.Paint;
import java.awt.TexturePaint;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;

/**
 * Fill brush for the {@link org.bytedeco.pytorch.vision.draw} module.
 *
 * <p>Beyond a flat {@link DColor} the brush can be:
 * <ul>
 *   <li>a vertical/horizontal/radial gradient (multi-stop)</li>
 *   <li>a tiled texture (any {@link BufferedImage})</li>
 *   <li>a hatch (line / cross / dot) drawn from a 1-channel procedural pattern</li>
 *   <li>a checker pattern (debug-friendly)</li>
 * </ul>
 *
 * <p>Bridges to AWT {@link Paint} when handed to a {@link DrawingCanvas}.
 */
public final class DBrush {

    public enum Style { COLOR, LINEAR_GRADIENT, RADIAL_GRADIENT, TEXTURE, HATCH, CHECKER }

    public final Style style;
    public final DColor color;
    public final float[] stops;       // gradient stops in [0,1]
    public final DColor[] colors;     // gradient colors (parallel to stops)
    public final DPoint from;         // gradient anchor (logical pixels)
    public final DPoint to;
    public final float radius;        // radial radius
    public final BufferedImage texture;
    public final Rectangle2D anchor;  // texture tile anchor
    public final HatchPattern hatch;

    public enum HatchPattern { LINES, CROSS, DOTS, DIAGONAL_FWD, DIAGONAL_BOTH }

    private DBrush(Style style, DColor color, float[] stops, DColor[] colors, DPoint from, DPoint to,
                   float radius, BufferedImage texture, Rectangle2D anchor, HatchPattern hatch) {
        this.style = style;
        this.color = color;
        this.stops = stops;
        this.colors = colors;
        this.from = from;
        this.to = to;
        this.radius = radius;
        this.texture = texture;
        this.anchor = anchor;
        this.hatch = hatch;
    }

    public static DBrush solid(DColor c) { return new DBrush(Style.COLOR, c, null, null, null, null, 0, null, null, null); }
    public static DBrush transparent() { return solid(DColor.of(0, 0, 0, 0)); }

    public static DBrush linearGradient(float x1, float y1, float x2, float y2, DColor... cs) {
        if (cs == null || cs.length < 2) throw new IllegalArgumentException("need >= 2 colors");
        return linearGradient(DPoint.of(x1, y1), DPoint.of(x2, y2), cs);
    }

    public static DBrush linearGradient(DPoint from, DPoint to, DColor... cs) {
        if (cs == null || cs.length < 2) throw new IllegalArgumentException("need >= 2 colors");
        float[] stops = new float[cs.length];
        for (int i = 0; i < cs.length; i++) stops[i] = (float) i / (cs.length - 1);
        return new DBrush(Style.LINEAR_GRADIENT, null, stops, cs.clone(), from, to, 0, null, null, null);
    }

    public static DBrush linearGradient(DPoint from, DPoint to, float[] stops, DColor... cs) {
        if (cs == null || cs.length < 2) throw new IllegalArgumentException("need >= 2 colors");
        if (stops == null || stops.length != cs.length) throw new IllegalArgumentException("stops must match colors");
        return new DBrush(Style.LINEAR_GRADIENT, null, stops.clone(), cs.clone(), from, to, 0, null, null, null);
    }

    public static DBrush radialGradient(float cx, float cy, float radius, DColor... cs) {
        if (cs == null || cs.length < 2) throw new IllegalArgumentException("need >= 2 colors");
        return radialGradient(DPoint.of(cx, cy), radius, cs);
    }

    public static DBrush radialGradient(DPoint center, float radius, DColor... cs) {
        if (cs == null || cs.length < 2) throw new IllegalArgumentException("need >= 2 colors");
        float[] stops = new float[cs.length];
        for (int i = 0; i < cs.length; i++) stops[i] = (float) i / (cs.length - 1);
        return new DBrush(Style.RADIAL_GRADIENT, null, stops, cs.clone(), center, center, radius, null, null, null);
    }

    public static DBrush texture(BufferedImage img) {
        return texture(img, new Rectangle2D.Float(0, 0, img.getWidth(), img.getHeight()));
    }

    public static DBrush texture(BufferedImage img, Rectangle2D anchor) {
        return new DBrush(Style.TEXTURE, null, null, null, null, null, 0, img, anchor, null);
    }

    public static DBrush hatch(HatchPattern pattern, DColor bg, DColor fg, int spacingPx) {
        if (pattern == null) throw new IllegalArgumentException("pattern required");
        int s = Math.max(2, spacingPx);
        BufferedImage img = new BufferedImage(s * 2, s * 2, BufferedImage.TYPE_INT_ARGB);
        java.awt.Graphics2D g = img.createGraphics();
        g.setColor(bg.toAwt());
        g.fillRect(0, 0, s * 2, s * 2);
        g.setColor(fg.toAwt());
        g.setStroke(new java.awt.BasicStroke(Math.max(1f, s / 6f)));
        switch (pattern) {
            case LINES:
                for (int y = 0; y < s * 2; y += s) g.drawLine(0, y, s * 2, y);
                break;
            case CROSS:
                for (int y = 0; y < s * 2; y += s) g.drawLine(0, y, s * 2, y);
                for (int x = 0; x < s * 2; x += s) g.drawLine(x, 0, x, s * 2);
                break;
            case DOTS:
                g.fillOval(s / 2, s / 2, Math.max(2, s / 3), Math.max(2, s / 3));
                g.fillOval(s + s / 2, s + s / 2, Math.max(2, s / 3), Math.max(2, s / 3));
                break;
            case DIAGONAL_FWD:
                for (int i = -s; i < s * 2; i += s / 2) g.drawLine(i, 0, i + s * 2, s * 2);
                break;
            case DIAGONAL_BOTH:
                for (int i = -s; i < s * 2; i += s / 2) g.drawLine(i, 0, i + s * 2, s * 2);
                for (int i = -s; i < s * 2; i += s / 2) g.drawLine(i, s * 2, i + s * 2, 0);
                break;
            default: throw new IllegalArgumentException("unknown pattern: " + pattern);
        }
        g.dispose();
        return new DBrush(Style.HATCH, null, null, null, null, null, 0, img,
                new Rectangle2D.Float(0, 0, s * 2, s * 2), pattern);
    }

    /** Two-color checker tile. Useful for transparency visualisation. */
    public static DBrush checker(DColor a, DColor b, int cellPx) {
        int s = Math.max(2, cellPx);
        BufferedImage img = new BufferedImage(s * 2, s * 2, BufferedImage.TYPE_INT_ARGB);
        java.awt.Graphics2D g = img.createGraphics();
        g.setColor(a.toAwt()); g.fillRect(0, 0, s, s);   g.fillRect(s, s, s, s);
        g.setColor(b.toAwt()); g.fillRect(s, 0, s, s);   g.fillRect(0, s, s, s);
        g.dispose();
        return new DBrush(Style.CHECKER, null, null, null, null, null, 0, img,
                new Rectangle2D.Float(0, 0, s * 2, s * 2), null);
    }

    public Paint toAwt() {
        switch (style) {
            case COLOR: return color.toAwt();
            case TEXTURE:
            case HATCH:
            case CHECKER: {
                Rectangle2D a = anchor == null
                        ? new Rectangle2D.Float(0, 0, texture.getWidth(), texture.getHeight())
                        : anchor;
                return new TexturePaint(texture, a);
            }
            case LINEAR_GRADIENT: {
                java.awt.Color[] awtColors = new java.awt.Color[colors.length];
                for (int i = 0; i < colors.length; i++) awtColors[i] = colors[i].toAwt();
                return new java.awt.LinearGradientPaint(
                        new java.awt.geom.Point2D.Float(from.x, from.y),
                        new java.awt.geom.Point2D.Float(to.x, to.y),
                        stops, awtColors);
            }
            case RADIAL_GRADIENT: {
                java.awt.Color[] awtColors = new java.awt.Color[colors.length];
                for (int i = 0; i < colors.length; i++) awtColors[i] = colors[i].toAwt();
                return new java.awt.RadialGradientPaint(
                        new java.awt.geom.Point2D.Float(from.x, from.y), Math.max(0.0001f, radius),
                        stops, awtColors);
            }
            default: throw new IllegalStateException("unknown style");
        }
    }
}