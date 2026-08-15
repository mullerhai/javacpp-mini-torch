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

import java.util.Objects;

/**
 * Annotations helper — draws bounding boxes, keypoints, segmentation polygons,
 * class labels (with auto-coloring by class id), and arrows on a
 * {@link DrawingCanvas}.
 */
public final class DAnnotations {

    private DAnnotations() {}

    /** Bounding box in {@code [x1, y1, x2, y2]} format. */
    public static final class Box {
        public final float x1, y1, x2, y2;
        public final String label;
        public final DColor color;
        public final float score;
        public Box(float x1, float y1, float x2, float y2, String label, DColor color, float score) {
            this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2; this.label = label; this.color = color; this.score = score;
        }
        public static Box of(float x1, float y1, float x2, float y2, String label, DColor color, float score) {
            return new Box(x1, y1, x2, y2, label, color, score);
        }
    }

    /** Draw a single bounding box (rectangle + label badge). */
    public static void drawBox(DrawingCanvas c, Box b, float strokeWidth, boolean filled) {
        Objects.requireNonNull(c, "canvas");
        Objects.requireNonNull(b, "box");
        DRect r = DRect.of(Math.min(b.x1, b.x2), Math.min(b.y1, b.y2), Math.abs(b.x2 - b.x1), Math.abs(b.y2 - b.y1));
        c.save();
        if (filled) {
            DColor fill = b.color.withAlpha(0.15f);
            c.setFillColor(fill);
            c.fillRect(r);
        }
        c.setPen(DPen.solid(strokeWidth, b.color).colored(b.color));
        c.setStrokeColor(b.color);
        c.drawRect(r);
        c.restore();
        if (b.label != null) {
            drawLabelBadge(c, b.label, b.score, r.x, Math.max(0, r.y - 18), b.color);
        }
    }

    /** Draw a filled label badge. */
    public static void drawLabelBadge(DrawingCanvas c, String text, float score, float x, float y, DColor color) {
        c.save();
        DFont f = DFont.sans(11f).bold();
        DRect textRect = c.measure(text + (score > 0 ? String.format(" %.2f", score) : ""));
        DRect pad = new DRect(x, y, textRect.width() + 12, 18);
        c.setFillColor(color);
        c.fillRoundedRect(pad, 4);
        c.setFont(f);
        c.text(text + (score > 0 ? String.format(" %.2f", score) : ""), x + 6, y + 14, DColor.of("white"));
        c.restore();
    }

    /** Draw a keypoint set (line connecting consecutive pairs + circles). */
    public static void drawKeypoints(DrawingCanvas c, float[] xys, float radius, DColor color, boolean lines) {
        if (xys.length == 0) return;
        if (xys.length % 2 != 0) throw new IllegalArgumentException("xys.length must be even");
        int n = xys.length / 2;
        c.save();
        c.setFillColor(color);
        c.setStrokeColor(color);
        if (lines) {
            c.setPen(DPen.solid(1.5f, color).colored(color));
            for (int i = 1; i < n; i++) {
                c.drawLine(xys[(i - 1) * 2], xys[(i - 1) * 2 + 1], xys[i * 2], xys[i * 2 + 1]);
            }
        }
        for (int i = 0; i < n; i++) {
            c.fillEllipse(DEllipse.of(xys[i * 2], xys[i * 2 + 1], radius, radius));
        }
        c.restore();
    }

    /** Draw a segmentation polygon outline. */
    public static void drawPolygon(DrawingCanvas c, float[] xys, DColor color, float strokeWidth, boolean fill) {
        if (xys.length < 6 || xys.length % 2 != 0) return;
        DPoint[] pts = new DPoint[xys.length / 2];
        for (int i = 0; i < pts.length; i++) pts[i] = DPoint.of(xys[i * 2], xys[i * 2 + 1]);
        c.save();
        if (fill) {
            c.setFillColor(color.withAlpha(0.25f));
            c.fillPolygon(pts);
        }
        c.setPen(DPen.solid(strokeWidth, color).colored(color));
        c.setStrokeColor(color);
        c.drawPolygon(pts);
        c.restore();
    }

    /** Draw an arrow from {@code (x1,y1)} → {@code (x2,y2)}. */
    public static void drawArrow(DrawingCanvas c, float x1, float y1, float x2, float y2, DColor color, float width) {
        D3DRenderer r3d = new D3DRenderer(c, org.bytedeco.pytorch.vision.draw.D3D.Camera.defaultCamera(1));
        r3d.drawArrow(org.bytedeco.pytorch.vision.draw.D3D.V3.class.cast(null) == null
                ? new org.bytedeco.pytorch.vision.draw.D3D.V3(x1, y1, 0)
                : new org.bytedeco.pytorch.vision.draw.D3D.V3(x1, y1, 0),
                new org.bytedeco.pytorch.vision.draw.D3D.V3(x2, y2, 0), color, width);
    }

    /** Color picker for arbitrary class id (HSV-spaced). */
    public static DColor colorForClass(int classId) {
        float hue = ((classId * 2654435761L) & 0xFFFF) / 65536f;
        return DColor.hsv(hue, 0.75f, 0.95f);
    }
}