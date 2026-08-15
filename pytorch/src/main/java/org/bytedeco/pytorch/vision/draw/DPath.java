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

import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Path2D;
import java.awt.geom.PathIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Vector path used by the {@link org.bytedeco.pytorch.vision.draw} module.
 *
 * <p>Wraps an AWT {@link Path2D.Float} for zero-overhead bridging to
 * {@link java.awt.Graphics2D} while exposing a friendly builder API on top.
 * Sub-path tracking, {@code closePath()}, hit-testing and bounding boxes are
 * supported.
 *
 * <pre>{@code
 * DPath p = new DPath()
 *     .moveTo(10, 10)
 *     .lineTo(100, 10)
 *     .quadTo(150, 50, 100, 100)
 *     .closePath();
 * canvas.stroke(p);
 * }</pre>
 */
public final class DPath {

    /** Segment kind constants matching the AWT {@link PathIterator} SEG_* values. */
    public static final int SEG_MOVETO  = PathIterator.SEG_MOVETO;
    public static final int SEG_LINETO  = PathIterator.SEG_LINETO;
    public static final int SEG_QUADTO  = PathIterator.SEG_QUADTO;
    public static final int SEG_CUBICTO = PathIterator.SEG_CUBICTO;
    public static final int SEG_CLOSE   = PathIterator.SEG_CLOSE;

    private final Path2D.Float path;
    private final List<float[]> segments = new ArrayList<>(); // for serialization / inspection
    private final List<Integer> kinds = new ArrayList<>();

    public DPath() {
        this.path = new Path2D.Float(Path2D.WIND_NON_ZERO, 64);
    }

    // ---- builder API ------------------------------------------------------

    public DPath moveTo(float x, float y) {
        path.moveTo(x, y);
        kinds.add(SEG_MOVETO); segments.add(new float[]{x, y});
        return this;
    }

    public DPath lineTo(float x, float y) {
        path.lineTo(x, y);
        kinds.add(SEG_LINETO); segments.add(new float[]{x, y});
        return this;
    }

    public DPath quadTo(float cx, float cy, float x, float y) {
        path.quadTo(cx, cy, x, y);
        kinds.add(SEG_QUADTO); segments.add(new float[]{cx, cy, x, y});
        return this;
    }

    public DPath curveTo(float c1x, float c1y, float c2x, float c2y, float x, float y) {
        path.curveTo(c1x, c1y, c2x, c2y, x, y);
        kinds.add(SEG_CUBICTO); segments.add(new float[]{c1x, c1y, c2x, c2y, x, y});
        return this;
    }

    public DPath closePath() {
        path.closePath();
        kinds.add(SEG_CLOSE); segments.add(new float[0]);
        return this;
    }

    // ---- queries ----------------------------------------------------------

    public boolean contains(DPoint p) { return path.contains(p.toAwt()); }
    public boolean contains(float x, float y) { return path.contains(x, y); }

    public DRect bounds() {
        java.awt.geom.Rectangle2D b = path.getBounds2D();
        return new DRect((float) b.getX(), (float) b.getY(), (float) b.getWidth(), (float) b.getHeight());
    }

    public boolean isEmpty() { return segments.isEmpty(); }
    public int segmentCount() { return segments.size(); }

    // ---- conversion -------------------------------------------------------

    /** Convert to AWT {@link Path2D.Float}. Reuses backing storage. */
    public Path2D.Float toAwt() { return path; }

    /** Returns a deep copy. */
    public DPath copy() {
        DPath p = new DPath();
        p.path.reset();
        for (int i = 0; i < kinds.size(); i++) {
            int k = kinds.get(i);
            float[] s = segments.get(i);
            switch (k) {
                case SEG_MOVETO:  p.path.moveTo(s[0], s[1]); break;
                case SEG_LINETO:  p.path.lineTo(s[0], s[1]); break;
                case SEG_QUADTO:  p.path.quadTo(s[0], s[1], s[2], s[3]); break;
                case SEG_CUBICTO: p.path.curveTo(s[0], s[1], s[2], s[3], s[4], s[5]); break;
                case SEG_CLOSE:   p.path.closePath(); break;
                default: throw new AssertionError(k);
            }
        }
        p.kinds.addAll(kinds);
        for (float[] s : segments) p.segments.add(s.clone());
        return p;
    }

    public DPath transform(AffineTransform at) {
        DPath p = copy();
        p.path.transform(at);
        return p;
    }

    public PathIterator iterator(AffineTransform at) { return path.getPathIterator(at); }

    public List<DPoint> vertices() {
        List<DPoint> out = new ArrayList<>();
        for (int i = 0; i < kinds.size(); i++) {
            float[] s = segments.get(i);
            int k = kinds.get(i);
            if (k == SEG_MOVETO || k == SEG_LINETO) out.add(DPoint.of(s[0], s[1]));
        }
        return out;
    }

    // ---- helpers from primitives -----------------------------------------

    /** Build a polygon path from the given points (auto-closes). */
    public static DPath polygon(DPoint... pts) {
        Objects.requireNonNull(pts, "pts");
        if (pts.length == 0) return new DPath();
        DPath p = new DPath().moveTo(pts[0].x, pts[0].y);
        for (int i = 1; i < pts.length; i++) p.lineTo(pts[i].x, pts[i].y);
        p.closePath();
        return p;
    }

    /** Build a polyline path (open). */
    public static DPath polyline(DPoint... pts) {
        Objects.requireNonNull(pts, "pts");
        if (pts.length == 0) return new DPath();
        DPath p = new DPath().moveTo(pts[0].x, pts[0].y);
        for (int i = 1; i < pts.length; i++) p.lineTo(pts[i].x, pts[i].y);
        return p;
    }

    /** Build a rectangular path. */
    public static DPath rectangle(DRect r) {
        return new DPath()
                .moveTo(r.x, r.y)
                .lineTo(r.x2(), r.y)
                .lineTo(r.x2(), r.y2())
                .lineTo(r.x, r.y2())
                .closePath();
    }

    @Override public String toString() {
        return "DPath[" + kinds.size() + " segments]";
    }
}