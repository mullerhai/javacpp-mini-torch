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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Pure-Java contour utilities for the {@link org.bytedeco.pytorch.vision.draw}
 * module. Implements marching squares on a float grid, plus post-processing
 * (simplification, smoothing).
 *
 * <p>Use {@link #extractIsoContours(float[], int, int, float)} for a set of
 * polylines at a fixed threshold.
 */
public final class DContour {

    private DContour() {}

    /** Contour polyline. */
    public static final class Contour {
        public final List<DPoint> points;
        public final boolean closed;
        public Contour(List<DPoint> points, boolean closed) {
            this.points = points; this.closed = closed;
        }
    }

    /**
     * Marching squares on a 2D scalar field to extract all iso-contours at
     * {@code threshold}. Returns a list of polylines (closed when the loop
     * returns to its origin).
     */
    public static List<Contour> extractIsoContours(float[] field, int h, int w, float threshold) {
        Objects.requireNonNull(field, "field");
        if (field.length != h * w) throw new IllegalArgumentException("field.length != h*w");

        // Pre-compute cell codes.
        byte[] codes = new byte[(h - 1) * (w - 1)];
        for (int y = 0; y < h - 1; y++) {
            for (int x = 0; x < w - 1; x++) {
                float tl = field[y * w + x];
                float tr = field[y * w + x + 1];
                float br = field[(y + 1) * w + x + 1];
                float bl = field[(y + 1) * w + x];
                int code = 0;
                if (tl >= threshold) code |= 1;
                if (tr >= threshold) code |= 2;
                if (br >= threshold) code |= 4;
                if (bl >= threshold) code |= 8;
                codes[y * (w - 1) + x] = (byte) code;
            }
        }
        // Use simple per-cell line segment emission — collects an unordered set of
        // segments, then stitches them into polylines.
        List<float[]> segs = new ArrayList<>(); // [x1,y1,x2,y2]
        for (int y = 0; y < h - 1; y++) {
            for (int x = 0; x < w - 1; x++) {
                int code = codes[y * (w - 1) + x] & 0xF;
                if (code == 0 || code == 15) continue;
                float tl = field[y * w + x];
                float tr = field[y * w + x + 1];
                float br = field[(y + 1) * w + x + 1];
                float bl = field[(y + 1) * w + x];
                // edge crossing points (single (x,y) each)
                float[] top = lerp(tl, tr, threshold, x, x + 1, y);
                float[] right = lerp(tr, br, threshold, x + 1, x + 1, y);
                float[] bottom = lerp(bl, br, threshold, x, x + 1, y + 1);
                float[] left = lerp(tl, bl, threshold, x, x, y);
                switch (code) {
                    case 1: case 14: segs.add(seg(top[0], top[1], left[0], left[1])); break;
                    case 2: case 13: segs.add(seg(top[0], top[1], right[0], right[1])); break;
                    case 3: case 12: segs.add(seg(left[0], left[1], right[0], right[1])); break;
                    case 4: case 11: segs.add(seg(bottom[0], bottom[1], right[0], right[1])); break;
                    case 5: // saddle (ambiguous): emit both diagonals
                        segs.add(seg(top[0], top[1], left[0], left[1]));
                        segs.add(seg(bottom[0], bottom[1], right[0], right[1]));
                        break;
                    case 6: case 9: segs.add(seg(top[0], top[1], bottom[0], bottom[1])); break;
                    case 7: case 8: segs.add(seg(bottom[0], bottom[1], left[0], left[1])); break;
                    case 10: // saddle (ambiguous): emit both diagonals
                        segs.add(seg(top[0], top[1], right[0], right[1]));
                        segs.add(seg(bottom[0], bottom[1], left[0], left[1]));
                        break;
                    default: break;
                }
            }
        }
        return stitch(segs);
    }

    private static float[] seg(float x1, float y1, float x2, float y2) {
        return new float[]{x1, y1, x2, y2};
    }

    private static float[] lerp(float a, float b, float t, float x1, float x2, float y) {
        float k = (t - a) / (b - a);
        if (Float.isNaN(k) || Float.isInfinite(k)) k = 0.5f;
        if (k < 0f) k = 0f; else if (k > 1f) k = 1f;
        return new float[]{x1 + (x2 - x1) * k, y};
    }

    private static List<Contour> stitch(List<float[]> segs) {
        List<Contour> out = new ArrayList<>();
        boolean[] used = new boolean[segs.size()];
        for (int i = 0; i < segs.size(); i++) {
            if (used[i]) continue;
            List<DPoint> poly = new ArrayList<>();
            float[] head = segs.get(i);
            poly.add(DPoint.of(head[0], head[1]));
            poly.add(DPoint.of(head[2], head[3]));
            used[i] = true;
            boolean closed = false;
            while (true) {
                DPoint tail = poly.get(poly.size() - 1);
                boolean found = false;
                for (int j = 0; j < segs.size(); j++) {
                    if (used[j]) continue;
                    float[] s = segs.get(j);
                    if (eq(s[0], tail.x) && eq(s[1], tail.y)) {
                        poly.add(DPoint.of(s[2], s[3]));
                        used[j] = true;
                        found = true; break;
                    } else if (eq(s[2], tail.x) && eq(s[3], tail.y)) {
                        poly.add(DPoint.of(s[0], s[1]));
                        used[j] = true;
                        found = true; break;
                    }
                }
                if (!found) break;
                // check if last point equals first (closed loop)
                DPoint first = poly.get(0);
                DPoint last = poly.get(poly.size() - 1);
                if (eq(first.x, last.x) && eq(first.y, last.y)) { closed = true; break; }
            }
            // ensure at least 2 points
            if (poly.size() < 2) continue;
            out.add(new Contour(poly, closed));
        }
        return out;
    }

    private static boolean eq(float a, float b) { return Math.abs(a - b) < 1e-4f; }

    /**
     * Ramer-Douglas-Peucker polyline simplification. {@code eps} is the maximum
     * perpendicular distance from the original polyline.
     */
    public static List<DPoint> simplify(List<DPoint> pts, float eps) {
        if (pts.size() < 3 || eps <= 0f) return new ArrayList<>(pts);
        boolean[] keep = new boolean[pts.size()];
        keep[0] = keep[pts.size() - 1] = true;
        rdp(pts, 0, pts.size() - 1, eps, keep);
        List<DPoint> out = new ArrayList<>();
        for (int i = 0; i < pts.size(); i++) if (keep[i]) out.add(pts.get(i));
        return out;
    }

    private static void rdp(List<DPoint> pts, int a, int b, float eps, boolean[] keep) {
        if (b <= a + 1) return;
        double maxD = 0; int idx = -1;
        DPoint pa = pts.get(a), pb = pts.get(b);
        for (int i = a + 1; i < b; i++) {
            double d = DrawingMath.pointLineDistance(pts.get(i).x, pts.get(i).y, pa.x, pa.y, pb.x, pb.y);
            if (d > maxD) { maxD = d; idx = i; }
        }
        if (maxD > eps && idx > -1) {
            keep[idx] = true;
            rdp(pts, a, idx, eps, keep);
            rdp(pts, idx, b, eps, keep);
        }
    }

    /**
     * Draw contours on a {@link DrawingCanvas}. {@code thickness} is the line
     * width; pass {@code 0} for hairline.
     */
    public static void drawContours(DrawingCanvas c, List<Contour> contours, DColor color, float thickness) {
        c.save();
        c.setPen(DPen.solid(thickness, color).colored(color));
        c.setStrokeColor(color);
        for (Contour co : contours) {
            for (int i = 1; i < co.points.size(); i++) {
                DPoint a = co.points.get(i - 1);
                DPoint b = co.points.get(i);
                c.drawLine(a, b);
            }
            if (co.closed && co.points.size() > 1) {
                c.drawLine(co.points.get(co.points.size() - 1), co.points.get(0));
            }
        }
        c.restore();
    }
}