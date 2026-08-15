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

/**
 * Geometry helpers used by the {@link org.bytedeco.pytorch.vision.draw} module.
 * Pure-math, dependency-free.
 */
public final class DrawingMath {
    private DrawingMath() {}

    public static double deg2rad(double d) { return Math.toRadians(d); }
    public static double rad2deg(double r) { return Math.toDegrees(r); }

    public static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    public static int clamp(int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /**
     * Rotate {@code r} by {@code angleRad} around {@code center}; result is the
     * axis-aligned bounding box of the rotated rectangle.
     */
    public static DRect rotateRect(DRect r, double angleRad, DPoint center) {
        double c = Math.cos(angleRad), s = Math.sin(angleRad);
        float[][] corners = new float[][]{
                {r.x, r.y}, {r.x2(), r.y},
                {r.x2(), r.y2()}, {r.x, r.y2()}
        };
        float minX = Float.POSITIVE_INFINITY, minY = Float.POSITIVE_INFINITY;
        float maxX = Float.NEGATIVE_INFINITY, maxY = Float.NEGATIVE_INFINITY;
        for (float[] corner : corners) {
            float dx = corner[0] - center.x;
            float dy = corner[1] - center.y;
            float rx = (float) (c * dx - s * dy) + center.x;
            float ry = (float) (s * dx + c * dy) + center.y;
            if (rx < minX) minX = rx;
            if (ry < minY) minY = ry;
            if (rx > maxX) maxX = rx;
            if (ry > maxY) maxY = ry;
        }
        return new DRect(minX, minY, maxX - minX, maxY - minY);
    }

    /** Cubic Bezier evaluation. */
    public static DPoint bezier(float t, DPoint p0, DPoint p1, DPoint p2, DPoint p3) {
        float u = 1f - t;
        float b0 = u * u * u;
        float b1 = 3f * u * u * t;
        float b2 = 3f * u * t * t;
        float b3 = t * t * t;
        return new DPoint(b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x,
                b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y);
    }

    /** Linear map {@code value} from {@code [a,b]} → {@code [c,d]}. */
    public static float map(float value, float a, float b, float c, float d) {
        if (a == b) return c;
        float t = (value - a) / (b - a);
        return c + t * (d - c);
    }

    /** Approximate Euclidean distance from {@code (x,y)} to line through {@code a→b}. */
    public static double pointLineDistance(double x, double y,
                                           double ax, double ay,
                                           double bx, double by) {
        double dx = bx - ax, dy = by - ay;
        double len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1e-9) {
            dx = x - ax; dy = y - ay;
            return Math.sqrt(dx * dx + dy * dy);
        }
        return Math.abs((by - ay) * x - (bx - ax) * y + bx * ay - by * ax) / len;
    }

    /** Catmull-Rom spline → cubic Bezier conversion; returns control points as a 4-tuple per segment. */
    public static DPoint[] catmullRomToBezier(DPoint p0, DPoint p1, DPoint p2, DPoint p3) {
        // tension = 0.5 (centripetal-ish default)
        float c1x = p1.x + (p2.x - p0.x) / 6f;
        float c1y = p1.y + (p2.y - p0.y) / 6f;
        float c2x = p2.x - (p3.x - p1.x) / 6f;
        float c2y = p2.y - (p3.y - p1.y) / 6f;
        return new DPoint[]{DPoint.of(c1x, c1y), DPoint.of(c2x, c2y), p2};
    }

    /** Simple polynomial smooth — three-point average — for drawing denoising. */
    public static float[] smooth(float[] in, int passes) {
        if (in == null || in.length < 3 || passes <= 0) return in == null ? new float[0] : in.clone();
        float[] buf = in.clone();
        for (int p = 0; p < passes; p++) {
            float prev = buf[0];
            for (int i = 1; i < buf.length - 1; i++) {
                float cur = buf[i];
                buf[i] = (prev + cur + buf[i + 1]) / 3f;
                prev = cur;
            }
        }
        return buf;
    }
}