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

import java.awt.geom.Point2D;
import java.util.Objects;

/**
 * Floating-point 2D point used by the {@link org.bytedeco.pytorch.vision.draw}
 * module. Distinct from {@link java.awt.geom.Point2D} to provide a stable
 * value-type API surface and easy integration with tensors.
 */
public final class DPoint {

    public final float x;
    public final float y;

    public DPoint(float x, float y) { this.x = x; this.y = y; }
    public DPoint(double x, double y) { this.x = (float) x; this.y = (float) y; }

    public static DPoint of(float x, float y) { return new DPoint(x, y); }
    public static DPoint of(double x, double y) { return new DPoint(x, y); }

    public DPoint translate(float dx, float dy) { return new DPoint(x + dx, y + dy); }

    public DPoint scale(float sx, float sy) { return new DPoint(x * sx, y * sy); }

    public DPoint rotate(double angleRad, DPoint center) {
        float cx = center.x, cy = center.y;
        double c = Math.cos(angleRad), s = Math.sin(angleRad);
        float dx = x - cx, dy = y - cy;
        return new DPoint((float) (c * dx - s * dy) + cx, (float) (s * dx + c * dy) + cy);
    }

    public DPoint rotate(double angleRad) { return rotate(angleRad, DPoint.of(0, 0)); }

    public Point2D.Float toAwt() { return new Point2D.Float(x, y); }

    /** Euclidean distance. */
    public double distance(DPoint o) {
        double dx = x - o.x, dy = y - o.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /** Linear interpolation. */
    public DPoint lerp(DPoint o, float t) {
        if (t <= 0f) return this;
        if (t >= 1f) return o;
        return new DPoint(x + (o.x - x) * t, y + (o.y - y) * t);
    }

    @Override public int hashCode() { return Float.floatToRawIntBits(x) * 31 + Float.floatToRawIntBits(y); }

    @Override public boolean equals(Object o) {
        if (!(o instanceof DPoint)) return false;
        DPoint p = (DPoint) o;
        return p.x == x && p.y == y;
    }

    @Override public String toString() { return String.format("DPoint[%.3f, %.3f]", x, y); }
}