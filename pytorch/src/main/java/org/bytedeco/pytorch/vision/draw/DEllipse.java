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
 * Floating-point ellipse (bounding-box centric) for the drawing module.
 *
 * <p>Stored as a center point and the two radii. Use {@link #toPath()} to obtain
 * an AWT {@link java.awt.geom.Ellipse2D}-equivalent {@link DPath}.
 */
public final class DEllipse {

    public final DPoint center;
    public final float rx;
    public final float ry;

    public DEllipse(DPoint center, float rx, float ry) {
        Objects.requireNonNull(center, "center");
        this.center = center;
        this.rx = rx;
        this.ry = ry;
    }

    public DEllipse(float cx, float cy, float rx, float ry) {
        this(DPoint.of(cx, cy), rx, ry);
    }

    public static DEllipse of(float cx, float cy, float rx, float ry) {
        return new DEllipse(cx, cy, rx, ry);
    }

    public DRect bounds() {
        return new DRect(center.x - rx, center.y - ry, 2 * rx, 2 * ry);
    }

    public boolean contains(DPoint p) {
        float dx = (p.x - center.x) / rx;
        float dy = (p.y - center.y) / ry;
        return dx * dx + dy * dy <= 1f;
    }

    /** Approximate ellipse as a cubic-Bezier {@link DPath} (4-segment). */
    public DPath toPath() {
        // magic constant for cubic bezier approximation: (4/3)(sqrt(2)-1) ≈ 0.5522847498
        float k = 0.5522847498f * rx;
        float ky = 0.5522847498f * ry;
        DPath p = new DPath();
        // start at rightmost point
        p.moveTo(center.x + rx, center.y);
        p.curveTo(center.x + rx, center.y + ky,
                  center.x + k, center.y + ry,
                  center.x, center.y + ry);
        p.curveTo(center.x - k, center.y + ry,
                  center.x - rx, center.y + ky,
                  center.x - rx, center.y);
        p.curveTo(center.x - rx, center.y - ky,
                  center.x - k, center.y - ry,
                  center.x, center.y - ry);
        p.curveTo(center.x + k, center.y - ry,
                  center.x + rx, center.y - ky,
                  center.x + rx, center.y);
        p.closePath();
        return p;
    }

    @Override public int hashCode() {
        return center.hashCode() * 31 + Float.floatToRawIntBits(rx) + Float.floatToRawIntBits(ry);
    }

    @Override public boolean equals(Object o) {
        if (!(o instanceof DEllipse)) return false;
        DEllipse e = (DEllipse) o;
        return e.rx == rx && e.ry == ry && e.center.equals(center);
    }

    @Override public String toString() {
        return String.format("DEllipse[%.3f,%.3f r=%.3fx%.3f]", center.x, center.y, rx, ry);
    }
}