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

import java.awt.geom.Rectangle2D;

/**
 * Floating-point rectangle for the {@link org.bytedeco.pytorch.vision.draw}
 * module. Stored as {@code (x, y, width, height)}.
 */
public final class DRect {

    public final float x;
    public final float y;
    public final float width;
    public final float height;

    public DRect(float x, float y, float width, float height) {
        this.x = x; this.y = y; this.width = width; this.height = height;
    }

    public DRect(double x, double y, double w, double h) {
        this((float) x, (float) y, (float) w, (float) h);
    }

    public static DRect of(float x, float y, float w, float h) { return new DRect(x, y, w, h); }
    public static DRect of(double x, double y, double w, double h) { return new DRect(x, y, w, h); }

    /** Inclusive of the left/top edge, exclusive of the right/bottom edge. */
    public float x2() { return x + width; }
    public float y2() { return y + height; }

    public float width() { return width; }
    public float height() { return height; }

    public boolean contains(DPoint p) {
        return p.x >= x && p.x < x2() && p.y >= y && p.y < y2();
    }

    public boolean contains(DRect o) {
        return o.x >= x && o.y >= y && o.x2() <= x2() && o.y2() <= y2();
    }

    public boolean intersects(DRect o) {
        return o.x < x2() && o.x2() > x && o.y < y2() && o.y2() > y;
    }

    public DRect intersection(DRect o) {
        float nx = Math.max(x, o.x), ny = Math.max(y, o.y);
        float nx2 = Math.min(x2(), o.x2()), ny2 = Math.min(y2(), o.y2());
        if (nx2 <= nx || ny2 <= ny) return new DRect(0, 0, 0, 0);
        return new DRect(nx, ny, nx2 - nx, ny2 - ny);
    }

    public DRect union(DRect o) {
        float nx = Math.min(x, o.x), ny = Math.min(y, o.y);
        float nx2 = Math.max(x2(), o.x2()), ny2 = Math.max(y2(), o.y2());
        return new DRect(nx, ny, nx2 - nx, ny2 - ny);
    }

    public DRect inflate(float dx, float dy) { return new DRect(x - dx, y - dy, width + 2 * dx, height + 2 * dy); }

    public DRect scale(float sx, float sy) { return new DRect(x * sx, y * sy, width * sx, height * sy); }

    public DRect translate(float dx, float dy) { return new DRect(x + dx, y + dy, width, height); }

    public Rectangle2D.Float toAwt() { return new Rectangle2D.Float(x, y, width, height); }

    public DRect rotate(double angleRad, DPoint center) {
        DRect rot = DrawingMath.rotateRect(this, angleRad, center);
        return new DRect(rot.x, rot.y, rot.width, rot.height);
    }

    @Override public int hashCode() {
        long h = Float.floatToRawIntBits(x);
        h = h * 31 + Float.floatToRawIntBits(y);
        h = h * 31 + Float.floatToRawIntBits(width);
        h = h * 31 + Float.floatToRawIntBits(height);
        return (int) (h ^ (h >>> 32));
    }

    @Override public boolean equals(Object o) {
        if (!(o instanceof DRect)) return false;
        DRect r = (DRect) o;
        return r.x == x && r.y == y && r.width == width && r.height == height;
    }

    @Override public String toString() { return String.format("DRect[%.3f,%.3f %.3fx%.3f]", x, y, width, height); }
}