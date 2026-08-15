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

import java.awt.geom.AffineTransform;

/**
 * 2D affine transform used by the {@link org.bytedeco.pytorch.vision.draw}
 * module.
 *
 * <p>Wraps AWT {@link AffineTransform} for {@link java.awt.Graphics2D} bridging.
 * Provides a friendlier builder API for common cases (translate, scale, rotate,
 * skew) without losing any of AWT's expressive power.
 *
 * <pre>{@code
 * DTransform t = DTransform.identity()
 *     .translate(100, 100)
 *     .rotate(Math.PI / 4)
 *     .scale(2, 2);
 * canvas.setTransform(t);
 * }</pre>
 */
public final class DTransform {

    public final double a; // m00
    public final double b; // m10
    public final double c; // m01
    public final double d; // m11
    public final double tx; // m02
    public final double ty; // m12

    private DTransform(double a, double b, double c, double d, double tx, double ty) {
        this.a = a; this.b = b; this.c = c; this.d = d; this.tx = tx; this.ty = ty;
    }

    public static DTransform identity() { return new DTransform(1, 0, 0, 1, 0, 0); }

    public static DTransform translate(double tx, double ty) { return new DTransform(1, 0, 0, 1, tx, ty); }

    public static DTransform scale(double sx, double sy) { return new DTransform(sx, 0, 0, sy, 0, 0); }

    public static DTransform scale(double s) { return scale(s, s); }

    public static DTransform rotate(double radians) {
        double c = Math.cos(radians), s = Math.sin(radians);
        return new DTransform(c, s, -s, c, 0, 0);
    }

    public static DTransform rotate(double radians, double cx, double cy) {
        double c = Math.cos(radians), si = Math.sin(radians);
        return new DTransform(c, si, -si, c, cx - c * cx + si * cy, cy - si * cx - c * cy);
    }

    public static DTransform skew(double sx, double sy) { return new DTransform(1, sy, sx, 1, 0, 0); }

    /** Compose: {@code this ∘ rhs} — apply {@code rhs} first, then {@code this}. */
    public DTransform compose(DTransform rhs) {
        return new DTransform(
                a * rhs.a + b * rhs.c, a * rhs.b + b * rhs.d,
                c * rhs.a + d * rhs.c, c * rhs.b + d * rhs.d,
                a * rhs.tx + b * rhs.ty + tx, c * rhs.tx + d * rhs.ty + ty);
    }

    /** Apply {@code rhs} after {@code this}; same as {@link #compose(DTransform)}. */
    public DTransform andThen(DTransform rhs) { return compose(rhs); }

    public DTransform andThenTranslate(double tx, double ty) { return compose(translate(tx, ty)); }
    public DTransform andThenScale(double sx, double sy) { return compose(scale(sx, sy)); }
    public DTransform andThenRotate(double radians) { return compose(rotate(radians)); }

    /** Inverse transform; throws if determinant is zero. */
    public DTransform inverse() {
        double det = a * d - b * c;
        if (Math.abs(det) < 1e-12) throw new ArithmeticException("non-invertible transform");
        double inv = 1.0 / det;
        return new DTransform(d * inv, -b * inv, -c * inv, a * inv,
                (b * ty - d * tx) * inv, (c * tx - a * ty) * inv);
    }

    public DPoint apply(DPoint p) {
        return new DPoint(a * p.x + c * p.y + tx, b * p.x + d * p.y + ty);
    }

    public DRect apply(DRect r) {
        DPoint p1 = apply(DPoint.of(r.x, r.y));
        DPoint p2 = apply(DPoint.of(r.x2(), r.y));
        DPoint p3 = apply(DPoint.of(r.x2(), r.y2()));
        DPoint p4 = apply(DPoint.of(r.x, r.y2()));
        float minX = Math.min(Math.min(p1.x, p2.x), Math.min(p3.x, p4.x));
        float minY = Math.min(Math.min(p1.y, p2.y), Math.min(p3.y, p4.y));
        float maxX = Math.max(Math.max(p1.x, p2.x), Math.max(p3.x, p4.x));
        float maxY = Math.max(Math.max(p1.y, p2.y), Math.max(p3.y, p4.y));
        return new DRect(minX, minY, maxX - minX, maxY - minY);
    }

    public AffineTransform toAwt() {
        AffineTransform at = new AffineTransform();
        at.setTransform(a, b, c, d, tx, ty);
        return at;
    }

    public static DTransform fromAwt(AffineTransform at) {
        double[] m = new double[6];
        at.getMatrix(m);
        return new DTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
    }

    public boolean isIdentity() {
        return a == 1d && b == 0d && c == 0d && d == 1d && tx == 0d && ty == 0d;
    }

    @Override public String toString() {
        return String.format("DTransform[[%.3f %.3f %.3f][%.3f %.3f %.3f]]", a, c, tx, b, d, ty);
    }
}