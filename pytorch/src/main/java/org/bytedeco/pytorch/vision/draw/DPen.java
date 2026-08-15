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

import java.awt.BasicStroke;

/**
 * Stroke (a.k.a. pen) used to outline shapes in the {@link org.bytedeco.pytorch.vision.draw}
 * module.
 *
 * <p>Wraps the line width, dash pattern and end / join style of an AWT
 * {@link BasicStroke}, exposing a fluent builder.
 *
 * <pre>{@code
 * DPen pen = DPen.solid(2.0f, DColor.of("red"))
 *     .dashed(8, 4)
 *     .roundCaps()
 *     .roundJoin();
 * }</pre>
 */
public final class DPen {

    public enum Cap { BUTT, ROUND, SQUARE }
    public enum Join { MITER, ROUND, BEVEL }

    public final float width;
    public final Cap cap;
    public final Join join;
    public final float miterLimit;
    /** Dash array in pixels (in stroke units); {@code null} means solid. */
    public final float[] dash;
    /** Dash phase offset. */
    public final float dashPhase;

    private DPen(float width, Cap cap, Join join, float miterLimit, float[] dash, float dashPhase) {
        this.width = Math.max(0f, width);
        this.cap = cap == null ? Cap.BUTT : cap;
        this.join = join == null ? Join.MITER : join;
        this.miterLimit = miterLimit <= 0f ? 10f : miterLimit;
        this.dash = dash;
        this.dashPhase = dashPhase;
    }

    public static DPen solid(float width, DColor color) { return new DPen(width, Cap.BUTT, Join.MITER, 10f, null, 0f).colored(color); }
    public static DPen none() { return new DPen(0f, Cap.BUTT, Join.MITER, 10f, null, 0f).colored(DColor.of(0,0,0,0)); }

    public DPen colored(DColor color) { return new DPen(width, cap, join, miterLimit, dash, dashPhase); /* color is on the call site */ }

    public DPen dashed(float... dashOnOff) {
        return new DPen(width, cap, join, miterLimit, dashOnOff == null || dashOnOff.length == 0 ? null : dashOnOff.clone(), 0f);
    }

    public DPen dashPhase(float phase) { return new DPen(width, cap, join, miterLimit, dash, phase); }

    public DPen withCap(Cap cap) { return new DPen(width, cap, join, miterLimit, dash, dashPhase); }
    public DPen withJoin(Join join) { return new DPen(width, cap, join, miterLimit, dash, dashPhase); }
    public DPen withMiterLimit(float limit) { return new DPen(width, cap, join, limit, dash, dashPhase); }
    public DPen withWidth(float width) { return new DPen(width, cap, join, miterLimit, dash, dashPhase); }

    public DPen roundCaps() { return withCap(Cap.ROUND); }
    public DPen squareCaps() { return withCap(Cap.SQUARE); }
    public DPen roundJoin() { return withJoin(Join.ROUND); }
    public DPen bevelJoin() { return withJoin(Join.BEVEL); }

    public BasicStroke toAwt() {
        int awtCap = cap == Cap.ROUND ? BasicStroke.CAP_ROUND
                : cap == Cap.SQUARE ? BasicStroke.CAP_SQUARE : BasicStroke.CAP_BUTT;
        int awtJoin = join == Join.ROUND ? BasicStroke.JOIN_ROUND
                : join == Join.BEVEL ? BasicStroke.JOIN_BEVEL : BasicStroke.JOIN_MITER;
        if (dash == null || dash.length < 2) {
            return new BasicStroke(width, awtCap, awtJoin, miterLimit);
        }
        return new BasicStroke(width, awtCap, awtJoin, miterLimit, dash, dashPhase);
    }
}