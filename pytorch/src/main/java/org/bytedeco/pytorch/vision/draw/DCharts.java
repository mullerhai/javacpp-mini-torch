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

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Lightweight chart primitives (bar / scatter / pie / line). Built on top of
 * {@link DrawingCanvas} so they share the same color / font / antialiasing
 * conventions.
 */
public final class DCharts {

    private DCharts() {}

    public static BufferedImage bar(float[] values, String[] labels,
                                    DColor barColor, DColor axisColor, DColor labelColor,
                                    int widthPx, int heightPx, int paddingPx, DColor background) {
        Objects.requireNonNull(values, "values");
        BufferedImage out = new BufferedImage(widthPx, heightPx, BufferedImage.TYPE_INT_ARGB);
        try (DrawingCanvas c = DrawingCanvas.on(out)) {
            c.clear(background == null ? DColor.of("white") : background);
            int chartW = widthPx - 2 * paddingPx;
            int chartH = heightPx - 2 * paddingPx;
            // axes
            c.save();
            c.setPen(DPen.solid(1f, axisColor).colored(axisColor));
            c.setStrokeColor(axisColor);
            int x0 = paddingPx, y0 = heightPx - paddingPx;
            int x1 = widthPx - paddingPx, y1 = paddingPx;
            c.drawLine(x0, y0, x1, y0);
            c.drawLine(x0, y0, x0, y1);
            c.restore();

            float max = 0;
            for (float v : values) if (v > max) max = v;
            if (max <= 0) max = 1f;

            int slot = chartW / values.length;
            int barW = (int) (slot * 0.7f);
            for (int i = 0; i < values.length; i++) {
                int bh = (int) ((values[i] / max) * chartH);
                int x = x0 + i * slot + (slot - barW) / 2;
                int y = y0 - bh;
                c.save();
                c.setFillColor(barColor);
                c.fillRect(DRect.of(x, y, barW, bh));
                c.restore();
                if (labels != null && i < labels.length) {
                    c.save();
                    c.setFont(DFont.sans(10f));
                    c.text(labels[i], x, y0 + 12, labelColor);
                    c.restore();
                }
            }
        }
        return out;
    }

    public static BufferedImage scatter(float[] xs, float[] ys, DColor dotColor,
                                        int widthPx, int heightPx, int paddingPx, DColor background,
                                        DColor axisColor) {
        Objects.requireNonNull(xs, "xs");
        Objects.requireNonNull(ys, "ys");
        BufferedImage out = new BufferedImage(widthPx, heightPx, BufferedImage.TYPE_INT_ARGB);
        try (DrawingCanvas c = DrawingCanvas.on(out)) {
            c.clear(background == null ? DColor.of("white") : background);
            int chartW = widthPx - 2 * paddingPx;
            int chartH = heightPx - 2 * paddingPx;
            c.save();
            c.setPen(DPen.solid(1f, axisColor).colored(axisColor));
            c.setStrokeColor(axisColor);
            int x0 = paddingPx, y0 = heightPx - paddingPx;
            int x1 = widthPx - paddingPx, y1 = paddingPx;
            c.drawLine(x0, y0, x1, y0);
            c.drawLine(x0, y0, x0, y1);
            c.restore();

            float minX = Float.POSITIVE_INFINITY, maxX = Float.NEGATIVE_INFINITY;
            float minY = Float.POSITIVE_INFINITY, maxY = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < xs.length; i++) {
                if (xs[i] < minX) minX = xs[i]; if (xs[i] > maxX) maxX = xs[i];
                if (ys[i] < minY) minY = ys[i]; if (ys[i] > maxY) maxY = ys[i];
            }
            float rx = maxX - minX, ry = maxY - minY;
            if (rx <= 0) rx = 1;
            if (ry <= 0) ry = 1;
            c.save();
            c.setFillColor(dotColor);
            for (int i = 0; i < xs.length; i++) {
                int x = x0 + (int) ((xs[i] - minX) / rx * chartW);
                int y = y0 - (int) ((ys[i] - minY) / ry * chartH);
                c.fillEllipse(DEllipse.of(x, y, 3, 3));
            }
            c.restore();
        }
        return out;
    }

    public static BufferedImage pie(float[] values, DColor[] colors,
                                    int diameterPx, DColor background, DColor borderColor) {
        Objects.requireNonNull(values, "values");
        BufferedImage out = new BufferedImage(diameterPx, diameterPx, BufferedImage.TYPE_INT_ARGB);
        try (DrawingCanvas c = DrawingCanvas.on(out)) {
            c.clear(background == null ? DColor.of("white") : background);
            float total = 0; for (float v : values) total += v;
            if (total <= 0) return out;
            int cx = diameterPx / 2, cy = diameterPx / 2;
            int r = diameterPx / 2 - 1;
            double start = -90;
            for (int i = 0; i < values.length; i++) {
                double sweep = values[i] / total * 360.0;
                DColor col = colors == null ? DColor.hsv((float) i / values.length, 0.8f, 0.95f) : colors[i % colors.length];
                c.save();
                c.setFillColor(col);
                c.fillArc(DRect.of(cx - r, cy - r, 2 * r, 2 * r), start, sweep);
                c.restore();
                if (borderColor != null) {
                    c.save();
                    c.setPen(DPen.solid(1f, borderColor).colored(borderColor));
                    c.setStrokeColor(borderColor);
                    c.drawArc(DRect.of(cx - r, cy - r, 2 * r, 2 * r), start, sweep);
                    c.restore();
                }
                start += sweep;
            }
        }
        return out;
    }
}