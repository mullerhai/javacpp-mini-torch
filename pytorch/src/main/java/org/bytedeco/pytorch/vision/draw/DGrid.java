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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Grid drawing helper for charts, matrix visualization, confusion matrices,
 * activation maps, etc.
 */
public final class DGrid {

    private DGrid() {}

    /** Draw an N×N (or M×N) grid of cells with optional labels. */
    public static BufferedImage renderGrid(double[][] matrix, String[] rowLabels, String[] colLabels,
                                           DColor cellColor, DColor borderColor, DColor labelColor,
                                           DFont labelFont, int cellSizePx, int paddingPx, DColor background) {
        Objects.requireNonNull(matrix, "matrix");
        int rows = matrix.length;
        int cols = rows == 0 ? 0 : matrix[0].length;
        int w = cols * cellSizePx + (cols + 1) * paddingPx;
        int h = rows * cellSizePx + (rows + 1) * paddingPx;
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        try (DrawingCanvas c = DrawingCanvas.on(out)) {
            c.clear(background == null ? DColor.of("white") : background);
            double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
            for (double[] row : matrix) for (double v : row) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            if (max - min < 1e-9) { min = 0; max = 1; }
            for (int r = 0; r < rows; r++) {
                for (int cc = 0; cc < cols; cc++) {
                    double t = (matrix[r][cc] - min) / (max - min);
                    DColor cellFill = lerpColor(DColor.of("white"), cellColor, (float) t);
                    DRect rect = DRect.of(
                            paddingPx + cc * (cellSizePx + paddingPx),
                            paddingPx + r * (cellSizePx + paddingPx),
                            cellSizePx, cellSizePx);
                    c.save();
                    c.setFillColor(cellFill);
                    c.fillRect(rect);
                    c.setPen(DPen.solid(0.5f, borderColor).colored(borderColor));
                    c.setStrokeColor(borderColor);
                    c.drawRect(rect);
                    c.restore();
                    // optional label inside cell
                    String txt = String.format("%.2f", matrix[r][cc]);
                    c.save();
                    c.setFont(labelFont);
                    c.text(txt, rect.x + 4, rect.y + labelFont.size() + 2, labelColor);
                    c.restore();
                }
            }
            // row labels (left)
            if (rowLabels != null) {
                for (int r = 0; r < rows; r++) {
                    c.save();
                    c.setFont(labelFont);
                    c.text(rowLabels[r], 0, paddingPx + r * (cellSizePx + paddingPx) + labelFont.size() + 2, labelColor);
                    c.restore();
                }
            }
            if (colLabels != null) {
                for (int cc = 0; cc < cols; cc++) {
                    c.save();
                    c.setFont(labelFont);
                    c.text(colLabels[cc], paddingPx + cc * (cellSizePx + paddingPx), labelFont.size(), labelColor);
                    c.restore();
                }
            }
        }
        return out;
    }

    private static DColor lerpColor(DColor a, DColor b, float t) {
        if (t <= 0) return a;
        if (t >= 1) return b;
        return DColor.lerp(a, b, t);
    }

    /** Polyline plot, with optional axes and labels. */
    public static BufferedImage plot(float[][] series, String[] labels,
                                     DColor[] colors, DColor background, DColor axisColor,
                                     int widthPx, int heightPx, int paddingPx) {
        if (series.length == 0) return new BufferedImage(widthPx, heightPx, BufferedImage.TYPE_INT_ARGB);
        int maxLen = 0;
        for (float[] s : series) maxLen = Math.max(maxLen, s.length);
        BufferedImage out = new BufferedImage(widthPx, heightPx, BufferedImage.TYPE_INT_ARGB);
        try (DrawingCanvas c = DrawingCanvas.on(out)) {
            c.clear(background == null ? DColor.of("white") : background);
            // axes
            int x0 = paddingPx, y0 = heightPx - paddingPx;
            int x1 = widthPx - paddingPx, y1 = paddingPx;
            c.save();
            c.setPen(DPen.solid(1f, axisColor).colored(axisColor));
            c.setStrokeColor(axisColor);
            c.drawLine(x0, y0, x1, y0);
            c.drawLine(x0, y0, x0, y1);
            c.restore();

            float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
            for (float[] s : series) for (float v : s) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            if (max - min < 1e-9) { min = 0; max = 1; }

            for (int s = 0; s < series.length; s++) {
                float[] data = series[s];
                DColor col = colors == null ? DColor.of("steelblue") : colors[s % colors.length];
                c.save();
                c.setPen(DPen.solid(1.5f, col).colored(col));
                c.setStrokeColor(col);
                for (int i = 1; i < data.length; i++) {
                    float xa = x0 + ((float) (i - 1) / Math.max(1, maxLen - 1)) * (x1 - x0);
                    float ya = y0 - ((data[i - 1] - min) / (max - min)) * (y0 - y1);
                    float xb = x0 + ((float) i / Math.max(1, maxLen - 1)) * (x1 - x0);
                    float yb = y0 - ((data[i] - min) / (max - min)) * (y0 - y1);
                    c.drawLine(xa, ya, xb, yb);
                }
                c.restore();
            }
        }
        return out;
    }
}