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

import org.bytedeco.pytorch.vision.pillow.Image;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Pillow-style drawing API for the {@link org.bytedeco.pytorch.vision.draw} module.
 *
 * <p>Provides a familiar {@code ImageDraw} surface for users coming from
 * Pillow, while delegating the actual rendering to {@link DrawingCanvas}
 * (and therefore Java's {@link java.awt.Graphics2D}).
 *
 * <pre>{@code
 * Image im = Image.new_("RGB", new int[]{640, 480}, "white");
 * try (ImageDraw d = new ImageDraw(im)) {
 *     d.rectangle(new int[]{20, 20, 200, 100}, new String[]{"red", "blue"}, new int[]{0, 2});
 *     d.text("hello", new int[]{40, 80}, DColor.of("black"));
 *     d.line(new int[]{0, 0, 100, 100}, DColor.of("lime"), 2);
 * }
 * }</pre>
 *
 * <p>Color / fill / stroke arguments are overloaded by convention:
 * <ul>
 *   <li>{@code null} / empty fill = no fill</li>
 *   <li>{@code null} / empty stroke = no outline</li>
 *   <li>{@code width = 0} is treated as "1 pixel"</li>
 * </ul>
 */
public final class ImageDraw implements AutoCloseable {

    private final Image image;
    private final DrawingCanvas canvas;

    public ImageDraw(Image image) {
        Objects.requireNonNull(image, "image");
        this.image = image;
        this.canvas = DrawingCanvas.on(image.toBufferedImage());
    }

    public ImageDraw(BufferedImage bimg) {
        Objects.requireNonNull(bimg, "bimg");
        this.image = null;
        this.canvas = DrawingCanvas.on(bimg);
    }

    public DrawingCanvas canvas() { return canvas; }
    public BufferedImage bufferedImage() { return canvas.image(); }
    public Image image() { return image; }

    // ---- Shapes -----------------------------------------------------------

    public ImageDraw line(int[] xy, DColor color, int widthPx) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length != 4) throw new IllegalArgumentException("xy must be length 4");
        canvas.save();
        canvas.setPen(DPen.solid(Math.max(1, widthPx), color).colored(color));
        canvas.setStrokeColor(color);
        canvas.drawLine(xy[0], xy[1], xy[2], xy[3]);
        canvas.restore();
        return this;
    }

    public ImageDraw line(int[] xy) { return line(xy, DColor.of("black"), 1); }

    public ImageDraw rectangle(int[] xy, DColor fill, DColor stroke, int strokeWidthPx) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length != 4) throw new IllegalArgumentException("xy must be length 4");
        DRect r = DRect.of(Math.min(xy[0], xy[2]), Math.min(xy[1], xy[3]),
                Math.abs(xy[2] - xy[0]), Math.abs(xy[3] - xy[1]));
        canvas.save();
        if (fill != null) {
            canvas.setFillColor(fill);
            canvas.fillRect(r);
        }
        if (stroke != null) {
            canvas.setPen(DPen.solid(Math.max(1, strokeWidthPx), stroke).colored(stroke));
            canvas.setStrokeColor(stroke);
            canvas.drawRect(r);
        }
        canvas.restore();
        return this;
    }

    /** Pillow-style: {@code fill = null} → no fill; {@code width = null} → no stroke. */
    public ImageDraw rectangle(int[] xy, String fill, String outline, Integer width) {
        return rectangle(xy, fill == null ? null : DColor.of(fill), outline == null ? null : DColor.of(outline), width == null ? 1 : width);
    }

    public ImageDraw ellipse(int[] xy, DColor fill, DColor stroke, int strokeWidthPx) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length != 4) throw new IllegalArgumentException("xy must be length 4");
        float cx = (xy[0] + xy[2]) / 2f;
        float cy = (xy[1] + xy[3]) / 2f;
        float rx = Math.abs(xy[2] - xy[0]) / 2f;
        float ry = Math.abs(xy[3] - xy[1]) / 2f;
        DEllipse e = DEllipse.of(cx, cy, rx, ry);
        canvas.save();
        if (fill != null) { canvas.setFillColor(fill); canvas.fillEllipse(e); }
        if (stroke != null) {
            canvas.setPen(DPen.solid(Math.max(1, strokeWidthPx), stroke).colored(stroke));
            canvas.setStrokeColor(stroke);
            canvas.drawEllipse(e);
        }
        canvas.restore();
        return this;
    }

    public ImageDraw ellipse(int[] xy, String fill, String outline, Integer width) {
        return ellipse(xy, fill == null ? null : DColor.of(fill), outline == null ? null : DColor.of(outline), width == null ? 1 : width);
    }

    public ImageDraw polygon(int[] xs, int[] ys, DColor fill, DColor stroke, int strokeWidthPx) {
        Objects.requireNonNull(xs, "xs");
        Objects.requireNonNull(ys, "ys");
        if (xs.length != ys.length || xs.length < 3) throw new IllegalArgumentException("polygon: need >= 3 vertices");
        DPoint[] pts = new DPoint[xs.length];
        for (int i = 0; i < xs.length; i++) pts[i] = DPoint.of(xs[i], ys[i]);
        canvas.save();
        if (fill != null) { canvas.setFillColor(fill); canvas.fillPolygon(pts); }
        if (stroke != null) {
            canvas.setPen(DPen.solid(Math.max(1, strokeWidthPx), stroke).colored(stroke));
            canvas.setStrokeColor(stroke);
            canvas.drawPolygon(pts);
        }
        canvas.restore();
        return this;
    }

    public ImageDraw arc(int[] xy, int startDeg, int endDeg, DColor fill, DColor stroke, int strokeWidthPx) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length != 4) throw new IllegalArgumentException("xy must be length 4");
        DRect r = DRect.of(Math.min(xy[0], xy[2]), Math.min(xy[1], xy[3]),
                Math.abs(xy[2] - xy[0]), Math.abs(xy[3] - xy[1]));
        canvas.save();
        if (fill != null) {
            canvas.setFillColor(fill);
            canvas.fillArc(r, startDeg, endDeg - startDeg);
        }
        if (stroke != null) {
            canvas.setPen(DPen.solid(Math.max(1, strokeWidthPx), stroke).colored(stroke));
            canvas.setStrokeColor(stroke);
            canvas.drawArc(r, startDeg, endDeg - startDeg);
        }
        canvas.restore();
        return this;
    }

    public ImageDraw point(int[] xy, DColor color) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length != 2) throw new IllegalArgumentException("xy must be length 2");
        canvas.save();
        canvas.setFillColor(color);
        canvas.fillEllipse(DEllipse.of(xy[0], xy[1], 0.5f, 0.5f));
        canvas.restore();
        return this;
    }

    public ImageDraw roundedRectangle(int[] xy, float radius, DColor fill, DColor stroke, int strokeWidthPx) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length != 4) throw new IllegalArgumentException("xy must be length 4");
        DRect r = DRect.of(Math.min(xy[0], xy[2]), Math.min(xy[1], xy[3]),
                Math.abs(xy[2] - xy[0]), Math.abs(xy[3] - xy[1]));
        canvas.save();
        if (fill != null) { canvas.setFillColor(fill); canvas.fillRoundedRect(r, radius); }
        if (stroke != null) {
            canvas.setPen(DPen.solid(Math.max(1, strokeWidthPx), stroke).colored(stroke));
            canvas.setStrokeColor(stroke);
            canvas.drawRoundedRect(r, radius);
        }
        canvas.restore();
        return this;
    }

    // ---- Text -------------------------------------------------------------

    public ImageDraw text(String text, int[] xy, DColor color) {
        return text(text, xy, color, DFont.sans(12f));
    }

    public ImageDraw text(String text, int[] xy, DColor color, DFont font) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length < 2) throw new IllegalArgumentException("xy must be length >= 2");
        canvas.save();
        canvas.setFont(font);
        canvas.text(text, xy[0], xy[1], color);
        canvas.restore();
        return this;
    }

    public ImageDraw text(String text, int[] xy, DFont font, DColor fill, DColor stroke, float strokeWidth) {
        Objects.requireNonNull(xy, "xy");
        if (xy.length < 2) throw new IllegalArgumentException("xy must be length >= 2");
        canvas.save();
        canvas.setFont(font);
        if (stroke != null) {
            DText.outline(canvas, text, xy[0], xy[1], font, fill, stroke, strokeWidth);
        } else {
            canvas.text(text, xy[0], xy[1], fill);
        }
        canvas.restore();
        return this;
    }

    // ---- bitmap ops ------------------------------------------------------

    public ImageDraw bitmap(DColor color, int[] xy, BitmapFn bitmap) {
        Objects.requireNonNull(xy, "xy");
        Objects.requireNonNull(bitmap, "bitmap");
        if (xy.length != 2) throw new IllegalArgumentException("xy must be length 2");
        BufferedImage b = bitmap.render();
        canvas.drawImage(b, xy[0], xy[1]);
        return this;
    }

    @FunctionalInterface
    public interface BitmapFn {
        BufferedImage render();
    }

    @Override public void close() { canvas.close(); }

    /** No-op for in-memory rendering; reserved for streaming surfaces. */
    public void flush() { /* no-op */ }
}