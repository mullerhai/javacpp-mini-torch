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
 * Tile multiple images into a single canvas — used for inference dashboards,
 * batch visualization, attention-map grids, etc.
 *
 * <pre>{@code
 * BufferedImage mosaic = DMosaic.grid(images, 4, DMosaic.Align.CENTER, 8, DColor.of("#222"));
 * }</pre>
 */
public final class DMosaic {

    public enum Align { START, CENTER, END }

    private DMosaic() {}

    /** Lay out {@code imgs} in {@code colsPerRow} columns with even spacing. */
    public static BufferedImage grid(BufferedImage[] imgs, int colsPerRow,
                                     Align align, int paddingPx, DColor background) {
        Objects.requireNonNull(imgs, "imgs");
        if (colsPerRow <= 0) throw new IllegalArgumentException("colsPerRow must be > 0");
        if (imgs.length == 0) return new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
        // 1) Pick tile size = max dims across all images
        int tileW = 0, tileH = 0;
        for (BufferedImage i : imgs) {
            if (i == null) continue;
            if (i.getWidth() > tileW) tileW = i.getWidth();
            if (i.getHeight() > tileH) tileH = i.getHeight();
        }
        int rows = (imgs.length + colsPerRow - 1) / colsPerRow;
        int width = colsPerRow * tileW + (colsPerRow + 1) * paddingPx;
        int height = rows * tileH + (rows + 1) * paddingPx;
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        java.awt.Graphics2D g = out.createGraphics();
        g.setColor(background == null ? new java.awt.Color(0, 0, 0, 0) : background.toAwt());
        g.fillRect(0, 0, width, height);
        for (int idx = 0; idx < imgs.length; idx++) {
            BufferedImage img = imgs[idx];
            if (img == null) continue;
            int col = idx % colsPerRow;
            int row = idx / colsPerRow;
            int x = col * (tileW + paddingPx) + paddingPx;
            int y = row * (tileH + paddingPx) + paddingPx;
            int ix = x, iy = y;
            switch (align) {
                case CENTER:
                    ix = x + (tileW - img.getWidth()) / 2;
                    iy = y + (tileH - img.getHeight()) / 2;
                    break;
                case END:
                    ix = x + (tileW - img.getWidth());
                    iy = y + (tileH - img.getHeight());
                    break;
                default: break;
            }
            g.drawImage(img, ix, iy, null);
        }
        g.dispose();
        return out;
    }

    public static BufferedImage grid(BufferedImage[] imgs, int colsPerRow) {
        return grid(imgs, colsPerRow, Align.START, 0, DColor.of(0, 0, 0, 0));
    }

    /** Vertical strip / horizontal strip of equal-height images. */
    public static BufferedImage strip(BufferedImage[] imgs, int paddingPx, DColor background, boolean horizontal) {
        if (imgs.length == 0) return new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
        int total = paddingPx * (imgs.length + 1);
        int w = 0, h = 0, maxCross = 0;
        for (BufferedImage i : imgs) {
            if (i == null) continue;
            if (horizontal) { w += i.getWidth(); maxCross = Math.max(maxCross, i.getHeight()); }
            else { h += i.getHeight(); maxCross = Math.max(maxCross, i.getWidth()); }
        }
        if (horizontal) { w += total; h = maxCross + 2 * paddingPx; }
        else { h += total; w = maxCross + 2 * paddingPx; }
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        java.awt.Graphics2D g = out.createGraphics();
        g.setColor(background == null ? new java.awt.Color(0, 0, 0, 0) : background.toAwt());
        g.fillRect(0, 0, w, h);
        int cursor = paddingPx;
        for (BufferedImage i : imgs) {
            if (i == null) continue;
            if (horizontal) {
                int y = paddingPx + (maxCross - i.getHeight()) / 2;
                g.drawImage(i, cursor, y, null);
                cursor += i.getWidth() + paddingPx;
            } else {
                int x = paddingPx + (maxCross - i.getWidth()) / 2;
                g.drawImage(i, x, cursor, null);
                cursor += i.getHeight() + paddingPx;
            }
        }
        g.dispose();
        return out;
    }

    public static BufferedImage strip(BufferedImage[] imgs, int paddingPx) {
        return strip(imgs, paddingPx, DColor.of(0, 0, 0, 0), true);
    }
}