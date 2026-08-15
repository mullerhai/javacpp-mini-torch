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
 * Composer that places multiple images at specific rectangles onto a single
 * canvas — useful for dashboard-style overlays, annotation views, side-by-side
 * comparisons.
 */
public final class DOverlay {

    private DOverlay() {}

    /**
     * Place each {@code (image, rect)} pair onto a fresh canvas sized to fit all
     * rects plus {@code padding}. Returns the composed canvas.
     */
    public static BufferedImage compose(Entry[] entries, int paddingPx, DColor background) {
        Objects.requireNonNull(entries, "entries");
        if (entries.length == 0) return new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
        float maxX2 = 0, maxY2 = 0;
        for (Entry e : entries) {
            float x2 = e.rect.x + e.rect.width;
            float y2 = e.rect.y + e.rect.height;
            if (x2 > maxX2) maxX2 = x2;
            if (y2 > maxY2) maxY2 = y2;
        }
        int w = Math.round(maxX2 + paddingPx);
        int h = Math.round(maxY2 + paddingPx);
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        java.awt.Graphics2D g = out.createGraphics();
        g.setColor(background == null ? new java.awt.Color(0, 0, 0, 0) : background.toAwt());
        g.fillRect(0, 0, w, h);
        for (Entry e : entries) {
            g.drawImage(e.image, Math.round(e.rect.x), Math.round(e.rect.y), null);
        }
        g.dispose();
        return out;
    }

    /** Compose onto an existing canvas. */
    public static void composeOnto(DrawingCanvas c, Entry[] entries) {
        for (Entry e : entries) {
            c.drawImage(e.image, e.rect);
        }
    }

    /** Single overlay entry. */
    public static final class Entry {
        public final BufferedImage image;
        public final DRect rect;
        public Entry(BufferedImage image, DRect rect) {
            this.image = image; this.rect = rect;
        }
        public static Entry of(BufferedImage image, DRect rect) { return new Entry(image, rect); }
    }
}