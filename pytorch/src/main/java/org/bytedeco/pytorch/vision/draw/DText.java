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

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.font.LineBreakMeasurer;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Higher-level typography helpers built on top of {@link DrawingCanvas}.
 *
 * <p>Supports:
 * <ul>
 *   <li>Per-character colored text (e.g. rainbow / typewriter effects)</li>
 *   <li>Outline / shadow / glow text effects</li>
 *   <li>Multi-line block layout with alignment, indent, bullet, leading</li>
 *   <li>Vertical CJK writing modes</li>
 *   <li>Bitmap text raster — produces a tight-cropped ARGB image (used for
 *       on-the-fly font glyph atlases by model serving layers)</li>
 * </ul>
 */
public final class DText {

    private DText() {}

    // ---- per-character colored text --------------------------------------

    /** Draw {@code s} where each character's color is computed by {@code palette}. */
    public static void perChar(DrawingCanvas c, String s, float x, float y, DFont font, ColorFn palette) {
        Objects.requireNonNull(s, "s");
        Objects.requireNonNull(palette, "palette");
        if (s.isEmpty()) return;
        Font awt = font.toAwt();
        FontRenderContext frc = c.raw().getFontRenderContext();
        GlyphVector gv = awt.createGlyphVector(frc, s);
        float dx = x;
        for (int i = 0; i < s.length(); i++) {
            DColor color = palette.apply(s.charAt(i), i, (float) i / Math.max(1, s.length() - 1));
            c.setStrokeColor(color);
            c.setFillColor(color);
            c.raw().setFont(awt);
            c.raw().setPaint(color.toAwt());
            char ch = s.charAt(i);
            c.raw().drawString(String.valueOf(ch), dx, y);
            dx += gv.getGlyphMetrics(i).getAdvance() * (1f + font.tracking);
        }
    }

    /** Rainbow palette by character index. */
    public static void rainbow(DrawingCanvas c, String s, float x, float y, DFont font) {
        perChar(c, s, x, y, font, (ch, idx, t) -> DColor.hsv(t, 0.85f, 1f));
    }

    @FunctionalInterface
    public interface ColorFn {
        DColor apply(char ch, int idx, float t);
    }

    // ---- shadow / outline / glow -----------------------------------------

    /**
     * Draw text with a drop shadow at {@code (dx, dy)} with the given blur radius
     * (multi-pass alpha offset approximation; no real Gaussian in pure AWT).
     */
    public static void shadow(DrawingCanvas c, String s, float x, float y, DFont font, DColor text, DColor shadow, float dx, float dy) {
        if (s == null || s.isEmpty()) return;
        c.save();
        c.setOpacity(0.35f);
        c.setFont(font);
        c.text(s, x + dx, y + dy, shadow);
        c.setOpacity(1f);
        c.text(s, x, y, text);
        c.restore();
    }

    /**
     * Outline text: draw stroke around glyph outlines using {@link java.awt.font.GlyphVector}
     * shapes, then fill the inside.
     */
    public static void outline(DrawingCanvas c, String s, float x, float y, DFont font, DColor fill, DColor stroke, float strokeWidth) {
        if (s == null || s.isEmpty()) return;
        c.save();
        Font awt = font.toAwt();
        FontRenderContext frc = c.raw().getFontRenderContext();
        GlyphVector gv = awt.createGlyphVector(frc, s);
        for (int i = 0; i < gv.getNumGlyphs(); i++) {
            java.awt.Shape outlineShape = gv.getGlyphOutline(i, x, y);
            c.raw().setColor(stroke.toAwt());
            c.raw().setStroke(new java.awt.BasicStroke(strokeWidth, java.awt.BasicStroke.CAP_ROUND, java.awt.BasicStroke.JOIN_ROUND));
            c.raw().draw(outlineShape);
            c.raw().setColor(fill.toAwt());
            c.raw().fill(outlineShape);
        }
        c.restore();
    }

    /** Glow: draw text multiple times with progressively larger transparent strokes. */
    public static void glow(DrawingCanvas c, String s, float x, float y, DFont font, DColor color, float radius) {
        if (s == null || s.isEmpty()) return;
        c.save();
        c.setFont(font);
        float r = Math.max(1f, radius);
        for (int i = (int) r; i >= 1; i--) {
            float alpha = 0.18f / (i + 1);
            c.setOpacity(alpha);
            outline(c, s, x, y, font, color, color, i * 2f);
        }
        c.setOpacity(1f);
        c.text(s, x, y, color);
        c.restore();
    }

    // ---- block layout -----------------------------------------------------

    /**
     * Wrap text inside {@code rect}; supports alignment, indent, leading.
     * Returns total height drawn.
     */
    public static float block(DrawingCanvas c, String text, DRect rect, DFont font, int hAlign, float leading, float firstLineIndent) {
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) return 0f;
        c.save();
        c.setFont(font);
        FontRenderContext frc = c.raw().getFontRenderContext();
        AttributedString as = new AttributedString(text);
        as.addAttribute(TextAttribute.FONT, font.toAwt());
        AttributedCharacterIterator it = as.getIterator();
        int start = it.getBeginIndex();
        int end = it.getEndIndex();
        LineBreakMeasurer lbm = new LineBreakMeasurer(it, frc);
        lbm.setPosition(start);
        float y = rect.y + font.ascent(frc);
        float maxWidth = rect.width;
        boolean first = true;
        while (lbm.getPosition() < end) {
            int next = lbm.nextOffset(maxWidth - (first ? firstLineIndent : 0));
            int limit = next;
            if (limit < end) {
                int lastSpace = text.lastIndexOf(' ', limit - 1);
                if (lastSpace >= start) limit = lastSpace + 1;
            }
            TextLayout layout = lbm.nextLayout(maxWidth - (first ? firstLineIndent : 0), limit, false);
            float x = rect.x + (first ? firstLineIndent : 0);
            float advance = layout.getAdvance();
            if (hAlign == 1) x += (maxWidth - (first ? firstLineIndent : 0) - advance) / 2;
            else if (hAlign == 2) x += maxWidth - (first ? firstLineIndent : 0) - advance;
            layout.draw(c.raw(), x, y);
            y += layout.getAscent() + layout.getDescent() + layout.getLeading();
            first = false;
        }
        c.restore();
        return y - rect.y;
    }

    /** Bulleted list inside {@code rect}; {@code bullets} may be {@code null} for "•". */
    public static float bulleted(DrawingCanvas c, List<String> items, DRect rect, DFont font, String bullet) {
        if (items == null || items.isEmpty()) return 0f;
        StringBuilder sb = new StringBuilder();
        String b = bullet == null ? "• " : (bullet + " ");
        for (String it : items) sb.append(b).append(it).append('\n');
        return block(c, sb.toString(), rect, font, 0, 1.0f, font.size() * 1.5f);
    }

    // ---- vertical writing (CJK) -------------------------------------------

    /**
     * Render text vertically, one character per line, top-to-bottom, right-to-left.
     * Each line is at {@code (x, y + i*lineSpacing)}.
     */
    public static void vertical(DrawingCanvas c, String s, float x, float y, DFont font, DColor color, float lineSpacing) {
        if (s == null || s.isEmpty()) return;
        c.save();
        c.setFont(font);
        for (int i = 0; i < s.length(); i++) {
            c.text(String.valueOf(s.charAt(i)), x, y + i * lineSpacing, color);
        }
        c.restore();
    }

    // ---- bitmap text raster ----------------------------------------------

    /**
     * Render {@code text} as a tightly-cropped ARGB bitmap. Useful for building
     * small glyph atlases for model pipelines (e.g. DALL·E label rendering).
     *
     * <p>Returns a {@link Bitmap} holding the image plus the offset of the
     * baseline so the caller can compose the glyph correctly.
     */
    public static Bitmap raster(String text, DFont font, DColor color) {
        Objects.requireNonNull(text, "text");
        if (text.isEmpty()) return new Bitmap(new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB), 0, 0);
        Font awt = font.toAwt();
        BufferedImage probe = new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB);
        FontRenderContext frc = probe.createGraphics().getFontRenderContext();
        Rectangle2D bounds = awt.getStringBounds(text, frc);
        int w = Math.max(1, (int) Math.ceil(bounds.getWidth()));
        int h = Math.max(1, (int) Math.ceil(bounds.getHeight()));
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(java.awt.RenderingHints.KEY_TEXT_ANTIALIASING,
                java.awt.RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(java.awt.RenderingHints.KEY_FRACTIONALMETRICS,
                java.awt.RenderingHints.VALUE_FRACTIONALMETRICS_ON);
        g.setFont(awt);
        g.setPaint(color.toAwt());
        g.drawString(text, -Math.round((float) bounds.getX()), -Math.round((float) bounds.getY()));
        g.dispose();
        return new Bitmap(img, Math.round((float) bounds.getX()), Math.round((float) bounds.getY()));
    }

    /** Tight-cropped bitmap output. */
    public static final class Bitmap {
        public final BufferedImage image;
        /** Offset of the original glyph's left baseline relative to {@code image}. */
        public final int offsetX;
        public final int offsetY;
        public Bitmap(BufferedImage image, int offsetX, int offsetY) {
            this.image = image; this.offsetX = offsetX; this.offsetY = offsetY;
        }
        public int width() { return image.getWidth(); }
        public int height() { return image.getHeight(); }
    }

    // ---- measure utilities -----------------------------------------------

    /** Per-character advances for the given text in the current font. */
    public static float[] advances(String s, DFont font, FontRenderContext frc) {
        if (s == null) return new float[0];
        GlyphVector gv = font.toAwt().createGlyphVector(frc, s);
        float[] adv = new float[s.length()];
        for (int i = 0; i < s.length(); i++) {
            adv[i] = gv.getGlyphMetrics(i).getAdvance() * (1f + font.tracking);
        }
        return adv;
    }

    /** Break the given text into lines that fit inside {@code maxWidth}. */
    public static List<String> wrap(String text, float maxWidth, DFont font, FontRenderContext frc) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isEmpty()) return out;
        Font awt = font.toAwt();
        AttributedString as = new AttributedString(text);
        as.addAttribute(TextAttribute.FONT, awt);
        LineBreakMeasurer lbm = new LineBreakMeasurer(as.getIterator(), frc);
        int start = as.getIterator().getBeginIndex();
        int end = as.getIterator().getEndIndex();
        lbm.setPosition(start);
        StringBuilder cur = new StringBuilder();
        while (lbm.getPosition() < end) {
            int next = lbm.nextOffset(maxWidth);
            int limit = next < end ? next : end;
            TextLayout layout = lbm.nextLayout(maxWidth, limit, false);
            cur.append(text, lbm.getPosition() - layout.getCharacterCount(), lbm.getPosition());
            if (lbm.getPosition() < end) cur.append('\n');
        }
        for (String line : cur.toString().split("\n")) out.add(line);
        return out;
    }
}