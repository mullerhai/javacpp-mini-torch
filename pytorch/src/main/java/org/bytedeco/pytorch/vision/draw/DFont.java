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

import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.awt.font.FontRenderContext;
import java.awt.font.TextAttribute;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Font handle used by the {@link org.bytedeco.pytorch.vision.draw} module.
 *
 * <p>Bridges AWT {@link Font} with a richer fluent builder:
 * <ul>
 *   <li>{@code family} (logical name with platform fallback), {@code style}, {@code size}</li>
 *   <li>tracking / leading for fine typography</li>
 *   <li>fallback chain across multiple families (CJK / emoji / math)</li>
 * </ul>
 *
 * <pre>{@code
 * DFont f = DFont.builder("SansSerif")
 *     .size(28f)
 *     .bold()
 *     .tracking(0.02f)
 *     .fallback("Noto Sans CJK SC", "Noto Emoji")
 *     .build();
 * }</pre>
 */
public final class DFont {

    public enum Style { PLAIN, BOLD, ITALIC, BOLD_ITALIC }

    public final String family;
    public final Style style;
    /** Font size in points; accessible via {@link #size()}. */
    private final float sizeValue;
    public final float tracking;     // letter spacing, fraction of em
    public final float leading;      // line height multiplier
    public final List<String> fallback;

    private DFont(String family, Style style, float size, float tracking, float leading, List<String> fallback) {
        this.family = family == null || family.isEmpty() ? "SansSerif" : family;
        this.style = style == null ? Style.PLAIN : style;
        this.sizeValue = size <= 0f ? 12f : size;
        this.tracking = tracking;
        this.leading = leading <= 0f ? 1.2f : leading;
        this.fallback = fallback == null ? List.of() : List.copyOf(fallback);
    }

    /** Public accessor for the immutable size field. */
    public float size() { return sizeValue; }

    /** Legacy compatibility — the size field is now exposed via {@link #size()}. */
    public float pt() { return sizeValue; }

    public static DFont of(String family, float size) { return builder(family).size(size).build(); }
    public static DFont sans(float size) { return of("SansSerif", size); }
    public static DFont serif(float size) { return of("Serif", size); }
    public static DFont mono(float size) { return of("Monospaced", size); }

    /** Returns a new {@link DFont} with the same family but bold. */
    public DFont bold() {
        return new DFont(family, style == Style.ITALIC ? Style.BOLD_ITALIC : Style.BOLD,
                sizeValue, tracking, leading, fallback);
    }

    /** Returns a new {@link DFont} with the same family but italic. */
    public DFont italic() {
        return new DFont(family, style == Style.BOLD ? Style.BOLD_ITALIC : Style.ITALIC,
                sizeValue, tracking, leading, fallback);
    }

    public DFont withSize(float size) {
        return new DFont(family, style, size, tracking, leading, fallback);
    }

    public static Builder builder(String family) { return new Builder(family); }

    public static final class Builder {
        private String family;
        private Style style = Style.PLAIN;
        private float size = 12f;
        private float tracking = 0f;
        private float leading = 1.2f;
        private final List<String> fallback = new ArrayList<>();

        Builder(String family) { this.family = family; }

        public Builder family(String family) { this.family = family; return this; }
        public Builder size(float size) { this.size = size; return this; }
        public Builder bold() { this.style = style == Style.ITALIC ? Style.BOLD_ITALIC : Style.BOLD; return this; }
        public Builder italic() { this.style = style == Style.BOLD ? Style.BOLD_ITALIC : Style.ITALIC; return this; }
        public Builder plain() { this.style = Style.PLAIN; return this; }
        public Builder style(Style s) { this.style = s; return this; }
        public Builder tracking(float t) { this.tracking = t; return this; }
        public Builder leading(float l) { this.leading = l; return this; }
        public Builder fallback(String... fams) { fallback.addAll(Arrays.asList(fams)); return this; }

        public DFont build() { return new DFont(family, style, size, tracking, leading, fallback); }
    }

    // ---- measurement ------------------------------------------------------

    /** Width in pixels of {@code text} using the current font and a default render context. */
    public float stringWidth(String text, FontRenderContext frc) {
        if (text == null || text.isEmpty()) return 0f;
        return (float) toAwt().getStringBounds(text, frc).getWidth();
    }

    /** Bounding box (logical pixels). */
    public Rectangle2D stringBounds(String text, FontRenderContext frc) {
        return toAwt().getStringBounds(text, frc);
    }

    /** Line height in pixels (ascent + descent + leading padding). */
    public float lineHeight(FontRenderContext frc) {
        java.awt.font.LineMetrics lm = toAwt().getLineMetrics("Mg", frc);
        return lm.getHeight() * leading;
    }

    /** Ascent in pixels. */
    public float ascent(FontRenderContext frc) {
        return toAwt().getLineMetrics("Mg", frc).getAscent();
    }

    public Font toAwt() {
        int s = Font.PLAIN;
        if (style == Style.BOLD) s = Font.BOLD;
        else if (style == Style.ITALIC) s = Font.ITALIC;
        else if (style == Style.BOLD_ITALIC) s = Font.BOLD | Font.ITALIC;
        Font primary = available(family, s) ? new Font(family, s, Math.round(sizeValue))
                : new Font(Font.SANS_SERIF, s, Math.round(sizeValue));
        if (fallback.isEmpty()) return primary;
        Map<TextAttribute, Object> attrs = new HashMap<>();
        attrs.put(TextAttribute.FAMILY, primary.getFamily());
        attrs.put(TextAttribute.SIZE, sizeValue);
        attrs.put(TextAttribute.KERNING, TextAttribute.KERNING_ON);
        return primary.deriveFont(attrs);
    }

    private static boolean available(String family, int style) {
        if (family == null || family.isEmpty()) return false;
        String[] avail = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
        for (String f : avail) if (f.equalsIgnoreCase(family)) return true;
        // logical names
        if (family.equalsIgnoreCase("SansSerif")) return true;
        if (family.equalsIgnoreCase("Serif")) return true;
        if (family.equalsIgnoreCase("Monospaced")) return true;
        if (family.equalsIgnoreCase("Dialog")) return true;
        if (family.equalsIgnoreCase("DialogInput")) return true;
        return false;
    }

    /** List available font family names on the current JVM. */
    public static String[] availableFamilies() {
        return GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
    }

    @Override public boolean equals(Object o) {
        if (!(o instanceof DFont)) return false;
        DFont f = (DFont) o;
        return f.sizeValue == sizeValue && f.style == style && Objects.equals(f.family, family) && f.tracking == tracking;
    }

    @Override public int hashCode() {
        return Objects.hash(family, style, sizeValue, tracking);
    }

    @Override public String toString() {
        return "DFont[" + family + " " + style + " " + sizeValue + "pt]";
    }
}