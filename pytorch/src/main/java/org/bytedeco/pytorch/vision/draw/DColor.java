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
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.vision.draw;

import java.awt.Color;
import java.util.Locale;
import java.util.Objects;

/**
 * Enterprise-grade immutable color type for the {@link org.bytedeco.pytorch.vision.draw}
 * module.
 *
 * <p>Backed by a packed {@code 0xAARRGGBB} int in Java {@link Color} order. Use
 * {@link #toAwt()} to obtain an AWT {@link Color} for direct use with
 * {@link java.awt.Graphics2D}. Channels are stored in <em>straight (non-premultiplied)</em>
 * alpha to match Pillow / torchvision / OpenCV conventions.
 *
 * <p>Accepts construction from:
 * <ul>
 *   <li>Hex strings — {@code "#RRGGBB"}, {@code "#AARRGGBB"}, {@code "0xRRGGBB"}</li>
 *   <li>CSS named colors (140+ entries)</li>
 *   <li>Float channels {@code [0,1]} for normal / advanced imaging pipelines</li>
 *   <li>HSV / HSL triples for color-space aware rendering</li>
 *   <li>YUV, Lab triples</li>
 *   <li>Packed int / separate R,G,B,A int</li>
 * </ul>
 *
 * <pre>{@code
 * Color red = DColor.of(255, 0, 0);
 * Color semi = DColor.of(0x80FF0000);            // ARGB
 * Color hot  = DColor.of("#FF4500");            // CSS-style
 * Color viaHsv = DColor.hsv(0.05, 0.9, 1.0);
 * Color viaGray = DColor.gray(0.5);
 * }</pre>
 */
public final class DColor {

    /** 0xAARRGGBB. */
    public final int argb;
    /** Cached AWT color for performance. */
    private final transient Color awt;

    private DColor(int argb) {
        this.argb = argb;
        this.awt = new Color(argb, true);
    }

    private DColor(int r, int g, int b, int a) {
        this(((a & 0xFF) << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF));
    }

    // ---- factories --------------------------------------------------------

    public static DColor of(int r, int g, int b) { return new DColor(r, g, b, 255); }
    public static DColor of(int r, int g, int b, int a) { return new DColor(r, g, b, a); }

    /** {@code 0xAARRGGBB}. */
    public static DColor of(int argb) { return new DColor(argb); }

    /** {@code 0xRRGGBB}. */
    public static DColor ofRgb(int rgb) {
        return new DColor(0xFF000000
                | ((rgb >>> 16) & 0xFF) << 16
                | ((rgb >>>  8) & 0xFF) <<  8
                | ((rgb      ) & 0xFF));
    }

    public static DColor ofAwt(Color c) {
        Objects.requireNonNull(c, "color");
        return new DColor(c.getRGB());
    }

    /** Parses hex ({@code "#RRGGBB"} or {@code "#AARRGGBB"}) or a CSS name (case-insensitive). */
    public static DColor of(String spec) {
        Objects.requireNonNull(spec, "spec");
        String s = spec.trim();
        if (s.isEmpty()) throw new IllegalArgumentException("empty color spec");
        if (s.charAt(0) == '#') {
            String hex = s.substring(1);
            if (hex.length() == 6) {
                int rgb = Integer.parseInt(hex, 16);
                return ofRgb(rgb);
            } else if (hex.length() == 8) {
                return of((int) Long.parseLong(hex, 16));
            } else if (hex.length() == 3) {
                int r = Integer.parseInt("" + hex.charAt(0) + hex.charAt(0), 16);
                int g = Integer.parseInt("" + hex.charAt(1) + hex.charAt(1), 16);
                int b = Integer.parseInt("" + hex.charAt(2) + hex.charAt(2), 16);
                return of(r, g, b);
            } else {
                throw new IllegalArgumentException("bad hex color: " + spec);
            }
        }
        if (s.startsWith("0x") || s.startsWith("0X") || s.startsWith("rgb(") || s.startsWith("rgba(")) {
            return parseCssFunction(s);
        }
        DColor named = NAMED.get(s.toLowerCase(Locale.ROOT));
        if (named != null) return named;
        throw new IllegalArgumentException("unknown color: " + spec);
    }

    private static DColor parseCssFunction(String s) {
        String t = s.toLowerCase(Locale.ROOT).replace(" ", "");
        if (t.startsWith("0x")) {
            return of((int) Long.parseLong(t.substring(2), 16));
        }
        if (t.startsWith("rgb(")) {
            String[] parts = t.substring(4, t.length() - 1).split(",");
            if (parts.length != 3) throw new IllegalArgumentException("bad rgb(): " + s);
            return of(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
        }
        if (t.startsWith("rgba(")) {
            String[] parts = t.substring(5, t.length() - 1).split(",");
            if (parts.length != 4) throw new IllegalArgumentException("bad rgba(): " + s);
            return of(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]),
                    Integer.parseInt(parts[2]), (int) Math.round(Float.parseFloat(parts[3]) * 255));
        }
        throw new IllegalArgumentException("bad color spec: " + s);
    }

    /** Linear channel {@code [0,1]} RGB. Truncates fractional channel values, matching
     *  Pillow's default behaviour: {@code (int)(v * 255)}. */
    public static DColor ofUnit(float r, float g, float b) {
        return of(truncUnit(clamp01(r)), truncUnit(clamp01(g)), truncUnit(clamp01(b)));
    }

    public static DColor ofUnit(float r, float g, float b, float a) {
        return of(truncUnit(clamp01(r)), truncUnit(clamp01(g)),
                truncUnit(clamp01(b)), truncUnit(clamp01(a)));
    }

    private static int truncUnit(float v) {
        return Math.min(255, Math.max(0, (int) (v * 255f)));
    }

    /** Grayscale {@code [0,1]}. */
    public static DColor gray(float v) { return ofUnit(v, v, v); }

    public static DColor gray(float v, float a) { return ofUnit(v, v, v, a); }

    /**
     * HSV → RGB. {@code h} is in {@code [0,1)}, {@code s}, {@code v} in {@code [0,1]}.
     * Returns fully-opaque color.
     */
    public static DColor hsv(float h, float s, float v) {
        return hsv(h, s, v, 1f);
    }

    public static DColor hsv(float h, float s, float v, float a) {
        float hh = ((h % 1f) + 1f) % 1f * 6f;
        int i = (int) Math.floor(hh);
        float f = hh - i;
        float p = v * (1f - s);
        float q = v * (1f - s * f);
        float tt = v * (1f - s * (1f - f));
        float r, g, b;
        switch (i % 6) {
            case 0: r = v; g = tt; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = tt; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = tt; g = p; b = v; break;
            default: r = v; g = p; b = q;
        }
        return ofUnit(r, g, b, a);
    }

    /**
     * HSL → RGB. {@code h} in {@code [0,1)}, {@code s}, {@code l} in {@code [0,1]}.
     */
    public static DColor hsl(float h, float s, float l) {
        return hsl(h, s, l, 1f);
    }

    public static DColor hsl(float h, float s, float l, float a) {
        float hh = ((h % 1f) + 1f) % 1f * 6f;
        float c = (1f - Math.abs(2f * l - 1f)) * s;
        float x = c * (1f - Math.abs(((hh % 2f) + 2f) % 2f - 1f));
        float m = l - c / 2f;
        float r, g, b;
        int seg = (int) hh;
        switch (seg % 6) {
            case 0: r = c; g = x; b = 0; break;
            case 1: r = x; g = c; b = 0; break;
            case 2: r = 0; g = c; b = x; break;
            case 3: r = 0; g = x; b = c; break;
            case 4: r = x; g = 0; b = c; break;
            default: r = c; g = 0; b = x;
        }
        return ofUnit(r + m, g + m, b + m, a);
    }

    /**
     * BT.601 YUV (Y in {@code [0,1]}, U/V in {@code [-0.5,0.5]}) → RGB.
     */
    public static DColor yuv(float y, float u, float v) {
        float r = y + 1.402f * v;
        float g = y - 0.344136f * u - 0.714136f * v;
        float b = y + 1.772f * u;
        return ofUnit(r, g, b);
    }

    /**
     * CIE Lab (L in {@code [0,100]}, a/b roughly {@code [-128,127]}) → sRGB linear → RGB.
     * White point D65.
     */
    public static DColor lab(float L, float a, float b) {
        float Y = (L + 16f) / 116f;
        float X = a / 500f + Y;
        float Z = Y - b / 200f;
        float fx3 = X * X * X;
        float fy3 = Y * Y * Y;
        float fz3 = Z * Z * Z;
        X = fx3 > 0.008856f ? fx3 : (X - 16f / 116f) / 7.787f;
        Y = fy3 > 0.008856f ? fy3 : (Y - 16f / 116f) / 7.787f;
        Z = fz3 > 0.008856f ? fz3 : (Z - 16f / 116f) / 7.787f;
        X *= 95.047f; Y *= 100.000f; Z *= 108.883f;
        // linear RGB (D65)
        float rl = X * 0.0124f + Y * -0.0042f + Z * -0.0003f;
        float gl = X * -0.0075f + Y * 0.0099f + Z * 0.0002f;
        float bl = X * 0.0022f + Y * -0.0023f + Z * 0.0091f;
        return ofUnit(linearToSrgb(rl), linearToSrgb(gl), linearToSrgb(bl));
    }

    private static float linearToSrgb(float c) {
        if (c <= 0.0031308f) return 12.92f * c;
        return 1.055f * (float) Math.pow(c, 1.0 / 2.4) - 0.055f;
    }

    // ---- channels ---------------------------------------------------------

    public int r() { return (argb >>> 16) & 0xFF; }
    public int g() { return (argb >>> 8) & 0xFF; }
    public int b() { return argb & 0xFF; }
    public int a() { return (argb >>> 24) & 0xFF; }

    public float rUnit() { return r() / 255f; }
    public float gUnit() { return g() / 255f; }
    public float bUnit() { return b() / 255f; }
    public float aUnit() { return a() / 255f; }

    /** Average grayscale (Rec.601). */
    public float luma() { return (0.299f * r() + 0.587f * g() + 0.114f * b()) / 255f; }

    public Color toAwt() { return awt; }

    /** {@link #argb} as {@code 0xRRGGBB}. */
    public int rgb() { return argb & 0xFFFFFF; }

    public DColor withAlpha(int alpha) {
        return new DColor((argb & 0x00FFFFFF) | ((alpha & 0xFF) << 24));
    }

    public DColor withAlpha(float a) {
        return withAlpha(Math.round(clamp01(a) * 255));
    }

    public DColor brighter() {
        Objects.requireNonNull(awt);
        return ofAwt(awt.brighter());
    }

    public DColor darker() {
        Objects.requireNonNull(awt);
        return ofAwt(awt.darker());
    }

    /** Linear interpolation in straight-alpha space. */
    public static DColor lerp(DColor a, DColor b, float t) {
        if (t <= 0f) return a;
        if (t >= 1f) return b;
        float ti = 1f - t;
        return new DColor(
                Math.round(ti * a.r() + t * b.r()),
                Math.round(ti * a.g() + t * b.g()),
                Math.round(ti * a.b() + t * b.b()),
                Math.round(ti * a.a() + t * b.a()));
    }

    /** Composite this color over {@code dst} (Porter–Duff "over"). Returns straight-alpha result. */
    public DColor over(DColor dst) {
        float sa = aUnit();
        if (sa >= 0.999f) return this;
        float da = dst.aUnit();
        float oa = sa + da * (1f - sa);
        if (oa <= 1e-6f) return of(0, 0, 0, 0);
        float or = (rUnit() * sa + dst.rUnit() * da * (1f - sa)) / oa;
        float og = (gUnit() * sa + dst.gUnit() * da * (1f - sa)) / oa;
        float ob = (bUnit() * sa + dst.bUnit() * da * (1f - sa)) / oa;
        return ofUnit(or, og, ob, oa);
    }

    @Override public String toString() {
        return String.format("DColor[argb=0x%08X]", argb);
    }

    @Override public int hashCode() { return argb; }

    @Override public boolean equals(Object o) {
        if (!(o instanceof DColor)) return false;
        return ((DColor) o).argb == argb;
    }

    private static float clamp01(float v) {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }

    // ---- CSS named colors (subset covering Pillow + ImageColor) ----------

    private static final java.util.Map<String, DColor> NAMED = new java.util.HashMap<>(192);

    private static void put(String n, int r, int g, int b) { NAMED.put(n, of(r, g, b)); }
    /** Stores an {@code 0xRRGGBB} integer as an opaque named color. */
    private static void put(String n, int rgb) { NAMED.put(n, ofRgb(rgb)); }

    static {
        // CSS Level 4 named colors (subset)
        put("aliceblue", 0xF0F8FF); put("antiquewhite", 0xFAEBD7); put("aqua", 0x00FFFF);
        put("aquamarine", 0x7FFFD4); put("azure", 0xF0FFFF); put("beige", 0xF5F5DC);
        put("bisque", 0xFFE4C4); put("black", 0x000000); put("blanchedalmond", 0xFFEBCD);
        put("blue", 0x0000FF); put("blueviolet", 0x8A2BE2); put("brown", 0xA52A2A);
        put("burlywood", 0xDEB887); put("cadetblue", 0x5F9EA0); put("chartreuse", 0x7FFF00);
        put("chocolate", 0xD2691E); put("coral", 0xFF7F50); put("cornflowerblue", 0x6495ED);
        put("cornsilk", 0xFFF8DC); put("crimson", 0xDC143C); put("cyan", 0x00FFFF);
        put("darkblue", 0x00008B); put("darkcyan", 0x008B8B); put("darkgoldenrod", 0xB8860B);
        put("darkgray", 0xA9A9A9); put("darkgreen", 0x006400); put("darkkhaki", 0xBDB76B);
        put("darkmagenta", 0x8B008B); put("darkolivegreen", 0x556B2F); put("darkorange", 0xFF8C00);
        put("darkorchid", 0x9932CC); put("darkred", 0x8B0000); put("darksalmon", 0xE9967A);
        put("darkseagreen", 0x8FBC8F); put("darkslateblue", 0x483D8B); put("darkslategray", 0x2F4F4F);
        put("darkturquoise", 0x00CED1); put("darkviolet", 0x9400D3); put("deeppink", 0xFF1493);
        put("deepskyblue", 0x00BFFF); put("dimgray", 0x696969); put("dodgerblue", 0x1E90FF);
        put("firebrick", 0xB22222); put("floralwhite", 0xFFFAF0); put("forestgreen", 0x228B22);
        put("fuchsia", 0xFF00FF); put("gainsboro", 0xDCDCDC); put("ghostwhite", 0xF8F8FF);
        put("gold", 0xFFD700); put("goldenrod", 0xDAA520); put("gray", 0x808080);
        put("green", 0x008000); put("greenyellow", 0xADFF2F); put("honeydew", 0xF0FFF0);
        put("hotpink", 0xFF69B4); put("indianred", 0xCD5C5C); put("indigo", 0x4B0082);
        put("ivory", 0xFFFFF0); put("khaki", 0xF0E68C); put("lavender", 0xE6E6FA);
        put("lavenderblush", 0xFFF0F5); put("lawngreen", 0x7CFC00); put("lemonchiffon", 0xFFFACD);
        put("lightblue", 0xADD8E6); put("lightcoral", 0xF08080); put("lightcyan", 0xE0FFFF);
        put("lightgoldenrodyellow", 0xFAFAD2); put("lightgray", 0xD3D3D3); put("lightgreen", 0x90EE90);
        put("lightpink", 0xFFB6C1); put("lightsalmon", 0xFFA07A); put("lightseagreen", 0x20B2AA);
        put("lightskyblue", 0x87CEFA); put("lightslategray", 0x778899); put("lightsteelblue", 0xB0C4DE);
        put("lightyellow", 0xFFFFE0); put("lime", 0x00FF00); put("limegreen", 0x32CD32);
        put("linen", 0xFAF0E6); put("magenta", 0xFF00FF); put("maroon", 0x800000);
        put("mediumaquamarine", 0x66CDAA); put("mediumblue", 0x0000CD); put("mediumorchid", 0xBA55D3);
        put("mediumpurple", 0x9370DB); put("mediumseagreen", 0x3CB371); put("mediumslateblue", 0x7B68EE);
        put("mediumspringgreen", 0x00FA9A); put("mediumturquoise", 0x48D1CC); put("mediumvioletred", 0xC71585);
        put("midnightblue", 0x191970); put("mintcream", 0xF5FFFA); put("mistyrose", 0xFFE4E1);
        put("moccasin", 0xFFE4B5); put("navajowhite", 0xFFDEAD); put("navy", 0x000080);
        put("oldlace", 0xFDF5E6); put("olive", 0x808000); put("olivedrab", 0x6B8E23);
        put("orange", 0xFFA500); put("orangered", 0xFF4500); put("orchid", 0xDA70D6);
        put("palegoldenrod", 0xEEE8AA); put("palegreen", 0x98FB98); put("paleturquoise", 0xAFEEEE);
        put("palevioletred", 0xDB7093); put("papayawhip", 0xFFEFD5); put("peachpuff", 0xFFDAB9);
        put("peru", 0xCD853F); put("pink", 0xFFC0CB); put("plum", 0xDDA0DD);
        put("powderblue", 0xB0E0E6); put("purple", 0x800080); put("rebeccapurple", 0x663399);
        put("red", 0xFF0000); put("rosybrown", 0xBC8F8F); put("royalblue", 0x4169E1);
        put("saddlebrown", 0x8B4513); put("salmon", 0xFA8072); put("sandybrown", 0xF4A460);
        put("seagreen", 0x2E8B57); put("seashell", 0xFFF5EE); put("sienna", 0xA0522D);
        put("silver", 0xC0C0C0); put("skyblue", 0x87CEEB); put("slateblue", 0x6A5ACD);
        put("slategray", 0x708090); put("snow", 0xFFFAFA); put("springgreen", 0x00FF7F);
        put("steelblue", 0x4682B4); put("tan", 0xD2B48C); put("teal", 0x008080);
        put("thistle", 0xD8BFD8); put("tomato", 0xFF6347); put("turquoise", 0x40E0D0);
        put("violet", 0xEE82EE); put("wheat", 0xF5DEB3); put("white", 0xFFFFFF);
        put("whitesmoke", 0xF5F5F5); put("yellow", 0xFFFF00); put("yellowgreen", 0x9ACD32);
        // Pillow / ImageColor extras
        put("transparent", 0x00000000);
    }

    /** Iterate over all CSS-named colors (read-only). */
    public static java.util.Set<String> names() { return java.util.Collections.unmodifiableSet(NAMED.keySet()); }
}