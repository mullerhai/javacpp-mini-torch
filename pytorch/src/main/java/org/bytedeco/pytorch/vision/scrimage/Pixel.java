package org.bytedeco.pytorch.vision.scrimage;

import org.bytedeco.pytorch.vision.draw.DColor;

/**
 * Pixel = (r, g, b, a) per channel, 0-255 straight alpha.
 * Scrimage Pixel stand-in. Pure POJO; cheap to allocate.
 */
public final class Pixel {
    public final int r;
    public final int g;
    public final int b;
    public final int a;

    public Pixel(int r, int g, int b) { this(r, g, b, 255); }

    public Pixel(int r, int g, int b, int a) {
        this.r = clamp8(r);
        this.g = clamp8(g);
        this.b = clamp8(b);
        this.a = clamp8(a);
    }

    public Pixel(int argb) {
        this.a = (argb >>> 24) & 0xff;
        this.r = (argb >>> 16) & 0xff;
        this.g = (argb >>> 8) & 0xff;
        this.b = argb & 0xff;
    }

    public static Pixel of(int r, int g, int b) { return new Pixel(r, g, b); }
    public static Pixel of(int r, int g, int b, int a) { return new Pixel(r, g, b, a); }
    public static Pixel of(DColor c) {
        return new Pixel((c.argb >>> 16) & 0xff, (c.argb >>> 8) & 0xff, c.argb & 0xff, (c.argb >>> 24) & 0xff);
    }

    public int argb() {
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    public int rgb() {
        return (r << 16) | (g << 8) | b;
    }

    public DColor toColor() { return DColor.of(argb()); }

    public float rf() { return r / 255f; }
    public float gf() { return g / 255f; }
    public float bf() { return b / 255f; }
    public float af() { return a / 255f; }

    public static Pixel fromFloat(float r, float g, float b) {
        return new Pixel(Math.round(r * 255f), Math.round(g * 255f), Math.round(b * 255f));
    }

    public static Pixel fromFloat(float r, float g, float b, float a) {
        return new Pixel(Math.round(r * 255f), Math.round(g * 255f), Math.round(b * 255f), Math.round(a * 255f));
    }

    private static int clamp8(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    @Override public boolean equals(Object o) {
        if (!(o instanceof Pixel p)) return false;
        return p.r == r && p.g == g && p.b == b && p.a == a;
    }

    @Override public int hashCode() {
        return ((r << 24) | (g << 16) | (b << 8) | a);
    }

    @Override public String toString() {
        return "Pixel(" + r + "," + g + "," + b + "," + a + ")";
    }
}