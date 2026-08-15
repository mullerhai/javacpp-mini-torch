package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;
import org.bytedeco.pytorch.vision.scrimage.Pixel;

import java.util.Objects;

/**
 * Color operations (scrimage {@code ColorMethods} + Pillow ImageOps).
 *
 * <p>Each op converts to RGB(A) first, applies the math, and returns a new
 * {@link ImmutableImage}. Operations are pure Java; no native dependencies.
 *
 * <p>Performance notes:
 * <ul>
 *   <li>Band-aware (L/RGB/RGBA) — uses byte-plane access when possible.
 *   <li>Single-pass where the op is a per-pixel function.
 *   <li>Lookup-table (LUT) versions for {@link #brightness(float)} /
 *       {@link #contrast(float)} / {@link #gamma(double)} to amortize clamp +
 *       multiply for batch use.
 * </ul>
 */
public final class Colors {

    private Colors() {}

    // ── simple unaries ──────────────────────────────────────────────────────

    /** Convert to grayscale (mode "L"). */
    public static ImmutableImage grayscale(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        return new ImmutableImage(src.image().convert("L"));
    }

    /** Monochrome: threshold at mid-gray, then to L mode. */
    public static ImmutableImage monochrome(ImmutableImage src, int threshold) {
        Objects.requireNonNull(src, "src");
        Image g = src.image().convert("L");
        Image out = Image.new_("L", g.width(), g.height());
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = i < threshold ? 0 : 255;
        return new ImmutableImage(g.point(lut));
    }

    /** Sepia toning (RGB): classic warm overlay. */
    public static ImmutableImage sepia(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata();
        int[] opx = out.getdata();
        for (int i = 0; i < px.length; i += 3) {
            int r = px[i], g = px[i + 1], b = px[i + 2];
            int oR = clamp8((int) (0.393 * r + 0.769 * g + 0.189 * b));
            int oG = clamp8((int) (0.349 * r + 0.686 * g + 0.168 * b));
            int oB = clamp8((int) (0.272 * r + 0.534 * g + 0.131 * b));
            opx[i] = oR;
            opx[i + 1] = oG;
            opx[i + 2] = oB;
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Invert colors: 255 - x per channel. */
    public static ImmutableImage invert(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        return new ImmutableImage(src.image().point(v -> 255 - v));
    }

    /** Threshold: each channel replaced by 0 or 255 based on threshold. */
    public static ImmutableImage threshold(ImmutableImage src, int threshold) {
        Objects.requireNonNull(src, "src");
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = i >= threshold ? 255 : 0;
        return new ImmutableImage(src.image().point(lut));
    }

    /** Posteriorize: round each channel to {@code bits} levels (1..8). */
    public static ImmutableImage posterize(ImmutableImage src, int bits) {
        Objects.requireNonNull(src, "src");
        if (bits < 1 || bits > 8) throw new IllegalArgumentException("bits 1..8");
        int levels = 1 << bits;
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = (i / (256 / levels)) * (256 / levels);
        return new ImmutableImage(src.image().point(lut));
    }

    /** Solarize: invert pixel when above threshold. */
    public static ImmutableImage solarize(ImmutableImage src, int threshold) {
        Objects.requireNonNull(src, "src");
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = i < threshold ? i : 255 - i;
        return new ImmutableImage(src.image().point(lut));
    }

    // ── brightness / contrast / gamma ──────────────────────────────────────

    /** Brightness factor: 0 = black, 1 = unchanged, >1 brighter. */
    public static ImmutableImage brightness(ImmutableImage src, float factor) {
        Objects.requireNonNull(src, "src");
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = clamp8((int) (i * factor));
        return new ImmutableImage(src.image().point(lut));
    }

    /**
     * Contrast factor. {@code factor = 1.0} unchanged. The implementation uses
     * Pillow's standard formula: {@code out = (c - 127.5) * factor + 127.5}.
     */
    public static ImmutableImage contrast(ImmutableImage src, float factor) {
        Objects.requireNonNull(src, "src");
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = clamp8((int) ((i - 127.5) * factor + 127.5));
        return new ImmutableImage(src.image().point(lut));
    }

    /** Gamma correction: {@code out = (c/255)^(1/gamma) * 255}. */
    public static ImmutableImage gamma(ImmutableImage src, double gamma) {
        Objects.requireNonNull(src, "src");
        if (gamma <= 0) throw new IllegalArgumentException("gamma > 0");
        double inv = 1.0 / gamma;
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) lut[i] = clamp8((int) Math.round(Math.pow(i / 255.0, inv) * 255.0));
        return new ImmutableImage(src.image().point(lut));
    }

    /** Adjust hue rotation in degrees, in HSL space. */
    public static ImmutableImage hue(ImmutableImage src, float degrees) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata();
        int[] opx = out.getdata();
        for (int i = 0; i < px.length; i += 3) {
            float[] hsl = rgbToHsl(px[i], px[i + 1], px[i + 2]);
            hsl[0] = ((hsl[0] + degrees / 360f) % 1f + 1f) % 1f;
            int[] out2 = hslToRgb(hsl[0], hsl[1], hsl[2]);
            opx[i] = out2[0];
            opx[i + 1] = out2[1];
            opx[i + 2] = out2[2];
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Saturation factor. {@code factor=0} → grayscale, {@code 1}=unchanged. */
    public static ImmutableImage saturation(ImmutableImage src, float factor) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata();
        int[] opx = out.getdata();
        for (int i = 0; i < px.length; i += 3) {
            int gray = (px[i] * 299 + px[i + 1] * 587 + px[i + 2] * 114) / 1000;
            int oR = clamp8((int) (gray + (px[i] - gray) * factor));
            int oG = clamp8((int) (gray + (px[i + 1] - gray) * factor));
            int oB = clamp8((int) (gray + (px[i + 2] - gray) * factor));
            opx[i] = oR;
            opx[i + 1] = oG;
            opx[i + 2] = oB;
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Blend two images by {@code alpha} (0..1) per pixel using straight alpha. */
    public static ImmutableImage overlay(ImmutableImage bottom, ImmutableImage top, float alpha) {
        return blend(bottom, top, BlendMode.OVERLAY, alpha);
    }

    /** Multiply. */
    public static ImmutableImage multiply(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.MULTIPLY, 1f);
    }

    /** Screen blend. */
    public static ImmutableImage screen(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.SCREEN, 1f);
    }

    /** Lighten (per-channel max). */
    public static ImmutableImage lighten(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.LIGHTEN, 1f);
    }

    /** Darken (per-channel min). */
    public static ImmutableImage darken(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.DARKEN, 1f);
    }

    /** Difference blend. */
    public static ImmutableImage difference(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.DIFFERENCE, 1f);
    }

    /** Exclusion blend. */
    public static ImmutableImage exclusion(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.EXCLUSION, 1f);
    }

    /** Soft-light (W3C variant). */
    public static ImmutableImage softLight(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.SOFT_LIGHT, 1f);
    }

    /** Hard-light (mirrored soft-light). */
    public static ImmutableImage hardLight(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.HARD_LIGHT, 1f);
    }

    /** Color-burn. */
    public static ImmutableImage colorBurn(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.COLOR_BURN, 1f);
    }

    /** Color-dodge. */
    public static ImmutableImage colorDodge(ImmutableImage bottom, ImmutableImage top) {
        return blend(bottom, top, BlendMode.COLOR_DODGE, 1f);
    }

    /** General blend using {@link BlendMode}. */
    public static ImmutableImage blend(ImmutableImage bottom, ImmutableImage top, BlendMode mode) {
        return blend(bottom, top, mode, 1f);
    }

    public static ImmutableImage blend(ImmutableImage bottom, ImmutableImage top, BlendMode mode, float alpha) {
        Objects.requireNonNull(bottom, "bottom");
        Objects.requireNonNull(top, "top");
        Objects.requireNonNull(mode, "mode");
        if (bottom.width() != top.width() || bottom.height() != top.height()) {
            throw new IllegalArgumentException("size mismatch");
        }
        Image b = bottom.image().convert("RGB");
        Image t = top.image().convert("RGB");
        Image out = Image.new_("RGB", b.width(), b.height());
        int[] bp = b.getdata(), tp = t.getdata(), op = out.getdata();
        for (int i = 0; i < bp.length; i += 3) {
            int bbR = bp[i], bbG = bp[i + 1], bbB = bp[i + 2];
            int ttR = tp[i], ttG = tp[i + 1], ttB = tp[i + 2];
            int[] o = mode.combine(bbR, bbG, bbB, ttR, ttG, ttB);
            int aR = o[0], aG = o[1], aB = o[2];
            op[i]     = (int) (bbR * (1f - alpha) + aR * alpha);
            op[i + 1] = (int) (bbG * (1f - alpha) + aG * alpha);
            op[i + 2] = (int) (bbB * (1f - alpha) + aB * alpha);
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    /** Tint image by {@code color}, mixing by {@code mix} (0..1). */
    public static ImmutableImage tint(ImmutableImage src, DColor color, float mix) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(color, "color");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata(), opx = out.getdata();
        int cR = (color.argb >>> 16) & 0xff;
        int cG = (color.argb >>> 8) & 0xff;
        int cB = color.argb & 0xff;
        for (int i = 0; i < px.length; i += 3) {
            int lum = (px[i] * 299 + px[i + 1] * 587 + px[i + 2] * 114) / 1000;
            opx[i]     = clamp8((int) (px[i] * (1 - mix) + cR * lum / 255f * mix));
            opx[i + 1] = clamp8((int) (px[i + 1] * (1 - mix) + cG * lum / 255f * mix));
            opx[i + 2] = clamp8((int) (px[i + 2] * (1 - mix) + cB * lum / 255f * mix));
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Replace any pixel exactly matching {@code from} with {@code to}. */
    public static ImmutableImage replaceColor(ImmutableImage src, DColor from, DColor to) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata(), opx = out.getdata();
        int fR = (from.argb >>> 16) & 0xff, fG = (from.argb >>> 8) & 0xff, fB = from.argb & 0xff;
        int tR = (to.argb >>> 16) & 0xff, tG = (to.argb >>> 8) & 0xff, tB = to.argb & 0xff;
        for (int i = 0; i < px.length; i += 3) {
            opx[i]     = px[i] == fR && px[i + 1] == fG && px[i + 2] == fB ? tR : px[i];
            opx[i + 1] = px[i] == fR && px[i + 1] == fG && px[i + 2] == fB ? tG : px[i + 1];
            opx[i + 2] = px[i] == fR && px[i + 1] == fG && px[i + 2] == fB ? tB : px[i + 2];
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    // ── HSV/HSL math ────────────────────────────────────────────────────────

    static float[] rgbToHsl(int r, int g, int b) {
        float rf = r / 255f, gf = g / 255f, bf = b / 255f;
        float max = Math.max(rf, Math.max(gf, bf));
        float min = Math.min(rf, Math.min(gf, bf));
        float h, s, l = (max + min) / 2f;
        float d = max - min;
        if (d == 0) { h = s = 0; }
        else {
            s = l > 0.5f ? d / (2f - max - min) : d / (max + min);
            if (max == rf) h = (gf - bf) / d + (gf < bf ? 6 : 0);
            else if (max == gf) h = (bf - rf) / d + 2;
            else h = (rf - gf) / d + 4;
            h /= 6f;
        }
        return new float[]{h, s, l};
    }

    static int[] hslToRgb(float h, float s, float l) {
        if (s == 0) {
            int v = clamp8((int) (l * 255));
            return new int[]{v, v, v};
        }
        float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
        float p = 2f * l - q;
        int[] rgb = new int[3];
        float[] hh = {h + 1f / 3f, h, h - 1f / 3f};
        for (int i = 0; i < 3; i++) {
            float t = hh[i];
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            float v;
            if (t < 1f / 6f) v = p + (q - p) * 6f * t;
            else if (t < 1f / 2f) v = q;
            else if (t < 2f / 3f) v = p + (q - p) * (2f / 3f - t) * 6f;
            else v = p;
            rgb[i] = clamp8((int) (v * 255));
        }
        return rgb;
    }

    static int clamp8(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }
}