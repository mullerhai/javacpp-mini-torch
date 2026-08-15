package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;

import java.util.Objects;

/**
 * Padding operations (scrimage {@code Pad} / Pillow ImageOps pad / torchvision v2 Pad).
 *
 * <p>Padding modes:
 * <ul>
 *   <li>{@link PaddingMode#CONSTANT} — fill with {@code fillColor}</li>
 *   <li>{@link PaddingMode#EDGE}     — replicate border pixel</li>
 *   <li>{@link PaddingMode#REFLECT}  — mirror around edge (exclude edge)</li>
 *   <li>{@link PaddingMode#SYMMETRIC} — mirror including edge</li>
 * </ul>
 *
 * <p>Padding spec accepts {@code int} (single value for all sides), {@code int[4]} as
 * {@code (left, top, right, bottom)}, or {@code int[2]} as {@code (horizontal, vertical)}.
 */
public final class Pad {

    private Pad() {}

    public enum PaddingMode { CONSTANT, EDGE, REFLECT, SYMMETRIC }

    public static ImmutableImage pad(ImmutableImage src, int all) {
        return pad(src, all, all, all, all, null, PaddingMode.CONSTANT);
    }

    public static ImmutableImage pad(ImmutableImage src, int left, int top, int right, int bottom) {
        return pad(src, left, top, right, bottom, null, PaddingMode.CONSTANT);
    }

    public static ImmutableImage pad(ImmutableImage src, int all, DColor fillColor) {
        return pad(src, all, all, all, all, fillColor, PaddingMode.CONSTANT);
    }

    public static ImmutableImage pad(ImmutableImage src, int left, int top, int right, int bottom,
                                     DColor fillColor, PaddingMode mode) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(mode, "mode");
        int sw = src.width(), sh = src.height();
        int nw = sw + left + right;
        int nh = sh + top + bottom;
        if (nw <= 0 || nh <= 0) throw new IllegalArgumentException("padding produces zero/negative size");
        Image srcBuf = src.image();
        int sb = srcBuf.getImagingBuffer().bands();
        Image dst = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(srcBuf.getImagingBuffer().modeInfo(), nw, nh));
        // Initialize to fill if CONSTANT
        if (mode == PaddingMode.CONSTANT && fillColor != null) {
            int[] fillBands = colorToBands(fillColor, sb);
            dst.getImagingBuffer().fill(fillBands);
        }
        int[] sp = srcBuf.getdata();
        int[] dp = dst.getdata();
        for (int y = 0; y < nh; y++) {
            int sy = y - top;
            for (int x = 0; x < nw; x++) {
                int sx = x - left;
                int ssx = clampSource(sx, sw, mode);
                int ssy = clampSource(sy, sh, mode);
                if (ssx < 0 || ssy < 0 || ssx >= sw || ssy >= sh) continue;
                int srcIdx = (ssy * sw + ssx) * sb;
                int dstIdx = (y * nw + x) * sb;
                System.arraycopy(sp, srcIdx, dp, dstIdx, sb);
            }
        }
        dst.getImagingBuffer().putdata(dp, 0, 1);
        return new ImmutableImage(dst);
    }

    private static int[] colorToBands(DColor c, int bands) {
        int r = (c.argb >>> 16) & 0xff, g = (c.argb >>> 8) & 0xff, b = c.argb & 0xff;
        int a = (c.argb >>> 24) & 0xff;
        return bands == 1 ? new int[]{r} : bands == 3 ? new int[]{r, g, b} : new int[]{r, g, b, a};
    }

    /**
     * {@code padTo(size)} — pad up to a minimum width/height with edge-padding.
     * Equivalent to torchvision's PadIfNeeded-like behavior, using EDGE padding.
     */
    public static ImmutableImage padTo(ImmutableImage src, int targetW, int targetH) {
        Objects.requireNonNull(src, "src");
        if (src.width() >= targetW && src.height() >= targetH) return src.copy();
        int left = Math.max(0, (targetW - src.width()) / 2);
        int top = Math.max(0, (targetH - src.height()) / 2);
        int right = Math.max(0, targetW - src.width() - left);
        int bottom = Math.max(0, targetH - src.height() - top);
        return pad(src, left, top, right, bottom, null, PaddingMode.EDGE);
    }

    /** Padding spec from {@code int[1,2,4]}. */
    static int[] resolveSpec(int[] spec) {
        if (spec.length == 1) return new int[]{spec[0], spec[0], spec[0], spec[0]};
        if (spec.length == 2) return new int[]{spec[0], spec[1], spec[0], spec[1]};
        if (spec.length == 4) return spec;
        throw new IllegalArgumentException("padding spec must be length 1, 2, or 4");
    }

    private static int clampSource(int v, int size, PaddingMode mode) {
        if (v >= 0 && v < size) return v;
        switch (mode) {
            case CONSTANT:
            case EDGE:
                if (v < 0) return -1; // sentinel → skip
                return v;
            case REFLECT: {
                // mirror excluding edge: [-1..size) -> [1..size-2]
                int n = size;
                int r = ((v % (2 * n)) + 2 * n) % (2 * n);
                return r < n ? r : 2 * n - r - 2;
            }
            case SYMMETRIC: {
                int n = size;
                int r = ((v % (2 * n)) + 2 * n) % (2 * n);
                return r < n ? r : 2 * n - r - 1;
            }
            default:
                throw new IllegalArgumentException("mode " + mode);
        }
    }
}