package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;

import java.util.Objects;

/**
 * Effects / geometric transformations (scrimage {@code AwtMethods} + Pillow
 * {@code ImageOps}). Stateless wrappers that delegate to {@link Image} when
 * possible, and add convenience overloads that return
 * {@link ImmutableImage} for fluent chaining.
 */
public final class Effects {

    private Effects() {}

    /** Rotate by an exact 90-degree quadrant. */
    public static ImmutableImage rotateLeft(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.ROTATE_90));
    }

    public static ImmutableImage rotateRight(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.ROTATE_270));
    }

    /** Rotate 180. */
    public static ImmutableImage rotate180(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.ROTATE_180));
    }

    /** Arbitrary angle rotation with resampling and optional canvas expand. */
    public static ImmutableImage rotate(ImmutableImage src, double degrees) {
        return rotate(src, degrees, false, null);
    }

    public static ImmutableImage rotate(ImmutableImage src, double degrees, boolean expand, org.bytedeco.pytorch.vision.draw.DColor fill) {
        Objects.requireNonNull(src, "src");
        Object fillObj = fill == null ? null : (Object) (fill.argb & 0xFFFFFF);
        return new ImmutableImage(src.image().rotate(degrees, org.bytedeco.pytorch.vision.pillow.enums.Resampling.BICUBIC, expand, fillObj));
    }

    /** Mirror (flip vertically). */
    public static ImmutableImage flip(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.FLIP_TOP_BOTTOM));
    }

    /** Mirror left-right (flip horizontally). */
    public static ImmutableImage mirror(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.FLIP_LEFT_RIGHT));
    }

    /** Mirror around main diagonal. */
    public static ImmutableImage transpose(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.TRANSPOSE));
    }

    /** Mirror around anti-diagonal. */
    public static ImmutableImage transverse(ImmutableImage src) {
        return new ImmutableImage(src.image().transpose(org.bytedeco.pytorch.vision.pillow.enums.Transpose.TRANSVERSE));
    }

    /** Anti-alias scale. Default uses BICUBIC resampling. */
    public static ImmutableImage antiAlias(ImmutableImage src, double scale) {
        int nw = Math.max(1, (int) Math.round(src.width() * scale));
        int nh = Math.max(1, (int) Math.round(src.height() * scale));
        return new ImmutableImage(src.image().resize(nw, nh, org.bytedeco.pytorch.vision.pillow.enums.Resampling.BICUBIC));
    }

    /** Apply a 256-entry lookup table (LUT) to each channel. */
    public static ImmutableImage lut(ImmutableImage src, int[] table) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(table, "table");
        if (table.length < 256) throw new IllegalArgumentException("table length >= 256");
        return new ImmutableImage(src.image().point(table));
    }

    /** Apply an arbitrary curve: array of (input → output) breakpoints, sorted by input. */
    public static ImmutableImage applyCurve(ImmutableImage src, int[] xs, int[] ys) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(xs, "xs");
        Objects.requireNonNull(ys, "ys");
        if (xs.length != ys.length || xs.length == 0) throw new IllegalArgumentException("xs/ys mismatch");
        int[] lut = new int[256];
        for (int v = 0; v < 256; v++) {
            // find segment
            int i = 0;
            while (i < xs.length - 1 && xs[i + 1] < v) i++;
            int x0 = xs[i], x1 = xs[Math.min(i + 1, xs.length - 1)];
            int y0 = ys[i], y1 = ys[Math.min(i + 1, xs.length - 1)];
            if (x0 == x1) {
                lut[v] = y0;
            } else {
                double t = (v - x0) / (double) (x1 - x0);
                lut[v] = Colors.clamp8((int) Math.round(y0 + (y1 - y0) * t));
            }
        }
        return new ImmutableImage(src.image().point(lut));
    }

    /** Adjust levels: blackPoint/whitePoint stretch, gamma applies. */
    public static ImmutableImage levels(ImmutableImage src, int blackPoint, int whitePoint, double gamma) {
        Objects.requireNonNull(src, "src");
        if (blackPoint < 0 || blackPoint >= whitePoint) throw new IllegalArgumentException("0 <= blackPoint < whitePoint");
        int range = whitePoint - blackPoint;
        double inv = gamma <= 0 ? 1.0 : 1.0 / gamma;
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) {
            double t = (i - blackPoint) / (double) range;
            t = Math.max(0.0, Math.min(1.0, t));
            lut[i] = Colors.clamp8((int) Math.round(Math.pow(t, inv) * 255.0));
        }
        return new ImmutableImage(src.image().point(lut));
    }

    /** Auto contrast: stretch histogram ignoring top/bottom 1%. */
    public static ImmutableImage autoContrast(ImmutableImage src) {
        return autoContrast(src, 1.0f);
    }

    public static ImmutableImage autoContrast(ImmutableImage src, float cutoff) {
        Objects.requireNonNull(src, "src");
        int[] hist = src.image().histogram();
        int bands = Math.max(1, src.image().getImagingBuffer().modeInfo().bands());
        int cutoffN = (int) ((src.width() * src.height()) * cutoff / 100f);
        int min = 0, max = 255;
        for (int c = 0; c < bands; c++) {
            int count = 0;
            int lo = 0, hi = 255;
            for (int v = 0; v < 256; v++) {
                count += hist[c * 256 + v];
                if (count >= cutoffN) { lo = v; break; }
            }
            count = 0;
            for (int v = 255; v >= 0; v--) {
                count += hist[c * 256 + v];
                if (count >= cutoffN) { hi = v; break; }
            }
            min = Math.max(min, lo);
            max = Math.min(max, hi);
        }
        if (max <= min) return src.copy();
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) {
            lut[i] = Colors.clamp8((int) ((i - min) * 255.0 / (max - min)));
        }
        return new ImmutableImage(src.image().point(lut));
    }

    /** Equalize histogram. Per-band cumulative distribution. */
    public static ImmutableImage equalize(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        int[] hist = src.image().histogram();
        int bands = Math.max(1, src.image().getImagingBuffer().modeInfo().bands());
        int total = src.width() * src.height();
        int[][] luts = new int[bands][256];
        for (int c = 0; c < bands; c++) {
            long cum = 0;
            int[] lut = luts[c];
            for (int v = 0; v < 256; v++) {
                cum += hist[c * 256 + v];
                lut[v] = Colors.clamp8((int) ((cum * 255L) / total));
            }
        }
        int[] sp = src.image().getdata();
        int[] out = new int[sp.length];
        for (int i = 0; i < sp.length; i++) {
            int c = i % bands;
            out[i] = luts[c][sp[i] & 0xff];
        }
        Image result = Image.fromBuffer(new ImagingBuffer(src.image().getImagingBuffer().modeInfo(), src.width(), src.height()));
        result.putdata(out);
        return new ImmutableImage(result);
    }
}