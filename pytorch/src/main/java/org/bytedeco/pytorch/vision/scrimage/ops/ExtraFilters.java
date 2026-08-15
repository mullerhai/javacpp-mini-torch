package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;

import java.util.Objects;
import java.util.Random;

/**
 * Extra scrimage filters (scrimage {@code ImageFilter} surface) — channel
 * shifts, plasma, marble, offset, edge, stroke, tile, gcd, convolution.
 * All pure-Java; no external native deps.
 */
public final class ExtraFilters {

    private ExtraFilters() {}

    /** Channel shift by per-channel pixel offset (scrimage {@code ChannelShiftFilter}). */
    public static ImmutableImage channelShift(ImmutableImage src, int rShift, int gShift, int bShift) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        int[] sp = rgb.getdata(), op = out.getdata();
        for (int y = 0; y < h; y++) {
            int sy = ((y - rShift) % h + h) % h;
            for (int x = 0; x < w; x++) {
                int sxR = ((x - rShift) % w + w) % w;
                int sxG = ((x - gShift) % w + w) % w;
                int sxB = ((x - bShift) % w + w) % w;
                int sidxR = (sy * w + sxR) * 3;
                int sidxG = (sy * w + sxG) * 3;
                int sidxB = (sy * w + sxB) * 3;
                int didx = (y * w + x) * 3;
                op[didx]     = sp[sidxR];      // red row offset by rShift
                op[didx + 1] = sp[sidxG + 1];  // green
                op[didx + 2] = sp[sidxB + 2];  // blue
            }
        }
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Generic convolution with arbitrary odd-sized kernel (scrimage {@code ConvolutionFilter}). */
    public static ImmutableImage convolve(ImmutableImage src, float[] kernel, int kw, int kh, float scale, float bias) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(kernel, "kernel");
        if (kernel.length != kw * kh) throw new IllegalArgumentException("kernel size mismatch");
        if (kw % 2 == 0 || kh % 2 == 0) throw new IllegalArgumentException("kernel must be odd");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        int[] sp = rgb.getdata(), op = out.getdata();
        int hRx = kw / 2, hRy = kh / 2;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int didx = (y * w + x) * 3;
                double accR = 0, accG = 0, accB = 0;
                for (int ky = -hRy; ky <= hRy; ky++) {
                    int yy = Math.min(h - 1, Math.max(0, y + ky));
                    for (int kx = -hRx; kx <= hRx; kx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + kx));
                        int sidx = (yy * w + xx) * 3;
                        float k = kernel[(ky + hRy) * kw + (kx + hRx)];
                        accR += (sp[sidx]     & 0xff) * k;
                        accG += (sp[sidx + 1] & 0xff) * k;
                        accB += (sp[sidx + 2] & 0xff) * k;
                    }
                }
                op[didx]     = Colors.clamp8((int) (accR * scale + bias));
                op[didx + 1] = Colors.clamp8((int) (accG * scale + bias));
                op[didx + 2] = Colors.clamp8((int) (accB * scale + bias));
            }
        }
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Edge filter (scrimage {@code EdgeFilter}): highlights edges with a 3x3 kernel. */
    public static ImmutableImage edge(ImmutableImage src) {
        float[] k = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        return convolve(src, k, 3, 3, 1f, 0f);
    }

    /** Sharpen filter (scrimage {@code SharpenFilter}). */
    public static ImmutableImage sharpenFilter(ImmutableImage src) {
        float[] k = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        return convolve(src, k, 3, 3, 1f, 0f);
    }

    /** Stroke filter — outline effect (scrimage {@code StrokeFilter}). */
    public static ImmutableImage stroke(ImmutableImage src, int thickness) {
        Objects.requireNonNull(src, "src");
        if (thickness < 1) throw new IllegalArgumentException("thickness");
        // Approximate via difference between image and its morphologically-eroded version.
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        int[] sp = rgb.getdata();
        int[] erode = new int[sp.length];
        int r = thickness;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int eR = 255, eG = 255, eB = 255;
                for (int ky = -r; ky <= r; ky++) {
                    int yy = Math.min(h - 1, Math.max(0, y + ky));
                    for (int kx = -r; kx <= r; kx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + kx));
                        int sidx = (yy * w + xx) * 3;
                        eR = Math.min(eR, sp[sidx]     & 0xff);
                        eG = Math.min(eG, sp[sidx + 1] & 0xff);
                        eB = Math.min(eB, sp[sidx + 2] & 0xff);
                    }
                }
                int didx = (y * w + x) * 3;
                erode[didx]     = eR;
                erode[didx + 1] = eG;
                erode[didx + 2] = eB;
            }
        }
        // output = abs(src - erode)
        int[] op = new int[sp.length];
        for (int i = 0; i < sp.length; i++) op[i] = Math.min(255, Math.abs((sp[i] & 0xff) - (erode[i] & 0xff)) * 4);
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Tile filter (scrimage {@code TileFilter}) — wrap image as a repeating tile. */
    public static ImmutableImage tileFilter(ImmutableImage src, int tiles) {
        Objects.requireNonNull(src, "src");
        if (tiles < 1) throw new IllegalArgumentException("tiles");
        int sw = src.width(), sh = src.height();
        int nw = sw / tiles, nh = sh / tiles;
        if (nw <= 0 || nh <= 0) return src.copy();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(src.image().getImagingBuffer().modeInfo(), sw, sh));
        int[] op = out.getdata();
        int sb = src.image().getImagingBuffer().bands();
        int[] sp = src.image().getdata();
        for (int y = 0; y < sh; y++) {
            int sy = (y / tiles) % nh;
            for (int x = 0; x < sw; x++) {
                int sx = (x / tiles) % nw;
                int sidx = (sy * nw + sx) * sb;
                int didx = (y * sw + x) * sb;
                System.arraycopy(sp, sidx, op, didx, sb);
            }
        }
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Offset filter (scrimage {@code OffsetFilter}). */
    public static ImmutableImage offset(ImmutableImage src, int dx, int dy) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        for (int y = 0; y < h; y++) {
            int sy = ((y + dy) % h + h) % h;
            for (int x = 0; x < w; x++) {
                int sx = ((x + dx) % w + w) % w;
                int sidx = (sy * w + sx) * 3;
                int didx = (y * w + x) * 3;
                op[didx]     = sp[sidx];
                op[didx + 1] = sp[sidx + 1];
                op[didx + 2] = sp[sidx + 2];
            }
        }
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Marble noise filter (scrimage {@code MarbleFilter}) — turbulent perlin-like texture. */
    public static ImmutableImage marble(ImmutableImage src, double scale, long seed) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        int[] sp = rgb.getdata(), op = out.getdata();
        Random rnd = new Random(seed);
        // simple pseudo-turbulence: 3 octaves of sine
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double v = Math.sin((x + Math.sin(y * 0.05 + rnd.nextDouble())) * scale * 0.05)
                         + 0.5 * Math.sin((x + y) * scale * 0.07)
                         + 0.3 * Math.sin((x - y) * scale * 0.03);
                int t = (int) ((v + 2.0) * 60);
                int sidx = (y * w + x) * 3;
                int didx = (y * w + x) * 3;
                int lum = (sp[sidx] + sp[sidx + 1] + sp[sidx + 2]) / 3;
                int n = Math.max(0, Math.min(255, lum + t));
                op[didx]     = n;
                op[didx + 1] = n;
                op[didx + 2] = n;
            }
        }
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Plasma noise (scrimage {@code PlasmaFilter}). */
    public static ImmutableImage plasma(ImmutableImage src, long seed) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        int[] op = out.getdata();
        Random rnd = new Random(seed);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double nx = x * 6.0 / w;
                double ny = y * 6.0 / h;
                double v = Math.sin(nx + rnd.nextDouble())
                         + Math.sin(ny + rnd.nextDouble())
                         + Math.sin(nx + ny + rnd.nextDouble());
                int n = (int) ((v + 3.0) * (255.0 / 6.0));
                int didx = (y * w + x) * 3;
                op[didx]     = n;
                op[didx + 1] = n;
                op[didx + 2] = n;
            }
        }
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** GCD filter (scrimage {@code GcdFilter}) — Greatest Common Divisor effect. */
    public static ImmutableImage gcd(ImmutableImage src, int gcd) {
        Objects.requireNonNull(src, "src");
        if (gcd < 2 || gcd > 256) throw new IllegalArgumentException("gcd 2..256");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        int[] sp = rgb.getdata(), op = out.getdata();
        for (int i = 0; i < sp.length; i++) op[i] = (sp[i] / gcd) * gcd;
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Dilation (max in window) — used internally and exposed for stroke / morphology. */
    public static ImmutableImage dilate(ImmutableImage src, int radius) {
        Objects.requireNonNull(src, "src");
        if (radius < 1) throw new IllegalArgumentException("radius");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int mR = 0, mG = 0, mB = 0;
                for (int ky = -radius; ky <= radius; ky++) {
                    int yy = Math.min(h - 1, Math.max(0, y + ky));
                    for (int kx = -radius; kx <= radius; kx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + kx));
                        int sidx = (yy * w + xx) * 3;
                        mR = Math.max(mR, sp[sidx]     & 0xff);
                        mG = Math.max(mG, sp[sidx + 1] & 0xff);
                        mB = Math.max(mB, sp[sidx + 2] & 0xff);
                    }
                }
                int didx = (y * w + x) * 3;
                op[didx]     = mR;
                op[didx + 1] = mG;
                op[didx + 2] = mB;
            }
        }
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Erosion (min in window). */
    public static ImmutableImage erode(ImmutableImage src, int radius) {
        Objects.requireNonNull(src, "src");
        if (radius < 1) throw new IllegalArgumentException("radius");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int mR = 255, mG = 255, mB = 255;
                for (int ky = -radius; ky <= radius; ky++) {
                    int yy = Math.min(h - 1, Math.max(0, y + ky));
                    for (int kx = -radius; kx <= radius; kx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + kx));
                        int sidx = (yy * w + xx) * 3;
                        mR = Math.min(mR, sp[sidx]     & 0xff);
                        mG = Math.min(mG, sp[sidx + 1] & 0xff);
                        mB = Math.min(mB, sp[sidx + 2] & 0xff);
                    }
                }
                int didx = (y * w + x) * 3;
                op[didx]     = mR;
                op[didx + 1] = mG;
                op[didx + 2] = mB;
            }
        }
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), w, h));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }
}