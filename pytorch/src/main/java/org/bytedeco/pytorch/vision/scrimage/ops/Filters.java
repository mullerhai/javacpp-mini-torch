package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;
import org.bytedeco.pytorch.vision.scrimage.Pixel;

import java.util.Objects;
import java.util.Random;

/**
 * Filters (scrimage {@code FilterMethods} + Pillow {@code ImageFilter} surface).
 *
 * <p>All implementations are pure Java convolution / pixel transforms over
 * the existing Pillow {@link Image}. For larger kernels we use separable
 * passes (e.g. Gaussian) for performance.
 */
public final class Filters {

    private Filters() {}

    // ── blur ────────────────────────────────────────────────────────────────

    /** Box blur with kernel size {@code radius}. Box kernel applied per channel. */
    public static ImmutableImage blur(ImmutableImage src, int radius) {
        Objects.requireNonNull(src, "src");
        if (radius < 1) throw new IllegalArgumentException("radius");
        return new ImmutableImage(convolveSeparable(src.image(), boxKernel(radius), boxKernel(radius)));
    }

    /** Gaussian blur. {@code sigma} controls falloff; {@code radius} derived (3*σ). */
    public static ImmutableImage gaussianBlur(ImmutableImage src, double sigma) {
        Objects.requireNonNull(src, "src");
        if (sigma <= 0) throw new IllegalArgumentException("sigma");
        int radius = Math.max(1, (int) Math.ceil(sigma * 3));
        return new ImmutableImage(convolveSeparable(src.image(), gaussianKernel(sigma, radius), gaussianKernel(sigma, radius)));
    }

    /** Motion blur in direction angle. */
    public static ImmutableImage motionBlur(ImmutableImage src, double angle, int length) {
        Objects.requireNonNull(src, "src");
        if (length < 1) throw new IllegalArgumentException("length");
        double rad = Math.toRadians(angle);
        double dx = Math.cos(rad), dy = Math.sin(rad);
        int cx = length / 2, cy = length / 2;
        float[] k = new float[length];
        for (int i = 0; i < length; i++) k[i] = 1f;
        Image out = applyLinearKernel(src.image(), k, (float) dx, (float) dy, cx, cy);
        return new ImmutableImage(out);
    }

    // ── sharpen / emboss / edge ─────────────────────────────────────────────

    /** Sharpen. Pillow's SHARPEN: 3x3 with center=5, edges=-1 (normalized). */
    public static ImmutableImage sharpen(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        float[] k = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        return new ImmutableImage(convolve3x3(src.image(), k));
    }

    /** Emboss: classic 3x3. */
    public static ImmutableImage emboss(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        float[] k = {-2, -1, 0, -1, 1, 1, 0, 1, 2};
        return new ImmutableImage(convolve3x3(src.image(), k));
    }

    /** Find edges: Sobel magnitude. */
    public static ImmutableImage edgeDetect(ImmutableImage src) {
        return sobel(src);
    }

    /** Sobel magnitude. Output single-channel. */
    public static ImmutableImage sobel(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        Image g = src.image().convert("L");
        int w = g.width(), h = g.height();
        Image out = Image.new_("L", w, h);
        int[] gp = g.getdata();
        int[] op = out.getdata();
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int i = y * w + x;
                int tl = gp[i - w - 1] & 0xff, tm = gp[i - w] & 0xff, tr = gp[i - w + 1] & 0xff;
                int ml = gp[i - 1] & 0xff, mr = gp[i + 1] & 0xff;
                int bl = gp[i + w - 1] & 0xff, bm = gp[i + w] & 0xff, br = gp[i + w + 1] & 0xff;
                int gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
                int gy = (bl + 2 * bm + br) - (tl + 2 * tm + tr);
                int mag = Math.min(255, Math.abs(gx) + Math.abs(gy));
                op[i] = mag;
            }
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    /** Laplacian edge detection. */
    public static ImmutableImage laplacian(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        Image g = src.image().convert("L");
        float[] k = {0, 1, 0, 1, -4, 1, 0, 1, 0};
        return new ImmutableImage(convolve3x3(g, k));
    }

    /** Charcoal: grayscale + edge enhance. */
    public static ImmutableImage charcoal(ImmutableImage src) {
        return Colors.grayscale(sobel(src));
    }

    // ── noise / pixelate / vignette / swirl ─────────────────────────────────

    /** Uniform random noise (0..amplitude). */
    public static ImmutableImage noise(ImmutableImage src, int amplitude, long seed) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata(), opx = out.getdata();
        Random rnd = new Random(seed);
        for (int i = 0; i < px.length; i++) {
            int noise = rnd.nextInt(2 * amplitude + 1) - amplitude;
            opx[i] = Colors.clamp8(px[i] + noise);
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Snow: sprinkle white pixels randomly with {@code threshold} probability 0..1. */
    public static ImmutableImage snow(ImmutableImage src, float threshold, long seed) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = rgb.copy();
        int[] opx = out.getdata();
        Random rnd = new Random(seed);
        for (int i = 0; i < opx.length; i += 3) {
            if (rnd.nextFloat() < threshold) {
                opx[i] = 255;
                opx[i + 1] = 255;
                opx[i + 2] = 255;
            }
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Pixelate: downscale then upscale nearest. */
    public static ImmutableImage pixelate(ImmutableImage src, int blockSize) {
        Objects.requireNonNull(src, "src");
        if (blockSize < 1) throw new IllegalArgumentException("blockSize");
        int nw = Math.max(1, src.width() / blockSize);
        int nh = Math.max(1, src.height() / blockSize);
        Image small = src.image().resize(nw, nh, org.bytedeco.pytorch.vision.pillow.enums.Resampling.BOX);
        return new ImmutableImage(small.resize(src.width(), src.height(), org.bytedeco.pytorch.vision.pillow.enums.Resampling.NEAREST));
    }

    /** Vignette: darken edges via radial gradient. */
    public static ImmutableImage vignette(ImmutableImage src, float strength) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Image out = Image.new_("RGB", rgb.width(), rgb.height());
        int[] px = rgb.getdata(), opx = out.getdata();
        int cx = rgb.width() / 2, cy = rgb.height() / 2;
        double maxR = Math.sqrt(cx * cx + cy * cy);
        for (int y = 0; y < rgb.height(); y++) {
            for (int x = 0; x < rgb.width(); x++) {
                int idx = (y * rgb.width() + x) * 3;
                double dx = x - cx, dy = y - cy;
                double r = Math.sqrt(dx * dx + dy * dy) / maxR;
                double k = Math.max(0, 1 - strength * r);
                opx[idx]     = Colors.clamp8((int) (px[idx] * k));
                opx[idx + 1] = Colors.clamp8((int) (px[idx + 1] * k));
                opx[idx + 2] = Colors.clamp8((int) (px[idx + 2] * k));
            }
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Swirl: rotate pixels around the center by angle that decays with radius. */
    public static ImmutableImage swirl(ImmutableImage src, double degrees, double radius) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = rgb.copy();
        int[] px = rgb.getdata(), opx = out.getdata();
        int cx = w / 2, cy = h / 2;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int dx = x - cx, dy = y - cy;
                double r = Math.sqrt(dx * dx + dy * dy);
                double a = Math.toRadians(degrees) * (1.0 - Math.min(1.0, r / radius));
                double c = Math.cos(a), s = Math.sin(a);
                int sx = (int) Math.round(cx + dx * c - dy * s);
                int sy = (int) Math.round(cy + dx * s + dy * c);
                if (sx >= 0 && sy >= 0 && sx < w && sy < h) {
                    int srcIdx = (sy * w + sx) * 3;
                    int dstIdx = (y * w + x) * 3;
                    opx[dstIdx] = px[srcIdx];
                    opx[dstIdx + 1] = px[srcIdx + 1];
                    opx[dstIdx + 2] = px[srcIdx + 2];
                }
            }
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Oil painting effect via per-pixel dominant channel over a window. */
    public static ImmutableImage oilPainting(ImmutableImage src, int radius, int intensity) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int w = rgb.width(), h = rgb.height();
        Image out = Image.new_("RGB", w, h);
        int[] px = rgb.getdata(), opx = out.getdata();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int[] hist = new int[intensity];
                int[] sumR = new int[intensity];
                int[] sumG = new int[intensity];
                int[] sumB = new int[intensity];
                int maxCount = 0, maxIdx = 0;
                for (int ky = -radius; ky <= radius; ky++) {
                    int yy = Math.min(h - 1, Math.max(0, y + ky));
                    for (int kx = -radius; kx <= radius; kx++) {
                        int xx = Math.min(w - 1, Math.max(0, x + kx));
                        int idx = (yy * w + xx) * 3;
                        int v = (px[idx] + px[idx + 1] + px[idx + 2]) / 3 * intensity / 256;
                        v = Math.min(intensity - 1, v);
                        hist[v]++;
                        sumR[v] += px[idx];
                        sumG[v] += px[idx + 1];
                        sumB[v] += px[idx + 2];
                        if (hist[v] > maxCount) { maxCount = hist[v]; maxIdx = v; }
                    }
                }
                int dst = (y * w + x) * 3;
                opx[dst] = maxCount > 0 ? sumR[maxIdx] / maxCount : 0;
                opx[dst + 1] = maxCount > 0 ? sumG[maxIdx] / maxCount : 0;
                opx[dst + 2] = maxCount > 0 ? sumB[maxIdx] / maxCount : 0;
            }
        }
        out.putdata(opx);
        return new ImmutableImage(out);
    }

    /** Brighten/darken edges (Pillow ImageFilter-enhance edges pattern). */
    public static ImmutableImage watercolor(ImmutableImage src) {
        return Colors.saturation(sobel(src), 1.5f);
    }

    // ── internal kernels ────────────────────────────────────────────────────

    /** Box kernel of length {@code 2*radius+1} centred at index {@code radius}. */
    static float[] boxKernel(int radius) {
        float[] k = new float[2 * radius + 1];
        java.util.Arrays.fill(k, 1f / (2 * radius + 1));
        return k;
    }

    static float[] gaussianKernel(double sigma, int radius) {
        float[] k = new float[radius * 2 + 1];
        double s2 = 2 * sigma * sigma;
        double sum = 0;
        for (int i = -radius; i <= radius; i++) {
            k[i + radius] = (float) Math.exp(-(i * i) / s2);
            sum += k[i + radius];
        }
        for (int i = 0; i < k.length; i++) k[i] /= sum;
        return k;
    }

    /** Convolve with 3x3 separable kernel over RGB or L buffer. */
    static Image convolve3x3(Image src, float[] k) {
        Image out = Image.fromBuffer(new ImagingBuffer(src.getImagingBuffer().modeInfo(), src.width(), src.height()));
        int w = src.width(), h = src.height();
        int b = src.getImagingBuffer().modeInfo().bands();
        int[] sp = src.getdata();
        int[] op = out.getdata();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int dst = (y * w + x) * b;
                for (int c = 0; c < b; c++) {
                    double acc = 0;
                    for (int ky = -1; ky <= 1; ky++) {
                        int yy = Math.min(h - 1, Math.max(0, y + ky));
                        for (int kx = -1; kx <= 1; kx++) {
                            int xx = Math.min(w - 1, Math.max(0, x + kx));
                            int srcIdx = (yy * w + xx) * b + c;
                            acc += (sp[srcIdx] & 0xff) * k[(ky + 1) * 3 + (kx + 1)];
                        }
                    }
                    op[dst + c] = Colors.clamp8((int) acc);
                }
            }
        }
        out.putdata(op);
        return out;
    }

    /** Separable convolution: passes horizontal then vertical 1D kernel. */
    static Image convolveSeparable(Image src, float[] hKernel, float[] vKernel) {
        // Horizontal pass into scratch, then vertical into output.
        int w = src.width(), h = src.height();
        int b = src.getImagingBuffer().modeInfo().bands();
        int[] sp = src.getdata();
        int[] scratch = new int[sp.length];
        int hN = hKernel.length, vN = vKernel.length;
        int hR = hN / 2, vR = vN / 2;
        // horizontal
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int dst = (y * w + x) * b;
                for (int c = 0; c < b; c++) {
                    double acc = 0;
                    for (int k = -hR; k <= hR; k++) {
                        int xx = Math.min(w - 1, Math.max(0, x + k));
                        int srcIdx = (y * w + xx) * b + c;
                        acc += (sp[srcIdx] & 0xff) * hKernel[k + hR];
                    }
                    scratch[dst + c] = Colors.clamp8((int) acc);
                }
            }
        }
        int[] out = new int[sp.length];
        // vertical
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int dst = (y * w + x) * b;
                for (int c = 0; c < b; c++) {
                    double acc = 0;
                    for (int k = -vR; k <= vR; k++) {
                        int yy = Math.min(h - 1, Math.max(0, y + k));
                        int srcIdx = (yy * w + x) * b + c;
                        acc += (scratch[srcIdx] & 0xff) * vKernel[k + vR];
                    }
                    out[dst + c] = Colors.clamp8((int) acc);
                }
            }
        }
        Image result = Image.fromBuffer(new ImagingBuffer(src.getImagingBuffer().modeInfo(), w, h));
        result.putdata(out);
        return result;
    }

    /** Apply a 1D linear kernel along a vector (dx, dy). */
    static Image applyLinearKernel(Image src, float[] k, float dx, float dy, int cx, int cy) {
        int w = src.width(), h = src.height();
        int b = src.getImagingBuffer().modeInfo().bands();
        Image out = Image.fromBuffer(new ImagingBuffer(src.getImagingBuffer().modeInfo(), w, h));
        int[] sp = src.getdata();
        int[] op = out.getdata();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int dst = (y * w + x) * b;
                for (int c = 0; c < b; c++) {
                    double acc = 0;
                    double sum = 0;
                    for (int i = 0; i < k.length; i++) {
                        int sx = (int) Math.round(x + dx * (i - cx));
                        int sy = (int) Math.round(y + dy * (i - cy));
                        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
                        int srcIdx = (sy * w + sx) * b + c;
                        acc += (sp[srcIdx] & 0xff) * k[i];
                        sum += k[i];
                    }
                    if (sum > 0) acc /= sum;
                    op[dst + c] = Colors.clamp8((int) acc);
                }
            }
        }
        out.putdata(op);
        return out;
    }
}