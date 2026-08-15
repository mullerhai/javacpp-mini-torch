package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;
import org.bytedeco.pytorch.vision.scrimage.Scrimage;

import java.util.Objects;
import java.util.Random;

/**
 * Torchvision v2 augmentation ops ported to the scrimage facade.
 *
 * <p>All ops are pure-Java and produce {@link ImmutableImage}. Where the
 * original Torchvision op is random, we provide a non-random functional API
 * plus a stable "Random" variant taking a seeded {@link Random}.
 *
 * <p>Coverage:
 * <ul>
 *   <li>Geometry: {@code centerCrop}, {@code fiveCrop}, {@code tenCrop},
 *       {@code randomCrop}, {@code randomResizedCrop}, {@code randomHorizontalFlip},
 *       {@code randomVerticalFlip}, {@code randomAffine}, {@code randomPerspective},
 *       {@code elasticTransform}, {@code scaleJitter}, {@code randomZoomOut}</li>
 *   <li>Color: {@code colorJitter}, {@code randomGrayscale},
 *       {@code randomPhotometricDistort}, {@code randomChannelPermutation},
 *       {@code gaussianNoise}, {@code randomAdjustSharpness},
 *       {@code normalize}, {@code linearTransformation}</li>
 *   <li>Cut/Drop: {@code randomErasing}, {@code cutout},
 *       {@code hideAndSeek}, {@code jpegQuality}</li>
 *   <li>Composition: {@code compose}, {@code randomApply}, {@code randomChoice},
 *       {@code randomOrder}</li>
 *   <li>AutoAug / RandAug / TrivialAugment / AugMix</li>
 * </ul>
 */
public final class TorchVision {

    private TorchVision() {}

    /** Random source. Create your own and pass explicitly for deterministic results. */
    public static Random rng() { return new Random(); }

    // ── Geometry: deterministic transforms ──────────────────────────────────

    /** CenterCrop to {@code size}x{@code size}. */
    public static ImmutableImage centerCrop(ImmutableImage src, int size) {
        Objects.requireNonNull(src, "src");
        int w = src.width(), h = src.height();
        int lx = Math.max(0, (w - size) / 2);
        int ly = Math.max(0, (h - size) / 2);
        return Dimensions.crop(src, lx, ly, lx + Math.min(size, w), ly + Math.min(size, h));
    }

    /** CenterCrop to {@code width} x {@code height}. */
    public static ImmutableImage centerCrop(ImmutableImage src, int width, int height) {
        Objects.requireNonNull(src, "src");
        int w = src.width(), h = src.height();
        int lx = Math.max(0, (w - width) / 2);
        int ly = Math.max(0, (h - height) / 2);
        return Dimensions.crop(src, lx, ly, lx + Math.min(width, w), ly + Math.min(height, h));
    }

    /** FiveCrop: returns array of 5 crops (tl, tr, bl, br, center). */
    public static ImmutableImage[] fiveCrop(ImmutableImage src, int size) {
        Objects.requireNonNull(src, "src");
        int w = src.width(), h = src.height();
        int cx = (w - size) / 2, cy = (h - size) / 2;
        return new ImmutableImage[]{
                Dimensions.crop(src, 0, 0, size, size),
                Dimensions.crop(src, w - size, 0, w, size),
                Dimensions.crop(src, 0, h - size, size, h),
                Dimensions.crop(src, w - size, h - size, w, h),
                Dimensions.crop(src, cx, cy, cx + size, cy + size)
        };
    }

    /** FiveCrop with explicit width/height. */
    public static ImmutableImage[] fiveCrop(ImmutableImage src, int width, int height) {
        Objects.requireNonNull(src, "src");
        int w = src.width(), h = src.height();
        int cx = (w - width) / 2, cy = (h - height) / 2;
        return new ImmutableImage[]{
                Dimensions.crop(src, 0, 0, width, height),
                Dimensions.crop(src, w - width, 0, w, height),
                Dimensions.crop(src, 0, h - height, width, h),
                Dimensions.crop(src, w - width, h - height, w, h),
                Dimensions.crop(src, cx, cy, cx + width, cy + height)
        };
    }

    /** TenCrop: {@code fiveCrop} plus each flipped horizontally. */
    public static ImmutableImage[] tenCrop(ImmutableImage src, int size, boolean verticalFlip) {
        ImmutableImage[] five = fiveCrop(src, size);
        ImmutableImage[] out = new ImmutableImage[10];
        for (int i = 0; i < 5; i++) {
            out[i] = five[i];
            ImmutableImage flipped = verticalFlip ? Effects.flip(five[i]) : Effects.mirror(five[i]);
            out[i + 5] = flipped;
        }
        return out;
    }

    /** Random crop at a uniform random position, target {@code size}. */
    public static ImmutableImage randomCrop(ImmutableImage src, int size, long seed) {
        return randomCrop(src, size, size, seed);
    }

    public static ImmutableImage randomCrop(ImmutableImage src, int width, int height, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        int lx = rnd.nextInt(Math.max(1, src.width() - width + 1));
        int ly = rnd.nextInt(Math.max(1, src.height() - height + 1));
        return Dimensions.crop(src, lx, ly, lx + width, ly + height);
    }

    /**
     * RandomResizedCrop: pick a random area and aspect-ratio, then crop + resize to {@code size}.
     * Default {@code scale = (0.08, 1.0)} and {@code ratio = (3/4, 4/3)} matching torchvision.
     */
    public static ImmutableImage randomResizedCrop(ImmutableImage src, int size, long seed) {
        return randomResizedCrop(src, size, size, 0.08, 1.0, 0.75, 1.3333333, seed);
    }

    public static ImmutableImage randomResizedCrop(ImmutableImage src, int tw, int th,
                                                    double scaleMin, double scaleMax,
                                                    double ratioMin, double ratioMax, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        int w = src.width(), h = src.height();
        int area = w * h;
        for (int attempt = 0; attempt < 10; attempt++) {
            double targetArea = area * (scaleMin + rnd.nextDouble() * (scaleMax - scaleMin));
            double aspect = Math.exp(Math.log(ratioMin) + rnd.nextDouble() * (Math.log(ratioMax) - Math.log(ratioMin)));
            int cropW = (int) Math.round(Math.sqrt(targetArea * aspect));
            int cropH = (int) Math.round(Math.sqrt(targetArea / aspect));
            if (cropW <= w && cropH <= h && cropW > 0 && cropH > 0) {
                int lx = rnd.nextInt(w - cropW + 1);
                int ly = rnd.nextInt(h - cropH + 1);
                ImmutableImage cropped = Dimensions.crop(src, lx, ly, lx + cropW, ly + cropH);
                return Dimensions.resize(cropped, tw, th);
            }
        }
        // Fallback: centerCrop
        ImmutableImage cc = centerCrop(src, Math.min(w, h));
        return Dimensions.resize(cc, tw, th);
    }

    /** Flip horizontally with probability {@code p}. */
    public static ImmutableImage randomHorizontalFlip(ImmutableImage src, float p, long seed) {
        return new Random(seed).nextFloat() < p ? Effects.mirror(src) : src.copy();
    }

    /** Flip vertically with probability {@code p}. */
    public static ImmutableImage randomVerticalFlip(ImmutableImage src, float p, long seed) {
        return new Random(seed).nextFloat() < p ? Effects.flip(src) : src.copy();
    }

    /** Random rotation in {@code [-degrees, degrees]}. */
    public static ImmutableImage randomRotation(ImmutableImage src, double degrees, long seed) {
        Objects.requireNonNull(src, "src");
        double d = 2 * degrees * new Random(seed).nextDouble() - degrees;
        return Effects.rotate(src, d, true, org.bytedeco.pytorch.vision.draw.DColor.of("white"));
    }

    /**
     * Random affine: rotate by {@code degrees[]}, translate {@code translate[]},
     * scale {@code scale[]}, shear {@code shear[]} (degrees).
     */
    public static ImmutableImage randomAffine(ImmutableImage src,
                                              double[] degrees, double[] translate,
                                              double[] scale, double[] shear,
                                              long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        double theta = degrees == null ? 0 : (degrees[0] + (degrees.length > 1 ? rnd.nextDouble() * (degrees[1] - degrees[0]) : 0));
        double[] t = translate != null ? translate : new double[]{0, 0};
        double[] s = scale != null ? scale : new double[]{1.0};
        double[] sh = shear != null ? shear : new double[]{0};
        double scaleV = s[rnd.nextInt(s.length)];
        double shearV = (sh.length > 1 ? sh[0] + rnd.nextDouble() * (sh[1] - sh[0]) : sh[0]);
        return affine(src, theta, t[0], t[1], scaleV, shearV);
    }

    /** Affine transformation matrix. */
    public static ImmutableImage affine(ImmutableImage src, double angle, double translateX, double translateY,
                                        double scale, double shearDegrees) {
        Objects.requireNonNull(src, "src");
        double rad = Math.toRadians(angle);
        double cos = Math.cos(rad) / Math.max(1e-9, scale);
        double sin = Math.sin(rad) / Math.max(1e-9, scale);
        double sSh = Math.tan(Math.toRadians(shearDegrees));
        // Combined forward: x' = cos*x - sin*y + tx;  y' = sin*x + cos*y + ty + sSh*(cos*x - sin*y)
        double a = cos;
        double b = -sin;
        double c = sin + sSh * cos;
        double d = cos + sSh * (-sin); // = cos - sin*sSh
        int w = src.width(), h = src.height();
        org.bytedeco.pytorch.vision.draw.DColor fill = org.bytedeco.pytorch.vision.draw.DColor.of("white");
        org.bytedeco.pytorch.vision.pillow.Image out = org.bytedeco.pytorch.vision.pillow.Image.new_("RGB", w, h, fill.argb & 0xFFFFFF);
        double cx = w / 2.0, cy = h / 2.0;
        int[] sp = src.image().getdata();
        int[] op = out.getdata();
        // Inverse of M = [[a, b],[c, d]]
        double det = a * d - b * c;
        if (Math.abs(det) < 1e-12) return src.copy();
        double invA = d / det, invB = -b / det, invC = -c / det, invD = a / det;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double dx = x - cx;
                double dy = y - cy;
                double sx = cx + invA * dx + invB * dy - translateX;
                double sy = cy + invC * dx + invD * dy - translateY;
                int ix = (int) Math.round(sx);
                int iy = (int) Math.round(sy);
                if (ix >= 0 && iy >= 0 && ix < w && iy < h) {
                    int sidx = (iy * w + ix) * 3;
                    int didx = (y * w + x) * 3;
                    op[didx]     = sp[sidx];
                    op[didx + 1] = sp[sidx + 1];
                    op[didx + 2] = sp[sidx + 2];
                }
            }
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    /** Random perspective with {@code distortionScale} (0..1). */
    public static ImmutableImage randomPerspective(ImmutableImage src, double distortionScale, float p, long seed) {
        Objects.requireNonNull(src, "src");
        if (new Random(seed).nextFloat() >= p) return src.copy();
        return perspective(src, distortionScale);
    }

    public static ImmutableImage perspective(ImmutableImage src, double distortionScale) {
        Objects.requireNonNull(src, "src");
        int w = src.width(), h = src.height();
        if (distortionScale <= 0) return src.copy();
        Random rnd = new Random(0);
        double half = distortionScale / 2.0;
        int[][] from = new int[][]{{0, 0}, {w, 0}, {w, h}, {0, h}};
        int[][] to = new int[4][2];
        for (int i = 0; i < 4; i++) {
            to[i][0] = (int) (from[i][0] + (rnd.nextDouble() * 2 - 1) * half * w);
            to[i][1] = (int) (from[i][1] + (rnd.nextDouble() * 2 - 1) * half * h);
        }
        // Solve 8x8 linear system per row/col, then apply inverse
        // Simple approach: bilinear warp from `to` -> `from` for every destination pixel
        org.bytedeco.pytorch.vision.draw.DColor fill = org.bytedeco.pytorch.vision.draw.DColor.of("white");
        org.bytedeco.pytorch.vision.pillow.Image out = org.bytedeco.pytorch.vision.pillow.Image.new_("RGB", w, h, fill.argb & 0xFFFFFF);
        int[] sp = src.image().getdata();
        int[] op = out.getdata();
        // Use the simple variant: compute perspective H so that H @ to[i] = from[i]
        // 8x8 system solving using Cramer's rule for 3 rows at a time
        double[] H = computePerspective(from, to);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double wD = H[6] * x + H[7] * y + 1;
                int sx = (int) Math.round((H[0] * x + H[1] * y + H[2]) / wD);
                int sy = (int) Math.round((H[3] * x + H[4] * y + H[5]) / wD);
                if (sx >= 0 && sy >= 0 && sx < w && sy < h) {
                    int sidx = (sy * w + sx) * 3;
                    int didx = (y * w + x) * 3;
                    op[didx]     = sp[sidx];
                    op[didx + 1] = sp[sidx + 1];
                    op[didx + 2] = sp[sidx + 2];
                }
            }
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    /** Compute perspective H so that H * from[i] (homogeneous) ~ to[i]. */
    private static double[] computePerspective(int[][] from, int[][] to) {
        // Set up 8 equations in 8 unknowns (h0..h7).
        // to[i] ~ H * from[i].
        double[] A = new double[8 * 9];
        for (int i = 0; i < 4; i++) {
            int x = from[i][0], y = from[i][1];
            double u = to[i][0], v = to[i][1];
            int r0 = i * 2, r1 = i * 2 + 1;
            A[r0 * 9 + 0] = x; A[r0 * 9 + 1] = y; A[r0 * 9 + 2] = 1; A[r0 * 9 + 6] = -u * x; A[r0 * 9 + 7] = -u * y; A[r0 * 9 + 8] = u;
            A[r1 * 9 + 3] = x; A[r1 * 9 + 4] = y; A[r1 * 9 + 5] = 1; A[r1 * 9 + 6] = -v * x; A[r1 * 9 + 7] = -v * y; A[r1 * 9 + 8] = v;
        }
        double[] H = new double[8];
        // Gauss elimination
        for (int i = 0; i < 8; i++) {
            int pivot = i;
            for (int j = i; j < 8; j++) if (Math.abs(A[j * 9 + i]) > Math.abs(A[pivot * 9 + i])) pivot = j;
            for (int k = 0; k < 9; k++) { double t = A[i * 9 + k]; A[i * 9 + k] = A[pivot * 9 + k]; A[pivot * 9 + k] = t; }
            for (int j = i + 1; j < 8; j++) {
                double f = A[j * 9 + i] / A[i * 9 + i];
                for (int k = i; k < 9; k++) A[j * 9 + k] -= f * A[i * 9 + k];
            }
        }
        for (int i = 7; i >= 0; i--) {
            double s = A[i * 9 + 8];
            for (int j = i + 1; j < 8; j++) s -= A[i * 9 + j] * H[j];
            H[i] = s / A[i * 9 + i];
        }
        return H;
    }

    /** Elastic transform (torchvision-style). */
    public static ImmutableImage randomElastic(ImmutableImage src, double alpha, double sigma, long seed) {
        return elasticTransform(src, alpha, sigma, seed);
    }

    public static ImmutableImage elasticTransform(ImmutableImage src, double alpha, double sigma, long seed) {
        Objects.requireNonNull(src, "src");
        int w = src.width(), h = src.height();
        Random rnd = new Random(seed);
        double[] dxField = smoothRandomField(w, h, sigma, rnd);
        double[] dyField = smoothRandomField(w, h, sigma, rnd);
        org.bytedeco.pytorch.vision.draw.DColor fill = org.bytedeco.pytorch.vision.draw.DColor.of("white");
        org.bytedeco.pytorch.vision.pillow.Image out = org.bytedeco.pytorch.vision.pillow.Image.new_("RGB", w, h, fill.argb & 0xFFFFFF);
        int[] sp = src.image().getdata();
        int[] op = out.getdata();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int sx = (int) (x + alpha * dxField[y * w + x]);
                int sy = (int) (y + alpha * dyField[y * w + x]);
                if (sx >= 0 && sy >= 0 && sx < w && sy < h) {
                    int sidx = (sy * w + sx) * 3;
                    int didx = (y * w + x) * 3;
                    op[didx]     = sp[sidx];
                    op[didx + 1] = sp[sidx + 1];
                    op[didx + 2] = sp[sidx + 2];
                }
            }
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    private static double[] smoothRandomField(int w, int h, double sigma, Random rnd) {
        double[] raw = new double[w * h];
        for (int i = 0; i < raw.length; i++) raw[i] = rnd.nextDouble() * 2 - 1;
        // Apply gaussian smoothing via separable conv
        double[] smoothed = new double[raw.length];
        int r = (int) Math.max(1, Math.ceil(sigma * 3));
        float[] k = Filters.gaussianKernel(sigma, r);
        // horizontal
        double[] scratch = new double[raw.length];
        int hR = k.length / 2;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double s = 0;
                for (int i = -hR; i <= hR; i++) {
                    int xx = Math.min(w - 1, Math.max(0, x + i));
                    s += raw[y * w + xx] * k[i + hR];
                }
                scratch[y * w + x] = s;
            }
        }
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double s = 0;
                for (int j = -hR; j <= hR; j++) {
                    int yy = Math.min(h - 1, Math.max(0, y + j));
                    s += scratch[yy * w + x] * k[j + hR];
                }
                smoothed[y * w + x] = s;
            }
        }
        return smoothed;
    }

    /** ScaleJitter: like RandomResizedCrop but with random ratio only. */
    public static ImmutableImage scaleJitter(ImmutableImage src, int targetSize, double[] scaleRange, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        double lo = scaleRange == null ? 0.5 : scaleRange[0];
        double hi = scaleRange == null ? 1.5 : scaleRange[1];
        double s = lo + rnd.nextDouble() * (hi - lo);
        int nw = Math.max(1, (int) (src.width() * s));
        int nh = Math.max(1, (int) (src.height() * s));
        return Dimensions.resize(src, nw, nh);
    }

    /** RandomZoomOut: scale down then pad. */
    public static ImmutableImage randomZoomOut(ImmutableImage src, double[] sideRange, org.bytedeco.pytorch.vision.draw.DColor fill, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        double lo = sideRange == null ? 1.0 : sideRange[0];
        double hi = sideRange == null ? 4.0 : sideRange[1];
        double s = lo + rnd.nextDouble() * (hi - lo);
        int nw = (int) (src.width() * s);
        int nh = (int) (src.height() * s);
        ImmutableImage scaled = Dimensions.resize(src, nw, nh);
        return Dimensions.pad(scaled, nw, nh, fill == null ? org.bytedeco.pytorch.vision.draw.DColor.of("white") : fill, Dimensions.PaddingPosition.CENTER);
    }

    // ── Color / photometric ops ─────────────────────────────────────────────

    /**
     * ColorJitter: brightness, contrast, saturation, hue factors.
     * Each factor can be {@code float} (single value) or {@code float[2]} (lo, hi).
     */
    public static ImmutableImage colorJitter(ImmutableImage src, float brightness, float contrast, float saturation, float hue, long seed) {
        Random rnd = new Random(seed);
        ImmutableImage cur = src;
        cur = Colors.brightness(cur, brightnessFactor(brightness, rnd));
        cur = Colors.contrast(cur, contrastFactor(contrast, rnd));
        cur = Colors.saturation(cur, satFactor(saturation, rnd));
        cur = Colors.hue(cur, hueFactor(hue, rnd));
        return cur;
    }

    private static float brightnessFactor(float v, Random rnd) { return v <= 0 ? 1f : 1f + (rnd.nextFloat() * 2 - 1) * v; }
    private static float contrastFactor(float v, Random rnd)   { return v <= 0 ? 1f : 1f + (rnd.nextFloat() * 2 - 1) * v; }
    private static float satFactor(float v, Random rnd)         { return v <= 0 ? 1f : 1f + (rnd.nextFloat() * 2 - 1) * v; }
    private static float hueFactor(float v, Random rnd)         { return v <= 0 ? 0f : (rnd.nextFloat() * 2 - 1) * v; }

    /** Random grayscale: convert to grayscale with probability {@code p}. */
    public static ImmutableImage randomGrayscale(ImmutableImage src, float p, long seed) {
        return new Random(seed).nextFloat() < p ? Colors.grayscale(src) : src.copy();
    }

    /** RandomPhotometricDistort (SSD-style). */
    public static ImmutableImage randomPhotometricDistort(ImmutableImage src,
                                                          float brightness, float contrast, float saturation, float hue,
                                                          long seed) {
        return colorJitter(src, brightness, contrast, saturation, hue, seed);
    }

    /** RandomChannelPermutation: shuffle channel order. */
    public static ImmutableImage randomChannelPermutation(ImmutableImage src, long seed) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        Random rnd = new Random(seed);
        int[] perm = new int[]{0, 1, 2};
        for (int i = 2; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        for (int i = 0; i < sp.length; i += 3) {
            op[i + perm[0]] = sp[i];
            op[i + perm[1]] = sp[i + 1];
            op[i + perm[2]] = sp[i + 2];
        }
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), rgb.width(), rgb.height()));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Add Gaussian noise. */
    public static ImmutableImage gaussianNoise(ImmutableImage src, double mean, double sigma, boolean clip, long seed) {
        Objects.requireNonNull(src, "src");
        Image rgb = src.image().convert("RGB");
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        Random rnd = new Random(seed);
        for (int i = 0; i < sp.length; i++) {
            double g = (rnd.nextGaussian() * sigma + mean);
            int v = sp[i] + (int) g;
            if (clip) v = Math.max(0, Math.min(255, v));
            op[i] = v;
        }
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), rgb.width(), rgb.height()));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** Adjust sharpness by factor (1.0 unchanged, {@code <1} blur, {@code >1} sharpen). */
    public static ImmutableImage adjustSharpness(ImmutableImage src, float factor) {
        Objects.requireNonNull(src, "src");
        if (factor <= 0) throw new IllegalArgumentException("factor > 0");
        if (factor == 1f) return src.copy();
        if (factor > 1f) {
            // scale the SHARPEN kernel strength
            return Filters.sharpen(src);
        }
        // factor<1: box blur by (1/factor), approximation
        return Filters.blur(src, Math.max(1, (int) Math.round(1.0 / factor)));
    }

    public static ImmutableImage randomAdjustSharpness(ImmutableImage src, float factor, float p, long seed) {
        return new Random(seed).nextFloat() < p ? adjustSharpness(src, factor) : src.copy();
    }

    /** Normalize (only valid for L/RGB): subtract mean, divide std. */
    public static ImmutableImage normalize(ImmutableImage src, float[] mean, float[] std) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(mean, "mean");
        Objects.requireNonNull(std, "std");
        Image rgb = src.image().convert("RGB");
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        for (int i = 0; i < sp.length; i += 3) {
            op[i]     = Colors.clamp8((int) ((sp[i]     - mean[0]) / std[0]));
            op[i + 1] = Colors.clamp8((int) ((sp[i + 1] - mean[1]) / std[1]));
            op[i + 2] = Colors.clamp8((int) ((sp[i + 2] - mean[2]) / std[2]));
        }
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), rgb.width(), rgb.height()));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    /** LinearTransformation: apply a 3x3 matrix to RGB pixels (PCA-style). */
    public static ImmutableImage linearTransformation(ImmutableImage src, float[][] matrix) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(matrix, "matrix");
        Image rgb = src.image().convert("RGB");
        int[] sp = rgb.getdata();
        int[] op = new int[sp.length];
        for (int i = 0; i < sp.length; i += 3) {
            for (int r = 0; r < 3; r++) {
                op[i + r] = Colors.clamp8((int) (sp[i]     * matrix[r][0]
                                                + sp[i + 1] * matrix[r][1]
                                                + sp[i + 2] * matrix[r][2]));
            }
        }
        Image out = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(rgb.getImagingBuffer().modeInfo(), rgb.width(), rgb.height()));
        out.getImagingBuffer().putdata(op, 0, 1);
        return new ImmutableImage(out);
    }

    // ── Erase / Cutout ─────────────────────────────────────────────────────

    /** Random erasing (torchvision): zero-out a random rectangle. */
    public static ImmutableImage randomErasing(ImmutableImage src, float p, double[] scaleRange, double[] ratioRange,
                                               org.bytedeco.pytorch.vision.draw.DColor fillColor, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        if (rnd.nextFloat() >= p) return src.copy();
        double sMin = scaleRange == null ? 0.02 : scaleRange[0];
        double sMax = scaleRange == null ? 0.33 : scaleRange[1];
        double rMin = ratioRange == null ? 0.3 : ratioRange[0];
        double rMax = ratioRange == null ? 3.3 : ratioRange[1];
        int area = src.width() * src.height();
        for (int attempt = 0; attempt < 10; attempt++) {
            double targetArea = area * (sMin + rnd.nextDouble() * (sMax - sMin));
            double aspect = Math.exp(Math.log(rMin) + rnd.nextDouble() * (Math.log(rMax) - Math.log(rMin)));
            int ew = (int) Math.round(Math.sqrt(targetArea * aspect));
            int eh = (int) Math.round(Math.sqrt(targetArea / aspect));
            if (ew >= src.width()) ew = src.width();
            if (eh >= src.height()) eh = src.height();
            int lx = rnd.nextInt(Math.max(1, src.width() - ew + 1));
            int ly = rnd.nextInt(Math.max(1, src.height() - eh + 1));
            return eraseRect(src, lx, ly, ew, eh, fillColor);
        }
        return src.copy();
    }

    /** Alias for randomErasing. */
    public static ImmutableImage cutout(ImmutableImage src, int size, org.bytedeco.pytorch.vision.draw.DColor fillColor, long seed) {
        Random rnd = new Random(seed);
        int lx = rnd.nextInt(Math.max(1, src.width() - size + 1));
        int ly = rnd.nextInt(Math.max(1, src.height() - size + 1));
        return eraseRect(src, lx, ly, size, size, fillColor);
    }

    /** Hide-and-seek: zero out {@code nPatches} random patches. */
    public static ImmutableImage hideAndSeek(ImmutableImage src, int grid, float p, org.bytedeco.pytorch.vision.draw.DColor fillColor, long seed) {
        Objects.requireNonNull(src, "src");
        Image out = src.copy().image();
        int cw = src.width() / grid, ch = src.height() / grid;
        Random rnd = new Random(seed);
        int[] op = out.getdata();
        int fb = fillColor == null ? 0 : (fillColor.argb & 0xFFFFFF);
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                if (rnd.nextFloat() < p) {
                    int x0 = gx * cw, y0 = gy * ch;
                    for (int y = y0; y < y0 + ch && y < out.height(); y++) {
                        int didx = y * out.width() * 3 + x0 * 3;
                        int nFill = Math.min(cw, out.width() - x0);
                        for (int k = 0; k < nFill; k++) {
                            op[didx++] = (fb >> 16) & 0xff;
                            op[didx++] = (fb >> 8) & 0xff;
                            op[didx++] = fb & 0xff;
                        }
                    }
                }
            }
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    /** JPEG encode-decode with quality in [0,100]. */
    public static ImmutableImage jpegQuality(ImmutableImage src, int quality) throws java.io.IOException {
        Objects.requireNonNull(src, "src");
        byte[] data = Scrimage.toBytes(src, "JPEG");
        java.util.HashMap<String, Object> opts = new java.util.HashMap<>();
        opts.put("quality", quality / 100f);
        byte[] enc = org.bytedeco.pytorch.vision.scrimage.io.ScrimageIO.write(new ImmutableImage(src.image().copy()), "JPEG", opts);
        return Scrimage.open(new java.io.ByteArrayInputStream(enc));
    }

    static ImmutableImage eraseRect(ImmutableImage src, int lx, int ly, int ew, int eh, org.bytedeco.pytorch.vision.draw.DColor fillColor) {
        Image out = src.copy().image();
        int[] op = out.getdata();
        org.bytedeco.pytorch.vision.draw.DColor c = fillColor == null ? org.bytedeco.pytorch.vision.draw.DColor.of(0, 0, 0) : fillColor;
        int r = (c.argb >> 16) & 0xff, g = (c.argb >> 8) & 0xff, b = c.argb & 0xff;
        for (int y = ly; y < ly + eh && y < out.height(); y++) {
            for (int x = lx; x < lx + ew && x < out.width(); x++) {
                int idx = (y * out.width() + x) * 3;
                op[idx]     = r;
                op[idx + 1] = g;
                op[idx + 2] = b;
            }
        }
        out.putdata(op);
        return new ImmutableImage(out);
    }

    // ── AutoAug / RandAug / TrivialAugment / AugMix ────────────────────────

    /** AutoAugment with policy IMAGENET/CIFAR10/SVHN — sample a sub-policy and apply. */
    public static ImmutableImage autoAugment(ImmutableImage src, String policy, long seed) {
        Objects.requireNonNull(src, "src");
        java.util.List<OpEntry[]> ps = policy.equals("CIFAR10") ? AUG_CIFAR : policy.equals("SVHN") ? AUG_SVHN : AUG_IMAGENET;
        Random rnd = new Random(seed);
        OpEntry[] sub = ps.get(rnd.nextInt(ps.size()));
        ImmutableImage cur = src;
        for (OpEntry e : sub) cur = e.apply(cur, rnd);
        return cur;
    }

    /** RandAugment: pick {@code n} random ops, each applied at magnitude {@code m/numBins}. */
    public static ImmutableImage randAugment(ImmutableImage src, int n, int m, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        ImmutableImage cur = src;
        for (int i = 0; i < n; i++) {
            OpEntry entry = AUG_OPS.get(rnd.nextInt(AUG_OPS.size()));
            cur = entry.apply(cur, rnd, m, 30);
        }
        return cur;
    }

    /** TrivialAugmentWide: a single random op at a random magnitude. */
    public static ImmutableImage trivialAugmentWide(ImmutableImage src, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        OpEntry entry = AUG_OPS.get(rnd.nextInt(AUG_OPS.size()));
        return entry.apply(src, rnd, rnd.nextInt(31), 30);
    }

    /** AugMix: chain of random ops + a final mix with the original. */
    public static ImmutableImage augMix(ImmutableImage src, int severity, int mixtureWidth, float alpha, long seed) {
        Objects.requireNonNull(src, "src");
        Random rnd = new Random(seed);
        java.util.List<ImmutableImage> chain = new java.util.ArrayList<>();
        chain.add(src);
        for (int i = 0; i < mixtureWidth; i++) {
            ImmutableImage cur = src;
            for (int j = 0; j < 2; j++) {
                OpEntry e = AUG_OPS.get(rnd.nextInt(AUG_OPS.size()));
                cur = e.apply(cur, rnd, severity, 30);
            }
            chain.add(cur);
        }
        // Mix originals
        java.util.Random rngBeta = new Random(seed);
        float[] weights = new float[chain.size()];
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) Math.pow(rngBeta.nextDouble(), alpha == 0 ? 1 : 1.0 / alpha);
            sum += weights[i];
        }
        for (int i = 0; i < weights.length; i++) weights[i] /= sum;
        int w = src.width(), h = src.height();
        Image sumImg = Image.fromBuffer(new org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer(src.image().getImagingBuffer().modeInfo(), w, h));
        int[] sp = new int[sumImg.getdata().length];
        for (int k = 0; k < chain.size(); k++) {
            int[] cp = chain.get(k).image().getdata();
            for (int i = 0; i < sp.length; i++) sp[i] += cp[i] * weights[k];
        }
        for (int i = 0; i < sp.length; i++) sp[i] = Colors.clamp8(sp[i]);
        sumImg.getImagingBuffer().putdata(sp, 0, 1);
        return new ImmutableImage(sumImg);
    }

    // ── Aug entries ────────────────────────────────────────────────────────

    /** Per-op enum with an {@code apply(image, rng, level, bins)} implementation. */
    public enum OpEntry {
        Identity { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { return src.copy(); } },
        ShearX   { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { double s = ((level / (double) bins) * 0.3) * (rnd.nextBoolean() ? 1 : -1); return affine(src, 0, 0, 0, 1.0, Math.toDegrees(s)); } },
        ShearY   { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { double s = ((level / (double) bins) * 0.3) * (rnd.nextBoolean() ? 1 : -1); return affine(src, 0, 0, 0, 1.0, Math.toDegrees(s)); } },
        TranslateX { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { int dx = (int) ((level / (double) bins) * 0.5 * src.width()) * (rnd.nextBoolean() ? 1 : -1); return affine(src, 0, dx, 0, 1, 0); } },
        TranslateY { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { int dy = (int) ((level / (double) bins) * 0.5 * src.height()) * (rnd.nextBoolean() ? 1 : -1); return affine(src, 0, 0, dy, 1, 0); } },
        Rotate   { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { double deg = ((level / (double) bins) * 30) * (rnd.nextBoolean() ? 1 : -1); return Effects.rotate(src, deg, true, org.bytedeco.pytorch.vision.draw.DColor.of("white")); } },
        Brightness { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { float f = (float)(1.0 + (level / (double) bins) * 0.6 * (rnd.nextBoolean() ? 1 : -1)); return Colors.brightness(src, f); } },
        Color    { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { float f = (float)(1.0 + (level / (double) bins) * 0.9 * (rnd.nextBoolean() ? 1 : -1)); return Colors.saturation(src, f); } },
        Contrast { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { float f = (float)(1.0 + (level / (double) bins) * 0.9 * (rnd.nextBoolean() ? 1 : -1)); return Colors.contrast(src, f); } },
        Sharpness { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { float f = (float)(1.0 + (level / (double) bins) * 0.9 * (rnd.nextBoolean() ? 1 : -1)); return adjustSharpness(src, f); } },
        Posterize { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { int bits = Math.max(1, 8 - (int) Math.round((level / (double) bins) * 4)); return Colors.posterize(src, bits); } },
        Solarize  { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { int threshold = 256 - (int) (level / (double) bins) * 256; return Colors.solarize(src, threshold); } },
        Equalize  { public ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins) { return Effects.equalize(src); } };

        public abstract ImmutableImage apply(ImmutableImage src, Random rnd, int level, int bins);
        public ImmutableImage apply(ImmutableImage src, Random rnd) { return apply(src, rnd, 5, 10); }
    }

    private static final java.util.List<OpEntry> AUG_OPS = java.util.Arrays.asList(
            OpEntry.ShearX, OpEntry.ShearY, OpEntry.TranslateX, OpEntry.TranslateY,
            OpEntry.Rotate, OpEntry.Brightness, OpEntry.Color, OpEntry.Contrast,
            OpEntry.Sharpness, OpEntry.Posterize, OpEntry.Solarize, OpEntry.Equalize
    );

    /** AutoAugment policies (sublist of ops). For brevity we use OpEntry entries. */
    private static final java.util.List<OpEntry[]> AUG_IMAGENET = new java.util.ArrayList<>();
    private static final java.util.List<OpEntry[]> AUG_CIFAR = new java.util.ArrayList<>();
    private static final java.util.List<OpEntry[]> AUG_SVHN = new java.util.ArrayList<>();
    static {
        AUG_IMAGENET.add(new OpEntry[]{OpEntry.ShearY, OpEntry.Equalize});
        AUG_IMAGENET.add(new OpEntry[]{OpEntry.ShearX, OpEntry.Posterize});
        AUG_IMAGENET.add(new OpEntry[]{OpEntry.Equalize, OpEntry.Rotate});
        AUG_IMAGENET.add(new OpEntry[]{OpEntry.Posterize, OpEntry.Brightness});
        AUG_CIFAR.add(new OpEntry[]{OpEntry.ShearY, OpEntry.Equalize});
        AUG_CIFAR.add(new OpEntry[]{OpEntry.Contrast, OpEntry.TranslateX});
        AUG_SVHN.add(new OpEntry[]{OpEntry.ShearY, OpEntry.Equalize});
        AUG_SVHN.add(new OpEntry[]{OpEntry.Contrast, OpEntry.Rotate});
    }

    // ── Composition ────────────────────────────────────────────────────────

    @FunctionalInterface
    public interface Transform {
        ImmutableImage apply(ImmutableImage src, Random rnd);
    }

    public static ImmutableImage compose(ImmutableImage src, Transform... ts) {
        ImmutableImage cur = src;
        Random rnd = new Random();
        for (Transform t : ts) cur = t.apply(cur, rnd);
        return cur;
    }

    public static Transform randomApply(Transform t, float p, long seed) {
        return (im, rnd) -> rnd.nextFloat() < p ? t.apply(im, rnd) : im.copy();
    }

    public static Transform randomChoice(Transform[] opts, long seed) {
        return (im, rnd) -> opts[rnd.nextInt(opts.length)].apply(im, rnd);
    }

    public static Transform randomOrder(Transform[] opts, long seed) {
        return (im, rnd) -> {
            java.util.List<Transform> list = new java.util.ArrayList<>(java.util.Arrays.asList(opts));
            java.util.Collections.shuffle(list, rnd);
            ImmutableImage cur = im;
            for (Transform t : list) cur = t.apply(cur, rnd);
            return cur;
        };
    }
}