package org.bytedeco.pytorch.vision.scrimage;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.enums.Resampling;
import org.bytedeco.pytorch.vision.scrimage.io.ScrimageIO;
import org.bytedeco.pytorch.vision.scrimage.ops.BlendMode;
import org.bytedeco.pytorch.vision.scrimage.ops.Colors;
import org.bytedeco.pytorch.vision.scrimage.ops.Composite;
import org.bytedeco.pytorch.vision.scrimage.ops.Dimensions;
import org.bytedeco.pytorch.vision.scrimage.ops.Effects;
import org.bytedeco.pytorch.vision.scrimage.ops.ExtraFilters;
import org.bytedeco.pytorch.vision.scrimage.ops.Filters;
import org.bytedeco.pytorch.vision.scrimage.ops.Pad;
import org.bytedeco.pytorch.vision.scrimage.ops.ScaleMethod;
import org.bytedeco.pytorch.vision.scrimage.ops.TorchVision;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Objects;

/**
 * {@code Scrimage} — high-level facade offering a Pillow/Scrimage-style API
 * covering 60+ image operations. Zero external dependencies; all operations
 * compose the existing {@link Image} / {@link org.bytedeco.pytorch.vision.draw}
 * infrastructure.
 *
 * <p>Each method returns an {@link ImmutableImage} for fluent composition:
 *
 * <pre>{@code
 * ImmutableImage out = Scrimage.grayscale(Scrimage.open(path))
 *                                  .resize(800, 600)
 *                                  .sepia()
 *                                  .blur(3);
 * }</pre>
 *
 * <p>Or as a one-shot builder:
 *
 * <pre>{@code
 * ImmutableImage out = Scrimage.of(Scrimage.open(path))
 *     .with("sepia")
 *     .with("resize", 800, 600)
 *     .get();
 * }</pre>
 */
public final class Scrimage {

    private Scrimage() {}

    // ── factories ───────────────────────────────────────────────────────────

    public static ImmutableImage open(String path) throws IOException {
        return new ImmutableImage(Image.open(path));
    }

    public static ImmutableImage open(Path path) throws IOException {
        return new ImmutableImage(Image.open(path));
    }

    public static ImmutableImage open(InputStream in) throws IOException {
        return new ImmutableImage(Image.open(in));
    }

    public static ImmutableImage open(byte[] data) throws IOException {
        return new ImmutableImage(Image.open(data));
    }

    public static ImmutableImage filled(int width, int height, Pixel fill) {
        return ImmutableImage.filled(width, height, fill);
    }

    public static ImmutableImage filled(int width, int height, DColor fill) {
        return filled(width, height, Pixel.of(fill));
    }

    public static ImmutableImage of(Image img) {
        return new ImmutableImage(img);
    }

    public static ImmutableImage of(BufferedImage img) {
        return new ImmutableImage(Image.fromBufferedImage(img));
    }

    public static Builder builder(ImmutableImage base) { return new Builder(base); }

    // ── pass-throughs (single method, common tasks) ────────────────────────

    public static ImmutableImage grayscale(ImmutableImage src) { return Colors.grayscale(src); }
    public static ImmutableImage invert(ImmutableImage src) { return Colors.invert(src); }
    public static ImmutableImage sepia(ImmutableImage src) { return Colors.sepia(src); }
    public static ImmutableImage brightness(ImmutableImage src, float factor) { return Colors.brightness(src, factor); }
    public static ImmutableImage contrast(ImmutableImage src, float factor) { return Colors.contrast(src, factor); }
    public static ImmutableImage gamma(ImmutableImage src, double g) { return Colors.gamma(src, g); }
    public static ImmutableImage hue(ImmutableImage src, float degrees) { return Colors.hue(src, degrees); }
    public static ImmutableImage saturation(ImmutableImage src, float factor) { return Colors.saturation(src, factor); }
    public static ImmutableImage posterize(ImmutableImage src, int bits) { return Colors.posterize(src, bits); }
    public static ImmutableImage solarize(ImmutableImage src, int threshold) { return Colors.solarize(src, threshold); }
    public static ImmutableImage threshold(ImmutableImage src, int threshold) { return Colors.threshold(src, threshold); }
    public static ImmutableImage monochrome(ImmutableImage src, int threshold) { return Colors.monochrome(src, threshold); }
    public static ImmutableImage tint(ImmutableImage src, DColor color, float mix) { return Colors.tint(src, color, mix); }

    public static ImmutableImage blur(ImmutableImage src, int radius) { return Filters.blur(src, radius); }
    public static ImmutableImage gaussianBlur(ImmutableImage src, double sigma) { return Filters.gaussianBlur(src, sigma); }
    public static ImmutableImage motionBlur(ImmutableImage src, double angle, int length) { return Filters.motionBlur(src, angle, length); }
    public static ImmutableImage sharpen(ImmutableImage src) { return Filters.sharpen(src); }
    public static ImmutableImage emboss(ImmutableImage src) { return Filters.emboss(src); }
    public static ImmutableImage edgeDetect(ImmutableImage src) { return Filters.edgeDetect(src); }
    public static ImmutableImage sobel(ImmutableImage src) { return Filters.sobel(src); }
    public static ImmutableImage laplacian(ImmutableImage src) { return Filters.laplacian(src); }
    public static ImmutableImage charcoal(ImmutableImage src) { return Filters.charcoal(src); }
    public static ImmutableImage pixelate(ImmutableImage src, int blockSize) { return Filters.pixelate(src, blockSize); }
    public static ImmutableImage vignette(ImmutableImage src, float strength) { return Filters.vignette(src, strength); }
    public static ImmutableImage swirl(ImmutableImage src, double degrees, double radius) { return Filters.swirl(src, degrees, radius); }
    public static ImmutableImage oilPainting(ImmutableImage src, int radius, int intensity) { return Filters.oilPainting(src, radius, intensity); }
    public static ImmutableImage snow(ImmutableImage src, float threshold, long seed) { return Filters.snow(src, threshold, seed); }
    public static ImmutableImage noise(ImmutableImage src, int amplitude, long seed) { return Filters.noise(src, amplitude, seed); }

    public static ImmutableImage scale(ImmutableImage src, double r) { return Dimensions.scale(src, r); }
    public static ImmutableImage scale(ImmutableImage src, double rx, double ry) { return Dimensions.scale(src, rx, ry); }
    public static ImmutableImage scale(ImmutableImage src, double rx, double ry, Resampling f) { return Dimensions.scale(src, rx, ry, f); }
    public static ImmutableImage resize(ImmutableImage src, int w, int h) { return Dimensions.resize(src, w, h); }
    public static ImmutableImage resize(ImmutableImage src, int w, int h, Resampling f) { return Dimensions.resize(src, w, h, f); }
    public static ImmutableImage fit(ImmutableImage src, int w, int h) { return Dimensions.fit(src, w, h); }
    public static ImmutableImage cover(ImmutableImage src, int w, int h) { return Dimensions.cover(src, w, h); }
    public static ImmutableImage contain(ImmutableImage src, int w, int h, DColor color) { return Dimensions.contain(src, w, h, color); }
    public static ImmutableImage crop(ImmutableImage src, int l, int u, int r, int d) { return Dimensions.crop(src, l, u, r, d); }
    public static ImmutableImage pad(ImmutableImage src, int w, int h, DColor c, Dimensions.PaddingPosition p) { return Dimensions.pad(src, w, h, c, p); }
    public static ImmutableImage autocrop(ImmutableImage src) { return Dimensions.autocrop(src, 0); }
    public static ImmutableImage trim(ImmutableImage src, int color) { return Dimensions.trim(src, color); }
    public static ImmutableImage thumbnail(ImmutableImage src, int w, int h) { return Dimensions.thumbnail(src, w, h); }

    public static ImmutableImage rotateLeft(ImmutableImage src) { return Effects.rotateLeft(src); }
    public static ImmutableImage rotateRight(ImmutableImage src) { return Effects.rotateRight(src); }
    public static ImmutableImage rotate(ImmutableImage src, double degrees) { return Effects.rotate(src, degrees); }
    public static ImmutableImage flip(ImmutableImage src) { return Effects.flip(src); }
    public static ImmutableImage mirror(ImmutableImage src) { return Effects.mirror(src); }
    public static ImmutableImage transverse(ImmutableImage src) { return Effects.transverse(src); }
    public static ImmutableImage lut(ImmutableImage src, int[] table) { return Effects.lut(src, table); }
    public static ImmutableImage applyCurve(ImmutableImage src, int[] xs, int[] ys) { return Effects.applyCurve(src, xs, ys); }
    public static ImmutableImage levels(ImmutableImage src, int black, int white, double gamma) { return Effects.levels(src, black, white, gamma); }
    public static ImmutableImage autoContrast(ImmutableImage src) { return Effects.autoContrast(src); }
    public static ImmutableImage equalize(ImmutableImage src) { return Effects.equalize(src); }
    public static ImmutableImage antiAlias(ImmutableImage src, double scale) { return Effects.antiAlias(src, scale); }

    public static ImmutableImage overlay(ImmutableImage b, ImmutableImage t, int x, int y, float a) { return Composite.overlay(b, t, x, y, a); }
    public static ImmutableImage alphaComposite(ImmutableImage b, ImmutableImage t) { return Composite.alphaComposite(b, t); }
    public static ImmutableImage watermark(ImmutableImage src, ImmutableImage mark, float alpha) { return Composite.watermark(src, mark, alpha); }
    public static ImmutableImage mask(ImmutableImage src, ImmutableImage m) { return Composite.mask(src, m); }
    public static ImmutableImage flatten(ImmutableImage src, DColor bg) { return Composite.flatten(src, bg); }
    public static ImmutableImage tile(ImmutableImage src, ImmutableImage pattern) { return Composite.tile(src, pattern); }
    public static ImmutableImage spriteSheet(ImmutableImage sheet, int col, int row, int tileW, int tileH) { return Composite.spriteSheet(sheet, col, row, tileW, tileH); }

    // ── extra scrimage filters ───────────────────────────────────────────────

    public static ImmutableImage channelShift(ImmutableImage src, int r, int g, int b) { return ExtraFilters.channelShift(src, r, g, b); }
    public static ImmutableImage convolve(ImmutableImage src, float[] k, int kw, int kh, float scale, float bias) { return ExtraFilters.convolve(src, k, kw, kh, scale, bias); }
    public static ImmutableImage edgeFilter(ImmutableImage src) { return ExtraFilters.edge(src); }
    public static ImmutableImage sharpenFilter(ImmutableImage src) { return ExtraFilters.sharpenFilter(src); }
    public static ImmutableImage stroke(ImmutableImage src, int thickness) { return ExtraFilters.stroke(src, thickness); }
    public static ImmutableImage tileFilter(ImmutableImage src, int tiles) { return ExtraFilters.tileFilter(src, tiles); }
    public static ImmutableImage marble(ImmutableImage src, double scale, long seed) { return ExtraFilters.marble(src, scale, seed); }
    public static ImmutableImage plasma(ImmutableImage src, long seed) { return ExtraFilters.plasma(src, seed); }
    public static ImmutableImage gcd(ImmutableImage src, int gcd) { return ExtraFilters.gcd(src, gcd); }
    public static ImmutableImage dilate(ImmutableImage src, int radius) { return ExtraFilters.dilate(src, radius); }
    public static ImmutableImage erode(ImmutableImage src, int radius) { return ExtraFilters.erode(src, radius); }
    public static ImmutableImage offset(ImmutableImage src, int dx, int dy) { return ExtraFilters.offset(src, dx, dy); }

    // ── pad / padTo with padding modes ──────────────────────────────────────

    public static ImmutableImage pad(ImmutableImage src, int all) { return Pad.pad(src, all); }
    public static ImmutableImage pad(ImmutableImage src, int left, int top, int right, int bottom) { return Pad.pad(src, left, top, right, bottom); }
    public static ImmutableImage pad(ImmutableImage src, int left, int top, int right, int bottom, DColor fill, Pad.PaddingMode mode) { return Pad.pad(src, left, top, right, bottom, fill, mode); }
    public static ImmutableImage padTo(ImmutableImage src, int w, int h) { return Pad.padTo(src, w, h); }

    // ── scale method overloads ──────────────────────────────────────────────

    public static ImmutableImage scale(ImmutableImage src, double rx, double ry, ScaleMethod method) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(method, "method");
        return Dimensions.scale(src, rx, ry, method.toResampling());
    }
    public static ImmutableImage resize(ImmutableImage src, int w, int h, ScaleMethod method) {
        Objects.requireNonNull(src, "src");
        return Dimensions.resize(src, w, h, method.toResampling());
    }

    // ── torchvision port ───────────────────────────────────────────────────

    public static ImmutableImage centerCrop(ImmutableImage src, int size) { return TorchVision.centerCrop(src, size); }
    public static ImmutableImage centerCrop(ImmutableImage src, int w, int h) { return TorchVision.centerCrop(src, w, h); }
    public static ImmutableImage[] fiveCrop(ImmutableImage src, int size) { return TorchVision.fiveCrop(src, size); }
    public static ImmutableImage randomCrop(ImmutableImage src, int size, long seed) { return TorchVision.randomCrop(src, size, seed); }
    public static ImmutableImage randomResizedCrop(ImmutableImage src, int size, long seed) { return TorchVision.randomResizedCrop(src, size, seed); }
    public static ImmutableImage randomHorizontalFlip(ImmutableImage src, float p, long seed) { return TorchVision.randomHorizontalFlip(src, p, seed); }
    public static ImmutableImage randomVerticalFlip(ImmutableImage src, float p, long seed) { return TorchVision.randomVerticalFlip(src, p, seed); }
    public static ImmutableImage randomRotation(ImmutableImage src, double degrees, long seed) { return TorchVision.randomRotation(src, degrees, seed); }
    public static ImmutableImage randomAffine(ImmutableImage src, double[] deg, double[] translate, double[] scale, double[] shear, long seed) { return TorchVision.randomAffine(src, deg, translate, scale, shear, seed); }
    public static ImmutableImage affine(ImmutableImage src, double angle, double tx, double ty, double scale, double shear) { return TorchVision.affine(src, angle, tx, ty, scale, shear); }
    public static ImmutableImage randomPerspective(ImmutableImage src, double distortion, float p, long seed) { return TorchVision.randomPerspective(src, distortion, p, seed); }
    public static ImmutableImage perspective(ImmutableImage src, double distortion) { return TorchVision.perspective(src, distortion); }
    public static ImmutableImage randomElastic(ImmutableImage src, double alpha, double sigma, long seed) { return TorchVision.randomElastic(src, alpha, sigma, seed); }
    public static ImmutableImage colorJitter(ImmutableImage src, float brightness, float contrast, float saturation, float hue, long seed) { return TorchVision.colorJitter(src, brightness, contrast, saturation, hue, seed); }
    public static ImmutableImage randomGrayscale(ImmutableImage src, float p, long seed) { return TorchVision.randomGrayscale(src, p, seed); }
    public static ImmutableImage gaussianNoise(ImmutableImage src, double mean, double sigma, boolean clip, long seed) { return TorchVision.gaussianNoise(src, mean, sigma, clip, seed); }
    public static ImmutableImage randomAdjustSharpness(ImmutableImage src, float factor, float p, long seed) { return TorchVision.randomAdjustSharpness(src, factor, p, seed); }
    public static ImmutableImage normalize(ImmutableImage src, float[] mean, float[] std) { return TorchVision.normalize(src, mean, std); }
    public static ImmutableImage randomErasing(ImmutableImage src, float p, double[] scaleRange, double[] ratioRange, DColor fill, long seed) { return TorchVision.randomErasing(src, p, scaleRange, ratioRange, fill, seed); }
    public static ImmutableImage cutout(ImmutableImage src, int size, DColor fill, long seed) { return TorchVision.cutout(src, size, fill, seed); }
    public static ImmutableImage hideAndSeek(ImmutableImage src, int grid, float p, DColor fill, long seed) { return TorchVision.hideAndSeek(src, grid, p, fill, seed); }
    public static ImmutableImage randomZoomOut(ImmutableImage src, double[] sideRange, DColor fill, long seed) { return TorchVision.randomZoomOut(src, sideRange, fill, seed); }
    public static ImmutableImage scaleJitter(ImmutableImage src, int targetSize, double[] scaleRange, long seed) { return TorchVision.scaleJitter(src, targetSize, scaleRange, seed); }
    public static ImmutableImage randomChannelPermutation(ImmutableImage src, long seed) { return TorchVision.randomChannelPermutation(src, seed); }
    public static ImmutableImage autoAugment(ImmutableImage src, String policy, long seed) { return TorchVision.autoAugment(src, policy, seed); }
    public static ImmutableImage randAugment(ImmutableImage src, int n, int m, long seed) { return TorchVision.randAugment(src, n, m, seed); }
    public static ImmutableImage trivialAugmentWide(ImmutableImage src, long seed) { return TorchVision.trivialAugmentWide(src, seed); }
    public static ImmutableImage augMix(ImmutableImage src, int severity, int mixtureWidth, float alpha, long seed) { return TorchVision.augMix(src, severity, mixtureWidth, alpha, seed); }

    public static ImmutableImage blend(ImmutableImage b, ImmutableImage t, BlendMode mode) { return Colors.blend(b, t, mode); }
    public static ImmutableImage multiply(ImmutableImage b, ImmutableImage t) { return Colors.multiply(b, t); }
    public static ImmutableImage screen(ImmutableImage b, ImmutableImage t) { return Colors.screen(b, t); }
    public static ImmutableImage darken(ImmutableImage b, ImmutableImage t) { return Colors.darken(b, t); }
    public static ImmutableImage lighten(ImmutableImage b, ImmutableImage t) { return Colors.lighten(b, t); }
    public static ImmutableImage difference(ImmutableImage b, ImmutableImage t) { return Colors.difference(b, t); }
    public static ImmutableImage exclusion(ImmutableImage b, ImmutableImage t) { return Colors.exclusion(b, t); }

    // ── IO ──────────────────────────────────────────────────────────────────

    public static void save(ImmutableImage img, Path path) throws IOException { org.bytedeco.pytorch.vision.scrimage.io.ScrimageIO.write(img, path); }
    public static void save(ImmutableImage img, Path path, String fmt) throws IOException { org.bytedeco.pytorch.vision.scrimage.io.ScrimageIO.write(img, path, fmt, new java.util.HashMap<>()); }
    public static byte[] toBytes(ImmutableImage img, String fmt) throws IOException { return org.bytedeco.pytorch.vision.scrimage.io.ScrimageIO.write(img, fmt); }

    // ── chainable builder ───────────────────────────────────────────────────

    /**
     * Lightweight builder that accepts {@code String} op names + values and
     * chains them. Useful when the op set is driven by configuration.
     */
    public static final class Builder {
        private ImmutableImage current;

        public Builder(ImmutableImage base) { this.current = base; }

        public ImmutableImage get() { return current; }

        public Builder with(String op, Object... args) {
            current = applyOp(current, op, args);
            return this;
        }

        private static ImmutableImage applyOp(ImmutableImage in, String op, Object[] args) {
            switch (op) {
                case "grayscale": return Colors.grayscale(in);
                case "invert":    return Colors.invert(in);
                case "sepia":     return Colors.sepia(in);
                case "blur":      return Filters.blur(in, (int) args[0]);
                case "gaussian":  return Filters.gaussianBlur(in, ((Number) args[0]).doubleValue());
                case "sharpen":   return Filters.sharpen(in);
                case "edge":      return Filters.edgeDetect(in);
                case "sobel":     return Filters.sobel(in);
                case "emboss":    return Filters.emboss(in);
                case "pixelate":  return Filters.pixelate(in, (int) args[0]);
                case "vignette":  return Filters.vignette(in, ((Number) args[0]).floatValue());
                case "swirl":     return Filters.swirl(in, ((Number) args[0]).doubleValue(), ((Number) args[1]).doubleValue());
                case "posterize": return Colors.posterize(in, (int) args[0]);
                case "solarize":  return Colors.solarize(in, (int) args[0]);
                case "threshold": return Colors.threshold(in, (int) args[0]);
                case "rotate":    return Effects.rotate(in, ((Number) args[0]).doubleValue());
                case "flip":      return Effects.flip(in);
                case "mirror":    return Effects.mirror(in);
                case "resize":    return Dimensions.resize(in, (int) args[0], (int) args[1]);
                case "scale":     return Dimensions.scale(in, ((Number) args[0]).doubleValue());
                case "fit":       return Dimensions.fit(in, (int) args[0], (int) args[1]);
                case "cover":     return Dimensions.cover(in, (int) args[0], (int) args[1]);
                case "crop":      return Dimensions.crop(in, (int) args[0], (int) args[1], (int) args[2], (int) args[3]);
                case "pad":       return Dimensions.pad(in, (int) args[0], (int) args[1], DColor.of((String) args[2]), Dimensions.PaddingPosition.CENTER);
                case "autoContrast": return Effects.autoContrast(in);
                case "equalize":  return Effects.equalize(in);
                case "levels":    return Effects.levels(in, (int) args[0], (int) args[1], ((Number) args[2]).doubleValue());
                case "tint":      return Colors.tint(in, DColor.of((String) args[0]), ((Number) args[1]).floatValue());
                default:
                    throw new IllegalArgumentException("Unknown op: " + op);
            }
        }
    }
}