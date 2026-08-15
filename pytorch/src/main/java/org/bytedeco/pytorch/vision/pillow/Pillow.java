package org.bytedeco.pytorch.vision.pillow;

import org.bytedeco.pytorch.vision.pillow.features.Features;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;

/**
 * Facade entry: version / open / new / init / features shortcuts.
 */
public final class Pillow {
    private Pillow() {}

    public static String version() {
        return PillowVersion.VERSION;
    }

    public static String upstream_ref() {
        return PillowVersion.UPSTREAM_REF;
    }

    public static void preinit() {
        Image.preinit();
    }

    public static void init() {
        Image.init();
    }

    public static Image open(String path) throws IOException {
        return Image.open(path);
    }

    public static Image open(Path path) throws IOException {
        return Image.open(path);
    }

    public static Image open(InputStream in) throws IOException {
        return Image.open(in);
    }

    public static Image open(byte[] data) throws IOException {
        return Image.open(data);
    }

    public static Image new_(String mode, int width, int height) {
        return Image.new_(mode, width, height);
    }

    public static Image new_(String mode, int width, int height, Object color) {
        return Image.new_(mode, width, height, color);
    }

    public static Image create(String mode, int width, int height) {
        return Image.create(mode, width, height);
    }

    public static void pilinfo() {
        Features.pilinfo();
    }

    public static boolean check_codec(String name) {
        return Features.check_codec(name);
    }

    public static boolean checkCodec(String name) {
        return Features.checkCodec(name);
    }

    // ── Drawing API entry points ──────────────────────────────────────────
    // Mirrors Pillow's `ImageDraw.Draw(im)` pattern.

    /**
     * Open (or re-attach to) a draw handle for the given image.
     * The returned object draws directly into the underlying {@code BufferedImage}.
     */
    public static org.bytedeco.pytorch.vision.draw.ImageDraw ImageDraw(Image image) {
        return image.getDraw();
    }

    /**
     * Convenience: allocate a fresh RGB image with the given color and return
     * a ready-to-use draw handle on it.
     */
    public static org.bytedeco.pytorch.vision.draw.ImageDraw newImageDraw(String mode, int width, int height, Object color) {
        Image im = new_(mode, width, height, color);
        return im.getDraw();
    }

    // ── Scrimage bridge ───────────────────────────────────────────────────
    //
    // 1:1 scrimage-style ops exposed directly on Pillow. All methods delegate
    // to {@link org.bytedeco.pytorch.vision.scrimage.Scrimage}. Returning
    // {@code Image} (instead of {@code ImmutableImage}) keeps backward
    // compatibility with existing Pillow call sites.

    /** Wrap an Image into the scrimage immutable type. */
    public static org.bytedeco.pytorch.vision.scrimage.ImmutableImage asImmutable(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.of(im);
    }

    public static Image grayscale(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.grayscale(asImmutable(im)).image();
    }

    public static Image invert(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.invert(asImmutable(im)).image();
    }

    public static Image sepia(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.sepia(asImmutable(im)).image();
    }

    public static Image blur(Image im, int radius) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.blur(asImmutable(im), radius).image();
    }

    public static Image sharpen(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.sharpen(asImmutable(im)).image();
    }

    public static Image edgeDetect(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.edgeDetect(asImmutable(im)).image();
    }

    public static Image emboss(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.emboss(asImmutable(im)).image();
    }

    public static Image rotate(Image im, double degrees) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.rotate(asImmutable(im), degrees).image();
    }

    public static Image flip(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.flip(asImmutable(im)).image();
    }

    public static Image mirror(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.mirror(asImmutable(im)).image();
    }

    public static Image scale(Image im, double r) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.scale(asImmutable(im), r).image();
    }

    public static Image scale(Image im, double rx, double ry) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.scale(asImmutable(im), rx, ry).image();
    }

    public static Image fit(Image im, int w, int h) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.fit(asImmutable(im), w, h).image();
    }

    public static Image cover(Image im, int w, int h) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.cover(asImmutable(im), w, h).image();
    }

    public static Image contain(Image im, int w, int h, org.bytedeco.pytorch.vision.draw.DColor color) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.contain(asImmutable(im), w, h, color).image();
    }

    public static Image brightness(Image im, float factor) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.brightness(asImmutable(im), factor).image();
    }

    public static Image contrast(Image im, float factor) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.contrast(asImmutable(im), factor).image();
    }

    public static Image gamma(Image im, double g) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.gamma(asImmutable(im), g).image();
    }

    public static Image hue(Image im, float degrees) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.hue(asImmutable(im), degrees).image();
    }

    public static Image saturation(Image im, float factor) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.saturation(asImmutable(im), factor).image();
    }

    public static Image posterize(Image im, int bits) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.posterize(asImmutable(im), bits).image();
    }

    public static Image solarize(Image im, int threshold) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.solarize(asImmutable(im), threshold).image();
    }

    public static Image pixelate(Image im, int blockSize) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.pixelate(asImmutable(im), blockSize).image();
    }

    public static Image vignette(Image im, float strength) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.vignette(asImmutable(im), strength).image();
    }

    public static Image autoContrast(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.autoContrast(asImmutable(im)).image();
    }

    public static Image equalize(Image im) {
        return org.bytedeco.pytorch.vision.scrimage.Scrimage.equalize(asImmutable(im)).image();
    }
}
