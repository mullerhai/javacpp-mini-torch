package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.enums.Resampling;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;

import java.util.Objects;

/**
 * Sizing / geometry family (scrimage {@code ScaleMethods} + Pillow ImageOps).
 *
 * <p>Pure-Java implementations over the existing {@link Image} API.
 * Each method is stateless and returns a new {@link ImmutableImage}.
 */
public final class Dimensions {

    private Dimensions() {}

    /** Pillow: scale by fractional ratio (e.g. 0.5 = half). */
    public static ImmutableImage scale(ImmutableImage src, double ratio) {
        return scale(src, ratio, ratio, Resampling.BICUBIC);
    }

    public static ImmutableImage scale(ImmutableImage src, double rx, double ry) {
        return scale(src, rx, ry, Resampling.BICUBIC);
    }

    public static ImmutableImage scale(ImmutableImage src, double rx, double ry, Resampling filter) {
        Objects.requireNonNull(src, "src");
        int nw = Math.max(1, (int) Math.round(src.width() * rx));
        int nh = Math.max(1, (int) Math.round(src.height() * ry));
        return new ImmutableImage(src.image().resize(nw, nh, filter));
    }

    /** Pillow: scale so width = targetW (height proportional). */
    public static ImmutableImage scaleToWidth(ImmutableImage src, int targetW, Resampling filter) {
        Objects.requireNonNull(src, "src");
        double r = (double) targetW / src.width();
        return scale(src, r, r, filter);
    }

    /** Pillow: scale so height = targetH. */
    public static ImmutableImage scaleToHeight(ImmutableImage src, int targetH, Resampling filter) {
        Objects.requireNonNull(src, "src");
        double r = (double) targetH / src.height();
        return scale(src, r, r, filter);
    }

    /** Resize to exact size (Pillow's resize()). */
    public static ImmutableImage resize(ImmutableImage src, int width, int height) {
        Objects.requireNonNull(src, "src");
        return new ImmutableImage(src.image().resize(width, height));
    }

    public static ImmutableImage resize(ImmutableImage src, int width, int height, Resampling filter) {
        Objects.requireNonNull(src, "src");
        return new ImmutableImage(src.image().resize(width, height, filter));
    }

    /** Fit: resize so the image fits inside (w, h); preserves aspect; do not crop. */
    public static ImmutableImage fit(ImmutableImage src, int w, int h) {
        return fit(src, w, h, Resampling.BICUBIC);
    }

    public static ImmutableImage fit(ImmutableImage src, int w, int h, Resampling filter) {
        Objects.requireNonNull(src, "src");
        double r = Math.min((double) w / src.width(), (double) h / src.height());
        return scale(src, r, r, filter);
    }

    /** Cover: resize to fill (w, h), center-crop to exact size. */
    public static ImmutableImage cover(ImmutableImage src, int w, int h) {
        return cover(src, w, h, Resampling.BICUBIC);
    }

    public static ImmutableImage cover(ImmutableImage src, int w, int h, Resampling filter) {
        Objects.requireNonNull(src, "src");
        double r = Math.max((double) w / src.width(), (double) h / src.height());
        ImmutableImage scaled = scale(src, r, r, filter);
        // center-crop
        int sx = Math.max(0, (scaled.width() - w) / 2);
        int sy = Math.max(0, (scaled.height() - h) / 2);
        return new ImmutableImage(scaled.image().crop(sx, sy, sx + w, sy + h));
    }

    /** Contain: fit inside then pad to exact (w, h) with {@code color}. */
    public static ImmutableImage contain(ImmutableImage src, int w, int h, org.bytedeco.pytorch.vision.draw.DColor color) {
        return contain(src, w, h, color, Resampling.BICUBIC);
    }

    public static ImmutableImage contain(ImmutableImage src, int w, int h, org.bytedeco.pytorch.vision.draw.DColor color, Resampling filter) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(color, "color");
        ImmutableImage fitted = fit(src, w, h, filter);
        Image canvas = Image.new_("RGB", w, h, color.argb & 0xFFFFFF);
        int px = Math.max(0, (w - fitted.width()) / 2);
        int py = Math.max(0, (h - fitted.height()) / 2);
        canvas.paste(fitted.image(), px, py);
        return new ImmutableImage(canvas);
    }

    /** Resize to exact size (Pillow: resizeTo). */
    public static ImmutableImage resizeTo(ImmutableImage src, int w, int h) {
        return resize(src, w, h);
    }

    public static ImmutableImage resizeTo(ImmutableImage src, int w, int h, Resampling filter) {
        return resize(src, w, h, filter);
    }

    /** Crop (Pillow: crop((left, upper, right, lower))). */
    public static ImmutableImage crop(ImmutableImage src, int left, int upper, int right, int lower) {
        Objects.requireNonNull(src, "src");
        return new ImmutableImage(src.image().crop(left, upper, right, lower));
    }

    /** Auto-crop (Pillow: autocrop or getbbox + crop). */
    public static ImmutableImage autocrop(ImmutableImage src, int threshold) {
        Objects.requireNonNull(src, "src");
        int[] box = src.image().getbbox();
        if (box == null) return src.copy();
        // extend by 1 to mimic Pillow's autocrop padding
        int left = Math.max(0, box[0] - 1);
        int upper = Math.max(0, box[1] - 1);
        int right = Math.min(src.width(), box[2] + 1);
        int lower = Math.min(src.height(), box[3] + 1);
        return crop(src, left, upper, right, lower);
    }

    public static ImmutableImage trim(ImmutableImage src, int color) {
        Objects.requireNonNull(src, "src");
        return autocrop(src, color);
    }

    /** Pad to exact (w, h) with {@code color} using position from {@link PaddingPosition}. */
    public static ImmutableImage pad(ImmutableImage src, int w, int h, org.bytedeco.pytorch.vision.draw.DColor color, PaddingPosition position) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(color, "color");
        if (src.width() == w && src.height() == h) return src.copy();
        Image canvas = Image.new_("RGB", w, h, color.argb & 0xFFFFFF);
        int px = switch (position) {
            case TOP_LEFT, CENTER_LEFT, BOTTOM_LEFT -> 0;
            case TOP_CENTER, CENTER, BOTTOM_CENTER -> Math.max(0, (w - src.width()) / 2);
            case TOP_RIGHT, CENTER_RIGHT, BOTTOM_RIGHT -> Math.max(0, w - src.width());
        };
        int py = switch (position) {
            case TOP_LEFT, TOP_CENTER, TOP_RIGHT -> 0;
            case CENTER_LEFT, CENTER, CENTER_RIGHT -> Math.max(0, (h - src.height()) / 2);
            case BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT -> Math.max(0, h - src.height());
        };
        canvas.paste(src.image(), px, py);
        return new ImmutableImage(canvas);
    }

    /** Pillow: thumbnail (in-place reduces size, never enlarges). */
    public static ImmutableImage thumbnail(ImmutableImage src, int w, int h) {
        Objects.requireNonNull(src, "src");
        Image copy = src.image().copy();
        copy.thumbnail(new int[]{w, h});
        return new ImmutableImage(copy);
    }

    /** Aspect ratio as float (w/h). */
    public static double aspectRatio(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        return (double) src.width() / src.height();
    }

    /** Bounds [x, y, width, height]. */
    public static int[] bounds(ImmutableImage src) {
        Objects.requireNonNull(src, "src");
        return new int[]{0, 0, src.width(), src.height()};
    }

    public enum PaddingPosition {
        TOP_LEFT, TOP_CENTER, TOP_RIGHT,
        CENTER_LEFT, CENTER, CENTER_RIGHT,
        BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT
    }
}