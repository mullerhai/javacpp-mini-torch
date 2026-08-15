package org.bytedeco.pytorch.vision.scrimage;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;

import java.util.Objects;

/**
 * ImmutableImage: scrimage ImmutableImage stand-in.
 *
 * <p>Wraps an {@link Image}. All operations that would mutate state copy first,
 * so this object is safe to share across threads (as long as the underlying
 * {@code ImagingBuffer} byte array is not exposed).
 *
 * <p>All methods return a new {@code ImmutableImage} when the result differs
 * from the input. {@link #image()} exposes the underlying {@link Image} for
 * interop with existing Pillow APIs.
 */
public final class ImmutableImage {

    private final Image image;

    public ImmutableImage(Image image) {
        this.image = Objects.requireNonNull(image);
    }

    public static ImmutableImage of(Image src) { return new ImmutableImage(src); }

    public static ImmutableImage filled(int width, int height, Pixel fill) {
        Image im = Image.new_("RGB", width, height, fill.argb() & 0xFFFFFF);
        return new ImmutableImage(im);
    }

    public Image image() { return image; }
    public int width() { return image.width(); }
    public int height() { return image.height(); }
    public String mode() { return image.mode(); }

    public ImmutableImage copy() { return new ImmutableImage(image.copy()); }

    public MutableImage toMutable() { return new MutableImage(image.copy()); }

    public Pixel pixel(int x, int y) {
        int[] p = image.getpixel(x, y);
        return new Pixel(p[0], p.length > 1 ? p[1] : p[0], p.length > 2 ? p[2] : p[0], p.length > 3 ? p[3] : 255);
    }
}