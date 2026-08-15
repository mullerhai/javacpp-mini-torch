package org.bytedeco.pytorch.vision.scrimage;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;

import java.util.Objects;

/**
 * MutableImage: scrimage MutableImage stand-in.
 *
 * <p>Wraps an {@link Image} but provides a {@code foreach}-style pixel mutator:
 * {@link #updatePixels(PixelUpdater)}. Internally makes a copy of the buffer
 * and writes it back so callers never modify the source.
 *
 * <p>This is preferred when many pixels change in place; {@code ImmutableImage}
 * is used for chained pure-function transforms.
 */
public final class MutableImage {

    private final Image image;

    public MutableImage(Image image) {
        this.image = Objects.requireNonNull(image);
    }

    public Image image() { return image; }
    public int width() { return image.width(); }
    public int height() { return image.height(); }
    public String mode() { return image.mode(); }

    /** Apply a transformation function to each pixel (in-place copy). */
    public MutableImage updatePixels(PixelUpdater updater) {
        Objects.requireNonNull(updater, "updater");
        int[] pixels = image.getdata();
        int bands = pixels.length / (width() * height());
        for (int i = 0; i < pixels.length; i += bands) {
            int r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
            int a = bands >= 4 ? pixels[i + 3] : 255;
            Pixel p = new Pixel(r, g, b, a);
            Pixel out = updater.apply(p, (i / bands) % width(), (i / bands) / width());
            pixels[i] = out.r;
            pixels[i + 1] = out.g;
            pixels[i + 2] = out.b;
            if (bands >= 4) pixels[i + 3] = out.a;
        }
        image.putdata(pixels);
        return this;
    }

    public MutableImage copy() { return new MutableImage(image.copy()); }

    public ImmutableImage toImmutable() { return new ImmutableImage(image.copy()); }

    /** Functional interface for per-pixel mutation. */
    @FunctionalInterface
    public interface PixelUpdater {
        Pixel apply(Pixel p, int x, int y);
    }
}