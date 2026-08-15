package org.bytedeco.pytorch.vision.scrimage.ops;

import org.bytedeco.pytorch.vision.draw.DColor;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;

import java.util.Objects;

/**
 * Composite / overlay operations.
 *
 * <p>Scrimage exposes {@code overlay()} and Pillow has {@code paste()},
 * {@code alpha_composite()}, and {@code composite()}. This class unifies them
 * behind a small fluent surface that returns {@link ImmutableImage}.
 */
public final class Composite {

    private Composite() {}

    /** Place {@code top} over {@code bottom} with {@code x,y} offset and alpha. */
    public static ImmutableImage overlay(ImmutableImage bottom, ImmutableImage top, int x, int y, float alpha) {
        Objects.requireNonNull(bottom, "bottom");
        Objects.requireNonNull(top, "top");
        Image b = bottom.image();
        Image t = top.image();
        // ensure alpha channel
        Image t2 = t.convert("RGBA");
        int[] tdata = t2.getdata();
        for (int i = 3; i < tdata.length; i += 4) {
            tdata[i] = Colors.clamp8((int) (tdata[i] * alpha));
        }
        t2.putdata(tdata);
        b.paste(t2, x, y);
        return new ImmutableImage(b);
    }

    /** Alpha composite (Pillow Image.alpha_composite): src over dst. */
    public static ImmutableImage alphaComposite(ImmutableImage bottom, ImmutableImage top) {
        Objects.requireNonNull(bottom, "bottom");
        Objects.requireNonNull(top, "top");
        return new ImmutableImage(Image.alphaComposite(bottom.image(), top.image()));
    }

    /** Watermark: place {@code mark} centered with given alpha. */
    public static ImmutableImage watermark(ImmutableImage src, ImmutableImage mark, float alpha) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(mark, "mark");
        int x = Math.max(0, (src.width() - mark.width()) / 2);
        int y = Math.max(0, (src.height() - mark.height()) / 2);
        return overlay(src, mark, x, y, alpha);
    }

    /** Mask composite: copy {@code src} pixels where {@code mask} is non-zero. */
    public static ImmutableImage mask(ImmutableImage src, ImmutableImage mask) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(mask, "mask");
        Image s = src.image().convert("RGB");
        Image m = mask.image().convert("L");
        if (s.width() != m.width() || s.height() != m.height()) {
            throw new IllegalArgumentException("size mismatch");
        }
        int[] sp = s.getdata();
        int[] mp = m.getdata();
        for (int i = 0; i < sp.length; i += 3) {
            int a = mp[i / 3] & 0xff;
            sp[i] = sp[i] * a / 255;
            sp[i + 1] = sp[i + 1] * a / 255;
            sp[i + 2] = sp[i + 2] * a / 255;
        }
        s.putdata(sp);
        return new ImmutableImage(s);
    }

    /** Tile {@code pattern} across the entire image. */
    public static ImmutableImage tile(ImmutableImage src, ImmutableImage pattern) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(pattern, "pattern");
        Image dst = src.image().copy();
        int pw = pattern.width(), ph = pattern.height();
        for (int y = 0; y < dst.height(); y += ph) {
            for (int x = 0; x < dst.width(); x += pw) {
                dst.paste(pattern.image(), x, y);
            }
        }
        return new ImmutableImage(dst);
    }

    /** Composite with a constant-color background (flatten alpha). */
    public static ImmutableImage flatten(ImmutableImage src, DColor bg) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(bg, "bg");
        Image rgba = src.image().convert("RGBA");
        int w = rgba.width(), h = rgba.height();
        Image canvas = Image.new_("RGB", w, h, bg.argb & 0xFFFFFF);
        canvas.paste(rgba, 0, 0);
        return new ImmutableImage(canvas);
    }

    /** Sprite-sheet composite: pull tile at (col,row) from sheet. */
    public static ImmutableImage spriteSheet(ImmutableImage sheet, int col, int row, int tileWidth, int tileHeight) {
        Objects.requireNonNull(sheet, "sheet");
        int x = col * tileWidth;
        int y = row * tileHeight;
        return new ImmutableImage(sheet.image().crop(x, y, x + tileWidth, y + tileHeight));
    }
}