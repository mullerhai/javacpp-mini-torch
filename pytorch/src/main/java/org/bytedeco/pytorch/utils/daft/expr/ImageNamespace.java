/*
 * Image namespace - operations on image columns.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.dtype.DataValue;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;

import java.util.Objects;

public final class ImageNamespace {

    private final Expression inner;
    public ImageNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner); }

    public Expression decode() {
        return new MediaFn(this, "decode", bytes -> {
            if (bytes == null) return null;
            if (bytes instanceof byte[]) return new ImageData((byte[]) bytes);
            return new ImageData(bytes.toString());
        });
    }

    public Expression resize(int width, int height) {
        return new MediaFn(this, "resize_" + width + "x" + height, value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            return img == null ? null : img.resize(width, height);
        });
    }

    public Expression centerCrop(int width, int height) {
        return new MediaFn(this, "crop_" + width + "x" + height, value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            return img == null ? null : img.crop(
                    (img.getWidth() - width) / 2,
                    (img.getHeight() - height) / 2,
                    width, height);
        });
    }

    public Expression rotate(double radians) {
        return new MediaFn(this, "rotate", value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            return img == null ? null : img.rotate(radians);
        });
    }

    public Expression grayscale() {
        return new MediaFn(this, "grayscale", value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            if (img == null) return null;
            return img.toGrayscale();
        });
    }

    public Expression width() {
        return new MediaFn(this, "width", value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            return img == null ? null : (long) img.getWidth();
        });
    }

    public Expression height() {
        return new MediaFn(this, "height", value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            return img == null ? null : (long) img.getHeight();
        });
    }

    public Expression toTensor() {
        return new MediaFn(this, "tensor", value -> {
            if (value == null) return null;
            ImageData img = asImage(value);
            return img == null ? null : MediaBridge.imageToTensor(img);
        });
    }

    private static ImageData asImage(Object v) {
        if (v instanceof ImageData) return (ImageData) v;
        if (v instanceof byte[]) return new ImageData((byte[]) v);
        if (v instanceof DataValue) {
            try {
                Object payload = ((DataValue) v).toArrowCompatible();
                if (payload instanceof byte[]) return new ImageData((byte[]) payload);
                return null;
            } catch (Exception e) { return null; }
        }
        return null;
    }

    static byte[] bytesOf(Object v) {
        if (v == null) return null;
        if (v instanceof byte[]) return (byte[]) v;
        if (v instanceof DataValue) {
            try {
                Object payload = ((DataValue) v).toArrowCompatible();
                if (payload instanceof byte[]) return (byte[]) payload;
                return null;
            } catch (Exception e) { return null; }
        }
        return null;
    }
}
