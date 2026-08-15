package org.bytedeco.pytorch.vision.scrimage.ops;

/**
 * Scale algorithms (scrimage {@code ScaleMethod} stand-in).
 *
 * <p>{@link #value()} mirrors the public enum field used by Pillow for IO
 * compatibility. Each value maps onto a {@link org.bytedeco.pytorch.vision.pillow.enums.Resampling}.
 */
public enum ScaleMethod {
    FastScale(0),
    Bilinear(2),
    Bicubic(3),
    BSpline(4),
    Lanczos3(1),
    Progressive(5);

    private final int value;
    ScaleMethod(int v) { this.value = v; }
    public int value() { return value; }

    public org.bytedeco.pytorch.vision.pillow.enums.Resampling toResampling() {
        return switch (this) {
            case FastScale, Progressive -> org.bytedeco.pytorch.vision.pillow.enums.Resampling.NEAREST;
            case Bilinear -> org.bytedeco.pytorch.vision.pillow.enums.Resampling.BILINEAR;
            case Bicubic -> org.bytedeco.pytorch.vision.pillow.enums.Resampling.BICUBIC;
            case BSpline -> org.bytedeco.pytorch.vision.pillow.enums.Resampling.BICUBIC;
            case Lanczos3 -> org.bytedeco.pytorch.vision.pillow.enums.Resampling.LANCZOS;
        };
    }
}