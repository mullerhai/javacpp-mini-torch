package org.bytedeco.pytorch.vision.scrimage;

import java.util.Objects;

/**
 * ImageMetadata: scrimage-aligned ImageMetadata stand-in.
 *
 * <p>Holds optional fields: format, mimeType, density (DPI), gamma (display),
 * iccProfile (raw bytes), orientation, xmp (raw bytes), exif (raw bytes),
 * pixelDensity ratio (pixels per inch / unit).
 *
 * <p>Pure POJO with builder. Image.getMetadata() returns one per Image;
 * saving into any codec updates relevant fields.
 */
public final class ImageMetadata {

    private final String format;
    private final String mimeType;
    private final double density;
    private final double gamma;
    private final byte[] iccProfile;
    private final byte[] xmp;
    private final byte[] exif;
    private final int orientation;
    private final double pixelDensityRatio;

    private ImageMetadata(Builder b) {
        this.format = b.format;
        this.mimeType = b.mimeType;
        this.density = b.density;
        this.gamma = b.gamma;
        this.iccProfile = b.iccProfile;
        this.xmp = b.xmp;
        this.exif = b.exif;
        this.orientation = b.orientation;
        this.pixelDensityRatio = b.pixelDensityRatio;
    }

    public String format() { return format; }
    public String mimeType() { return mimeType; }
    public double density() { return density; }
    public double gamma() { return gamma; }
    public byte[] iccProfile() { return iccProfile; }
    public byte[] xmp() { return xmp; }
    public byte[] exif() { return exif; }
    public int orientation() { return orientation; }
    public double pixelDensityRatio() { return pixelDensityRatio; }

    public Builder toBuilder() {
        return new Builder()
                .format(format).mimeType(mimeType).density(density).gamma(gamma)
                .iccProfile(iccProfile).xmp(xmp).exif(exif)
                .orientation(orientation).pixelDensityRatio(pixelDensityRatio);
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String format;
        private String mimeType;
        private double density = 1.0;
        private double gamma = 0.45455;
        private byte[] iccProfile;
        private byte[] xmp;
        private byte[] exif;
        private int orientation = 1;
        private double pixelDensityRatio = 1.0;

        public Builder format(String v) { Objects.requireNonNull(v); this.format = v; return this; }
        public Builder mimeType(String v) { this.mimeType = v; return this; }
        public Builder density(double v) { this.density = v; return this; }
        public Builder gamma(double v) { this.gamma = v; return this; }
        public Builder iccProfile(byte[] v) { this.iccProfile = v == null ? null : v.clone(); return this; }
        public Builder xmp(byte[] v) { this.xmp = v == null ? null : v.clone(); return this; }
        public Builder exif(byte[] v) { this.exif = v == null ? null : v.clone(); return this; }
        public Builder orientation(int v) { this.orientation = v; return this; }
        public Builder pixelDensityRatio(double v) { this.pixelDensityRatio = v; return this; }
        public ImageMetadata build() { return new ImageMetadata(this); }
    }
}