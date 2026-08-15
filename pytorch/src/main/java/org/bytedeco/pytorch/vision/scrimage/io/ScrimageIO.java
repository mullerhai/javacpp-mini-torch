package org.bytedeco.pytorch.vision.scrimage.io;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;
import org.bytedeco.pytorch.vision.scrimage.ImmutableImage;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.FileImageOutputStream;
import javax.imageio.stream.MemoryCacheImageOutputStream;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-format IO facade built on top of {@link Image} and the standard
 * javax.imageio codecs.
 *
 * <p>Format strings mirror Pillow: {@code "PNG", "JPEG", "GIF", "BMP", "TIFF",
 * "WBMP", "PNM"} (last two via javax.imageio). Format detection: header
 * sniffing for PNG/JPEG/GIF/BMP/TIFF; otherwise fallback to filename
 * extension.
 */
public final class ScrimageIO {

    private ScrimageIO() {}

    private static final Map<String, String> EXT_TO_FORMAT = new HashMap<>();
    static {
        EXT_TO_FORMAT.put("png", "PNG");
        EXT_TO_FORMAT.put("jpg", "JPEG");
        EXT_TO_FORMAT.put("jpeg", "JPEG");
        EXT_TO_FORMAT.put("gif", "GIF");
        EXT_TO_FORMAT.put("bmp", "BMP");
        EXT_TO_FORMAT.put("tif", "TIFF");
        EXT_TO_FORMAT.put("tiff", "TIFF");
        EXT_TO_FORMAT.put("wbmp", "WBMP");
        EXT_TO_FORMAT.put("pnm", "PNM");
        EXT_TO_FORMAT.put("ppm", "PNM");
        EXT_TO_FORMAT.put("pgm", "PNM");
    }

    /** Detect format from magic bytes. */
    public static String detectFormat(byte[] head) {
        Objects.requireNonNull(head, "head");
        if (head.length >= 8
                && (head[0] & 0xff) == 0x89 && head[1] == 'P' && head[2] == 'N' && head[3] == 'G') {
            return "PNG";
        }
        if (head.length >= 3
                && (head[0] & 0xff) == 0xff && (head[1] & 0xff) == 0xd8 && (head[2] & 0xff) == 0xff) {
            return "JPEG";
        }
        if (head.length >= 6
                && head[0] == 'G' && head[1] == 'I' && head[2] == 'F' && head[3] == '8') {
            return "GIF";
        }
        if (head.length >= 2
                && head[0] == 'B' && head[1] == 'M') {
            return "BMP";
        }
        if (head.length >= 4
                && ((head[0] == 'I' && head[1] == 'I' && head[2] == 0x2a && head[3] == 0x00)
                || (head[0] == 'M' && head[1] == 'M' && head[2] == 0x00 && head[3] == 0x2a))) {
            return "TIFF";
        }
        if (head.length >= 4
                && head[0] == 'R' && head[1] == 'I' && head[2] == 'F' && head[3] == 'F') {
            return "WEBP"; // RIFF WEBP
        }
        return null;
    }

    public static String detectFormat(InputStream in) throws IOException {
        byte[] head = new byte[12];
        int n = 0;
        while (n < head.length) {
            int r = in.read(head, n, head.length - n);
            if (r < 0) break;
            n += r;
        }
        if (n == 0) return null;
        byte[] used = new byte[n];
        System.arraycopy(head, 0, used, 0, n);
        return detectFormat(used);
    }

    public static String detectFormat(Path path) throws IOException {
        return detectFormat(path.toFile());
    }

    public static String detectFormat(File file) throws IOException {
        try (java.io.InputStream in = new java.io.FileInputStream(file)) {
            return detectFormat(in);
        }
    }

    /** Auto-detect format and read. */
    public static ImmutableImage read(Path path) throws IOException {
        Image img = readAsImage(path);
        return new ImmutableImage(img);
    }

    public static ImmutableImage read(byte[] data) throws IOException {
        Image img = readAsImage(data);
        return new ImmutableImage(img);
    }

    public static ImmutableImage read(InputStream in) throws IOException {
        Image img = readAsImage(in);
        return new ImmutableImage(img);
    }

    /** Read with format detection by extension when magic-byte detection fails. */
    public static Image readAsImage(Path path) throws IOException {
        String fmt = detectFormat(path.toFile());
        if (fmt == null) {
            String ext = ext(path.toString());
            fmt = EXT_TO_FORMAT.get(ext);
        }
        if (fmt == null) {
            throw new IOException("unknown image format: " + path);
        }
        BufferedImage bi = javax.imageio.ImageIO.read(path.toFile());
        if (bi == null) throw new IOException("decode failed: " + path);
        return Image.fromBufferedImage(bi);
    }

    public static Image readAsImage(byte[] data) throws IOException {
        String fmt = detectFormat(data);
        if (fmt == null) throw new IOException("unknown image format from bytes");
        return readAsImage(new ByteArrayInputStream(data));
    }

    public static Image readAsImage(InputStream in) throws IOException {
        BufferedImage bi = javax.imageio.ImageIO.read(in);
        if (bi == null) throw new IOException("decode failed");
        return Image.fromBufferedImage(bi);
    }

    /** Write with auto-detected format by extension. */
    public static void write(ImmutableImage img, Path path) throws IOException {
        Objects.requireNonNull(img, "img");
        String ext = ext(path.toString());
        String fmt = EXT_TO_FORMAT.get(ext);
        if (fmt == null) throw new IOException("unknown format for extension: " + ext);
        write(img, path, fmt, new HashMap<>());
    }

    public static void write(ImmutableImage img, Path path, String format) throws IOException {
        write(img, path, format, new HashMap<>());
    }

    /** {@code options} may contain quality (0..1), compression (0..1), interlaced (PNG), dpi (double). */
    public static void write(ImmutableImage img, Path path, String format, Map<String, Object> options) throws IOException {
        Objects.requireNonNull(img, "img");
        Objects.requireNonNull(format, "format");
        try (FileImageOutputStream fos = new FileImageOutputStream(path.toFile())) {
            writeToStream(img, fos, format, options);
        }
    }

    public static byte[] write(ImmutableImage img, String format, Map<String, Object> options) throws IOException {
        Objects.requireNonNull(img, "img");
        Objects.requireNonNull(format, "format");
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        writeToStream(img, new MemoryCacheImageOutputStream(bos), format, options);
        return bos.toByteArray();
    }

    public static byte[] write(ImmutableImage img, String format) throws IOException {
        return write(img, format, new HashMap<>());
    }

    private static void writeToStream(ImmutableImage img, javax.imageio.stream.ImageOutputStream out, String format, Map<String, Object> options) throws IOException {
        BufferedImage bi = img.image().toBufferedImage();
        Iterator<ImageWriter> writers = javax.imageio.ImageIO.getImageWritersByFormatName(format);
        if (!writers.hasNext()) throw new IOException("no writer for format " + format);
        ImageWriter w = writers.next();
        try {
            w.setOutput(out);
            ImageWriteParam param = w.getDefaultWriteParam();
            if ("JPEG".equalsIgnoreCase(format) || "TIFF".equalsIgnoreCase(format)) {
                if (param.canWriteCompressed()) {
                    param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                    Object q = options == null ? null : options.get("quality");
                    float quality = q instanceof Number ? ((Number) q).floatValue() : 0.92f;
                    param.setCompressionQuality(Math.max(0f, Math.min(1f, quality)));
                }
            } else if ("PNG".equalsIgnoreCase(format) && param.canWriteCompressed()) {
                param.setCompressionMode(ImageWriteParam.MODE_DEFAULT);
            }
            w.write(null, new IIOImage(bi, null, null), param);
        } finally {
            w.dispose();
        }
    }

    /** Convert RGB to indexed mode P (median-cut quantization over a fixed palette). */
    public static ImmutableImage toPalette(ImmutableImage src, int[] palette) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(palette, "palette");
        if (palette.length % 3 != 0) throw new IllegalArgumentException("palette not multiple of 3");
        int n = palette.length / 3;
        Image rgb = src.image().convert("RGB");
        Image out = Image.fromBuffer(new ImagingBuffer("P", rgb.width(), rgb.height()));
        int[] rp = rgb.getdata(), op = out.getdata();
        for (int i = 0; i < rp.length; i += 3) {
            int r = rp[i] & 0xff, g = rp[i + 1] & 0xff, b = rp[i + 2] & 0xff;
            int best = 0, bestD = Integer.MAX_VALUE;
            for (int k = 0; k < n; k++) {
                int kr = palette[k * 3], kg = palette[k * 3 + 1], kb = palette[k * 3 + 2];
                int dr = r - kr, dg = g - kg, db = b - kb;
                int d = dr * dr + dg * dg + db * db;
                if (d < bestD) { bestD = d; best = k; }
            }
            op[i / 3] = best;
        }
        out.putdata(op);
        byte[] pal = new byte[n * 3];
        for (int i = 0; i < pal.length; i++) pal[i] = (byte) palette[i];
        out.putpalette(pal);
        return new ImmutableImage(out);
    }

    private static String ext(String name) {
        int dot = name.lastIndexOf('.');
        return dot < 0 ? "" : name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }
}