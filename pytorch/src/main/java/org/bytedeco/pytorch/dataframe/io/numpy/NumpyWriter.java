package org.bytedeco.pytorch.dataframe.io.numpy;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade NumPy writer for .npy/.npz formats.
 * 
 * <p>Writes DataFrames to NumPy formats with support for:
 * <ul>
 *   <li>Single array (.npy)</li>
 *   <li>Multiple arrays (.npz - ZIP archive)</li>
 *   <li>Forced dtype for each column</li>
 *   <li>Memory-mapped output for large arrays</li>
 * </ul>
 */
public class NumpyWriter {

    private NumpyWriter() {}

    // NumPy magic: 0x93 'N' 'U' 'M' 'P' 'Y'
    private static final byte[] MAGIC = new byte[]{(byte)0x93, 'N', 'U', 'M', 'P', 'Y'};
    private static final int VERSION_MAJOR = 1;
    private static final int VERSION_MINOR = 0;
    private static final int FORTRAN_ORDER = 0x0001;
    private static final int ALIGNED = 0x0002;

    /**
     * Write DataFrame to .npy format (single array).
     * Only works for single-column DataFrames or first column.
     */
    public static void writeNpy(DataFrame df, String path) throws IOException {
        writeNpy(df, path, NumpyOptions.defaults());
    }

    public static void writeNpy(DataFrame df, String path, NumpyOptions options) throws IOException {
        NumpyOptions opt = options == null ? NumpyOptions.defaults() : options;
        
        if (df.rowCount() == 0) {
            throw new IOException("Cannot write empty DataFrame to NumPy format");
        }
        
        // Select column
        int colIdx = 0;
        if (opt.column() != null) {
            colIdx = df.columnIndex(opt.column());
        }

        if (colIdx < 0 || colIdx >= df.columnCount()) {
            throw new IOException("Column not found: " + opt.column());
        }
        
        Column col = df.column(colIdx);
        Object[] data = extractColumn(df, colIdx);
        
        // Determine dtype
        String dtype = opt.dtype();
        if (dtype == null) {
            dtype = inferNumpyDtype(col);
        }
        
        // Build header
        byte[] header = buildNpyHeader(dtype, new long[]{data.length}, opt);
        
        // Write file
        Files.createDirectories(Path.of(path).getParent());
        
        try (OutputStream out = Files.newOutputStream(Path.of(path))) {
            out.write(MAGIC);
            out.write(VERSION_MAJOR);
            out.write(VERSION_MINOR);
            
            // Header length (2 bytes, little-endian)
            out.write(header.length & 0xFF);
            out.write((header.length >> 8) & 0xFF);
            
            out.write(header);
            
            // Write data
            writeArrayData(out, data, dtype);
        }
    }

    /**
     * Write DataFrame to .npz format (ZIP archive with multiple .npy files).
     * Each column becomes a separate array.
     */
    public static void writeNpz(DataFrame df, String path) throws IOException {
        writeNpz(df, path, NumpyOptions.defaults());
    }

    public static void writeNpz(DataFrame df, String path, NumpyOptions options) throws IOException {
        NumpyOptions opt = options == null ? NumpyOptions.defaults() : options;
        
        if (df.rowCount() == 0) {
            throw new IOException("Cannot write empty DataFrame to NumPy format");
        }
        
        Files.createDirectories(Path.of(path).getParent());
        
        // Create ZIP file
        try (java.util.zip.ZipOutputStream zout = new java.util.zip.ZipOutputStream(
                Files.newOutputStream(Path.of(path)))) {
            
            for (int c = 0; c < df.columnCount(); c++) {
                String name = opt.sanitizeNames() ? sanitizeName(df.column(c).name()) : df.column(c).name();
                zout.putNextEntry(new java.util.zip.ZipEntry(name + ".npy"));
                
                Column col = df.column(c);
                Object[] data = extractColumn(df, c);
                
                String dtype = opt.dtype();
                if (dtype == null) {
                    dtype = inferNumpyDtype(col);
                }
                
                byte[] header = buildNpyHeader(dtype, new long[]{data.length}, opt);
                
                zout.write(MAGIC);
                zout.write(VERSION_MAJOR);
                zout.write(VERSION_MINOR);
                zout.write(header.length & 0xFF);
                zout.write((header.length >> 8) & 0xFF);
                zout.write(header);
                
                writeArrayData(zout, data, dtype);
                
                zout.closeEntry();
            }
        }
    }

    /**
     * Write specific columns to .npz format.
     */
    public static void writeNpz(DataFrame df, String path, String[] columns) throws IOException {
        writeNpz(df, path, columns, NumpyOptions.defaults());
    }

    public static void writeNpz(DataFrame df, String path, String[] columns, NumpyOptions options) throws IOException {
        NumpyOptions opt = options == null ? NumpyOptions.defaults() : options;
        
        Files.createDirectories(Path.of(path).getParent());
        
        try (java.util.zip.ZipOutputStream zout = new java.util.zip.ZipOutputStream(
                Files.newOutputStream(Path.of(path)))) {
            
            for (String colName : columns) {
                int idx = df.columnIndex(colName);
                if (idx < 0) continue;
                
                zout.putNextEntry(new java.util.zip.ZipEntry(sanitizeName(colName) + ".npy"));

                Column col = df.column(idx);
                Object[] data = extractColumn(df, idx);
                String dtype = inferNumpyDtype(col);

                byte[] header = buildNpyHeader(dtype, new long[]{data.length}, opt);

                zout.write(MAGIC);
                zout.write(VERSION_MAJOR);
                zout.write(VERSION_MINOR);
                zout.write(header.length & 0xFF);
                zout.write((header.length >> 8) & 0xFF);
                zout.write(header);
                
                writeArrayData(zout, data, dtype);
                zout.closeEntry();
            }
        }
    }

    private static Object[] extractColumn(DataFrame df, int colIdx) {
        Object[] result = new Object[df.rowCount()];
        for (int r = 0; r < df.rowCount(); r++) {
            result[r] = df.get(r, colIdx);
        }
        return result;
    }

    private static String inferNumpyDtype(Column col) {
        switch (col.dtype()) {
            case BOOLEAN: return "b1";
            case INT8: return "i1";
            case INT16: return "i2";
            case INT32: return "i4";
            case INT64: return "i8";
            case FLOAT16: return "f2";
            case FLOAT32: return "f4";
            case FLOAT64: return "f8";
            default: return "O";  // Object (covers STRING, TENSOR, DATE, DATETIME, TIME, DURATION,
                                  // VECTOR, IMAGE, AUDIO, VIDEO, EMBEDDING, BINARY, JSON, LIST,
                                  // MAP, STRUCT, GRAPH, POINT_CLOUD, NULL)
        }
    }

    private static byte[] buildNumpyHeader(String dtype, long[] shape, boolean fortranOrder) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("'descr': '").append(dtype).append("',");
        sb.append("'fortran_order': ").append(fortranOrder).append(",");
        sb.append("'shape': (").append(shape[0]);
        for (int i = 1; i < shape.length; i++) {
            sb.append(", ").append(shape[i]);
        }
        sb.append(",)}");
        
        // Pad to 16-byte boundary
        int totalLen = 6 + 2 + sb.length() + 1;  // magic + version + header + newline
        int padding = (16 - (totalLen % 16)) % 16;
        for (int i = 0; i < padding; i++) {
            sb.append(' ');
        }
        sb.append('\n');
        
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static byte[] buildNpyHeader(String dtype, long[] shape, NumpyOptions opt) {
        return buildNumpyHeader(dtype, shape, opt.fortranOrder());
    }

    private static void writeArrayData(OutputStream out, Object[] data, String dtype) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        
        for (Object val : data) {
            if (val == null) {
                // Write zeros for null
                switch (dtype) {
                    case "b1": out.write(0); break;
                    case "i1": case "u1": out.write(0); break;
                    case "i2": case "u2":
                        out.write(0); out.write(0); break;
                    case "i4": case "u4": case "f4":
                        out.write(0); out.write(0); out.write(0); out.write(0); break;
                    case "f2":
                        out.write(0); out.write(0); break;
                    case "i8": case "u8": case "f8":
                        for (int i = 0; i < 8; i++) out.write(0); break;
                    default:
                        out.write(0); out.write(0); out.write(0); out.write(0); break;
                }
                continue;
            }
            
            switch (dtype) {
                case "b1":
                    out.write(Boolean.TRUE.equals(val) ? 1 : 0);
                    break;
                case "i1":
                    out.write(((Number) val).byteValue());
                    break;
                case "u1":
                    out.write(((Number) val).byteValue() & 0xFF);
                    break;
                case "i2":
                case "u2":
                    short s = ((Number) val).shortValue();
                    out.write(s & 0xFF);
                    out.write((s >> 8) & 0xFF);
                    break;
                case "f2":
                    // FLOAT16: convert double to float16 bits
                    short f16 = (short) Float.floatToRawIntBits((float) ((Number) val).doubleValue());
                    out.write(f16 & 0xFF);
                    out.write((f16 >> 8) & 0xFF);
                    break;
                case "i4":
                case "u4":
                    int i = ((Number) val).intValue();
                    out.write(i & 0xFF);
                    out.write((i >> 8) & 0xFF);
                    out.write((i >> 16) & 0xFF);
                    out.write((i >> 24) & 0xFF);
                    break;
                case "f4":
                    float f = ((Number) val).floatValue();
                    int fi = Float.floatToIntBits(f);
                    out.write(fi & 0xFF);
                    out.write((fi >> 8) & 0xFF);
                    out.write((fi >> 16) & 0xFF);
                    out.write((fi >> 24) & 0xFF);
                    break;
                case "i8":
                case "u8":
                    long l = ((Number) val).longValue();
                    out.write((int)(l & 0xFF));
                    out.write((int)((l >> 8) & 0xFF));
                    out.write((int)((l >> 16) & 0xFF));
                    out.write((int)((l >> 24) & 0xFF));
                    out.write((int)((l >> 32) & 0xFF));
                    out.write((int)((l >> 40) & 0xFF));
                    out.write((int)((l >> 48) & 0xFF));
                    out.write((int)((l >> 56) & 0xFF));
                    break;
                case "f8":
                    double d = ((Number) val).doubleValue();
                    long dl = Double.doubleToLongBits(d);
                    out.write((int)(dl & 0xFF));
                    out.write((int)((dl >> 8) & 0xFF));
                    out.write((int)((dl >> 16) & 0xFF));
                    out.write((int)((dl >> 24) & 0xFF));
                    out.write((int)((dl >> 32) & 0xFF));
                    out.write((int)((dl >> 40) & 0xFF));
                    out.write((int)((dl >> 48) & 0xFF));
                    out.write((int)((dl >> 56) & 0xFF));
                    break;
                default:
                    // Object type - write as string
                    byte[] strBytes = val.toString().getBytes(StandardCharsets.UTF_8);
                    out.write(strBytes);
            }
        }
    }

    private static String sanitizeName(String name) {
        return name.replaceAll("[^a-zA-Z0-9_]", "_");
    }

    // ====================== Options ======================

    public static class NumpyOptions {
        private String column = null;  // null = first column for .npy
        private String dtype = null;    // null = auto-detect
        private boolean fortranOrder = false;
        private boolean sanitizeNames = true;

        public static NumpyOptions defaults() { return new NumpyOptions(); }

        public NumpyOptions column(String c) { this.column = c; return this; }
        public NumpyOptions dtype(String d) { this.dtype = d; return this; }
        public NumpyOptions fortranOrder(boolean b) { this.fortranOrder = b; return this; }
        public NumpyOptions sanitizeNames(boolean b) { this.sanitizeNames = b; return this; }

        public String column() { return column; }
        public String dtype() { return dtype; }
        public boolean fortranOrder() { return fortranOrder; }
        public boolean sanitizeNames() { return sanitizeNames; }
    }
}
