package org.bytedeco.pytorch.dataframe.io;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.ByteOrder;
import java.nio.channels.*;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade binary (.bin) file reader/writer for DataFrame.
 * 
 * <p>Supports multiple binary formats:</p>
 * <ul>
 *   <li>MicroLens/MMCTR custom format (.bin) - our optimized format</li>
 *   <li>NumPy .npy compatible format</li>
 *   <li>PyTorch tensor binary format</li>
 *   <li>Custom typed array format (BIN float32/int64)</li>
 * </ul>
 * 
 * <p>MicroLens Format:</p>
 * <pre>
 * [int32] num_fields
 * For each field:
 *   [int32] name_len
 *   [bytes] name (UTF-8)
 *   [4 bytes] dtype ("SNET"=float32, "DNET"=float64, "IN64"=int64)
 *   [int32] dim0
 *   [int32] dim1
 *   [data] binary data
 * </pre>
 */
public class BinReader {

    private static final byte[] MAGIC_ML = new byte[]{0x4D, 0x4C, 0x42, 0x49}; // "MLBI"
    private static final byte[] MAGIC_RAW = new byte[]{0x42, 0x49, 0x4E, 0x01}; // "BIN\1"
    private static final byte[] MAGIC_F32 = new byte[]{0x42, 0x49, 0x4E, 0x46}; // "BINF"
    private static final byte[] MAGIC_I64 = new byte[]{0x42, 0x49, 0x4E, 0x49}; // "BINI"
    
    private BinReader() {}

    // ====================== Public API ======================

    /**
     * Read binary file to DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, BinOptions.autoDetect());
    }

    public static DataFrame read(String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.autoDetect() : options;
        
        Path p = Path.of(path);
        byte[] header = new byte[32];
        
        try (FileInputStream fis = new FileInputStream(path)) {
            int read = fis.read(header);
            if (read < 4) {
                throw new IOException("File too small for binary format");
            }
        }
        
        if (opt.format() == BinFormat.AUTO) {
            BinFormat detected = detectFormat(header, path);
            opt = BinOptions.builder()
                .format(detected)
                .byteOrder(opt.byteOrder())
                .columns(opt.columns())
                .build();
        }
        
        return switch (opt.format()) {
            case MICROLENS -> readMicroLensFormat(path, opt);
            case RAW_F32 -> readFloat32(path, opt);
            case RAW_I64 -> readInt64(path, opt);
            case RAW -> readRaw(path, opt);
            case NPY -> readNpyFormat(path, opt);
            case PT -> readPtFormat(path, opt);
            default -> readMicroLensFormat(path, opt);
        };
    }

    public static void write(DataFrame df, String path) throws IOException {
        write(df, path, BinOptions.defaults());
    }

    public static void write(DataFrame df, String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.defaults() : options;
        
        if (df.columnCount() == 0) {
            throw new IOException("Cannot write empty DataFrame");
        }
        
        BinFormat format = opt.format();
        if (format == BinFormat.AUTO) {
            format = BinFormat.MICROLENS;
        }
        
        switch (format) {
            case MICROLENS -> writeMicroLensFormat(df, path, opt);
            case RAW_F32 -> writeFloat32(df, path, opt);
            case RAW_I64 -> writeInt64(df, path, opt);
            case RAW -> writeRaw(df, path, opt);
            case NPY -> writeNpyFormat(df, path, opt);
            case PT -> writePtFormat(df, path, opt);
        }
    }

    // ====================== Schema API ======================

    /**
     * Get schema information for binary file without loading all data.
     */
    public static BinSchema schema(String path) throws IOException {
        return schema(path, BinOptions.autoDetect());
    }

    public static BinSchema schema(String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.autoDetect() : options;
        
        byte[] header = new byte[64];
        try (FileInputStream fis = new FileInputStream(path)) {
            fis.read(header);
        }
        
        if (opt.format() == BinFormat.AUTO) {
            BinFormat detected = detectFormat(header, path);
            opt = BinOptions.builder()
                .format(detected)
                .byteOrder(opt.byteOrder())
                .build();
        }
        
        return switch (opt.format()) {
            case MICROLENS -> inferMicroLensSchema(path, opt);
            case RAW_F32 -> inferFloat32Schema(path, opt);
            case RAW_I64 -> inferInt64Schema(path, opt);
            case NPY -> inferNpySchema(path, opt);
            case PT -> inferPtSchema(path, opt);
            default -> inferMicroLensSchema(path, opt);
        };
    }

    /**
     * Print schema to stdout.
     */
    public static void printSchema(String path) throws IOException {
        BinSchema schema = schema(path);
        System.out.println(schema.toString());
    }

    /**
     * Get schema as DataFrame for preview.
     */
    public static DataFrame schemaAsDataFrame(String path) throws IOException {
        BinSchema s = schema(path);
        DataFrame df = DataFrame.create();
        df.addColumn("#", Column.DType.INT32);
        df.addColumn("field_name", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("rows", Column.DType.INT64);
        df.addColumn("cols", Column.DType.INT64);
        df.addColumn("size_bytes", Column.DType.INT64);

        int idx = 0;
        for (BinSchema.FieldInfo f : s.fields) {
            int ri = df.addEmptyRow();
            df.set(ri, "#", idx++);
            df.set(ri, "field_name", f.name);
            df.set(ri, "dtype", f.dtype);
            df.set(ri, "shape", f.shape);
            df.set(ri, "rows", f.rows);
            df.set(ri, "cols", f.cols);
            df.set(ri, "size_bytes", f.sizeBytes);
        }
        return df;
    }

    // ====================== Schema Classes ======================

    public static class BinSchema {
        public final String format;
        public final long fileSize;
        public final List<FieldInfo> fields = new ArrayList<>();

        public BinSchema(String format, long fileSize) {
            this.format = format;
            this.fileSize = fileSize;
        }

        public static class FieldInfo {
            public String name;
            public String dtype;
            public String shape;
            public long rows;
            public long cols;
            public long sizeBytes;

            public FieldInfo(String name, String dtype, long rows, long cols) {
                this.name = name;
                this.dtype = dtype;
                this.rows = rows;
                this.cols = cols;
                this.shape = "[" + rows + ", " + cols + "]";
                this.sizeBytes = rows * cols * elementSize(dtype);
            }

            private static int elementSize(String dtype) {
                return switch (dtype) {
                    case "float32", "int32" -> 4;
                    case "float64", "int64" -> 8;
                    case "string" -> 1;
                    default -> 8;
                };
            }
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("╔══════════════════════════════════════════════════════════════════════╗\n");
            sb.append(String.format("║ Binary Schema: %-46s ║\n", 
                format != null ? format : "Unknown"));
            sb.append(String.format("║ File size: %-50s ║\n", formatBytes(fileSize)));
            sb.append("╠══════════════════════════════════════════════════════════════════════╣\n");
            sb.append(String.format("║ %-3s │ %-20s │ %-10s │ %-15s ║\n", "#", "field_name", "dtype", "shape"));
            sb.append("╠═════╪══════════════════════╪════════════╪═════════════════╣\n");
            
            for (int i = 0; i < fields.size(); i++) {
                FieldInfo f = fields.get(i);
                sb.append(String.format("║ %3d │ %-20s │ %-10s │ %-15s ║\n",
                    i, 
                    f.name.length() > 20 ? f.name.substring(0, 17) + "..." : f.name,
                    f.dtype,
                    f.shape));
            }
            sb.append("╚═════╧══════════════════════╧════════════╧═════════════════╝\n");
            return sb.toString();
        }

        private static String formatBytes(long bytes) {
            if (bytes < 1024) return bytes + " B";
            if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
            if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
            return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
        }
    }

    // ====================== MicroLens Format ======================

    public static DataFrame readMicroLensFormat(String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.defaults() : options;
        
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            DataFrame df = DataFrame.create();
            
            int numFields = buf.getInt();
            
            for (int f = 0; f < numFields; f++) {
                int nameLen = buf.getInt();
                byte[] nameBytes = new byte[nameLen];
                buf.get(nameBytes);
                String fieldName = new String(nameBytes, StandardCharsets.UTF_8);
                
                byte[] typeMarker = new byte[4];
                buf.get(typeMarker);
                String type = new String(typeMarker, StandardCharsets.US_ASCII);
                
                int dim0 = buf.getInt();
                int dim1 = buf.getInt();
                
                Column.DType dtype = switch (type) {
                    case "SNET" -> Column.DType.FLOAT32;
                    case "DNET" -> Column.DType.FLOAT64;
                    case "IN16" -> Column.DType.INT16;
                    case "IN08" -> Column.DType.INT8;
                    case "IN64" -> Column.DType.INT64;
                    case "FP16" -> Column.DType.FLOAT16;
                    default -> Column.DType.FLOAT64;
                };
                
                int totalElems = dim0 * dim1;
                Column col = new Column(fieldName, dtype);
                
                for (int i = 0; i < totalElems; i++) {
                    switch (dtype) {
                        case FLOAT16 -> col.add((double) Float.intBitsToFloat(buf.getShort() & 0xFFFF));
                        case FLOAT32 -> col.add((double) buf.getFloat());
                        case FLOAT64 -> col.add(buf.getDouble());
                        case INT16 -> col.add((long) buf.getShort());
                        case INT32 -> col.add((long) buf.getInt());
                        case INT64 -> col.add(buf.getLong());
                        case INT8 -> col.add((long) buf.get());
                        default -> col.add(buf.getDouble());
                    }
                }
                
                df.addColumn(col);
            }
            
            return df;
        }
    }

    private static BinSchema inferMicroLensSchema(String path, BinOptions opt) throws IOException {
        long fileSize = Files.size(Path.of(path));
        BinSchema schema = new BinSchema("MicroLens", fileSize);
        
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate(64 * 1024).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            if (buf.remaining() < 4) return schema;
            
            int numFields = buf.getInt();
            
            for (int f = 0; f < numFields; f++) {
                if (buf.remaining() < 8) break;
                
                int nameLen = buf.getInt();
                if (nameLen <= 0 || nameLen > 1024 || buf.remaining() < nameLen + 12) break;
                
                byte[] nameBytes = new byte[nameLen];
                buf.get(nameBytes);
                String fieldName;
                try {
                    fieldName = new String(nameBytes, StandardCharsets.UTF_8);
                } catch (Exception e) {
                    fieldName = "field_" + f;
                }
                
                byte[] typeMarker = new byte[4];
                buf.get(typeMarker);
                String type = new String(typeMarker, StandardCharsets.US_ASCII);
                
                int dim0 = buf.getInt();
                int dim1 = buf.getInt();
                
                String dtype = switch (type) {
                    case "SNET" -> "float32";
                    case "DNET" -> "float64";
                    case "IN64" -> "int64";
                    default -> "unknown";
                };
                
                schema.fields.add(new BinSchema.FieldInfo(fieldName, dtype, dim0, dim1));
                
                // Skip data to next field
                long elemSize = type.equals("SNET") ? 4 : 8;
                long dataSize = (long) dim0 * dim1 * elemSize;
                if (buf.remaining() >= dataSize) {
                    buf.position(buf.position() + (int) dataSize);
                } else {
                    break;
                }
            }
        }
        
        return schema;
    }

    public static void writeMicroLensFormat(DataFrame df, String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.defaults() : options;
        
        try (FileOutputStream fos = new FileOutputStream(path);
             FileChannel ch = fos.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate(8192).order(opt.byteOrder());
            
            int numFields = df.columnCount();
            buf.putInt(numFields);
            
            for (int c = 0; c < numFields; c++) {
                Column col = df.column(c);
                String name = col.name();
                
                byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
                buf.putInt(nameBytes.length);
                buf.put(nameBytes);
                
                String typeMarker = switch (col.dtype()) {
                    case FLOAT32 -> "SNET";
                    case INT64, INT32 -> "IN64";
                    default -> "DNET";
                };
                buf.put(typeMarker.getBytes(StandardCharsets.US_ASCII));
                
                int rows = col.size();
                int cols = estimateCols(col);
                buf.putInt(rows);
                buf.putInt(cols);
                
                buf.flip();
                ch.write(buf);
                buf.clear();
                
                int elemSize = typeMarker.equals("SNET") ? 4 : 8;
                ByteBuffer dataBuf = ByteBuffer.allocate(rows * cols * elemSize).order(opt.byteOrder());
                writeColumnData(col, dataBuf, typeMarker);
                dataBuf.flip();
                ch.write(dataBuf);
            }
        }
    }

    private static int estimateCols(Column col) {
        Object first = col.get(0);
        if (first instanceof float[]) return ((float[]) first).length;
        if (first instanceof double[]) return ((double[]) first).length;
        if (first instanceof long[]) return ((long[]) first).length;
        if (first instanceof int[]) return ((int[]) first).length;
        return 1;
    }

    private static void writeColumnData(Column col, ByteBuffer buf, String type) {
        for (int r = 0; r < col.size(); r++) {
            Object v = col.get(r);
            if (v == null) {
                if ("SNET".equals(type)) buf.putFloat(0f);
                else if ("DNET".equals(type)) buf.putDouble(0.0);
                else buf.putLong(0L);
                continue;
            }
            
            if (v instanceof float[]) {
                for (float f : (float[]) v) buf.putFloat(f);
            } else if (v instanceof double[]) {
                for (double d : (double[]) v) buf.putDouble(d);
            } else if (v instanceof long[]) {
                for (long l : (long[]) v) buf.putLong(l);
            } else if (v instanceof int[]) {
                for (int i : (int[]) v) buf.putLong(i);
            } else if (v instanceof Number) {
                if ("SNET".equals(type)) buf.putFloat(((Number) v).floatValue());
                else if ("DNET".equals(type)) buf.putDouble(((Number) v).doubleValue());
                else buf.putLong(((Number) v).longValue());
            } else {
                if ("SNET".equals(type)) buf.putFloat(Float.parseFloat(String.valueOf(v)));
                else if ("DNET".equals(type)) buf.putDouble(Double.parseDouble(String.valueOf(v)));
                else buf.putLong(Long.parseLong(String.valueOf(v)));
            }
        }
    }

    // ====================== Float32/Int64 Formats ======================

    public static DataFrame readFloat32(String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.defaults() : options;
        
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            byte[] magic = new byte[4];
            buf.get(magic);
            
            if (Arrays.equals(magic, MAGIC_F32)) {
                int rows = buf.getInt();
                int cols = buf.getInt();
                int namesLen = buf.getInt();
                
                byte[] namesBytes = new byte[namesLen];
                buf.get(namesBytes);
                String namesStr = new String(namesBytes, StandardCharsets.UTF_8);
                List<String> colNames = Arrays.asList(namesStr.split(","));
                
                int floatCount = rows * cols;
                float[] data = new float[floatCount];
                buf.asFloatBuffer().get(data);
                
                DataFrame df = DataFrame.create();
                
                for (int c = 0; c < cols; c++) {
                    String name = c < colNames.size() ? colNames.get(c) : "col_" + c;
                    Column col = new Column(name, Column.DType.FLOAT64);
                    for (int r = 0; r < rows; r++) {
                        col.add((double) data[r * cols + c]);
                    }
                    df.addColumn(col);
                }
                return df;
            } else {
                throw new IOException("Not a float32 binary file");
            }
        }
    }

    private static BinSchema inferFloat32Schema(String path, BinOptions opt) throws IOException {
        long fileSize = Files.size(Path.of(path));
        BinSchema schema = new BinSchema("BINF (float32)", fileSize);
        
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate(1024).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            byte[] magic = new byte[4];
            buf.get(magic);
            
            if (Arrays.equals(magic, MAGIC_F32)) {
                int rows = buf.getInt();
                int cols = buf.getInt();
                int namesLen = buf.getInt();
                
                byte[] namesBytes = new byte[namesLen];
                buf.get(namesBytes);
                String namesStr = new String(namesBytes, StandardCharsets.UTF_8);
                List<String> colNames = Arrays.asList(namesStr.split(","));
                
                for (int c = 0; c < cols; c++) {
                    String name = c < colNames.size() ? colNames.get(c) : "col_" + c;
                    schema.fields.add(new BinSchema.FieldInfo(name, "float32", rows, 1));
                }
            }
        }
        
        return schema;
    }

    public static void writeFloat32(DataFrame df, String path, BinOptions opt) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path);
             FileChannel ch = fos.getChannel()) {
            
            ByteBuffer header = ByteBuffer.allocate(1024).order(opt.byteOrder());
            
            header.put(MAGIC_F32);
            
            int rows = df.rowCount();
            int cols = df.columnCount();
            
            StringBuilder names = new StringBuilder();
            for (int c = 0; c < cols; c++) {
                if (c > 0) names.append(",");
                names.append(df.column(c).name());
            }
            byte[] namesBytes = names.toString().getBytes(StandardCharsets.UTF_8);
            
            header.putInt(rows);
            header.putInt(cols);
            header.putInt(namesBytes.length);
            header.put(namesBytes);
            header.flip();
            ch.write(header);
            
            for (int c = 0; c < cols; c++) {
                Column col = df.column(c);
                float[] data = new float[rows];
                for (int r = 0; r < rows; r++) {
                    Object v = col.get(r);
                    data[r] = v == null ? 0f : ((Number) v).floatValue();
                }
                ByteBuffer dataBuf = ByteBuffer.allocate(rows * 4).order(opt.byteOrder());
                dataBuf.asFloatBuffer().put(data);
                ch.write(dataBuf);
            }
        }
    }

    public static DataFrame readInt64(String path, BinOptions options) throws IOException {
        BinOptions opt = options == null ? BinOptions.defaults() : options;
        
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate((int) ch.size()).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            byte[] magic = new byte[4];
            buf.get(magic);
            
            int rows = buf.getInt();
            int cols = buf.getInt();
            int namesLen = buf.getInt();
            
            byte[] namesBytes = new byte[namesLen];
            buf.get(namesBytes);
            String namesStr = new String(namesBytes, StandardCharsets.UTF_8);
            List<String> colNames = Arrays.asList(namesStr.split(","));
            
            long[] data = new long[rows * cols];
            buf.asLongBuffer().get(data);
            
            DataFrame df = DataFrame.create();
            
            for (int c = 0; c < cols; c++) {
                String name = c < colNames.size() ? colNames.get(c) : "col_" + c;
                Column col = new Column(name, Column.DType.INT64);
                for (int r = 0; r < rows; r++) {
                    col.add(data[r * cols + c]);
                }
                df.addColumn(col);
            }
            return df;
        }
    }

    private static BinSchema inferInt64Schema(String path, BinOptions opt) throws IOException {
        long fileSize = Files.size(Path.of(path));
        BinSchema schema = new BinSchema("BINI (int64)", fileSize);
        
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            ByteBuffer buf = ByteBuffer.allocate(1024).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            byte[] magic = new byte[4];
            buf.get(magic);
            
            if (Arrays.equals(magic, MAGIC_I64)) {
                int rows = buf.getInt();
                int cols = buf.getInt();
                int namesLen = buf.getInt();
                
                byte[] namesBytes = new byte[namesLen];
                buf.get(namesBytes);
                String namesStr = new String(namesBytes, StandardCharsets.UTF_8);
                List<String> colNames = Arrays.asList(namesStr.split(","));
                
                for (int c = 0; c < cols; c++) {
                    String name = c < colNames.size() ? colNames.get(c) : "col_" + c;
                    schema.fields.add(new BinSchema.FieldInfo(name, "int64", rows, 1));
                }
            }
        }
        
        return schema;
    }

    public static void writeInt64(DataFrame df, String path, BinOptions opt) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path);
             FileChannel ch = fos.getChannel()) {
            
            ByteBuffer header = ByteBuffer.allocate(1024).order(opt.byteOrder());
            
            header.put(MAGIC_I64);
            
            int rows = df.rowCount();
            int cols = df.columnCount();
            
            StringBuilder names = new StringBuilder();
            for (int c = 0; c < cols; c++) {
                if (c > 0) names.append(",");
                names.append(df.column(c).name());
            }
            byte[] namesBytes = names.toString().getBytes(StandardCharsets.UTF_8);
            
            header.putInt(rows);
            header.putInt(cols);
            header.putInt(namesBytes.length);
            header.put(namesBytes);
            header.flip();
            ch.write(header);
            
            for (int c = 0; c < cols; c++) {
                Column col = df.column(c);
                long[] data = new long[rows];
                for (int r = 0; r < rows; r++) {
                    Object v = col.get(r);
                    data[r] = v == null ? 0L : ((Number) v).longValue();
                }
                ByteBuffer dataBuf = ByteBuffer.allocate(rows * 8).order(opt.byteOrder());
                dataBuf.asLongBuffer().put(data);
                ch.write(dataBuf);
            }
        }
    }

    // ====================== Format Detection ======================

    private static BinFormat detectFormat(byte[] header, String path) {
        if (header.length < 4) return BinFormat.MICROLENS;
        
        // Check NPY
        if (header[0] == (byte) 0x93 && header[1] == 'N' && 
            header[2] == 'U' && header[3] == 'M') {
            return BinFormat.NPY;
        }
        
        // Check ZIP (PT files)
        if (header[0] == 0x50 && header[1] == 0x4B) {
            return BinFormat.PT;
        }
        
        // Check our custom formats
        if (Arrays.equals(Arrays.copyOf(header, 4), MAGIC_F32)) {
            return BinFormat.RAW_F32;
        }
        if (Arrays.equals(Arrays.copyOf(header, 4), MAGIC_I64)) {
            return BinFormat.RAW_I64;
        }
        if (Arrays.equals(Arrays.copyOf(header, 4), MAGIC_RAW)) {
            return BinFormat.RAW;
        }
        
        // Check MicroLens format
        if (isLikelyMicroLensFormat(header)) {
            return BinFormat.MICROLENS;
        }
        
        return BinFormat.MICROLENS;
    }

    private static boolean isLikelyMicroLensFormat(byte[] header) {
        ByteBuffer buf = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
        int numFields = buf.getInt();
        
        // Valid field counts are typically small numbers
        if (numFields < 0 || numFields > 1000) return false;
        
        return true;
    }

    // ====================== Raw Format ======================

    private static DataFrame readRaw(String path, BinOptions opt) throws IOException {
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel ch = fis.getChannel()) {
            
            long size = ch.size();
            ByteBuffer buf = ByteBuffer.allocate((int) size).order(opt.byteOrder());
            ch.read(buf);
            buf.flip();
            
            DataFrame df = DataFrame.create();
            
            int elemSize = opt.elemSize() > 0 ? opt.elemSize() : 8;
            int count = (int) (size / elemSize);
            
            Column col = new Column("data", 
                elemSize == 4 ? Column.DType.FLOAT32 : Column.DType.FLOAT64);
            
            if (elemSize == 8) {
                while (buf.hasRemaining()) {
                    col.add(buf.getDouble());
                }
            } else {
                while (buf.hasRemaining()) {
                    col.add((double) buf.getFloat());
                }
            }
            
            df.addColumn(col);
            return df;
        }
    }

    private static void writeRaw(DataFrame df, String path, BinOptions opt) throws IOException {
        writeFloat32(df, path, opt);
    }

    private static BinSchema inferNpySchema(String path, BinOptions opt) throws IOException {
        try {
            org.bytedeco.pytorch.data.numpy.NDArray arr = 
                org.bytedeco.pytorch.data.numpy.NP.load(path);
            long fileSize = Files.size(Path.of(path));
            BinSchema schema = new BinSchema("NumPy (.npy)", fileSize);
            
            String shape = arr.shape.length > 0 ? Arrays.toString(arr.shape) : "[]";
            long rows = arr.shape.length > 0 ? arr.shape[0] : 1;
            long cols = arr.shape.length > 1 ? arr.shape[1] : 1;
            
            schema.fields.add(new BinSchema.FieldInfo("data", arr.dtype.toString(), rows, cols));
            return schema;
        } catch (Exception e) {
            return new BinSchema("NumPy (.npy)", Files.size(Path.of(path)));
        }
    }

    private static BinSchema inferPtSchema(String path, BinOptions opt) throws IOException {
        try {
            Map<String, org.bytedeco.pytorch.data.pt.PT.TensorData> tensors = 
                org.bytedeco.pytorch.data.pt.PT.load(new File(path));
            long fileSize = Files.size(Path.of(path));
            BinSchema schema = new BinSchema("PyTorch (.pt)", fileSize);
            
            for (Map.Entry<String, org.bytedeco.pytorch.data.pt.PT.TensorData> e : tensors.entrySet()) {
                org.bytedeco.pytorch.data.pt.PT.TensorData td = e.getValue();
                long rows = td.shape.length > 0 ? td.shape[0] : 1;
                long cols = td.shape.length > 1 ? td.shape[1] : 1;
                schema.fields.add(new BinSchema.FieldInfo(e.getKey(), td.dtype.name(), rows, cols));
            }
            return schema;
        } catch (Exception e) {
            return new BinSchema("PyTorch (.pt)", Files.size(Path.of(path)));
        }
    }

    // ====================== NPY/PT Wrappers ======================

    private static DataFrame readNpyFormat(String path, BinOptions opt) throws IOException {
        try {
            return DataFrame.readNpy(path);
        } catch (Exception e) {
            throw new IOException("Failed to read NPY: " + e.getMessage(), e);
        }
    }

    private static void writeNpyFormat(DataFrame df, String path, BinOptions opt) throws IOException {
        try {
            df.toNumpy(path);
        } catch (Exception e) {
            throw new IOException("Failed to write NPY: " + e.getMessage(), e);
        }
    }

    private static DataFrame readPtFormat(String path, BinOptions opt) throws IOException {
        try {
            return DataFrame.readPT(path);
        } catch (Exception e) {
            throw new IOException("Failed to read PT: " + e.getMessage(), e);
        }
    }

    private static void writePtFormat(DataFrame df, String path, BinOptions opt) throws IOException {
        try {
            df.toPT(path);
        } catch (Exception e) {
            throw new IOException("Failed to write PT: " + e.getMessage(), e);
        }
    }

    // ====================== Options ======================

    public enum BinFormat {
        AUTO,
        MICROLENS,
        RAW,
        RAW_F32,
        RAW_I64,
        NPY,
        PT
    }

    public static class BinOptions {
        private BinFormat format = BinFormat.AUTO;
        private ByteOrder byteOrder = ByteOrder.LITTLE_ENDIAN;
        private int elemSize = 0;
        private List<String> columns = new ArrayList<>();
        
        public static BinOptions defaults() { return new BinOptions(); }
        
        public static BinOptions autoDetect() {
            return new BinOptions().format(BinFormat.AUTO);
        }
        
        public static BinOptions microlens() {
            return new BinOptions().format(BinFormat.MICROLENS);
        }
        
        public static Builder builder() { return new Builder(); }
        
        public BinOptions format(BinFormat f) { this.format = f; return this; }
        public BinOptions byteOrder(ByteOrder o) { this.byteOrder = o; return this; }
        public BinOptions elemSize(int s) { this.elemSize = s; return this; }
        public BinOptions columns(List<String> cols) { this.columns = cols; return this; }
        
        public BinFormat format() { return format; }
        public ByteOrder byteOrder() { return byteOrder; }
        public int elemSize() { return elemSize; }
        public List<String> columns() { return columns; }
        
        public static class Builder {
            private BinOptions opt = new BinOptions();
            
            public Builder format(BinFormat f) { opt.format = f; return this; }
            public Builder byteOrder(ByteOrder o) { opt.byteOrder = o; return this; }
            public Builder elemSize(int s) { opt.elemSize = s; return this; }
            public Builder columns(List<String> cols) { opt.columns = cols; return this; }
            
            public BinOptions build() { return opt; }
        }
    }
}
