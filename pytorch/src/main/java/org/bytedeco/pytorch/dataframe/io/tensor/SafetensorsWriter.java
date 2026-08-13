package org.bytedeco.pytorch.dataframe.io.tensor;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade SafeTensors writer for ML model weights.
 * 
 * <p>Writes DataFrames/arrays to SafeTensors format with:
 * <ul>
 *   <li>Header JSON with tensor metadata</li>
 *   <li>Aligned tensor data</li>
 *   <li>Custom metadata support</li>
 * </ul>
 */
public class SafetensorsWriter {

    private SafetensorsWriter() {}

    // Header size is 8 bytes (uint64 BE)
    private static final int HEADER_SIZE_BYTES = 8;
    private static final int ALIGNMENT = 8;

    /**
     * Write DataFrame to SafeTensors format.
     * For DataFrames, each column becomes a tensor.
     */
    public static void write(DataFrame df, String path) throws IOException {
        write(df, path, SafetensorsOptions.defaults());
    }

    public static void write(DataFrame df, String path, SafetensorsOptions options) throws IOException {
        SafetensorsOptions opt = options == null ? SafetensorsOptions.defaults() : options;
        
        Files.createDirectories(Path.of(path).getParent());
        
        // Build header JSON
        Map<String, TensorInfo> tensors = new LinkedHashMap<>();
        
        long currentOffset = 0;
        
        // Calculate offsets
        for (int c = 0; c < df.columnCount(); c++) {
            Column col = df.column(c);
            String name = opt.sanitizeNames() ? sanitizeName(col.name()) : col.name();
            
            long size = estimateTensorSize(df, c);
            long alignedSize = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
            
            TensorInfo info = new TensorInfo();
            info.shape = new long[]{df.rowCount()};
            info.dtype = mapDtype(col.dtype());
            
            if (opt.includeMetadata()) {
                info.metadata = new HashMap<>();
                info.metadata.put("column_index", c);
            }
            
            info.dataOffset = currentOffset;
            info.size = size;
            
            tensors.put(name, info);
            currentOffset += alignedSize;
        }
        
        // Build header JSON
        StringBuilder headerJson = new StringBuilder();
        headerJson.append("{");
        
        if (opt.includeMetadata()) {
            headerJson.append("\"__metadata__\":{");
            Map<String, Object> meta = new LinkedHashMap<>();
            meta.put("format", "safetensors");
            meta.put("version", "1.0");
            meta.put("row_count", df.rowCount());
            meta.put("column_count", df.columnCount());
            
            boolean firstMeta = true;
            for (Map.Entry<String, Object> e : meta.entrySet()) {
                if (!firstMeta) headerJson.append(",");
                headerJson.append("\"").append(e.getKey()).append("\":");
                appendJsonValue(headerJson, e.getValue());
                firstMeta = false;
            }
            headerJson.append("}");
            
            if (!tensors.isEmpty()) {
                headerJson.append(",");
            }
        }
        
        boolean first = true;
        for (Map.Entry<String, TensorInfo> e : tensors.entrySet()) {
            if (!first) headerJson.append(",");
            headerJson.append("\"").append(e.getKey()).append("\":");
            headerJson.append("{");
            headerJson.append("\"shape\":[");
            for (int i = 0; i < e.getValue().shape.length; i++) {
                if (i > 0) headerJson.append(",");
                headerJson.append(e.getValue().shape[i]);
            }
            headerJson.append("]");
            headerJson.append(",\"dtype\":\"").append(e.getValue().dtype).append("\"");
            headerJson.append(",\"data_offsets\":[");
            headerJson.append(e.getValue().dataOffset);
            headerJson.append(",");
            headerJson.append(e.getValue().dataOffset + e.getValue().size);
            headerJson.append("]");
            
            if (e.getValue().metadata != null && !e.getValue().metadata.isEmpty()) {
                headerJson.append(",\"__internal_metadata\":{");
                boolean firstMeta = true;
                for (Map.Entry<String, Object> me : e.getValue().metadata.entrySet()) {
                    if (!firstMeta) headerJson.append(",");
                    headerJson.append("\"").append(me.getKey()).append("\":");
                    appendJsonValue(headerJson, me.getValue());
                    firstMeta = false;
                }
                headerJson.append("}");
            }
            
            headerJson.append("}");
            first = false;
        }
        
        headerJson.append("}");
        
        byte[] headerBytes = headerJson.toString().getBytes(StandardCharsets.UTF_8);
        long headerSize = headerBytes.length;
        
        // Calculate total file size
        long dataOffset = HEADER_SIZE_BYTES + headerSize;
        long totalSize = dataOffset + currentOffset;
        
        // Write file
        try (RandomAccessFile raf = new RandomAccessFile(path, "rw")) {
            // Write header size (uint64 BE)
            raf.writeLong(Long.reverseBytes(headerSize));
            
            // Write header JSON
            raf.write(headerBytes);
            
            // Align to 8-byte boundary
            long padding = (ALIGNMENT - (raf.getFilePointer() % ALIGNMENT)) % ALIGNMENT;
            for (long i = 0; i < padding; i++) {
                raf.write(0);
            }
            
            // Write tensor data
            for (int c = 0; c < df.columnCount(); c++) {
                Column col = df.column(c);
                writeTensorData(raf, df, c, col.dtype(), opt);
                
                // Align after each tensor
                long pos = raf.getFilePointer();
                long alignPad = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT;
                for (long i = 0; i < alignPad; i++) {
                    raf.write(0);
                }
            }
        }
    }

    private static void writeTensorData(RandomAccessFile raf, DataFrame df, int colIdx, 
                                        Column.DType dtype, SafetensorsOptions opt) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8192);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        
        for (int r = 0; r < df.rowCount(); r++) {
            Object val = df.get(r, colIdx);
            buf.clear();
            
            if (val == null) {
                // Write zeros
                writeZeros(raf, dtype);
            } else {
                switch (dtype) {
                    case FLOAT32:
                        buf.putFloat(((Number) val).floatValue());
                        break;
                    case FLOAT64:
                        buf.putDouble(((Number) val).doubleValue());
                        break;
                    case INT32:
                        buf.putInt(((Number) val).intValue());
                        break;
                    case INT64:
                        buf.putLong(((Number) val).longValue());
                        break;
                    default:
                        // INT8/INT16/BOOLEAN and other narrow types -> writeFloat best-effort
                        try {
                            buf.putFloat(Float.parseFloat(val.toString()));
                        } catch (NumberFormatException e) {
                            writeZeros(raf, dtype);
                            continue;
                        }
                }

                raf.write(buf.array(), 0, buf.position());
            }
        }
    }

    private static void writeZeros(RandomAccessFile raf, Column.DType dtype) throws IOException {
        int size;
        switch (dtype) {
            case FLOAT64:
            case INT64:
                size = 8;
                break;
            case FLOAT32:
            case INT32:
                size = 4;
                break;
            default:
                // INT8/INT16/BOOLEAN and others: write 4 bytes of zeros
                size = 4;
        }
        for (int i = 0; i < size; i++) {
            raf.write(0);
        }
    }

    private static long estimateTensorSize(DataFrame df, int colIdx) {
        Column.DType dtype = df.column(colIdx).dtype();
        int rows = df.rowCount();

        switch (dtype) {
            case FLOAT32: return (long) rows * 4;
            case FLOAT64: return (long) rows * 8;
            case INT32: return (long) rows * 4;
            case INT64: return (long) rows * 8;
            case BOOLEAN: return rows;
            default: return (long) rows * 4;  // INT8/INT16 -> 2 bytes collapsed to 4 here
        }
    }

    private static String mapDtype(Column.DType dtype) {
        switch (dtype) {
            case FLOAT32: return "F32";
            case FLOAT64: return "F64";
            case INT32: return "I32";
            case INT64: return "I64";
            case BOOLEAN: return "BOOL";
            default: return "F32";
        }
    }

    private static void appendJsonValue(StringBuilder sb, Object val) {
        if (val == null) {
            sb.append("null");
        } else if (val instanceof String) {
            sb.append("\"").append(escapeJson((String) val)).append("\"");
        } else if (val instanceof Number) {
            sb.append(val);
        } else if (val instanceof Boolean) {
            sb.append(Boolean.TRUE.equals(val) ? "true" : "false");
        } else {
            sb.append("\"").append(escapeJson(val.toString())).append("\"");
        }
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t");
    }

    private static String sanitizeName(String name) {
        return name.replaceAll("[^a-zA-Z0-9_.-]", "_");
    }

    // ====================== Helper Classes ======================

    static class TensorInfo {
        long[] shape;
        String dtype;
        long dataOffset;
        long size;
        Map<String, Object> metadata;
    }

    // ====================== Options ======================

    public static class SafetensorsOptions {
        private boolean sanitizeNames = true;
        private boolean includeMetadata = true;

        public static SafetensorsOptions defaults() { return new SafetensorsOptions(); }

        public SafetensorsOptions sanitizeNames(boolean b) { this.sanitizeNames = b; return this; }
        public SafetensorsOptions includeMetadata(boolean b) { this.includeMetadata = b; return this; }

        public boolean sanitizeNames() { return sanitizeNames; }
        public boolean includeMetadata() { return includeMetadata; }
    }
}
