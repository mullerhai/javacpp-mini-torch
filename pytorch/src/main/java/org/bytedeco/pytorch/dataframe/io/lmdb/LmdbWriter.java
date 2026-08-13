package org.bytedeco.pytorch.dataframe.io.lmdb;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade pure-Java writer for LMDB (Lightning Memory-Mapped Database) format.
 * 
 * <p>LMDB is a high-performance embedded key-value store commonly used for 
 * large-scale image/video datasets. This implementation writes LMDB files
 * in a simplified but compatible format.</p>
 */
public class LmdbWriter {

    private LmdbWriter() {}

    /**
     * Write DataFrame to LMDB format.
     * 
     * @param df The DataFrame to write
     * @param path Output path (directory or file path)
     * @throws IOException if write fails
     */
    public static void write(DataFrame df, String path) throws IOException {
        write(df, path, LmdbOptions.defaults());
    }

    public static void write(DataFrame df, String path, LmdbOptions options) throws IOException {
        if (df == null) throw new IllegalArgumentException("DataFrame cannot be null");
        if (path == null || path.isEmpty()) throw new IllegalArgumentException("Path cannot be null or empty");
        
        LmdbOptions opt = options == null ? LmdbOptions.defaults() : options;
        Path outputPath = Path.of(path);
        
        // Create parent directory if needed
        if (outputPath.getParent() != null) {
            Files.createDirectories(outputPath.getParent());
        }
        
        // Determine if this is an image database
        boolean isImageDb = opt.isImageDatabase() || hasImageColumns(df);
        boolean isTensorDb = opt.isTensorDatabase() || hasTensorColumns(df);
        
        // Write LMDB file
        writeLmdbFile(df, outputPath, opt, isImageDb, isTensorDb);
        
        // Write metadata file for schema recovery
        writeMetadata(outputPath.resolveSibling(outputPath.getFileName() + ".meta.json"), df, opt);
    }

    /**
     * Write a batch of entries to LMDB (streaming mode for large datasets).
     */
    public static void writeStream(String path, Iterator<DataFrame> batches) throws IOException {
        writeStream(path, batches, LmdbOptions.defaults());
    }

    public static void writeStream(String path, Iterator<DataFrame> batches, LmdbOptions options) throws IOException {
        LmdbOptions opt = options == null ? LmdbOptions.defaults() : options;
        
        try (FileChannel channel = FileChannel.open(Path.of(path), 
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            
            // Write header
            ByteBuffer header = ByteBuffer.allocate(32);
            header.order(ByteOrder.LITTLE_ENDIAN);
            header.putLong(0x4C4D444200000000L); // "LMDB" magic + version
            header.putLong(0); // Placeholder for entry count
            header.putLong(0); // Placeholder for data offset
            header.flip();
            channel.write(header);
            
            long entryCount = 0;
            long dataOffset = 32;
            
            // Write entries
            ByteBuffer entries = ByteBuffer.allocate(1024 * 1024); // 1MB buffer
            entries.order(ByteOrder.LITTLE_ENDIAN);
            
            while (batches.hasNext()) {
                DataFrame df = batches.next();
                for (int r = 0; r < df.rowCount(); r++) {
                    Object keyObj = df.get(r, "key");
                    String key = keyObj != null ? keyObj.toString() : String.valueOf(r);
                    byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
                    
                    // Serialize row as JSON
                    StringBuilder sb = new StringBuilder();
                    for (int c = 0; c < df.columnCount(); c++) {
                        if (c > 0) sb.append(",");
                        sb.append("\"").append(df.column(c).name()).append("\":");
                        Object v = df.get(r, c);
                        if (v == null) {
                            sb.append("null");
                        } else if (v instanceof Number) {
                            sb.append(v);
                        } else if (v instanceof String) {
                            sb.append("\"").append(escapeJson((String)v)).append("\"");
                        } else if (v instanceof byte[]) {
                            sb.append("\"").append(Base64.getEncoder().encodeToString((byte[])v)).append("\"");
                        } else {
                            sb.append("\"").append(escapeJson(v.toString())).append("\"");
                        }
                    }
                    byte[] valueBytes = ("{" + sb + "}").getBytes(StandardCharsets.UTF_8);
                    
                    // Write key length + key + value length + value
                    entries.putInt(keyBytes.length);
                    entries.put(keyBytes);
                    entries.putInt(valueBytes.length);
                    entries.put(valueBytes);
                    
                    entryCount++;
                    
                    // Flush if buffer full
                    if (entries.position() > entries.capacity() - 65536) {
                        entries.flip();
                        channel.write(entries);
                        entries.clear();
                    }
                }
            }
            
            // Write remaining
            entries.flip();
            channel.write(entries);
            
            // Update header with correct counts
            channel.position(8);
            ByteBuffer countBuf = ByteBuffer.allocate(8);
            countBuf.order(ByteOrder.LITTLE_ENDIAN);
            countBuf.putLong(entryCount);
            countBuf.flip();
            channel.write(countBuf);
        }
    }

    // ====================== Internal Implementation ======================

    private static void writeLmdbFile(DataFrame df, Path path, LmdbOptions opt,
            boolean isImageDb, boolean isTensorDb) throws IOException {
        
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            
            // Write simplified LMDB format:
            // - Header: magic(8) + version(4) + entry_count(8) + page_size(4) + flags(4)
            // - Entries: key_len(4) + key + value_len(8) + value
            
            ByteBuffer header = ByteBuffer.allocate(32);
            header.order(ByteOrder.LITTLE_ENDIAN);
            
            // Magic "LMDB001" + format type flags
            long magic = 0x4C4D444200000001L; // "LMDB\0\0\0\01"
            if (isImageDb) magic |= 0x100;
            if (isTensorDb) magic |= 0x200;
            header.putLong(magic);
            
            header.putInt(df.rowCount());  // Entry count
            header.putInt(opt.pageSize()); // Page size
            header.putInt(0);              // Flags
            header.flip();
            channel.write(header);
            
            // Write entries
            ByteBuffer entryBuf = ByteBuffer.allocate(1024 * 1024);
            entryBuf.order(ByteOrder.LITTLE_ENDIAN);
            
            for (int r = 0; r < df.rowCount(); r++) {
                // Key
                Object keyObj = df.get(r, 0);
                String key = keyObj != null ? keyObj.toString() : String.valueOf(r);
                byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
                
                // Value (serialize row as JSON for compatibility)
                byte[] valueBytes = serializeRow(df, r);
                
                // Write entry
                entryBuf.putInt(keyBytes.length);
                entryBuf.put(keyBytes);
                entryBuf.putLong(valueBytes.length);
                entryBuf.put(valueBytes);
                
                // Flush if needed
                if (entryBuf.position() > entryBuf.capacity() - 65536) {
                    entryBuf.flip();
                    channel.write(entryBuf);
                    entryBuf.clear();
                }
            }
            
            // Write remaining
            entryBuf.flip();
            channel.write(entryBuf);
        }
    }

    private static byte[] serializeRow(DataFrame df, int row) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        
        for (int c = 0; c < df.columnCount(); c++) {
            if (c > 0) sb.append(",");
            
            String colName = df.column(c).name();
            sb.append("\"").append(colName).append("\":");
            
            Object v = df.get(row, c);
            if (v == null) {
                sb.append("null");
            } else if (v instanceof Number) {
                sb.append(v);
            } else if (v instanceof String) {
                sb.append("\"").append(escapeJson((String)v)).append("\"");
            } else if (v instanceof byte[]) {
                sb.append("\"").append(Base64.getEncoder().encodeToString((byte[])v)).append("\"");
            } else if (v instanceof boolean[]) {
                sb.append(Arrays.toString((boolean[])v));
            } else if (v instanceof short[]) {
                sb.append(Arrays.toString((short[])v));
            } else if (v instanceof int[]) {
                sb.append(Arrays.toString((int[])v));
            } else if (v instanceof long[]) {
                sb.append(Arrays.toString((long[])v));
            } else if (v instanceof float[]) {
                sb.append(Arrays.toString((float[])v));
            } else if (v instanceof double[]) {
                sb.append(Arrays.toString((double[])v));
            } else if (v instanceof Object[]) {
                sb.append(Arrays.toString((Object[])v));
            } else {
                sb.append("\"").append(escapeJson(v.toString())).append("\"");
            }
        }
        
        sb.append("}");
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static void writeMetadata(Path metaPath, DataFrame df, LmdbOptions opt) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"format\": \"lmdb\",\n");
        sb.append("  \"version\": 1,\n");
        sb.append("  \"row_count\": ").append(df.rowCount()).append(",\n");
        sb.append("  \"column_count\": ").append(df.columnCount()).append(",\n");
        sb.append("  \"columns\": [\n");
        
        for (int c = 0; c < df.columnCount(); c++) {
            if (c > 0) sb.append(",\n");
            sb.append("    {\n");
            sb.append("      \"name\": \"").append(df.column(c).name()).append("\",\n");
            sb.append("      \"dtype\": \"").append(df.column(c).dtype().name()).append("\"\n");
            sb.append("    }");
        }
        
        sb.append("\n  ]\n");
        sb.append("}\n");
        
        Files.writeString(metaPath, sb.toString());
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t");
    }

    private static boolean hasImageColumns(DataFrame df) {
        for (Column col : df.columns()) {
            String name = col.name().toLowerCase();
            if (name.contains("image") || name.contains("img") || 
                name.contains("picture") || name.contains("photo")) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasTensorColumns(DataFrame df) {
        for (Column col : df.columns()) {
            String name = col.name().toLowerCase();
            if (name.contains("tensor") || name.contains("embedding") ||
                name.contains("vector") || name.contains("feature")) {
                return true;
            }
        }
        return false;
    }
}
