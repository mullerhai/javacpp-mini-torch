package org.bytedeco.pytorch.dataframe.io.webdataset;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;
import java.util.jar.*;

/**
 * Enterprise-grade reader for WebDataset format.
 * 
 * <p>WebDataset is a dataset format for large-scale machine learning workloads.
 * It stores data as tar archives with naming conventions like:</p>
 * <pre>
 * dataset-000000.tar
 * ├── 000000.jpg
 * ├── 000000.json
 * ├── 000001.jpg
 * ├── 000001.json
 * └── ...
 * </pre>
 * 
 * <p>Each sample consists of multiple files with the same key (shard index + sample ID).
 * Common suffixes: .jpg, .png, .json, .txt, .cls, .pydpickle, .pt, .npy, .arrow</p>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   DataFrame df = WebDatasetReader.read("/path/to/dataset.tar");
 *   
 *   // Read multiple shards
 *   DataFrame df = WebDatasetReader.read("dataset-{000000..000099}.tar");
 *   
 *   // With options
 *   WebDatasetReader.WebDatasetOptions opts = WebDatasetReader.options()
 *       .decodeImages(true)
 *       .maxSamples(10000);
 *   DataFrame df = WebDatasetReader.read("/path/to/dataset.tar", opts);
 * </pre>
 */
public class WebDatasetReader {

    private WebDatasetReader() {}

    // Known WebDataset suffixes
    private static final Set<String> IMAGE_SUFFIXES = Set.of(
        ".jpg", ".jpeg", ".png", ".webp", ".ppm", ".cls"
    );
    private static final Set<String> TEXT_SUFFIXES = Set.of(
        ".txt", ".text", ".summary", ".caption"
    );
    private static final Set<String> JSON_SUFFIXES = Set.of(
        ".json", ".jsonl"
    );
    private static final Set<String> TENSOR_SUFFIXES = Set.of(
        ".pt", ".pydpickle", ".pickle", ".npy", ".arrow"
    );
    private static final Set<String> LABEL_SUFFIXES = Set.of(
        ".cls", ".label", ".class", ".target"
    );

    /**
     * Read a WebDataset tar file into a DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, WebDatasetOptions.defaults());
    }

    public static DataFrame read(String path, WebDatasetOptions options) throws IOException {
        WebDatasetOptions opts = options == null ? WebDatasetOptions.defaults() : options;
        
        DataFrame df = DataFrame.create();
        df.addColumn("__key__", Column.DType.STRING);
        df.addColumn("__shard__", Column.DType.INT32);
        df.addColumn("__tar_file__", Column.DType.STRING);
        
        // Add dynamic columns for common data types
        Set<String> foundSuffixes = new HashSet<>();
        
        // Handle glob patterns
        List<String> paths = expandGlobPatterns(path);
        
        int shardIndex = 0;
        int sampleCount = 0;
        
        for (String tarPath : paths) {
            if (opts.maxSamples() > 0 && sampleCount >= opts.maxSamples()) {
                break;
            }
            
            try {
                int shardCount = readTarFile(df, tarPath, shardIndex, opts, foundSuffixes);
                sampleCount += shardCount;
                shardIndex++;
            } catch (Exception e) {
                if (opts.failOnError()) {
                    throw new IOException("Failed to read tar file: " + tarPath, e);
                }
            }
        }
        
        return df;
    }

    private static List<String> expandGlobPatterns(String path) throws IOException {
        List<String> paths = new ArrayList<>();
        
        // Handle brace expansion like {000000..000099}
        if (path.contains("{") && path.contains("..") && path.contains("}")) {
            int braceStart = path.indexOf('{');
            int braceEnd = path.indexOf('}');
            if (braceStart < braceEnd) {
                String prefix = path.substring(0, braceStart);
                String suffix = path.substring(braceEnd + 1);
                String range = path.substring(braceStart + 1, braceEnd);
                
                int dotDot = range.indexOf("..");
                if (dotDot > 0) {
                    try {
                        int start = Integer.parseInt(range.substring(0, dotDot).trim());
                        int end = Integer.parseInt(range.substring(dotDot + 2).trim());
                        int padding = range.substring(0, dotDot).trim().length();
                        
                        for (int i = start; i <= end; i++) {
                            String expanded = prefix + String.format("%0" + padding + "d", i) + suffix;
                            if (Files.exists(Path.of(expanded))) {
                                paths.add(expanded);
                            }
                        }
                    } catch (NumberFormatException e) {
                        // Not a numeric range, treat as literal
                        paths.add(path);
                    }
                }
            }
        } else {
            paths.add(path);
        }
        
        return paths;
    }

    private static int readTarFile(DataFrame df, String tarPath, int shardIndex,
                                  WebDatasetOptions opts, Set<String> foundSuffixes) throws IOException {
        Path path = Path.of(tarPath);
        if (!Files.exists(path)) {
            throw new IOException("Tar file not found: " + tarPath);
        }
        
        int samplesInShard = 0;
        
        try (InputStream fis = Files.newInputStream(path);
             InputStream is = tarPath.endsWith(".gz") ? new GZIPInputStream(fis) : fis;
             TarArchiveInputStream tar = new TarArchiveInputStream(is)) {
            
            // Group entries by sample key
            Map<String, Map<String, byte[]>> samples = new LinkedHashMap<>();
            TarArchiveEntry entry;
            
            while ((entry = tar.getNextTarEntry()) != null) {
                if (entry.isDirectory()) continue;
                
                String name = entry.getName();
                int lastSlash = name.lastIndexOf('/');
                String baseName = lastSlash >= 0 ? name.substring(lastSlash + 1) : name;
                
                // Parse key and suffix: 000000.jpg -> key=000000, suffix=.jpg
                int dotIdx = baseName.lastIndexOf('.');
                if (dotIdx < 0) continue;
                
                String key = baseName.substring(0, dotIdx);
                String suffix = baseName.substring(dotIdx); // includes the dot
                
                // Read entry content
                byte[] content = readAllBytes(tar);
                
                samples.computeIfAbsent(key, k -> new HashMap<>()).put(suffix, content);
                foundSuffixes.add(suffix.toLowerCase());
            }
            
            // Convert samples to DataFrame rows
            for (Map.Entry<String, Map<String, byte[]>> sampleEntry : samples.entrySet()) {
                if (opts.maxSamples() > 0 && df.rowCount() >= opts.maxSamples()) {
                    break;
                }
                
                String key = sampleEntry.getKey();
                Map<String, byte[]> data = sampleEntry.getValue();
                
                int ri = df.addEmptyRow();
                df.set(ri, "__key__", key);
                df.set(ri, "__shard__", shardIndex);
                df.set(ri, "__tar_file__", tarPath);
                
                // Add data columns based on suffixes
                for (Map.Entry<String, byte[]> field : data.entrySet()) {
                    String suffix = field.getKey().toLowerCase();
                    String colName = "__" + suffix.substring(1) + "__";
                    
                    if (suffix.equals(".jpg") || suffix.equals(".jpeg") || 
                        suffix.equals(".png") || suffix.equals(".webp")) {
                        if (opts.decodeImages() && opts.includeImages()) {
                            df.addColumnIfAbsent(colName, Column.DType.BINARY);
                            df.set(ri, colName, field.getValue());
                        }
                    } else if (suffix.equals(".json") || suffix.equals(".jsonl")) {
                        if (opts.includeJson()) {
                            String jsonStr = new String(field.getValue());
                            df.addColumnIfAbsent(colName, Column.DType.STRING);
                            df.set(ri, colName, jsonStr);
                        }
                    } else if (suffix.equals(".txt") || suffix.equals(".text")) {
                        if (opts.includeText()) {
                            String text = new String(field.getValue());
                            df.addColumnIfAbsent(colName, Column.DType.STRING);
                            df.set(ri, colName, text);
                        }
                    } else if (suffix.equals(".pt") || suffix.equals(".pydpickle") || suffix.equals(".pickle")) {
                        if (opts.includeTensors()) {
                            df.addColumnIfAbsent(colName, Column.DType.BINARY);
                            df.set(ri, colName, field.getValue());
                        }
                    } else if (suffix.equals(".npy") || suffix.equals(".arrow")) {
                        if (opts.includeArrays()) {
                            df.addColumnIfAbsent(colName, Column.DType.BINARY);
                            df.set(ri, colName, field.getValue());
                        }
                    } else if (suffix.equals(".cls") || suffix.equals(".label") || 
                               suffix.equals(".class") || suffix.equals(".target")) {
                        if (opts.includeLabels()) {
                            String labelStr = new String(field.getValue()).trim();
                            try {
                                int label = Integer.parseInt(labelStr);
                                df.addColumnIfAbsent(colName, Column.DType.INT32);
                                df.set(ri, colName, label);
                            } catch (NumberFormatException e) {
                                df.addColumnIfAbsent(colName, Column.DType.STRING);
                                df.set(ri, colName, labelStr);
                            }
                        }
                    } else {
                        // Generic binary field
                        if (opts.includeOther()) {
                            String colNameBase = "__" + suffix.substring(1) + "__";
                            df.addColumnIfAbsent(colNameBase, Column.DType.BINARY);
                            df.set(ri, colNameBase, field.getValue());
                        }
                    }
                }
                
                samplesInShard++;
            }
        }
        
        return samplesInShard;
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = is.read(buffer)) >= 0) {
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    /**
     * Simple TarArchiveInputStream implementation.
     * Handles basic tar format without external dependencies.
     */
    public static class TarArchiveInputStream extends java.io.FilterInputStream {
        private byte[] header = new byte[512];
        private TarArchiveEntry currentEntry;
        private long entryRemaining;
        
        public TarArchiveInputStream(InputStream is) {
            super(is);
        }
        
        public TarArchiveEntry getNextTarEntry() throws IOException {
            // Read header
            int read = 0;
            while (read < 512) {
                int r = in.read(header, read, 512 - read);
                if (r < 0) return null;
                read += r;
            }
            
            // Check for null block (end of archive)
            boolean allZero = true;
            for (int i = 0; i < 512; i++) {
                if (header[i] != 0) {
                    allZero = false;
                    break;
                }
            }
            if (allZero) return null;
            
            // Parse tar header
            String name = new String(header, 0, 100).trim();
            String sizeStr = new String(header, 124, 12).trim();
            long size = 0;
            try {
                size = Long.parseLong(sizeStr, 8);
            } catch (NumberFormatException e) {}
            
            currentEntry = new TarArchiveEntry(name, size);
            entryRemaining = size;
            
            return currentEntry;
        }
        
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            if (entryRemaining <= 0) return -1;
            long toRead = Math.min(entryRemaining, len);
            int r = in.read(b, off, (int)toRead);
            if (r > 0) {
                entryRemaining -= r;
                // Skip to next block boundary if needed
                long pos = 512 - (entryRemaining % 512);
                if (entryRemaining == 0 && pos < 512) {
                    in.skip(pos);
                }
            }
            return r;
        }
        
        @Override
        public int read() throws IOException {
            byte[] b = new byte[1];
            int r = read(b);
            return r > 0 ? b[0] & 0xFF : -1;
        }
    }
    
    public static class TarArchiveEntry {
        private final String name;
        private final long size;
        
        TarArchiveEntry(String name, long size) {
            this.name = name;
            this.size = size;
        }
        
        public String getName() { return name; }
        public long getSize() { return size; }
        public boolean isDirectory() { return name.endsWith("/"); }
    }

    // ====================== Options ======================

    public static class WebDatasetOptions {
        private int maxSamples = 0;  // 0 = no limit
        private boolean decodeImages = false;
        private boolean includeImages = true;
        private boolean includeJson = true;
        private boolean includeText = true;
        private boolean includeTensors = true;
        private boolean includeArrays = true;
        private boolean includeLabels = true;
        private boolean includeOther = false;
        private boolean failOnError = false;

        public static WebDatasetOptions defaults() {
            return new WebDatasetOptions();
        }

        public WebDatasetOptions maxSamples(int v) { this.maxSamples = v; return this; }
        public WebDatasetOptions decodeImages(boolean v) { this.decodeImages = v; return this; }
        public WebDatasetOptions includeImages(boolean v) { this.includeImages = v; return this; }
        public WebDatasetOptions includeJson(boolean v) { this.includeJson = v; return this; }
        public WebDatasetOptions includeText(boolean v) { this.includeText = v; return this; }
        public WebDatasetOptions includeTensors(boolean v) { this.includeTensors = v; return this; }
        public WebDatasetOptions includeArrays(boolean v) { this.includeArrays = v; return this; }
        public WebDatasetOptions includeLabels(boolean v) { this.includeLabels = v; return this; }
        public WebDatasetOptions includeOther(boolean v) { this.includeOther = v; return this; }
        public WebDatasetOptions failOnError(boolean v) { this.failOnError = v; return this; }

        public int maxSamples() { return maxSamples; }
        public boolean decodeImages() { return decodeImages; }
        public boolean includeImages() { return includeImages; }
        public boolean includeJson() { return includeJson; }
        public boolean includeText() { return includeText; }
        public boolean includeTensors() { return includeTensors; }
        public boolean includeArrays() { return includeArrays; }
        public boolean includeLabels() { return includeLabels; }
        public boolean includeOther() { return includeOther; }
        public boolean failOnError() { return failOnError; }
    }
}
