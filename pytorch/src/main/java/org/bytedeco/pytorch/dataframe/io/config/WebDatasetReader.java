package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

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
 * <p><b>Three read modes</b> via {@link Mode}:
 * <ul>
 *   <li>{@code TAR} — original WebDataset tar shards (default; this is also what
 *       {@code AUTO} picks when the path ends in {@code .tar}, {@code .tar.gz}, or
 *       matches a brace range of tars).</li>
 *   <li>{@code HF_PARQUET} — HuggingFace {@code datasets} layout (one dataset per
 *       config/split, sharded parquet / arrow / csv / json / jsonl / txt files).
 *       Each row gets the same {@code __key__}, {@code __shard__}, {@code __tar_file__}
 *       columns as tar mode, plus two extras: {@code __hf_dataset__} and
 *       {@code __hf_split__} for traceability. Backed by the project's
 *       {@code org.bytedeco.pytorch.utils.datasets.HfDataset} / {@code HfDatasets}
 *       which already support parquet / arrow / csv / json / jsonl / orc / avro /
 *       text — i.e. many data formats out of the box.</li>
 *   <li>{@code AUTO} — sniff the path: tars → TAR, anything else → HF_PARQUET.</li>
 * </ul>
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
 *       .mode(WebDatasetReader.Mode.HF_PARQUET)
 *       .hfDataset("HF_DATASET_NAME", "config_name", "train")
 *       .maxSamples(10000);
 *   DataFrame df = WebDatasetReader.read("/path/to/hf_snapshot", opts);
 *
 *   // Or just point at a local parquet/arrow/csv/jsonl file
 *   DataFrame df = WebDatasetReader.read("/data/train-00000.parquet");
 * </pre>
 */
public class WebDatasetReader {

    /** Source geometry for {@link WebDatasetReader#read(String, WebDatasetOptions)}. */
    public enum Mode {
        /** Tar shards (with brace-expansion glob). Original WebDataset layout. */
        TAR,
        /** HuggingFace parquet/arrow/csv/jsonl/etc. shards via {@code HfDataset}. */
        HF_PARQUET,
        /** Auto-detect: tars → TAR, everything else → HF_PARQUET. */
        AUTO
    }

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
        Mode mode = opts.mode() == null ? Mode.AUTO : opts.mode();
        if (mode == Mode.AUTO) {
            mode = looksLikeTar(path) ? Mode.TAR : Mode.HF_PARQUET;
        }
        if (mode == Mode.HF_PARQUET) {
            return readHfDataset(path, opts);
        }
        // Falls through to TAR layout below.
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

    // ====================== HuggingFace parquet / etc. ======================

    /**
     * Heuristic: does {@code path} look like a tar shard (or a brace range of tars)?
     * Used by {@link Mode#AUTO} to pick between TAR and HF_PARQUET modes.
     */
    private static boolean looksLikeTar(String path) {
        if (path == null) return false;
        String lower = path.toLowerCase(java.util.Locale.ROOT);
        if (lower.endsWith(".tar") || lower.endsWith(".tar.gz") || lower.endsWith(".tgz")) {
            return true;
        }
        // brace pattern like dataset-{000000..000099}.tar
        if (lower.contains("{") && lower.contains("..") && lower.contains("}")) {
            int braceEnd = lower.indexOf('}');
            if (braceEnd > 0 && lower.substring(braceEnd).toLowerCase(java.util.Locale.ROOT)
                    .matches(".*\\.(tar|tar\\.gz|tgz)$")) {
                return true;
            }
        }
        return false;
    }

    /**
     * HuggingFace path: delegate to {@code org.bytedeco.pytorch.utils.datasets.HfDataset}
     * (single file) or {@code HfDatasets.loadDataset} (config/split selection), then
     * lift the result into a DataFrame with the same WebDataset bookkeeping columns.
     */
    private static DataFrame readHfDataset(String path, WebDatasetOptions opts) throws IOException {
        org.bytedeco.pytorch.utils.datasets.HfDataset hf;
        String dsId = opts.hfDataset();
        String config = opts.hfConfig();
        String split = opts.hfSplit();

        if (dsId != null && !dsId.isBlank()) {
            // Forward HF_TOKEN / etc. via builder hooks when provided.
            org.bytedeco.pytorch.utils.datasets.HfDatasets.LoadConfig lc =
                    org.bytedeco.pytorch.utils.datasets.HfDatasets.LoadConfig.builder()
                            .token(opts.hfToken())
                            .endpoint(opts.hfEndpoint())
                            .revision(opts.hfRevision())
                            .take(opts.maxSamples() > 0 ? opts.maxSamples() : -1)
                            .build();
            // The 4-arg overload always returns a DatasetDict; pick the requested
            // split if specified, else the first available.
            org.bytedeco.pytorch.utils.datasets.HfDataset.DatasetDict dict =
                    org.bytedeco.pytorch.utils.datasets.HfDatasets.loadDataset(
                            dsId, config, split, lc);
            if (split != null && !split.isBlank() && dict.splits().containsKey(split)) {
                hf = dict.get(split);
            } else {
                hf = dict.splits().values().iterator().next();
            }
        } else if (path == null) {
            throw new IOException("WebDatasetReader.read(HF_PARQUET) requires a path or hfDataset(...)");
        } else {
            java.nio.file.Path p = java.nio.file.Paths.get(path);
            if (java.nio.file.Files.isDirectory(p)) {
                hf = org.bytedeco.pytorch.utils.datasets.HfDataset.fromDirectory(p, true);
            } else {
                hf = org.bytedeco.pytorch.utils.datasets.HfDataset.fromFile(p,
                        opts.maxSamples() > 0 ? opts.maxSamples() : -1);
            }
        }

        DataFrame df = DataFrame.create();
        df.addColumn("__key__", Column.DType.STRING);
        df.addColumn("__shard__", Column.DType.INT32);
        df.addColumn("__tar_file__", Column.DType.STRING);
        df.addColumn("__hf_dataset__", Column.DType.STRING);
        df.addColumn("__hf_split__", Column.DType.STRING);

        // Source path forwarded to __tar_file__ in HF mode for column-shape parity.
        String source = path == null ? "<hub:" + dsId + ">" : path;
        String resolvedSplit = split == null ? "<default>" : split;

        int stop = hf.size();
        if (opts.maxSamples() > 0 && opts.maxSamples() < stop) stop = opts.maxSamples();
        for (int i = 0; i < stop; i++) {
            Map<String, Object> row = hf.get(i);
            int ri = df.addEmptyRow();
            df.set(ri, "__key__", String.valueOf(i));
            df.set(ri, "__shard__", 0);
            df.set(ri, "__tar_file__", source);
            df.set(ri, "__hf_dataset__", dsId == null ? "" : dsId);
            df.set(ri, "__hf_split__", resolvedSplit);

            // Fan-out: each row field becomes a column. Suffix-based gating reuses
            // the same flags as tar mode (includeJson, includeText, …) so callers
            // get consistent behaviour across the two read modes.
            for (Map.Entry<String, Object> e : row.entrySet()) {
                String key = e.getKey();
                if (key.startsWith("__")) continue;
                String lower = key.toLowerCase(java.util.Locale.ROOT);
                Object value = e.getValue();
                Column.DType dt = inferDtype(value);
                boolean include = decideInclude(lower, dt, opts);
                if (!include) continue;
                String colName = "__hf_" + key + "__";
                df.addColumnIfAbsent(colName, dt);
                df.set(ri, colName, value);
            }
        }
        return df;
    }

    /** Pick the most permissive {@link Column.DType} compatible with the value. */
    private static Column.DType inferDtype(Object value) {
        if (value == null) return Column.DType.STRING;
        if (value instanceof Boolean) return Column.DType.BOOLEAN;
        if (value instanceof Integer || value instanceof Long
                || value instanceof Short || value instanceof Byte) return Column.DType.INT64;
        if (value instanceof Float || value instanceof Double) return Column.DType.FLOAT64;
        if (value instanceof byte[]) return Column.DType.BINARY;
        if (value instanceof List) return Column.DType.LIST;
        if (value instanceof Map) return Column.DType.MAP;
        return Column.DType.STRING;
    }

    /**
     * Mirror the tar-mode inclusion logic so both readers honour the same
     * {@code include{X}} flags. Heuristic: JSON/JSONL → json, TXT/TEXT → text,
     * PT/PKL → tensors, NPY/ARROW → arrays, label-ish keys → labels.
     */
    private static boolean decideInclude(String lowerKey, Column.DType dt, WebDatasetOptions opts) {
        if (lowerKey.endsWith(".json") || lowerKey.endsWith(".jsonl")) return opts.includeJson();
        if (lowerKey.endsWith(".txt") || lowerKey.endsWith(".text")
                || lowerKey.endsWith(".summary") || lowerKey.endsWith(".caption")) return opts.includeText();
        if (lowerKey.endsWith(".pt") || lowerKey.endsWith(".pydpickle")
                || lowerKey.endsWith(".pickle")) return opts.includeTensors();
        if (lowerKey.endsWith(".npy") || lowerKey.endsWith(".arrow")) return opts.includeArrays();
        if (lowerKey.endsWith(".cls") || lowerKey.endsWith(".label")
                || lowerKey.endsWith(".class") || lowerKey.endsWith(".target")) return opts.includeLabels();
        if (lowerKey.endsWith(".jpg") || lowerKey.endsWith(".jpeg")
                || lowerKey.endsWith(".png") || lowerKey.endsWith(".webp")) return opts.includeImages();
        if (dt == Column.DType.BINARY) return opts.includeOther();
        return true;
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

        // ---- HF / multi-format routing ----
        private Mode mode = Mode.AUTO;
        private String hfDataset = null;
        private String hfConfig = null;
        private String hfSplit = null;
        private String hfToken = null;
        private String hfEndpoint = null;
        private String hfRevision = null;

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

        /** Force a read mode (default is {@link Mode#AUTO}). */
        public WebDatasetOptions mode(Mode v) { this.mode = v == null ? Mode.AUTO : v; return this; }
        /** HuggingFace dataset id (e.g. {@code glue}, {@code imdb}). When set, the path
         *  argument is unused and the loader tries to download a snapshot via
         *  {@code HfDatasets.loadDataset}. */
        public WebDatasetOptions hfDataset(String v) { this.hfDataset = v; return this; }
        public WebDatasetOptions hfConfig(String v) { this.hfConfig = v; return this; }
        public WebDatasetOptions hfSplit(String v) { this.hfSplit = v; return this; }
        public WebDatasetOptions hfToken(String v) { this.hfToken = v; return this; }
        public WebDatasetOptions hfEndpoint(String v) { this.hfEndpoint = v; return this; }
        public WebDatasetOptions hfRevision(String v) { this.hfRevision = v; return this; }

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

        public Mode mode() { return mode; }
        public String hfDataset() { return hfDataset; }
        public String hfConfig() { return hfConfig; }
        public String hfSplit() { return hfSplit; }
        public String hfToken() { return hfToken; }
        public String hfEndpoint() { return hfEndpoint; }
        public String hfRevision() { return hfRevision; }
    }
}
