package org.bytedeco.pytorch.utils.daft;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.io.SaveMode;
import org.bytedeco.pytorch.utils.daft.engine.ExecutionConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Daft-style fluent writer for {@link DaftDataFrame}.
 *
 * <p>Mirrors Python Daft's {@code df.write_lance(...)} / {@code df.write_parquet(...)}
 * surface and also the Spark-style API surface for maximum compatibility.
 *
 * <pre>{@code
 *   // Daft-style quick helpers (preferred)
 *   df.write().parquet("/data/out.parquet");
 *   df.write().csv("/data/out.csv");
 *   df.write().json("/data/out.json");
 *   df.write().jsonl("/data/out.jsonl");
 *   df.write().lance("/data/out.lance");
 *
 *   // Spark-style
 *   df.write().format("parquet").mode("overwrite").save("/data/out");
 *   df.write().option("compression", "zstd").parquet("/data/out.parquet");
 *   df.write().partitionBy("year", "month").parquet("/data/out");
 *
 *   // write_lance with vector columns
 *   df.write().option("vectorCols", "embedding").lance("/data/out.lance");
 * }</pre>
 *
 * <p>The writer materialises the lazy DaftDataFrame pipeline at the first
 * {@link #save(String)} / {@link #parquet} / etc. call via {@code collect()}.
 */
public final class DaftDataFrameWriter {

    /** Save mode semantics (matches Spark / DataFrameWriter). */
    public enum WriteMode {
        OVERWRITE,
        APPEND,
        IGNORE,
        ERROR_IF_EXISTS;

        public static WriteMode fromString(String s) {
            if (s == null || s.isEmpty()) return OVERWRITE;
            String n = s.trim().toLowerCase(Locale.ROOT).replace("-", "").replace("_", "");
            switch (n) {
                case "overwrite": return OVERWRITE;
                case "append":    return APPEND;
                case "ignore":    return IGNORE;
                case "error":
                case "errorifexists": return ERROR_IF_EXISTS;
                default: return OVERWRITE;
            }
        }

        /** Returns false (skip write) for IGNORE when path exists. */
        public boolean preCheck(String path) throws IOException {
            if (this == IGNORE) {
                Path p = Paths.get(path);
                return !Files.exists(p);
            }
            if (this == ERROR_IF_EXISTS) {
                Path p = Paths.get(path);
                if (Files.exists(p)) {
                    throw new IOException("Destination already exists: " + path
                        + " (WriteMode.ERROR_IF_EXISTS)");
                }
            }
            return true;
        }
    }

    private final DaftDataFrame df;
    private String format;
    private final Map<String, String> options = new LinkedHashMap<>();
    private WriteMode mode = WriteMode.OVERWRITE;
    private String[] partitionCols;
    private String deferredPath;

    DaftDataFrameWriter(DaftDataFrame df) {
        this.df = df;
    }

    // ---- format / options / mode / partition ----

    /** Set format by short name (case-insensitive): parquet, csv, json, jsonl, lance, etc. */
    public DaftDataFrameWriter format(String name) {
        this.format = name == null ? null : name.trim().toLowerCase(Locale.ROOT);
        return this;
    }

    /** Spark-style single key/value option. */
    public DaftDataFrameWriter option(String key, String value) {
        if (key == null || key.isEmpty()) return this;
        if ("format".equalsIgnoreCase(key)) this.format = value == null ? null : value.trim().toLowerCase(Locale.ROOT);
        else if ("mode".equalsIgnoreCase(key)) this.mode = WriteMode.fromString(value);
        else this.options.put(key, value == null ? "" : value);
        return this;
    }

    /** Boolean option as "true"/"false". */
    public DaftDataFrameWriter option(String key, boolean value) {
        return option(key, Boolean.toString(value));
    }

    /** Numeric option as string. */
    public DaftDataFrameWriter option(String key, int value) {
        return option(key, Integer.toString(value));
    }

    /** Numeric option as string. */
    public DaftDataFrameWriter option(String key, long value) {
        return option(key, Long.toString(value));
    }

    /** Bulk options from Map. */
    public DaftDataFrameWriter options(Map<String, String> map) {
        if (map == null) return this;
        map.forEach(this::option);
        return this;
    }

    /** Varargs key/value pairs. */
    public DaftDataFrameWriter options(String... kv) {
        if (kv == null || kv.length == 0) return this;
        if (kv.length % 2 != 0)
            throw new IllegalArgumentException("options(...) requires even number of args");
        for (int i = 0; i < kv.length; i += 2) option(kv[i], kv[i + 1]);
        return this;
    }

    /** Set write mode. */
    public DaftDataFrameWriter mode(WriteMode m) {
        if (m != null) this.mode = m;
        return this;
    }

    /** Set write mode by string (Spark-style: "overwrite", "append", "ignore", "error"). */
    public DaftDataFrameWriter mode(String name) {
        this.mode = WriteMode.fromString(name);
        return this;
    }

    /**
     * Partition by column(s) — mirrors Spark semantics. Creates a directory at
     * {@code path} and writes partition subdirectories (e.g. {@code year=2024/month=01/}).
     */
    public DaftDataFrameWriter partitionBy(String... cols) {
        if (cols == null || cols.length == 0) { this.partitionCols = null; return this; }
        java.util.LinkedHashSet<String> seen = new java.util.LinkedHashSet<>();
        for (String c : cols) if (c != null) seen.add(c);
        this.partitionCols = seen.toArray(new String[0]);
        return this;
    }

    // ---- accessors ----

    public String format() { return format; }
    public WriteMode mode() { return mode; }
    public Map<String, String> options() { return new LinkedHashMap<>(options); }

    // ---- save ----

    /**
     * Save to {@code path}. Format is resolved from:
     * 1. explicit {@link #format(String)} call,
     * 2. path extension (if format is null).
     */
    public void save(String path) throws Exception {
        saveImpl(path, null);
    }

    /** No-op if path already set and mode is IGNORE. */
    public void save() throws Exception {
        if (deferredPath == null)
            throw new IllegalStateException("path required: call save(path) or use a quick helper");
        saveImpl(deferredPath, null);
    }

    // ---- quick helpers (Daften + Spark-style) ----

    /** Write as Parquet. */
    public void parquet(String path) throws Exception { format("parquet"); save(path); }

    /** Write as CSV. */
    public void csv(String path) throws Exception { format("csv"); save(path); }

    /** Write as TSV. */
    public void tsv(String path) throws Exception { format("tsv"); save(path); }

    /** Write as JSON (one record per line — JSONL / NDJSON). */
    public void json(String path) throws Exception { format("jsonl"); save(path); }

    /** Write as JSONL / NDJSON. */
    public void jsonl(String path) throws Exception { format("jsonl"); save(path); }

    /** Write as LanceDB dataset. */
    public void lance(String path) throws Exception { format("lance"); save(path); }

    // ---- core dispatch ----

    private void saveImpl(String path, String preResolved) throws Exception {
        if (path == null) throw new IllegalArgumentException("path required");
        this.deferredPath = path;

        if (!mode.preCheck(path)) return; // IGNORE + exists → no-op

        // Resolve format from explicit → preResolved → extension
        String fmt = format;
        if (fmt == null || fmt.isEmpty()) {
            if (preResolved != null) fmt = preResolved.toLowerCase(Locale.ROOT);
            else fmt = detectFromPath(path);
        }

        // partitionBy support check
        if (partitionCols != null && partitionCols.length > 0 && !supportsPartition(fmt)) {
            throw new UnsupportedOperationException(
                "partitionBy() is not supported for format '" + fmt + "'");
        }

        // Materialise once
        DataFrame materialized = df.collect();

        if (partitionCols != null && partitionCols.length > 0) {
            savePartitioned(materialized, fmt, path);
        } else {
            saveSimple(materialized, fmt, path);
        }
    }

    private void saveSimple(DataFrame df, String fmt, String path) throws Exception {
        switch (fmt) {
            case "parquet":
                df.writeParquet(path);
                break;
            case "csv":
                applyAndWriteCsv(df, path, false);
                break;
            case "tsv":
                applyAndWriteCsv(df, path, true);
                break;
            case "json":
            case "jsonl":
            case "ndjson":
                applyAndWriteJson(df, path, "jsonl".equals(fmt) || "ndjson".equals(fmt));
                break;
            case "lance":
                writeLance(df, path);
                break;
            default:
                throw new IllegalArgumentException(
                    "Unsupported DaftDataFrame write format: '" + fmt + "'. "
                    + "Supported: parquet, csv, tsv, json, jsonl, lance");
        }
    }

    private void applyAndWriteCsv(DataFrame df, String path, boolean tsv)
            throws Exception {
        org.bytedeco.pytorch.dataframe.csv.CsvOptions.Builder b =
            tsv
                ? org.bytedeco.pytorch.dataframe.csv.CsvOptions.builder().delimiter('\t')
                : org.bytedeco.pytorch.dataframe.csv.CsvOptions.builder();
        b.header(true);
        applyCsvOptions(b);
        df.toCsv(path, b.build());
    }

    private void applyCsvOptions(org.bytedeco.pytorch.dataframe.csv.CsvOptions.Builder b) {
        String header = options.get("header");
        if (header != null) b.header(Boolean.parseBoolean(header));
        String delim = options.get("delimiter");
        if (delim != null && !delim.isEmpty()) b.delimiter(delim.charAt(0));
        String sep = options.get("sep");
        if (sep != null && !sep.isEmpty()) b.delimiter(sep.charAt(0));
        String charset = options.get("charset");
        if (charset != null && !charset.isEmpty())
            b.charset(java.nio.charset.Charset.forName(charset));
    }

    private void applyAndWriteJson(DataFrame df, String path, boolean lines)
            throws Exception {
        org.bytedeco.pytorch.dataframe.json.JsonOptions.Builder b =
            org.bytedeco.pytorch.dataframe.json.JsonOptions.builder()
                .orient(lines
                    ? org.bytedeco.pytorch.dataframe.json.JsonOptions.Orient.LINES
                    : org.bytedeco.pytorch.dataframe.json.JsonOptions.Orient.RECORDS);
        if (lines) df.toJsonl(path, b.build());
        else df.toJson(path, b.build());
    }

    private void writeLance(DataFrame df, String path) throws Exception {
        String vectorColsStr = options.get("vectorCols");
        if (vectorColsStr != null && !vectorColsStr.isEmpty()) {
            String[] cols = vectorColsStr.split(",");
            df.writeLance(path, cols);
        } else {
            df.writeLance(path);
        }
    }

    private boolean supportsPartition(String fmt) {
        return "parquet".equals(fmt) || "csv".equals(fmt) || "tsv".equals(fmt)
            || "json".equals(fmt) || "jsonl".equals(fmt) || "ndjson".equals(fmt);
    }

    // ---- partitionBy ----

    private void savePartitioned(DataFrame df, String fmt, String root) throws Exception {
        java.nio.file.Files.createDirectories(Paths.get(root));

        // Build partition buckets
        java.util.Map<java.util.List<String>, java.util.List<Integer>> buckets = new java.util.TreeMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            java.util.List<String> key = new java.util.ArrayList<>(partitionCols.length);
            for (String c : partitionCols) {
                Object v = df.get(i, c);
                key.add(v == null ? "__HIVE_NULL__" : v.toString());
            }
            buckets.computeIfAbsent(key, k -> new java.util.ArrayList<>()).add(i);
        }

        // Remaining (non-partition) columns
        java.util.Set<String> partSet = new java.util.LinkedHashSet<>();
        for (String c : partitionCols) partSet.add(c);
        String[] remaining = new String[df.columnCount() - partSet.size()];
        int ri = 0;
        for (org.bytedeco.pytorch.dataframe.Column c : df.columns())
            if (!partSet.contains(c.name())) remaining[ri++] = c.name();

        for (java.util.Map.Entry<java.util.List<String>, java.util.List<Integer>> e : buckets.entrySet()) {
            StringBuilder dir = new StringBuilder(root);
            for (int i = 0; i < partitionCols.length; i++) {
                dir.append('/').append(partitionCols[i])
                   .append('=').append(sanitizePartition(e.getKey().get(i)));
            }
            int[] idx = new int[e.getValue().size()];
            for (int k = 0; k < idx.length; k++) idx[k] = e.getValue().get(k);
            java.nio.file.Files.createDirectories(Paths.get(dir.toString()));
            DataFrame part = df.loc(idx).select(remaining);
            String filePath = dir + "/part-" + java.util.UUID.randomUUID().toString().replace("-", "")
                + extensionFor(fmt);
            saveSimple(part, fmt, filePath);
        }
    }

    private static String sanitizePartition(String v) {
        if (v == null) return "__HIVE_NULL__";
        StringBuilder sb = new StringBuilder(v.length());
        for (int i = 0; i < v.length(); i++) {
            char c = v.charAt(i);
            if (c <= 31 || c == '/' || c == ':' || c == '?' || c == '*'
                || c == '<' || c == '>' || c == '"' || c == '|' || c == '\\') {
                sb.append('%');
                if (c < 0x10) sb.append('0');
                sb.append(Integer.toHexString(c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static String extensionFor(String fmt) {
        switch (fmt) {
            case "parquet": return ".parquet";
            case "csv":     return ".csv";
            case "tsv":     return ".tsv";
            case "json":
            case "jsonl":
            case "ndjson": return ".jsonl";
            case "lance":  return ".lance";
            default:       return "";
        }
    }

    private static String detectFromPath(String path) {
        if (path == null) return "csv";
        String lower = path.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".parquet")) return "parquet";
        if (lower.endsWith(".csv")) return "csv";
        if (lower.endsWith(".tsv")) return "tsv";
        if (lower.endsWith(".json")) return "json";
        if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) return "jsonl";
        if (lower.endsWith(".lance")) return "lance";
        return "csv"; // sensible default
    }
}
