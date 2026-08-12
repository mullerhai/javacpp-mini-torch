package org.bytedeco.pytorch.utils.daft;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.daft.engine.ExecutionConfig;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Daft-style fluent reader for {@link DaftDataFrame}.
 *
 * <p>Mirrors Python Daft's {@code da.from_parquet(...)} / {@code da.from_csv(...)}
 * surface and also the Spark-style API surface.
 *
 * <pre>{@code
 *   // Daft-style quick helpers (preferred)
 *   DaftDataFrame df = DaftDataFrame.read().parquet("/data/*.parquet");
 *   DaftDataFrame df = DaftDataFrame.read().csv("/data/file.csv");
 *   DaftDataFrame df = DaftDataFrame.read().jsonl("/data/rows.jsonl");
 *   DaftDataFrame df = DaftDataFrame.read().json("/data/file.json");
 *   DaftDataFrame df = DaftDataFrame.read().text("/data/text.txt");
 *
 *   // Spark-style
 *   DaftDataFrame df = DaftDataFrame.read().format("parquet").load("/data/file.parquet");
 *   DaftDataFrame df = DaftDataFrame.read().option("header", "true").csv("/data/file.csv");
 *
 *   // then chain Daft transforms
 *   df.filter(col("age").gt(18)).select("name", "age").limit(1000).collect();
 * }</pre>
 *
 * <p>All load methods return a lazy {@link DaftDataFrame} that is not yet
 * materialised. Call {@code collect()} or {@code show(n)} to execute the pipeline.
 */
public final class DaftDataFrameReader {

    private String format;
    private final Map<String, String> options = new LinkedHashMap<>();
    private String[] loadPaths;

    DaftDataFrameReader() {}

    // ---- format / options ----

    /** Set format by short name (case-insensitive): parquet, csv, json, jsonl, text, etc. */
    public DaftDataFrameReader format(String name) {
        this.format = name == null ? null : name.trim().toLowerCase(Locale.ROOT);
        return this;
    }

    /** Spark-style key/value option. */
    public DaftDataFrameReader option(String key, String value) {
        if (key == null || key.isEmpty()) return this;
        if ("format".equalsIgnoreCase(key)) this.format = value == null ? null : value.trim().toLowerCase(Locale.ROOT);
        else this.options.put(key, value == null ? "" : value);
        return this;
    }

    /** Boolean option. */
    public DaftDataFrameReader option(String key, boolean value) {
        return option(key, Boolean.toString(value));
    }

    /** Int option. */
    public DaftDataFrameReader option(String key, int value) {
        return option(key, Integer.toString(value));
    }

    /** Bulk options from Map. */
    public DaftDataFrameReader options(Map<String, String> map) {
        if (map == null) return this;
        map.forEach(this::option);
        return this;
    }

    /** Varargs key/value pairs. */
    public DaftDataFrameReader options(String... kv) {
        if (kv == null || kv.length == 0) return this;
        if (kv.length % 2 != 0)
            throw new IllegalArgumentException("options(...) requires even number of args");
        for (int i = 0; i < kv.length; i += 2) option(kv[i], kv[i + 1]);
        return this;
    }

    // ---- load ----

    /**
     * Load from a single path. Format is resolved from explicit {@link #format(String)},
     * then from the path extension.
     */
    public DaftDataFrame load(String path) throws Exception {
        if (path == null) throw new IllegalArgumentException("path required");
        this.loadPaths = new String[]{path};
        return loadImpl(path);
    }

    /** Load from multiple paths (union/concatenation). */
    public DaftDataFrame load(String... paths) throws Exception {
        if (paths == null || paths.length == 0)
            throw new IllegalArgumentException("at least one path required");
        if (paths.length == 1) return load(paths[0]);
        this.loadPaths = paths;
        return loadImplMultiple(paths);
    }

    /** No-arg load when path was set via a quick helper. */
    public DaftDataFrame load() throws Exception {
        if (loadPaths == null || loadPaths.length == 0)
            throw new IllegalStateException("path required: call load(path) or use a quick helper");
        return loadPaths.length == 1 ? loadImpl(loadPaths[0]) : loadImplMultiple(loadPaths);
    }

    // ---- quick helpers (Daften + Spark-style) ----

    /** Read one or more Parquet files. */
    public DaftDataFrame parquet(String path) throws Exception { format("parquet"); return load(path); }
    /** Read one or more Parquet files (varargs). */
    public DaftDataFrame parquet(String... paths) throws Exception { format("parquet"); return load(paths); }

    /** Read a CSV file. */
    public DaftDataFrame csv(String path) throws Exception { format("csv"); return load(path); }
    /** Read CSV files (varargs). */
    public DaftDataFrame csv(String... paths) throws Exception { format("csv"); return load(paths); }

    /** Read a JSON file (array of records). */
    public DaftDataFrame json(String path) throws Exception { format("json"); return load(path); }
    /** Read JSON files (varargs). */
    public DaftDataFrame json(String... paths) throws Exception { format("json"); return load(paths); }

    /** Read a JSONL / NDJSON file (one JSON object per line). */
    public DaftDataFrame jsonl(String path) throws Exception { format("jsonl"); return load(path); }
    /** Read JSONL files (varargs). */
    public DaftDataFrame jsonl(String... paths) throws Exception { format("jsonl"); return load(paths); }

    /** Read an Arrow / Feather / IPC file. */
    public DaftDataFrame arrow(String path) throws Exception { format("arrow"); return load(path); }

    /** Read a plain-text file (one line per row, single "value" column). */
    public DaftDataFrame text(String path) throws Exception { format("text"); return load(path); }

    // ---- internal dispatch ----

    private DaftDataFrame loadImpl(String path) throws Exception {
        String fmt = resolveFormat(path);
        switch (fmt) {
            case "parquet":   return DaftDataFrame.fromParquet(path, options);
            case "csv":       return DaftDataFrame.fromCsv(path, options);
            case "json":      return DaftDataFrame.fromJson(path, options);
            case "jsonl":
            case "ndjson":    return DaftDataFrame.fromJson(path, options);  // JSONL also uses fromJson
            case "arrow":
            case "feather":
            case "ipc":       return DaftDataFrame.fromArrow(path);
            case "text":       return DaftDataFrame.fromText(path);
            default:
                // Fallback: try parquet first, then csv
                try { return DaftDataFrame.fromParquet(path, options); }
                catch (Exception e) { return DaftDataFrame.fromCsv(path, options); }
        }
    }

    private DaftDataFrame loadImplMultiple(String[] paths) throws Exception {
        DaftDataFrame result = null;
        for (String p : paths) {
            DaftDataFrame part = loadImpl(p);
            if (result == null) result = part;
            else result = result.concat(part);
        }
        return result != null ? result : loadImpl(paths[0]);
    }

    private String resolveFormat(String path) {
        if (format != null && !format.isEmpty()) return format;
        if (path == null) return "csv";
        String lower = path.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".parquet")) return "parquet";
        if (lower.endsWith(".csv")) return "csv";
        if (lower.endsWith(".tsv")) return "csv";  // TSV handled as CSV with tab delimiter
        if (lower.endsWith(".json")) return "json";
        if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) return "jsonl";
        if (lower.endsWith(".arrow") || lower.endsWith(".feather") || lower.endsWith(".ipc")) return "arrow";
        if (lower.endsWith(".txt")) return "text";
        return "csv"; // sensible default
    }
}
