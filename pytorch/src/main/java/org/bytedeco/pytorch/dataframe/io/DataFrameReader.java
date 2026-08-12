package org.bytedeco.pytorch.dataframe.io;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.csv.CsvOptions;
import org.bytedeco.pytorch.dataframe.excel.ExcelOptions;
import org.bytedeco.pytorch.dataframe.hdf5.Hdf5Options;
import org.bytedeco.pytorch.dataframe.json.JsonOptions;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Spark-style fluent reader for {@link DataFrame}.
 *
 * <pre>
 *   // explicit format
 *   DataFrame df = DataFrame.read()
 *       .format("parquet")
 *       .option("path", "/data/file.parquet")
 *       .load("/data/file.parquet");
 *
 *   // auto-detect from path extension
 *   DataFrame df = DataFrame.read().load("a.parquet");
 *
 *   // quick helpers
 *   DataFrame df = DataFrame.read().parquet("a.parquet");
 *   DataFrame df = DataFrame.read().csv("a.csv", true, ',');
 *   DataFrame df = DataFrame.read().csv("a.csv", CsvOptions.defaults());
 *   DataFrame df = DataFrame.read().json("a.json");
 *   DataFrame df = DataFrame.read().jsonl("rows.jsonl");
 *   DataFrame df = DataFrame.read().text("a.txt");
 *
 *   // with schema override
 *   DataFrame df = DataFrame.read()
 *       .format("csv")
 *       .schema(Map.of("id", Column.DType.INT64, "name", Column.DType.STRING))
 *       .load("a.csv");
 *
 *   // multiple paths
 *   DataFrame df = DataFrame.read().parquet("p1.parquet", "p2.parquet", "p3.parquet");
 * </pre>
 *
 * <p>Supported short formats match those in {@link DataFrameWriter}.
 */
public final class DataFrameReader {
    private String format;
    private final Map<String, String> options = new LinkedHashMap<>();
    private Map<String, Column.DType> schema;
    private String[] loadPaths;

    public DataFrameReader() {}

    // ---- format / options / schema ----

    /** Set format by short name (case-insensitive). */
    public DataFrameReader format(String name) {
        this.format = name == null ? null : name.trim().toLowerCase(Locale.ROOT);
        return this;
    }

    /** Set format by {@link FormatDetect.Format}. */
    public DataFrameReader format(FormatDetect.Format fmt) {
        if (fmt == null) { this.format = null; return this; }
        this.format = fmt.name().toLowerCase(Locale.ROOT);
        return this;
    }

    /** Spark-style single key/value option. */
    public DataFrameReader option(String key, String value) {
        if (key == null || key.isEmpty()) return this;
        String v = value == null ? "" : value;
        if ("format".equalsIgnoreCase(key)) this.format = v.trim().toLowerCase(Locale.ROOT);
        else options.put(key, v);
        return this;
    }

    /** Boolean option value. */
    public DataFrameReader option(String key, boolean value) {
        return option(key, Boolean.toString(value));
    }

    /** Numeric option value. */
    public DataFrameReader option(String key, long value) {
        return option(key, Long.toString(value));
    }

    /** Numeric option value. */
    public DataFrameReader option(String key, int value) {
        return option(key, Integer.toString(value));
    }

    /** Numeric option value. */
    public DataFrameReader option(String key, double value) {
        return option(key, Double.toString(value));
    }

    /** Bulk options from a Map. */
    public DataFrameReader options(Map<String, String> map) {
        if (map == null) return this;
        for (Map.Entry<String, String> e : map.entrySet()) option(e.getKey(), e.getValue());
        return this;
    }

    /**
     * Varargs key/value pairs:
     * {@code .options("header","true","delimiter","\t","inferSchema","false")}.
     */
    public DataFrameReader options(String... kv) {
        if (kv == null) return this;
        if (kv.length % 2 != 0)
            throw new IllegalArgumentException("options(...) requires even number of args");
        for (int i = 0; i < kv.length; i += 2) option(kv[i], kv[i + 1]);
        return this;
    }

    /**
     * Override or hint the schema for the load. Column names are matched case-sensitively.
     * For formats that infer schema (CSV, JSON, Excel), this replaces the inferred schema.
     * For formats with a fixed schema (Parquet, Arrow), this is ignored.
     */
    public DataFrameReader schema(Map<String, Column.DType> schema) {
        if (schema != null) this.schema = new LinkedHashMap<>(schema);
        return this;
    }

    /**
     * Column-by-column schema builder:
     * {@code .schema("id", Column.DType.INT64, "name", Column.DType.STRING)}.
     * Pairs of (columnName, dtype) are read in order.
     */
    public DataFrameReader schema(String... colNameAndTypes) {
        if (colNameAndTypes == null || colNameAndTypes.length == 0) return this;
        if (colNameAndTypes.length % 2 != 0)
            throw new IllegalArgumentException("schema(...) requires even number of args");
        Map<String, Column.DType> m = new LinkedHashMap<>();
        for (int i = 0; i < colNameAndTypes.length; i += 2) {
            String col = colNameAndTypes[i];
            Column.DType dt;
            try {
                dt = Column.DType.valueOf(colNameAndTypes[i + 1]);
            } catch (Exception e) {
                throw new IllegalArgumentException("Invalid DType at arg " + (i + 2)
                    + ": " + colNameAndTypes[i + 1]);
            }
            m.put(col, dt);
        }
        return schema(m);
    }

    public String format() { return format; }
    public Map<String, String> options() { return new LinkedHashMap<>(options); }
    public Map<String, Column.DType> schema() { return schema == null ? null : new LinkedHashMap<>(schema); }

    // ---- load() ----

    /**
     * Load from a single path. When {@link #format(String)} was set, that format is used.
     * Otherwise the format is auto-detected from the path extension (and content sniff).
     */
    public DataFrame load(String path) throws Exception {
        if (path == null) throw new IllegalArgumentException("path required");
        this.loadPaths = new String[]{path};
        return loadImpl(path, null);
    }

    /** Load from multiple paths (unioned/concatenated). */
    public DataFrame load(String... paths) throws Exception {
        if (paths == null || paths.length == 0)
            throw new IllegalArgumentException("at least one path required");
        if (paths.length == 1) return load(paths[0]);
        this.loadPaths = paths;
        return loadImplMultiple(paths, null);
    }

    /** Load from Path objects. */
    public DataFrame load(Path... paths) throws Exception {
        String[] spaths = new String[paths.length];
        for (int i = 0; i < paths.length; i++) spaths[i] = paths[i].toString();
        return load(spaths);
    }

    /** Load when format was already set (deferred path). */
    public DataFrame load() throws Exception {
        if (loadPaths == null || loadPaths.length == 0)
            throw new IllegalStateException("path required: call load(path) or use a quick helper");
        return loadPaths.length == 1
            ? loadImpl(loadPaths[0], null)
            : loadImplMultiple(loadPaths, null);
    }

    // ---- quick helpers (Spark-style format shortcuts) ----

    public DataFrame parquet(String path)     throws Exception { format("parquet"); return load(path); }
    public DataFrame csv(String path)         throws Exception { format("csv");     return load(path); }
    /** CSV with explicit header + delimiter. */
    public DataFrame csv(String path, boolean hasHeader, char delimiter) throws Exception {
        format("csv");
        CsvOptions opt = CsvOptions.builder()
            .header(hasHeader)
            .delimiter(delimiter)
            .build();
        return loadWithOpts(path, opt);
    }
    /** CSV with explicit CsvOptions. */
    public DataFrame csv(String path, CsvOptions opts) throws Exception {
        format("csv");
        return loadWithOpts(path, opts);
    }
    public DataFrame tsv(String path)         throws Exception { format("tsv");     return load(path); }
    public DataFrame json(String path)        throws Exception { format("json");    return load(path); }
    public DataFrame jsonl(String path)       throws Exception { format("jsonl");   return load(path); }
    public DataFrame ndjson(String path)      throws Exception { format("jsonl");   return load(path); }
    public DataFrame arrow(String path)        throws Exception { format("arrow");   return load(path); }
    public DataFrame feather(String path)     throws Exception { format("feather"); return load(path); }
    public DataFrame ipc(String path)         throws Exception { format("ipc");     return load(path); }
    public DataFrame pickle(String path)      throws Exception { format("pickle");  return load(path); }
    public DataFrame pkl(String path)         throws Exception { format("pickle");  return load(path); }
    public DataFrame npz(String path)         throws Exception { format("npz");     return load(path); }
    public DataFrame npy(String path)         throws Exception { format("npy");     return load(path); }
    public DataFrame safetensors(String path) throws Exception { format("safetensors"); return load(path); }
    public DataFrame gguf(String path)        throws Exception { format("gguf");    return load(path); }
    public DataFrame excel(String path)       throws Exception { format("excel");   return load(path); }
    public DataFrame hdf5(String path)        throws Exception { format("hdf5");    return load(path); }
    public DataFrame hdf(String path)         throws Exception { format("hdf5");    return load(path); }
    public DataFrame hdf(String path, String key) throws Exception {
        options("key", key); format("hdf5"); return load(path);
    }
    public DataFrame avro(String path)        throws Exception { format("avro");    return load(path); }
    public DataFrame orc(String path)         throws Exception { format("orc");     return load(path); }
    public DataFrame lance(String path)       throws Exception { format("lance");    return load(path); }
    public DataFrame toml(String path)        throws Exception { format("toml");    return load(path); }
    public DataFrame bin(String path)         throws Exception { format("bin");     return load(path); }

    /**
     * Plain text → single-column DataFrame with column {@code "value"}.
     * Lines are trimmed; blank lines are skipped by default.
     */
    public DataFrame text(String path) throws Exception {
        return DataFrame.readTextFolder(path);
    }

    // ---- internal helpers ----

    private DataFrame loadImpl(String path, FormatDetect.Format preResolved) throws Exception {
        String fmt = format;
        if (fmt == null || fmt.isEmpty()) {
            if (preResolved != null) fmt = preResolved.name().toLowerCase(Locale.ROOT);
            else fmt = FormatDetect.detectRobust(path).name().toLowerCase(Locale.ROOT);
        }
        switch (fmt) {
            case "csv":         return readCsv(path, buildCsvOptions());
            case "tsv":         return readCsv(path, buildTsvOptions());
            case "json":        return readJson(path, buildJsonOptions(JsonOptions.Orient.RECORDS));
            case "jsonl":
            case "ndjson":      return readJson(path, buildJsonOptions(JsonOptions.Orient.LINES));
            case "parquet":     return readParquet(path);
            case "arrow":
            case "feather":
            case "ipc":         return DataFrame.readArrow(path);
            case "pickle":
            case "pkl":         return readPickle(path);
            case "npz":         return DataFrame.readNpz(path);
            case "npy":         return DataFrame.readNpy(path);
            case "safetensors": return DataFrame.readSafetensors(path);
            case "gguf":        return DataFrame.readGguf(path);
            case "excel":
            case "xlsx":
            case "xls":         return readExcel(path);
            case "hdf5":
            case "hdf":         return readHdf(path);
            case "avro":        return readAvro(path);
            case "orc":         return readOrc(path);
            case "lance":       return DataFrame.readLance(path);
            case "toml":        return readToml(path);
            case "bin":
            case "binary":      return readBin(path);
            default:
                throw new IllegalArgumentException("Unknown read format: '" + fmt + "'");
        }
    }

    private DataFrame loadImplMultiple(String[] paths, FormatDetect.Format preResolved) throws Exception {
        String fmt = format;
        if (fmt == null || fmt.isEmpty()) {
            if (preResolved != null) fmt = preResolved.name().toLowerCase(Locale.ROOT);
            else fmt = FormatDetect.detectRobust(paths[0]).name().toLowerCase(Locale.ROOT);
        }
        // For formats that support concatenation, union all paths.
        // For formats with different column semantics, read and vstack.
        DataFrame result = null;
        for (String p : paths) {
            DataFrame part;
            if (fmt.equals("csv") || fmt.equals("tsv")) {
                part = loadImpl(p, FormatDetect.Format.valueOf(fmt.equals("tsv") ? "TSV" : "CSV"));
            } else {
                part = loadImpl(p, null);
            }
            if (result == null) result = part;
            else result = DataFrame.vstack(result, part);
        }
        return result;
    }

    private DataFrame loadWithOpts(String path, CsvOptions baseOpts) throws Exception {
        CsvOptions opts = baseOpts;
        if (schema != null) {
            CsvOptions.Builder b = CsvOptions.builder()
                .header(baseOpts.header())
                .delimiter(baseOpts.delimiter())
                .quote(baseOpts.quote())
                .escape(baseOpts.escape())
                .charset(baseOpts.charset())
                .comment(baseOpts.comment())
                .skipRows(baseOpts.skipRows())
                .maxRows(baseOpts.maxRows())
                .inferSchema(false)
                .inferSampleSize(baseOpts.inferSampleSize())
                .strict(baseOpts.strict())
                .typeHeader(baseOpts.typeHeader())
                .quoteMode(baseOpts.quoteMode())
                .writeNullToken(baseOpts.writeNullToken())
                .stripBom(baseOpts.stripBom())
                .schema(schema);
            for (String n : baseOpts.nullValues()) b.addNullValue(n);
            opts = b.build();
        }
        return DataFrame.readCsv(path, opts);
    }

    // ---- per-format options builder ----

    private CsvOptions buildCsvOptions() {
        CsvOptions.Builder b = CsvOptions.builder();
        applyCsvOptions(b);
        if (schema != null) {
            b.inferSchema(false).schema(schema);
        }
        return b.build();
    }

    private CsvOptions buildTsvOptions() {
        CsvOptions.Builder b = CsvOptions.builder().delimiter('\t');
        applyCsvOptions(b);
        if (schema != null) {
            b.inferSchema(false).schema(schema);
        }
        return b.build();
    }

    private void applyCsvOptions(CsvOptions.Builder b) {
        String header = options.get("header");
        if (header != null) b.header(Boolean.parseBoolean(header));
        String delim = options.get("delimiter");
        if (delim != null && !delim.isEmpty()) b.delimiter(delim.charAt(0));
        String sep = options.get("sep");
        if (sep != null && !sep.isEmpty()) b.delimiter(sep.charAt(0));
        String quote = options.get("quote");
        if (quote != null && !quote.isEmpty()) b.quote(quote.charAt(0));
        String charset = options.get("charset");
        if (charset != null && !charset.isEmpty()) b.charset(Charset.forName(charset));
        String nulls = options.get("nullValues");
        if (nulls != null && !nulls.isEmpty()) b.nullValues(nulls.split(","));
        String infer = options.get("inferSchema");
        if (infer != null) b.inferSchema(Boolean.parseBoolean(infer));
        String sample = options.get("inferSampleSize");
        if (sample != null) b.inferSampleSize(Integer.parseInt(sample));
        String maxRows = options.get("maxRows");
        if (maxRows != null) b.maxRows(Integer.parseInt(maxRows));
        String skipRows = options.get("skipRows");
        if (skipRows != null) b.skipRows(Integer.parseInt(skipRows));
        String strict = options.get("strict");
        if (strict != null) b.strict(Boolean.parseBoolean(strict));
        String typeHeader = options.get("typeHeader");
        if (typeHeader != null) b.typeHeader(Boolean.parseBoolean(typeHeader));
        String quoteMode = options.get("quoteMode");
        if (quoteMode != null) b.quoteMode(CsvOptions.QuoteMode.valueOf(quoteMode.toUpperCase()));
        String stripBom = options.get("stripBom");
        if (stripBom != null) b.stripBom(Boolean.parseBoolean(stripBom));
    }

    private JsonOptions buildJsonOptions(JsonOptions.Orient orient) {
        JsonOptions.Builder b = JsonOptions.builder().orient(orient);
        applyJsonOptions(b);
        if (schema != null) {
            b.inferSchema(false).schema(schema);
        }
        return b.build();
    }

    private void applyJsonOptions(JsonOptions.Builder b) {
        String orient = options.get("orient");
        if (orient != null) {
            try { b.orient(JsonOptions.Orient.valueOf(orient.toUpperCase())); }
            catch (Exception ignored) { /* default */ }
        }
        String flatten = options.get("flatten");
        if (flatten != null) b.flatten(Boolean.parseBoolean(flatten));
        String infer = options.get("inferSchema");
        if (infer != null) b.inferSchema(Boolean.parseBoolean(infer));
        String charset = options.get("charset");
        if (charset != null && !charset.isEmpty()) b.charset(Charset.forName(charset));
        String pretty = options.get("pretty");
        if (pretty != null) b.pretty(Boolean.parseBoolean(pretty));
        String maxRows = options.get("maxRows");
        if (maxRows != null) b.maxRows(Integer.parseInt(maxRows));
        String skipRows = options.get("skipRows");
        if (skipRows != null) b.skipRows(Integer.parseInt(skipRows));
        String recordPath = options.get("recordPath");
        if (recordPath != null && !recordPath.isEmpty()) b.recordPath(recordPath);
        String explode = options.get("explodeArrays");
        if (explode != null) b.explodeArrays(Boolean.parseBoolean(explode));
    }

    // ---- per-format dispatch ----

    private DataFrame readCsv(String path, CsvOptions opts) throws Exception {
        return DataFrame.readCsv(path, opts);
    }

    private DataFrame readJson(String path, JsonOptions opts) throws Exception {
        return DataFrame.readJson(path, opts);
    }

    private DataFrame readParquet(String path) throws Exception {
        return DataFrame.readParquet(path);
    }

    private DataFrame readPickle(String path) throws Exception {
        return DataFrame.readPickle(path);
    }

    private DataFrame readExcel(String path) throws Exception {
        ExcelOptions.Builder b = ExcelOptions.builder();
        String sheet = options.get("sheetName");
        if (sheet != null && !sheet.isEmpty()) b.sheet(sheet);
        String sheetIdx = options.get("sheetIndex");
        if (sheetIdx != null) b.sheetIndex(Integer.parseInt(sheetIdx));
        String header = options.get("header");
        if (header != null) b.header(Boolean.parseBoolean(header));
        String maxRows = options.get("maxRows");
        if (maxRows != null) b.maxRows(Integer.parseInt(maxRows));
        String skipRows = options.get("skipRows");
        if (skipRows != null) b.skipRows(Integer.parseInt(skipRows));
        String infer = options.get("inferSchema");
        if (infer != null) b.inferSchema(Boolean.parseBoolean(infer));
        return DataFrame.readExcel(path, b.build());
    }

    private DataFrame readHdf(String path) throws Exception {
        String key = options.getOrDefault("key", options.getOrDefault("group", "/df"));
        return DataFrame.readHdf(path, key);
    }

    private DataFrame readAvro(String path) throws Exception {
        return DataFrame.readAvro(path);
    }

    private DataFrame readOrc(String path) throws Exception {
        return DataFrame.readOrc(path);
    }

    private DataFrame readToml(String path) throws Exception {
        return TomlReader.read(path);
    }

    private DataFrame readBin(String path) throws Exception {
        return BinReader.read(path);
    }
}
