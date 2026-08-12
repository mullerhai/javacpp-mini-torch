package org.bytedeco.pytorch.dataframe.io;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.csv.CsvOptions;
import org.bytedeco.pytorch.dataframe.json.JsonOptions;
import org.bytedeco.pytorch.dataframe.pickle.PickleOptions;
import org.bytedeco.pytorch.dataframe.excel.ExcelOptions;
import org.bytedeco.pytorch.dataframe.hdf5.Hdf5Options;
import org.bytedeco.pytorch.data.avro.AvroOptions;
import org.bytedeco.pytorch.data.orc.OrcOptions;

import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Spark-style fluent writer for {@link DataFrame}.
 *
 * <pre>
 *   // explicit format
 *   df.write()
 *     .format("parquet")
 *     .option("compression", "snappy")
 *     .mode("overwrite")
 *     .save("/data/out");
 *
 *   // format inferred from path extension
 *   df.write().option("header", "true").save("out.csv");
 *
 *   // quick format helpers
 *   df.write().parquet("out.parquet");
 *   df.write().json("out.json");
 *   df.write().jsonl("out.jsonl");
 *
 *   // partitionBy() for parquet / orc / jsonl / csv
 *   df.write().format("parquet").partitionBy("year","month").save("/data/out");
 * </pre>
 *
 * <p>Supported short formats: {@code parquet}, {@code csv}, {@code tsv}, {@code json},
 * {@code jsonl}/{@code ndjson}, {@code arrow}/{@code feather}/{@code ipc}, {@code pickle}/{@code pkl},
 * {@code npz}, {@code npy}, {@code safetensors}, {@code gguf}, {@code excel}/{@code xlsx},
 * {@code hdf5}/{@code hdf}, {@code avro}, {@code orc}, {@code lance}.
 */
public final class DataFrameWriter {
    private final DataFrame df;
    private String format;
    private final Map<String, String> options = new LinkedHashMap<>();
    private SaveMode mode = SaveMode.OVERWRITE;
    private String[] partitionCols;
    private String deferredPath;

    public DataFrameWriter(DataFrame df) {
        if (df == null) throw new NullPointerException("df");
        this.df = df;
    }

    // ---- format / options / mode / partition ----

    /** Set format by short name (case-insensitive). */
    public DataFrameWriter format(String name) {
        this.format = name == null ? null : name.trim().toLowerCase(Locale.ROOT);
        return this;
    }

    /** Set format by {@link FormatDetect.Format}. */
    public DataFrameWriter format(FormatDetect.Format fmt) {
        if (fmt == null) { this.format = null; return this; }
        this.format = fmt.name().toLowerCase(Locale.ROOT);
        return this;
    }

    /** Spark-style single key/value option. */
    public DataFrameWriter option(String key, String value) {
        if (key == null || key.isEmpty()) return this;
        String v = value == null ? "" : value;
        // capture format / mode / path semantics for cheap accessors
        if ("format".equalsIgnoreCase(key)) this.format = v.trim().toLowerCase(Locale.ROOT);
        else if ("mode".equalsIgnoreCase(key)) this.mode = SaveMode.fromString(v);
        else options.put(key, v);
        return this;
    }

    /** Spark-style single key/value option (boolean value as "true"/"false"). */
    public DataFrameWriter option(String key, boolean value) {
        return option(key, Boolean.toString(value));
    }

    /** Spark-style single key/value option (numeric value). */
    public DataFrameWriter option(String key, long value) {
        return option(key, Long.toString(value));
    }

    /** Spark-style single key/value option (numeric value). */
    public DataFrameWriter option(String key, int value) {
        return option(key, Integer.toString(value));
    }

    /** Spark-style single key/value option (numeric value). */
    public DataFrameWriter option(String key, double value) {
        return option(key, Double.toString(value));
    }

    /** Bulk options from a Map (later entries overwrite earlier). */
    public DataFrameWriter options(Map<String, String> map) {
        if (map == null) return this;
        for (Map.Entry<String, String> e : map.entrySet()) option(e.getKey(), e.getValue());
        return this;
    }

    /**
     * Spark-style varargs of key/value pairs:
     * {@code .options("header","true","delimiter","\t","charset","UTF-8")}.
     */
    public DataFrameWriter options(String... kv) {
        if (kv == null) return this;
        if (kv.length % 2 != 0)
            throw new IllegalArgumentException("options(...) requires even number of args");
        for (int i = 0; i < kv.length; i += 2) option(kv[i], kv[i + 1]);
        return this;
    }

    /** Set save mode (overwrite / append / ignore / error[ifexists]). */
    public DataFrameWriter mode(SaveMode m) {
        if (m != null) this.mode = m;
        return this;
    }

    /** Set save mode by Spark-style string. */
    public DataFrameWriter mode(String name) {
        this.mode = SaveMode.fromString(name);
        return this;
    }

    /**
     * Declare partitioning columns. When set, the writer creates a directory at
     * {@code path} and partitions rows into subdirectories by column values.
     * Supported by parquet, csv, tsv, json, jsonl, arrow, hdf5, orc, avro, lance.
     */
    public DataFrameWriter partitionBy(String... cols) {
        if (cols == null) { this.partitionCols = null; return this; }
        if (cols.length == 0) { this.partitionCols = null; return this; }
        Set<String> seen = new LinkedHashSet<>();
        for (String c : cols) if (c != null) seen.add(c);
        this.partitionCols = seen.toArray(new String[0]);
        return this;
    }

    public SaveMode mode() { return mode; }
    public Map<String, String> options() { return new LinkedHashMap<>(options); }
    public String format() { return format; }

    /** Path accessor — populated after the first {@link #save(String)} or {@link #save()} call. */
    public String path() { return deferredPath; }

    // ---- save() ----

    /**
     * Save to {@code path}. When {@link #format(String)} was set, that format is used.
     * Otherwise the format is auto-detected from the path extension (and content sniff).
     */
    public void save(String path) throws Exception {
        saveImpl(path, null);
    }

    /**
     * Save when the format was inferred from the path. Equivalent to {@link #save(String)}.
     */
    public void save() throws Exception {
        if (deferredPath == null)
            throw new IllegalStateException("path required: call save(path) or use a quick helper");
        saveImpl(deferredPath, null);
    }

    // ---- quick helpers (Spark-style format shortcuts) ----

    public void parquet(String path) throws Exception { format("parquet").save(path); }
    public void csv(String path)    throws Exception { format("csv").save(path); }
    public void tsv(String path)    throws Exception { format("tsv").save(path); }
    public void json(String path)   throws Exception { format("json").save(path); }
    public void jsonl(String path)  throws Exception { format("jsonl").save(path); }
    public void arrow(String path)  throws Exception { format("arrow").save(path); }
    public void feather(String path)throws Exception { format("feather").save(path); }
    public void ipc(String path)    throws Exception { format("ipc").save(path); }
    public void pickle(String path) throws Exception { format("pickle").save(path); }
    public void pkl(String path)    throws Exception { format("pickle").save(path); }
    public void npz(String path)    throws Exception { format("npz").save(path); }
    public void npy(String path)    throws Exception { format("npy").save(path); }
    public void safetensors(String path) throws Exception { format("safetensors").save(path); }
    // gguf is read-only in this codebase; use duckdb-based writers for export.
    public void excel(String path)  throws Exception { format("excel").save(path); }
    public void xlsx(String path)   throws Exception { format("excel").save(path); }
    public void hdf5(String path)   throws Exception { format("hdf5").save(path); }
    public void hdf(String path)    throws Exception { format("hdf5").save(path); }
    public void avro(String path)   throws Exception { format("avro").save(path); }
    public void orc(String path)    throws Exception { format("orc").save(path); }
    public void lance(String path)  throws Exception { format("lance").save(path); }

    // ---- core dispatch ----

    private void saveImpl(String path, FormatDetect.Format preResolved) throws Exception {
        if (path == null) throw new IllegalArgumentException("path required");
        this.deferredPath = path;

        // SaveMode preCheck
        if (!mode.preCheck(path)) return; // IGNORE + exists → no-op

        // Resolve format
        String fmt = format;
        if (fmt == null || fmt.isEmpty()) {
            if (preResolved != null) fmt = preResolved.name().toLowerCase(Locale.ROOT);
            else fmt = FormatDetect.detectRobust(path).name().toLowerCase(Locale.ROOT);
        }

        // partitionBy support
        if (partitionCols != null && partitionCols.length > 0 && !supportsPartition(fmt)) {
            throw new UnsupportedOperationException(
                "partitionBy() is not supported for format '" + fmt + "'");
        }

        // Special path: partitionBy → directory layout
        if (partitionCols != null && partitionCols.length > 0) {
            savePartitioned(fmt, path);
            return;
        }

        // Simple format dispatch
        switch (fmt) {
            case "parquet":     writeParquet(path); break;
            case "csv":         writeCsv(path, CsvOptions.defaults()); break;
            case "tsv":         writeCsv(path, CsvOptions.tsv()); break;
            case "json":        writeJson(path, JsonOptions.defaults()); break;
            case "jsonl":
            case "ndjson":      writeJsonl(path, JsonOptions.lines()); break;
            case "arrow":
            case "feather":
            case "ipc":         df.writeArrow(path); break;
            case "pickle":
            case "pkl":         writePickle(path, PickleOptions.defaults()); break;
            case "npz":         df.toNpz(path); break;
            case "npy":         df.toNumpy(path); break;
            case "safetensors": df.writeSafetensors(path); break;
            case "excel":
            case "xlsx":
            case "xls":         writeExcel(path); break;
            case "hdf5":
            case "hdf":         writeHdf5(path); break;
            case "avro":        writeAvro(path); break;
            case "orc":         writeOrc(path); break;
            case "lance":       writeLance(path); break;
            default:
                throw new IllegalArgumentException("Unknown write format: '" + fmt + "'");
        }
    }

    private boolean supportsPartition(String fmt) {
        switch (fmt) {
            case "parquet": case "csv": case "tsv": case "json": case "jsonl":
            case "ndjson": case "arrow": case "feather": case "ipc":
            case "hdf5": case "hdf": case "orc": case "avro": case "lance":
                return true;
            default:
                return false;
        }
    }

    // ---- per-format option appliers ----

    private CsvOptions csvOptsWithOptions(CsvOptions base) {
        CsvOptions.Builder b = CsvOptions.builder()
            .header(base.header())
            .delimiter(base.delimiter())
            .quote(base.quote())
            .escape(base.escape())
            .charset(base.charset())
            .comment(base.comment())
            .skipRows(base.skipRows())
            .maxRows(base.maxRows())
            .inferSchema(base.inferSchema())
            .inferSampleSize(base.inferSampleSize())
            .strict(base.strict())
            .typeHeader(base.typeHeader())
            .quoteMode(base.quoteMode())
            .writeNullToken(base.writeNullToken())
            .stripBom(base.stripBom())
            .columnNames(base.columnNames())
            .schema(base.schema());
        for (String n : base.nullValues()) b.addNullValue(n);
        applyCsvOptions(b);
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
        if (nulls != null) b.nullValues(nulls.split(","));
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
        String writeNull = options.get("writeNullToken");
        if (writeNull != null) b.writeNullToken(writeNull);
        String stripBom = options.get("stripBom");
        if (stripBom != null) b.stripBom(Boolean.parseBoolean(stripBom));
    }

    private void writeCsv(String path, CsvOptions base) throws Exception {
        df.toCsv(path, csvOptsWithOptions(base));
    }

    private JsonOptions jsonOptsWith(JsonOptions base) {
        JsonOptions.Builder b = JsonOptions.builder()
            .orient(base.orient())
            .flatten(base.flatten())
            .flattenSeparator(base.flattenSeparator())
            .inferSchema(base.inferSchema())
            .inferSampleSize(base.inferSampleSize())
            .strict(base.strict())
            .maxRows(base.maxRows())
            .skipRows(base.skipRows())
            .keepNestedAsJson(base.keepNestedAsJson())
            .explodeArrays(base.explodeArrays())
            .recordPath(base.recordPath())
            .metaPaths(base.metaPaths())
            .pretty(base.pretty())
            .writeNulls(base.writeNulls())
            .dateFormat(base.dateFormat())
            .charset(base.charset())
            .stripBom(base.stripBom())
            .duplicateKeyPolicy(base.duplicateKeyPolicy())
            .allowComments(base.allowComments())
            .allowTrailingCommas(base.allowTrailingCommas())
            .allowMultiLineJsonl(base.allowMultiLineJsonl())
            .linesCommentPrefix(base.linesCommentPrefix())
            .dateUnit(base.dateUnit());
        for (String n : base.nullValues()) b.addNullValue(n);
        applyJsonOptions(b);
        return b.build();
    }

    private void applyJsonOptions(JsonOptions.Builder b) {
        String orient = options.get("orient");
        if (orient != null) b.orient(JsonOptions.Orient.valueOf(orient.toUpperCase()));
        String flatten = options.get("flatten");
        if (flatten != null) b.flatten(Boolean.parseBoolean(flatten));
        String infer = options.get("inferSchema");
        if (infer != null) b.inferSchema(Boolean.parseBoolean(infer));
        String charset = options.get("charset");
        if (charset != null) b.charset(Charset.forName(charset));
        String pretty = options.get("pretty");
        if (pretty != null) b.pretty(Boolean.parseBoolean(pretty));
        String writeNulls = options.get("writeNulls");
        if (writeNulls != null) b.writeNulls(Boolean.parseBoolean(writeNulls));
        String maxRows = options.get("maxRows");
        if (maxRows != null) b.maxRows(Integer.parseInt(maxRows));
    }

    private void writeJson(String path, JsonOptions base) throws Exception {
        df.toJson(path, jsonOptsWith(base));
    }

    private void writeJsonl(String path, JsonOptions base) throws Exception {
        df.toJsonl(path, jsonOptsWith(base));
    }

    private void writeParquet(String path) throws Exception {
        df.writeParquet(path);
    }

    private void writePickle(String path, PickleOptions base) throws Exception {
        PickleOptions opt = base;
        String records = options.get("records");
        if (records != null && Boolean.parseBoolean(records)) {
            opt = PickleOptions.records();
        }
        df.toPickle(path, opt);
    }

    private void writeExcel(String path) throws Exception {
        ExcelOptions opt = ExcelOptions.defaults();
        String sheetName = options.get("sheetName");
        String writeSheetName = options.get("writeSheetName");
        if ((sheetName != null && !sheetName.isEmpty())
            || (writeSheetName != null && !writeSheetName.isEmpty())
            || options.containsKey("freezeHeader")) {
            ExcelOptions.Builder b = ExcelOptions.builder()
                .sheet(sheetName)
                .header(opt.header())
                .skipRows(opt.skipRows())
                .maxRows(opt.maxRows())
                .inferSchema(opt.inferSchema())
                .inferSampleSize(opt.inferSampleSize())
                .strict(opt.strict())
                .evaluateFormulas(opt.evaluateFormulas())
                .dateAsLocalDate(opt.dateAsLocalDate())
                .writeNullToken(opt.writeNullToken())
                .freezeHeader(opt.freezeHeader())
                .writeSheetName(writeSheetName != null && !writeSheetName.isEmpty()
                    ? writeSheetName
                    : opt.writeSheetName());
            opt = b.build();
        }
        df.toExcel(path, opt);
    }

    private void writeHdf5(String path) throws Exception {
        String key = options.getOrDefault("key", options.getOrDefault("group", "/df"));
        Hdf5Options opt = Hdf5Options.defaults();
        df.toHdf(path, key, opt);
    }

    private void writeAvro(String path) throws Exception {
        AvroOptions.Builder b = AvroOptions.builder();
        String codec = options.get("codec");
        if (codec != null && !codec.isEmpty()) {
            try {
                b.codec(AvroOptions.Codec.valueOf(codec.toUpperCase()));
            } catch (Exception ignored) { /* fall back to defaults */ }
        }
        df.toAvro(path, b.build());
    }

    private void writeOrc(String path) throws Exception {
        OrcOptions opt = OrcOptions.defaults();
        String compression = options.get("compression");
        if (compression != null && !compression.isEmpty()) {
            try {
                OrcOptions.Compress compress = OrcOptions.Compress.valueOf(compression.toUpperCase());
                opt = OrcOptions.builder()
                    .compress(compress)
                    .build();
            } catch (Exception ignored) { /* fall back to defaults */ }
        }
        df.toOrc(path, opt);
    }

    private void writeLance(String path) throws Exception {
        String[] vectors = null;
        String v = options.get("vectorCols");
        if (v != null && !v.isEmpty()) vectors = v.split(",");
        if (vectors != null) df.writeLance(path, vectors);
        else df.writeLance(path);
    }

    // ---- partitionBy implementation ----

    private void savePartitioned(String fmt, String root) throws Exception {
        // Build distinct partition-value tuples and their row index lists.
        // For simplicity we partition by a single string-encoded key list.
        // For multi-column, the directory layout mirrors Spark: key=v1/key=v2/...
        Map<List<String>, List<Integer>> buckets = new TreeMap<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            List<String> key = new ArrayList<>(partitionCols.length);
            for (String c : partitionCols) {
                Object v = df.get(i, c);
                key.add(sanitizePartition(v));
            }
            buckets.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }

        // Prepare the partition column projection (without duplicate names)
        Set<String> partNames = new LinkedHashSet<>();
        for (String c : partitionCols) partNames.add(c);
        String[] remainingCols = new String[df.columnCount() - partNames.size()];
        int ri = 0;
        for (Column c : df.columns())
            if (!partNames.contains(c.name())) remainingCols[ri++] = c.name();
        DataFrame projectedAll = remainingCols.length == 0
            ? df.copy().iloc(0, 0)
            : df.select(remainingCols);

        java.nio.file.Path rootPath = java.nio.file.Paths.get(root);
        java.nio.file.Files.createDirectories(rootPath);

        String ext = extensionFor(fmt);

        for (Map.Entry<List<String>, List<Integer>> e : buckets.entrySet()) {
            StringBuilder dir = new StringBuilder(root);
            for (int i = 0; i < partitionCols.length; i++) {
                dir.append('/').append(partitionCols[i]).append('=').append(e.getKey().get(i));
            }
            String filePath = dir + "/part-" + java.util.UUID.randomUUID().toString().replace("-", "") + ext;
            int[] idx = new int[e.getValue().size()];
            for (int k = 0; k < idx.length; k++) idx[k] = e.getValue().get(k);

            java.nio.file.Files.createDirectories(java.nio.file.Paths.get(dir.toString()));

            DataFrame part = df.loc(idx).select(remainingCols.length == 0
                ? new String[0]
                : remainingCols);
            // Dispatch with the same options/mode but for this path
            String prevFmt = this.format;
            String prevPath = this.deferredPath;
            DataFrameWriter inner = new DataFrameWriter(part)
                .format(prevFmt == null ? fmt : prevFmt)
                .mode(this.mode);
            for (Map.Entry<String, String> opt : this.options.entrySet())
                inner.option(opt.getKey(), opt.getValue());
            inner.save(filePath);
            // Restore outer writer state (in case of chained calls)
            this.deferredPath = prevPath;
            this.format = prevFmt;
        }
    }

    private static String extensionFor(String fmt) {
        switch (fmt) {
            case "parquet": return ".parquet";
            case "csv":     return ".csv";
            case "tsv":     return ".tsv";
            case "json":    return ".json";
            case "jsonl":
            case "ndjson":  return ".jsonl";
            case "arrow":
            case "feather":
            case "ipc":     return ".arrow";
            case "pickle":
            case "pkl":     return ".pkl";
            case "npz":     return ".npz";
            case "npy":     return ".npy";
            case "safetensors": return ".safetensors";
            case "gguf":    return ".gguf";
            case "excel":
            case "xlsx":
            case "xls":     return ".xlsx";
            case "hdf5":
            case "hdf":     return ".h5";
            case "avro":    return ".avro";
            case "orc":     return ".orc";
            case "lance":   return ".lance";
            default:        return "";
        }
    }

    private static String sanitizePartition(Object v) {
        if (v == null) return "__HIVE_DEFAULT_PARTITION__";
        String s = v.toString();
        // Spark-style escape
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
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
}
