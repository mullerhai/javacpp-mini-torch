/*
 * DaftDataFrame — Python Daft DataFrame facade over Java DataFrame.
 *
 * 设计:
 *   - immutable: 每个 transform 方法返回新的 DaftDataFrame
 *   - lazy: 执行计划 (.plan) 延迟执行, 到 .collect() / .show() 才真正执行
 *   - chained: 全部方法返回 self
 */
package org.bytedeco.pytorch.utils.daft;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.utils.daft.engine.ExecutionConfig;
import org.bytedeco.pytorch.utils.daft.expr.Expression;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Lazy DataFrame with Daft-style fluent API.
 *
 * <pre>{@code
 *   DaftDataFrame df = DaftDataFrame.fromParquet("events/*.parquet")
 *       .filter(col("country").eq(lit("US")))
 *       .select(col("user_id"), col("ts"), col("score").alias("ctr"))
 *       .withColumn("embedding", col("text").embedding().encodeText())
 *       .limit(1_000_000);
 *   df.collect();       // materialise
 *   df.show(10);        // print first 10 rows
 * }</pre>
 */
public final class DaftDataFrame {

    private final DataFrame backing; // may be null for "deferred" mode
    private final List<Transform> transforms;
    private final ExecutionConfig config;
    private final String alias;

    private DaftDataFrame(DataFrame backing, List<Transform> transforms, ExecutionConfig config, String alias) {
        this.backing = backing;
        this.transforms = transforms;
        this.config = config;
        this.alias = alias;
    }

    /** Wrap an existing DataFrame (eager mode). */
    public static DaftDataFrame of(DataFrame df) {
        return new DaftDataFrame(Objects.requireNonNull(df), new ArrayList<>(),
                ExecutionConfig.defaults(), null);
    }

    /** Empty deferred DataFrame (used by fromGlobPath etc.). */
    static DaftDataFrame deferred(ExecutionConfig config) {
        return new DaftDataFrame(null, new ArrayList<>(), config, null);
    }

    // ---- read sources -------------------------------------------------------

    public static DaftDataFrame fromParquet(String glob) {
        return fromParquet(glob, java.util.Collections.emptyMap());
    }

    public static DaftDataFrame fromParquet(String glob, Map<String, String> options) {
        DeferredTransform t = new DeferredTransform(DeferredTransform.Kind.READ_PARQUET, glob, options);
        return deferred(ExecutionConfig.defaults()).withTransform(t);
    }

    public static DaftDataFrame fromCsv(String path) {
        return fromCsv(path, java.util.Collections.emptyMap());
    }

    public static DaftDataFrame fromCsv(String path, Map<String, String> options) {
        DeferredTransform t = new DeferredTransform(DeferredTransform.Kind.READ_CSV, path, options);
        return deferred(ExecutionConfig.defaults()).withTransform(t);
    }

    public static DaftDataFrame fromJson(String path) {
        return fromJson(path, java.util.Collections.emptyMap());
    }

    public static DaftDataFrame fromJson(String path, Map<String, String> options) {
        DeferredTransform t = new DeferredTransform(DeferredTransform.Kind.READ_JSON, path, options);
        return deferred(ExecutionConfig.defaults()).withTransform(t);
    }

    public static DaftDataFrame fromArrow(String path) {
        DeferredTransform t = new DeferredTransform(DeferredTransform.Kind.READ_ARROW, path,
                java.util.Collections.emptyMap());
        return deferred(ExecutionConfig.defaults()).withTransform(t);
    }

    public static DaftDataFrame fromText(String path) {
        DeferredTransform t = new DeferredTransform(DeferredTransform.Kind.READ_TEXT, path,
                java.util.Collections.emptyMap());
        return deferred(ExecutionConfig.defaults()).withTransform(t);
    }

    public static DaftDataFrame fromGlobPath(String glob) {
        DeferredTransform t = new DeferredTransform(DeferredTransform.Kind.READ_GLOB, glob,
                java.util.Collections.emptyMap());
        return deferred(ExecutionConfig.defaults()).withTransform(t);
    }

    public static DaftDataFrame fromPydict(Map<String, Object> data) {
        DataFrame df = new DataFrame();
        if (data != null) {
            for (Map.Entry<String, Object> e : data.entrySet()) {
                Column c = new Column(e.getKey(), Column.DType.STRING);
                if (e.getValue() instanceof List) {
                    for (Object v : (List<?>) e.getValue()) c.add(v);
                }
                df.addColumn(c);
            }
        }
        return of(df);
    }

    public static DaftDataFrame fromArrowTable(org.bytedeco.pytorch.dataframe.DataFrame arrowTable) {
        return of(arrowTable);
    }

    // ---- transform methods --------------------------------------------------

    public DaftDataFrame select(Expression... exprs) {
        Objects.requireNonNull(exprs, "exprs");
        return append(new Transform.TransformSelect(Arrays.asList(exprs)));
    }

    public DaftDataFrame select(String... columnNames) {
        Objects.requireNonNull(columnNames, "columnNames");
        Expression[] es = new Expression[columnNames.length];
        for (int i = 0; i < columnNames.length; i++) es[i] = Expression.col(columnNames[i]);
        return select(es);
    }

    public DaftDataFrame filter(Expression condition) {
        Objects.requireNonNull(condition, "condition");
        return append(new Transform.TransformFilter(condition));
    }

    public DaftDataFrame where(Expression condition) { return filter(condition); }

    public DaftDataFrame withColumn(String name, Expression expr) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(expr, "expr");
        return append(new Transform.TransformWithColumn(name, expr.alias(name)));
    }

    public DaftDataFrame withColumns(Map<String, Expression> exprs) {
        DaftDataFrame cur = this;
        for (Map.Entry<String, Expression> e : exprs.entrySet()) {
            cur = cur.withColumn(e.getKey(), e.getValue());
        }
        return cur;
    }

    public DaftDataFrame rename(String oldName, String newName) {
        return append(new Transform.TransformRename(oldName, newName));
    }

    public DaftDataFrame exclude(String... columns) {
        return append(new Transform.TransformExclude(Arrays.asList(columns)));
    }

    public DaftDataFrame drop(String... columns) { return exclude(columns); }

    public DaftDataFrame sort(Expression sortKey, boolean ascending) {
        return append(new Transform.TransformSort(sortKey, ascending));
    }

    public DaftDataFrame sort(String column, boolean ascending) {
        return sort(Expression.col(column), ascending);
    }

    public DaftDataFrame distinct() {
        return append(new Transform.TransformDistinct());
    }

    public DaftDataFrame dropDuplicates() { return distinct(); }

    public DaftDataFrame limit(long maxRows) {
        return append(new Transform.TransformLimit(maxRows));
    }

    public DaftDataFrame offset(long rows) {
        return append(new Transform.TransformOffset(rows));
    }

    public DaftDataFrame join(DaftDataFrame other, String on, String how) {
        Objects.requireNonNull(other, "other");
        return append(new Transform.TransformJoin(other, on, how));
    }

    public DaftDataFrame join(DaftDataFrame other, String leftOn, String rightOn, String how) {
        return append(new Transform.TransformJoinOn(other, leftOn, rightOn, how));
    }

    public DaftDataFrame concat(DaftDataFrame other) {
        return append(new Transform.TransformConcat(other));
    }

    public DaftDataFrame union(DaftDataFrame other) { return concat(other); }

    public DaftDataFrame groupBy(Expression... keys) {
        return append(new Transform.TransformGroupBy(Arrays.asList(keys)));
    }

    public DaftDataFrame groupBy(String... keys) {
        Expression[] es = new Expression[keys.length];
        for (int i = 0; i < keys.length; i++) es[i] = Expression.col(keys[i]);
        return groupBy(es);
    }

    public DaftDataFrame agg(Map<String, Expression> aggregations) {
        return append(new Transform.TransformAgg(aggregations));
    }

    public DaftDataFrame sql(String query) {
        return append(new Transform.TransformSql(query));
    }

    public DaftDataFrame alias(String name) {
        return new DaftDataFrame(backing, transforms, config, name);
    }

    public DaftDataFrame withExecutionConfig(ExecutionConfig cfg) {
        return new DaftDataFrame(backing, transforms, cfg, alias);
    }

    public DaftDataFrame withTransform(Transform t) {
        List<Transform> next = new ArrayList<>(transforms);
        next.add(t);
        return new DaftDataFrame(backing, next, config, alias);
    }

    private DaftDataFrame append(Transform t) {
        return withTransform(t);
    }

    // ---- execution ---------------------------------------------------------

    /** Materialise the lazy pipeline. */
    public DataFrame collect() {
        return org.bytedeco.pytorch.utils.daft.engine.DaftEngine.execute(this);
    }

    /** Print first n rows. */
    public void show(int n) {
        DataFrame df = collect();
        System.out.println("== daft dataframe (rows=" + df.rowCount() + ", cols=" + df.columnCount() + ") ==");
        // Print column names
        StringBuilder header = new StringBuilder();
        for (int i = 0; i < df.columnCount(); i++) {
            if (i > 0) header.append("\t");
            header.append(df.column(i).name());
        }
        System.out.println(header);
        int rows = Math.min(n, df.rowCount());
        for (int r = 0; r < rows; r++) {
            StringBuilder row = new StringBuilder();
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) row.append("\t");
                Object v = df.column(c).get(r);
                row.append(v == null ? "null" : truncate(v.toString()));
            }
            System.out.println(row);
        }
    }

    /** Count rows (executes the pipeline). */
    public long count() { return collect().rowCount(); }

    /** Iterate batched over the result (micro-batch streaming). */
    public Iterable<DataFrame> iterBatches(int batchRows) {
        return () -> new java.util.Iterator<DataFrame>() {
            DataFrame full;
            int pos = 0;
            @Override public boolean hasNext() {
                if (full == null) full = collect();
                return pos < full.rowCount();
            }
            @Override public DataFrame next() {
                if (full == null) full = collect();
                int end = Math.min(pos + batchRows, full.rowCount());
                DataFrame slice = full.iloc(pos, end);
                pos = end;
                return slice;
            }
        };
    }

    // ---- write -------------------------------------------------------------

    // ---- write -------------------------------------------------------------

    /**
     * Fluent writer entry point (Spark / Daft-style).
     *
     * <pre>{@code
     *   // Daft-style quick helpers
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
     * }</pre>
     *
     * @return a {@link DaftDataFrameWriter} configured for this DataFrame
     * @see DaftDataFrameWriter
     */
    public DaftDataFrameWriter write() {
        return new DaftDataFrameWriter(this);
    }

    // ---- read (static entry point) -------------------------------------------

    /**
     * Static fluent reader entry point (Daft / Spark-style).
     *
     * <pre>{@code
     *   // Daft-style quick helpers
     *   DaftDataFrame df = DaftDataFrame.read().parquet("/data/*.parquet");
     *   DaftDataFrame df = DaftDataFrame.read().csv("/data/file.csv");
     *   DaftDataFrame df = DaftDataFrame.read().json("/data/file.json");
     *   DaftDataFrame df = DaftDataFrame.read().jsonl("/data/rows.jsonl");
     *   DaftDataFrame df = DaftDataFrame.read().text("/data/file.txt");
     *
     *   // Spark-style
     *   DaftDataFrame df = DaftDataFrame.read().format("parquet").load("/data/file.parquet");
     *   DaftDataFrame df = DaftDataFrame.read().option("header", "true").csv("/data/file.csv");
     *   DaftDataFrame df = DaftDataFrame.read().parquet("p1.parquet", "p2.parquet");
     *
     *   // chain Daft transforms
     *   DaftDataFrame df = DaftDataFrame.read()
     *       .parquet("/data/*.parquet")
     *       .filter(Expression.col("age").gt(18))
     *       .select("name", "age")
     *       .limit(1000);
     *   df.collect();  // materialise
     * }</pre>
     *
     * @return a {@link DaftDataFrameReader} ready to configure and load
     * @see DaftDataFrameReader
     */
    public static DaftDataFrameReader read() {
        return new DaftDataFrameReader();
    }

    public void writeParquet(String path) throws Exception {
        DataFrame df = collect();
        df.writeParquet(path);
    }

    public void writeCsv(String path) throws Exception {
        DataFrame df = collect();
        df.toCsv(path);
//        df.writeCsv(path);
    }

    public void writeLance(String path) throws Exception {
        DataFrame df = collect();
        df.writeLance(path);
    }

    // ---- accessors used by engine ------------------------------------------

    public DataFrame backing() { return backing; }

    public List<Transform> transforms() { return transforms; }

    public ExecutionConfig config() { return config; }

    public String tableAlias() { return alias; }

    // ---- helpers ------------------------------------------------------------

    private static String truncate(String s) {
        if (s == null) return "null";
        if (s.length() > 80) return s.substring(0, 77) + "...";
        return s;
    }

    // ---- transform AST ------------------------------------------------------

    /**
     * Transform AST (one node per .filter / .select / etc. call).
     */
    public static abstract class Transform {
        public static final class TransformSelect extends Transform {
            public final List<Expression> exprs;
            public TransformSelect(List<Expression> e) { this.exprs = e; }
        }
        public static final class TransformFilter extends Transform {
            public final Expression predicate;
            public TransformFilter(Expression p) { this.predicate = p; }
        }
        public static final class TransformWithColumn extends Transform {
            public final String name;
            public final Expression expr;
            public TransformWithColumn(String n, Expression e) { this.name = n; this.expr = e; }
        }
        public static final class TransformRename extends Transform {
            public final String oldName;
            public final String newName;
            public TransformRename(String o, String n) { this.oldName = o; this.newName = n; }
        }
        public static final class TransformExclude extends Transform {
            public final List<String> columns;
            public TransformExclude(List<String> c) { this.columns = c; }
        }
        public static final class TransformSort extends Transform {
            public final Expression key;
            public final boolean ascending;
            public TransformSort(Expression k, boolean a) { this.key = k; this.ascending = a; }
        }
        public static final class TransformDistinct extends Transform {}
        public static final class TransformLimit extends Transform {
            public final long maxRows;
            public TransformLimit(long n) { this.maxRows = n; }
        }
        public static final class TransformOffset extends Transform {
            public final long rows;
            public TransformOffset(long r) { this.rows = r; }
        }
        public static final class TransformJoin extends Transform {
            public final DaftDataFrame other;
            public final String on;
            public final String how;
            public TransformJoin(DaftDataFrame o, String on, String how) {
                this.other = o; this.on = on; this.how = how;
            }
        }
        public static final class TransformJoinOn extends Transform {
            public final DaftDataFrame other;
            public final String leftOn;
            public final String rightOn;
            public final String how;
            public TransformJoinOn(DaftDataFrame o, String l, String r, String h) {
                this.other = o; this.leftOn = l; this.rightOn = r; this.how = h;
            }
        }
        public static final class TransformConcat extends Transform {
            public final DaftDataFrame other;
            public TransformConcat(DaftDataFrame o) { this.other = o; }
        }
        public static final class TransformGroupBy extends Transform {
            public final List<Expression> keys;
            public TransformGroupBy(List<Expression> k) { this.keys = k; }
        }
        public static final class TransformAgg extends Transform {
            public final Map<String, Expression> aggs;
            public TransformAgg(Map<String, Expression> a) { this.aggs = a; }
        }
        public static final class TransformSql extends Transform {
            public final String query;
            public TransformSql(String q) { this.query = q; }
        }
    }

    /** Internal: deferred source read recorded as a transform. */
    public static final class DeferredTransform extends Transform {
        public enum Kind { READ_PARQUET, READ_CSV, READ_JSON, READ_ARROW, READ_TEXT, READ_GLOB }
        public final Kind kind;
        public final String path;
        public final Map<String, String> options;
        DeferredTransform(Kind k, String p, Map<String, String> o) {
            this.kind = k; this.path = p; this.options = o;
        }
    }
}