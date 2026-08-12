/*
 * DaftEngine — executes a {@link DaftDataFrame} plan against the underlying
 * {@link DataFrame} infrastructure.
 *
 * 现阶段用一个简单 fold over transforms 实现：
 *   1. deferred transforms → 调 DataFrame.readParquet / readCsv / ...
 *   2. eager transforms   → 调 DataFrame.filter / select / withColumn / ...
 *   3. MediaFn 表达式     → 应用 MediaFn.apply
 *
 * 未来: 计划优化器 + 多阶段执行.
 */
package org.bytedeco.pytorch.utils.daft.engine;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.daft.DaftDataFrame;
import org.bytedeco.pytorch.utils.daft.expr.ColumnRef;
import org.bytedeco.pytorch.utils.daft.expr.Expression;
import org.bytedeco.pytorch.utils.daft.expr.MediaFn;
import org.bytedeco.pytorch.utils.daft.DaftDataFrame.DeferredTransform;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DaftEngine {

    private DaftEngine() {}

    public static DataFrame execute(DaftDataFrame plan) {
        DataFrame df = plan.backing();
        if (df == null) df = new DataFrame(); // empty starting frame for plans that begin with a deferred read
        for (DaftDataFrame.Transform t : plan.transforms()) {
            df = apply(df, t, plan);
        }
        return df;
    }

    private static DataFrame apply(DataFrame df, DaftDataFrame.Transform t, DaftDataFrame plan) {
        try {
            if (t instanceof DaftDataFrame.Transform.TransformSelect) {
                DaftDataFrame.Transform.TransformSelect sel = (DaftDataFrame.Transform.TransformSelect) t;
                return applySelect(df, sel.exprs);
            }
            if (t instanceof DaftDataFrame.Transform.TransformFilter) {
                DaftDataFrame.Transform.TransformFilter f = (DaftDataFrame.Transform.TransformFilter) t;
                // Evaluate the predicate to get a boolean mask column
                Column mask = f.predicate.eval(df);
                if (mask == null) {
                    throw new IllegalStateException("Filter predicate must be a ColumnRef-based expression");
                }
                // Use DataFrame.filter with the daft Expression
                return df.filter(toDataFrameExpr(f.predicate));
            }
            if (t instanceof DaftDataFrame.Transform.TransformWithColumn) {
                DaftDataFrame.Transform.TransformWithColumn wc = (DaftDataFrame.Transform.TransformWithColumn) t;
                Column c = evalExpression(df, wc.expr);
                if (c == null && wc.expr instanceof MediaFn) {
                    MediaFn mf = (MediaFn) wc.expr;
                    Column src = resolveMediaSource(df, mf);
                    if (src == null) {
                        c = new Column(wc.name, Column.DType.STRING);
                    } else {
                        c = new Column(wc.name, Column.DType.STRING);
                        for (int i = 0; i < src.size(); i++) {
                            try {
                                Object v = mf.apply(src.get(i));
                                c.add(v);
                            } catch (Throwable e) {
                                c.add(null);
                            }
                        }
                    }
                }
                if (c == null) {
                    c = new Column(wc.name, Column.DType.STRING);
                }
                // Build a list of values from the column to pass to DataFrame.withColumn
                java.util.List<Object> values = new java.util.ArrayList<>(c.size());
                for (int i = 0; i < c.size(); i++) {
                    values.add(c.get(i));
                }
                return df.withColumn(wc.name, values);
            }
            if (t instanceof DaftDataFrame.Transform.TransformRename) {
                DaftDataFrame.Transform.TransformRename rn = (DaftDataFrame.Transform.TransformRename) t;
                Column src = df.column(rn.oldName);
                Column renamed = src.rename(rn.newName);
                df = df.drop(rn.oldName);
                // Build a list of values from the column to pass to DataFrame.withColumn
                java.util.List<Object> values = new java.util.ArrayList<>(renamed.size());
                for (int i = 0; i < renamed.size(); i++) {
                    values.add(renamed.get(i));
                }
                return df.withColumn(renamed.name(), values);
            }
            if (t instanceof DaftDataFrame.Transform.TransformExclude) {
                DaftDataFrame.Transform.TransformExclude ex = (DaftDataFrame.Transform.TransformExclude) t;
                for (String col : ex.columns) df = df.drop(col);
                return df;
            }
            if (t instanceof DaftDataFrame.Transform.TransformSort) {
                DaftDataFrame.Transform.TransformSort s = (DaftDataFrame.Transform.TransformSort) t;
                if (!(s.key instanceof ColumnRef)) {
                    throw new IllegalArgumentException("Sort key must be a column ref");
                }
                ColumnRef cr = (ColumnRef) s.key;
                return df.sortValues(cr.name(), s.ascending);
            }
            if (t instanceof DaftDataFrame.Transform.TransformDistinct) {
                return df.dropDuplicates();
            }
            if (t instanceof DaftDataFrame.Transform.TransformLimit) {
                DaftDataFrame.Transform.TransformLimit l = (DaftDataFrame.Transform.TransformLimit) t;
                return df.limit((int) Math.min(Integer.MAX_VALUE, l.maxRows));
            }
            if (t instanceof DaftDataFrame.Transform.TransformOffset) {
                DaftDataFrame.Transform.TransformOffset o = (DaftDataFrame.Transform.TransformOffset) t;
                return df.iloc((int) o.rows, df.rowCount());
            }
            if (t instanceof DaftDataFrame.Transform.TransformJoin) {
                DaftDataFrame.Transform.TransformJoin j = (DaftDataFrame.Transform.TransformJoin) t;
                DataFrame otherDf = execute(j.other);
                return df.join(otherDf, j.on, j.how);
            }
            if (t instanceof DaftDataFrame.Transform.TransformJoinOn) {
                DaftDataFrame.Transform.TransformJoinOn j = (DaftDataFrame.Transform.TransformJoinOn) t;
                DataFrame otherDf = execute(j.other);
                return df.join(otherDf, j.leftOn, j.how);
            }
            if (t instanceof DaftDataFrame.Transform.TransformConcat) {
                DaftDataFrame.Transform.TransformConcat j = (DaftDataFrame.Transform.TransformConcat) t;
                DataFrame otherDf = execute(j.other);
                return DataFrameOps.concat(df, otherDf);
            }
            if (t instanceof DaftDataFrame.Transform.TransformGroupBy) {
                DaftDataFrame.Transform.TransformGroupBy g = (DaftDataFrame.Transform.TransformGroupBy) t;
                String[] keys = new String[g.keys.size()];
                for (int i = 0; i < keys.length; i++) {
                    Expression keyExpr = g.keys.get(i);
                    if (!(keyExpr instanceof ColumnRef)) throw new IllegalArgumentException("groupby only supports ColumnRef");
                    keys[i] = ((ColumnRef) keyExpr).name();
                }
                return DataFrameOps.groupBy(df, keys);
            }
            if (t instanceof DaftDataFrame.Transform.TransformAgg) {
                DaftDataFrame.Transform.TransformAgg agg = (DaftDataFrame.Transform.TransformAgg) t;
                // Look back at the last TransformGroupBy to capture keys
                String[] keys = lastGroupByKeys(plan);
                Map<String, String> out = new LinkedHashMap<>();
                for (Map.Entry<String, Expression> e : agg.aggs.entrySet()) {
                    out.put(e.getKey(), e.getValue().name());
                }
                return DataFrameOps.aggregate(df, keys, out);
            }
            if (t instanceof DaftDataFrame.Transform.TransformSql) {
                throw new UnsupportedOperationException("SQL transform requires a SQL engine; not yet wired");
            }
            if (t instanceof DeferredTransform) {
                return applyDeferred((DeferredTransform) t);
            }
            throw new IllegalStateException("Unknown transform: " + t.getClass().getName());
        } catch (Exception e) {
            throw new RuntimeException("daft engine: failed at " + t.getClass().getSimpleName() + ": " + e, e);
        }
    }

    private static String[] lastGroupByKeys(DaftDataFrame plan) {
        for (int i = plan.transforms().size() - 1; i >= 0; i--) {
            DaftDataFrame.Transform t = plan.transforms().get(i);
            if (t instanceof DaftDataFrame.Transform.TransformGroupBy) {
                DaftDataFrame.Transform.TransformGroupBy g = (DaftDataFrame.Transform.TransformGroupBy) t;
                String[] keys = new String[g.keys.size()];
                for (int k = 0; k < keys.length; k++) {
                    Expression keyExpr = g.keys.get(k);
                    if (!(keyExpr instanceof ColumnRef)) {
                        throw new IllegalArgumentException("groupby only supports ColumnRef");
                    }
                    keys[k] = ((ColumnRef) keyExpr).name();
                }
                return keys;
            }
        }
        throw new IllegalStateException("agg() without preceding groupBy()");
    }

    private static DataFrame applyDeferred(DeferredTransform t) throws Exception {
        switch (t.kind) {
            case READ_PARQUET: return DataFrame.readParquet(t.path);
            case READ_CSV:     return DataFrame.readCsv(t.path);
            case READ_JSON:    return DataFrame.readJson(t.path);
            case READ_ARROW:   return DataFrame.readArrow(t.path);
            case READ_TEXT: {
                // Read text file and create DataFrame with single "text" column
                java.util.List<String> lines = java.nio.file.Files.readAllLines(java.nio.file.Paths.get(t.path));
                DataFrame df = DataFrame.create();
                Column textCol = new Column("text", Column.DType.STRING);
                for (String line : lines) textCol.add(line);
                df.addColumn(textCol);
                return df;
            }
            case READ_GLOB:    return DataFrame.readImages(t.path);
            default: throw new IllegalStateException("unknown kind: " + t.kind);
        }
    }

    private static DataFrame applySelect(DataFrame df, List<Expression> exprs) throws Exception {
        if (exprs.size() == 1 && exprs.get(0) instanceof org.bytedeco.pytorch.utils.daft.expr.Star) {
            return df;
        }
        Column[] cols = new Column[exprs.size()];
        for (int i = 0; i < cols.length; i++) {
            Expression e = exprs.get(i);
            Column c = evalExpression(df, e);
            if (c == null) {
                throw new IllegalStateException("select expression returned null column: " + e);
            }
            cols[i] = c;
        }
        // Synthesize a frame with the selected columns
        DataFrame out = new DataFrame();
        for (Column c : cols) {
            out.addColumn(c);
        }
        return out;
    }

    private static Column evalExpression(DataFrame df, Expression e) {
        if (e instanceof MediaFn) {
            MediaFn mf = (MediaFn) e;
            Column src = resolveMediaSource(df, mf);
            if (src == null) return null;
            Column out = new Column(mf.name(), Column.DType.STRING);
            for (int i = 0; i < src.size(); i++) {
                try { out.add(mf.apply(src.get(i))); }
                catch (Throwable t) { out.add(null); }
            }
            return out;
        }
        return e.eval(df);
    }

    private static Column resolveMediaSource(DataFrame df, MediaFn mf) {
        for (int i = 0; i < df.columnCount(); i++) {
            Column c = df.column(i);
            Object v0 = c.size() > 0 ? c.get(0) : null;
            if (v0 == null) continue;
            String cls = v0.getClass().getName();
            if (cls.contains("ImageData") || cls.contains("AudioData") || cls.contains("VideoData")) {
                return c;
            }
        }
        return null;
    }

    /**
     * Convert daft Expression to DataFrame Expression.
     * Only supports ColumnRef-based expressions for now.
     */
    private static org.bytedeco.pytorch.dataframe.Expression toDataFrameExpr(Expression daftExpr) {
        if (daftExpr instanceof ColumnRef) {
            ColumnRef cr = (ColumnRef) daftExpr;
            return org.bytedeco.pytorch.dataframe.Expression.col(cr.name());
        }
        // For now, only support ColumnRef; add more as needed
        return null;
    }
}
