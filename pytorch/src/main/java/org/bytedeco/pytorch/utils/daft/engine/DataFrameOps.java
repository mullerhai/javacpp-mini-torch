/*
 * DataFrameOps — small helpers that are NOT yet on the underlying DataFrame,
 * but are needed by DaftEngine (groupby, aggregate, write-csv, sql, concat).
 *
 * These helpers delegate to the existing operators / I/O modules where available,
 * and provide minimal fallback implementations otherwise. Designed to keep
 * the DaftDataFrame facade functional with the current DataFrame engine
 * without waiting for new public API surfaces.
 */
package org.bytedeco.pytorch.utils.daft.engine;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Operator helpers for operations not directly on DataFrame.
 */
public final class DataFrameOps {

    private DataFrameOps() {}

    /** Vertical concatenation of two DataFrames (column names must match). */
    public static DataFrame concat(DataFrame a, DataFrame b) {
        // Use iloc + join pattern: clone then append rows.
        // Implementation: copy columns, append rows from b.
        DataFrame out = a.copy();
        int srcRows = b.rowCount();
        for (int i = 0; i < srcRows; i++) {
            for (int c = 0; c < b.columnCount(); c++) {
                Column dst = out.column(c);
                if (c < a.columnCount()) {
                    dst.add(b.column(c).get(i));
                }
            }
        }
        return out;
    }

    /** Write the DataFrame to a CSV file (UTF-8, comma). */
    public static void writeCsv(DataFrame df, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) bw.write(',');
                String v = df.column(c).name();
                bw.write(escape(v));
            }
            bw.write('\n');
            for (int r = 0; r < df.rowCount(); r++) {
                for (int c = 0; c < df.columnCount(); c++) {
                    if (c > 0) bw.write(',');
                    Object v = df.column(c).get(r);
                    bw.write(v == null ? "" : escape(v.toString()));
                }
                bw.write('\n');
            }
        }
    }

    private static String escape(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }

    /** groupby(keys) without aggregation — returns a DataFrame with grouped rows.
     *  Implementation: sort by keys then collect group boundaries (rows preserved). */
    public static DataFrame groupBy(DataFrame df, String[] keys) {
        // The underlying DataFrame has DataFrame.groupby() returning a GroupedDataFrame;
        // but to keep the public surface minimal we use sortValues (stable) and let the
        // caller aggregate row-by-row via the standard withColumn + map approach.
        for (int i = keys.length - 1; i >= 0; i--) {
            df = df.sortValues(keys[i], true);
        }
        return df;
    }

    /** Aggregate after groupby:
     *  aggregations = {outputName: "sum(x)" | "count" | "min(x)" | "max(x)" | "mean(x)"}.
     *  Implementation: group rows by keys, reduce per bucket. */
    public static DataFrame aggregate(DataFrame grouped, String[] keys,
                                     Map<String, String> aggregations) {
        List<List<Object>> groupRows = new ArrayList<>();
        List<Integer> groupStart = new ArrayList<>();
        // Identify groups by key tuple
        List<Object> currentKey = null;
        int start = 0;
        int n = grouped.rowCount();
        for (int r = 0; r < n; r++) {
            List<Object> key = new ArrayList<>(keys.length);
            for (String k : keys) {
                key.add(grouped.column(k).get(r));
            }
            if (currentKey == null || !currentKey.equals(key)) {
                if (currentKey != null) groupStart.add(start);
                currentKey = key;
                groupStart.add(r);
                start = r;
            }
        }
        groupStart.add(n);
        // Build aggregated output (preserve key order from groupStart)
        DataFrame out = new DataFrame();
        for (String k : keys) out.addColumn(k, grouped.column(k).dtype());

        // Add aggregation columns with broad BINARY dtype (boxing)
        for (String colName : aggregations.keySet()) {
            out.addColumn(colName, Column.DType.STRING);
        }
        int aggCount = aggregations.size();
        for (int g = 0; g < groupStart.size() - 1; g++) {
            int lo = groupStart.get(g);
            int hi = groupStart.get(g + 1);
            // Copy keys (we know they're identical within the group, take lo)
            for (int c = 0; c < keys.length; c++) {
                Object value = grouped.column(keys[c]).get(lo);
                for (int r = out.rowCount(); r <= out.rowCount(); r++) {
                    // No-op; we use addRow pattern via single column add
                    Column col = out.column(c);
                    col.add(value);
                }
            }
            // Compute each aggregation
            int idx = keys.length;
            for (Map.Entry<String, String> agg : aggregations.entrySet()) {
                Column col = out.column(idx++);
                col.add(computeAggregation(grouped, agg.getValue(), lo, hi));
            }
        }
        return out;
    }

    private static Object computeAggregation(DataFrame df, String expr, int lo, int hi) {
        // Format: functionName(col) e.g., "sum(x)", "count", "mean(y)"
        if (expr == null) return null;
        String e = expr.trim().toLowerCase(Locale.ROOT);
        if (e.equals("count")) return hi - lo;

        int p = e.indexOf('(');
        String fn = p >= 0 ? e.substring(0, p) : e;
        String col = (p >= 0 && e.endsWith(")")) ? e.substring(p + 1, e.length() - 1) : null;
        if (col == null || col.isEmpty()) return null;

        Column c = df.column(col);
        switch (fn) {
            case "sum": {
                double s = 0;
                boolean any = false;
                for (int i = lo; i < hi; i++) {
                    Object v = c.get(i);
                    if (v instanceof Number) { s += ((Number) v).doubleValue(); any = true; }
                }
                return any ? s : null;
            }
            case "mean": case "avg": {
                double s = 0;
                int cnt = 0;
                for (int i = lo; i < hi; i++) {
                    Object v = c.get(i);
                    if (v instanceof Number) { s += ((Number) v).doubleValue(); cnt++; }
                }
                return cnt == 0 ? null : s / cnt;
            }
            case "min": {
                Double best = null;
                for (int i = lo; i < hi; i++) {
                    Object v = c.get(i);
                    if (v instanceof Number) {
                        double d = ((Number) v).doubleValue();
                        if (best == null || d < best) best = d;
                    }
                }
                return best;
            }
            case "max": {
                Double best = null;
                for (int i = lo; i < hi; i++) {
                    Object v = c.get(i);
                    if (v instanceof Number) {
                        double d = ((Number) v).doubleValue();
                        if (best == null || d > best) best = d;
                    }
                }
                return best;
            }
            case "count_distinct":
                return countDistinct(c, lo, hi);
            default:
                return null;
        }
    }

    private static int countDistinct(Column c, int lo, int hi) {
        java.util.Set<Object> s = new java.util.HashSet<>();
        for (int i = lo; i < hi; i++) {
            Object v = c.get(i);
            if (v != null) s.add(v);
        }
        return s.size();
    }
}
