package org.bytedeco.pytorch.dataframe.io.lmdb;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade display for LMDB format files.
 */
public class LmdbShow {

    private LmdbShow() {}

    /**
     * Show LMDB database with full schema and data preview.
     */
    public static String show(String path) throws Exception {
        return show(path, new ShowOptions());
    }

    public static String show(String path, ShowOptions opts) throws Exception {
        LmdbReader.LmdbSchema schema = LmdbReader.schema(path);
        DataFrame df = LmdbReader.read(path, new LmdbReader.LmdbOptions().limit(opts.maxRows()));
        
        return formatShow(schema, df, path, opts);
    }

    /**
     * Show LMDB DataFrame with schema info.
     */
    public static String show(DataFrame df) {
        return show(df, new ShowOptions());
    }

    public static String show(DataFrame df, ShowOptions opts) {
        LmdbReader.LmdbSchema schema = inferSchema(df);
        return formatShow(schema, df, null, opts);
    }

    private static String formatShow(LmdbReader.LmdbSchema schema, DataFrame df, String path, ShowOptions opts) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append("║                        LMDB Format                                        ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append(String.format("║ Format: %-62s ║\n", schema.format != null ? schema.format : "Unknown"));
        sb.append(String.format("║ File size: %-58s ║\n", formatBytes(schema.fileSize)));
        sb.append(String.format("║ Entries: %-59s ║\n", formatCount(schema.entryCount)));
        sb.append(String.format("║ Key types: %-56s ║\n", String.join(", ", schema.keyTypes)));
        sb.append(String.format("║ Value types: %-54s ║\n", String.join(", ", schema.valueTypes)));
        sb.append(String.format("║ Avg value size: %-52s ║\n", formatBytes(schema.avgValueSize)));
        
        if (path != null) {
            File file = new File(path);
            sb.append(String.format("║ Path: %-61s ║\n", truncate(path, 61)));
        }
        
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║                           Schema                                           ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append(String.format("║ %-3s │ %-20s │ %-12s │ %-10s ║\n", "#", "field", "dtype", "count"));
        sb.append("╠═════╪══════════════════════╪═══════════════╪════════════╣\n");
        
        int idx = 0;
        for (LmdbReader.LmdbSchema.FieldInfo f : schema.fields) {
            String name = truncate(f.name, 20);
            sb.append(String.format("║ %3d │ %-20s │ %-12s │ %10d ║\n",
                idx++, name, f.dtype, f.count));
        }
        
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║                          Data Preview                                       ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Show column headers
        int maxCols = Math.min(df.columnCount(), opts.maxCols());
        sb.append("║  idx │");
        for (int c = 0; c < maxCols; c++) {
            sb.append(String.format(" %-18s│", truncate(df.column(c).name(), 18)));
        }
        if (df.columnCount() > maxCols) sb.append("        ... │");
        sb.append("\n");
        
        sb.append("╟──────┼");
        for (int c = 0; c < maxCols; c++) {
            sb.append("──────────────────────┼");
        }
        if (df.columnCount() > maxCols) sb.append("───────────────┤");
        sb.append("\n");
        
        // Show sample data rows
        int rows = Math.min(opts.maxRows(), df.rowCount());
        for (int r = 0; r < rows; r++) {
            sb.append(String.format("║ %4d │", r));
            for (int c = 0; c < maxCols; c++) {
                Object v = df.get(r, c);
                String str = formatValue(v, 18);
                sb.append(" ").append(str).append("│");
            }
            if (df.columnCount() > maxCols) sb.append("        ... │");
            sb.append("\n");
        }
        
        if (df.rowCount() > rows) {
            sb.append("║  ... │");
            for (int c = 0; c < maxCols; c++) sb.append("                ... │");
            if (df.columnCount() > maxCols) sb.append("           ... │");
            sb.append("\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    private static LmdbReader.LmdbSchema inferSchema(DataFrame df) {
        LmdbReader.LmdbSchema schema = new LmdbReader.LmdbSchema("LMDB (DataFrame)", 0);
        schema.entryCount = df.rowCount();
        
        for (Column c : df.columns()) {
            LmdbReader.LmdbSchema.FieldInfo f = new LmdbReader.LmdbSchema.FieldInfo(
                c.name(), c.dtype().name(), c.size());
            schema.fields.add(f);
        }
        
        return schema;
    }

    private static String formatValue(Object v, int maxLen) {
        if (v == null) return padRight("null", maxLen);
        String str;
        if (v instanceof Number n) {
            if (n instanceof Double || n instanceof Float) {
                str = String.format("%.4f", n.doubleValue());
            } else {
                str = String.valueOf(n.longValue());
            }
        } else if (v instanceof byte[] arr) {
            str = String.format("binary[%d]", arr.length);
        } else if (v instanceof String) {
            str = (String) v;
        } else {
            str = String.valueOf(v);
        }
        return padRight(truncate(str, maxLen), maxLen);
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 3)) + "...";
    }

    private static String padRight(String s, int len) {
        if (s.length() >= len) return s.substring(0, len);
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < len) sb.append(' ');
        return sb.toString();
    }

    private static String formatBytes(long bytes) {
        if (bytes < 0) return "unknown";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static String formatCount(long count) {
        if (count < 1000) return String.valueOf(count);
        if (count < 1000000) return String.format("%.1fK", count / 1000.0);
        if (count < 1000000000) return String.format("%.1fM", count / 1000000.0);
        return String.format("%.1fB", count / 1000000000.0);
    }

    // ====================== Options ======================

    public static class ShowOptions {
        private int maxRows = 10;
        private int maxCols = 5;

        public static ShowOptions defaults() { return new ShowOptions(); }

        public ShowOptions maxRows(int n) { this.maxRows = n; return this; }
        public ShowOptions maxCols(int n) { this.maxCols = n; return this; }

        public int maxRows() { return maxRows; }
        public int maxCols() { return maxCols; }
    }
}
