package org.bytedeco.pytorch.dataframe.io;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.File;
import java.util.*;

/**
 * Enterprise-grade display for Python IMDB format files.
 */
public class ImdbShow {

    private ImdbShow() {}

    /**
     * Show IMDB file with full schema and data preview.
     */
    public static String show(String path) throws Exception {
        return show(path, new ShowOptions());
    }

    public static String show(String path, ShowOptions opts) throws Exception {
        ImdbReader.ImdbSchema schema = ImdbReader.schema(path);
        DataFrame df = ImdbReader.read(path);
        
        return formatShow(schema, df, path, opts);
    }

    /**
     * Show IMDB DataFrame with schema info.
     */
    public static String show(DataFrame df) {
        return show(df, new ShowOptions());
    }

    public static String show(DataFrame df, ShowOptions opts) {
        ImdbReader.ImdbSchema schema = inferSchema(df);
        return formatShow(schema, df, null, opts);
    }

    private static String formatShow(ImdbReader.ImdbSchema schema, DataFrame df, String path, ShowOptions opts) {
        StringBuilder sb = new StringBuilder();
        File file = path != null ? new File(path) : null;
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ Python IMDB Format                                                             ║\n"));
        sb.append(String.format("║ Format: %-59s ║\n", schema.format != null ? schema.format : "Unknown"));
        if (file != null) {
            sb.append(String.format("║ File: %-63s ║\n", truncate(path, 63)));
            sb.append(String.format("║ Size: %-61s ║\n", formatBytes(file.length())));
        }
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Show schema table
        sb.append(String.format("║ %-3s │ %-20s │ %-12s │ %-10s ║\n", "#", "column", "dtype", "list"));
        sb.append("╠═════╪══════════════════════╪═══════════════╪════════════╣\n");
        
        int idx = 0;
        for (ImdbReader.ImdbSchema.FieldInfo f : schema.fields) {
            String name = truncate(f.name, 20);
            String sample = f.sample != null ? truncate(f.sample, 15) : "";
            sb.append(String.format("║ %3d │ %-20s │ %-12s │ %-10s ║\n",
                idx++, name, f.dtype, f.isList ? "true" : "false"));
        }
        
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║                         Data Preview                                         ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Show column headers
        int maxCols = Math.min(df.columnCount(), opts.maxCols());
        sb.append("║  idx │");
        for (int c = 0; c < maxCols; c++) {
            sb.append(String.format(" %-18s│", truncate(df.column(c).name(), 18)));
        }
        if (df.columnCount() > maxCols) sb.append("          ... │");
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
            if (df.columnCount() > maxCols) sb.append("          ... │");
            sb.append("\n");
        }
        
        // Trailing rows
        if (df.rowCount() > rows) {
            sb.append("║  ... │");
            for (int c = 0; c < maxCols; c++) {
                sb.append("                ... │");
            }
            if (df.columnCount() > maxCols) sb.append("          ... │");
            sb.append("\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    private static ImdbReader.ImdbSchema inferSchema(DataFrame df) {
        ImdbReader.ImdbSchema schema = new ImdbReader.ImdbSchema("IMDB (DataFrame)", 0);
        
        for (Column c : df.columns()) {
            ImdbReader.ImdbSchema.FieldInfo f = new ImdbReader.ImdbSchema.FieldInfo(
                c.name(), c.dtype().name(), c.size());
            f.isList = c.dtype() == Column.DType.LIST || c.dtype() == Column.DType.VECTOR;
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
        } else if (v instanceof String) {
            str = (String) v;
        } else if (v instanceof List) {
            List<?> list = (List<?>) v;
            if (list.isEmpty()) {
                str = "[]";
            } else if (list.size() <= 3) {
                str = "[" + list.stream().limit(3).map(String::valueOf).reduce((a, b) -> a + "," + b).orElse("") + "]";
            } else {
                str = "[" + list.get(0) + ", " + list.get(1) + ", ... (" + list.size() + ")]";
            }
        } else if (v instanceof float[] arr) {
            if (arr.length <= 3) {
                str = "[" + Arrays.toString(arr) + "]";
            } else {
                str = String.format("[%.3f, %.3f, ... (%d)]", arr[0], arr[1], arr.length);
            }
        } else if (v instanceof double[] arr) {
            if (arr.length <= 3) {
                str = "[" + Arrays.toString(arr) + "]";
            } else {
                str = String.format("[%.3f, %.3f, ... (%d)]", arr[0], arr[1], arr.length);
            }
        } else if (v instanceof Map) {
            str = "{...}";
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

    // ====================== Options ======================

    public static class ShowOptions {
        private int maxRows = 10;
        private int maxCols = 6;

        public static ShowOptions defaults() { return new ShowOptions(); }

        public ShowOptions maxRows(int n) { this.maxRows = n; return this; }
        public ShowOptions maxCols(int n) { this.maxCols = n; return this; }

        public int maxRows() { return maxRows; }
        public int maxCols() { return maxCols; }
    }
}
