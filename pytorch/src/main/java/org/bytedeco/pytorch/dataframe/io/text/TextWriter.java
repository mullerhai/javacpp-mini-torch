package org.bytedeco.pytorch.dataframe.io.text;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade Text writer for various text formats.
 * 
 * <p>Writes DataFrames to text files with support for:
 * <ul>
 *   <li>Plain text files</li>
 *   <li>Line-separated text (one row per line)</li>
 *   <li>Delimited text (custom separator)</li>
 *   <li>JSON Lines text</li>
 *   <li>HTML tables</li>
 *   <li>Markdown tables</li>
 * </ul>
 */
public class TextWriter {

    private TextWriter() {}

    /**
     * Write DataFrame to text file.
     */
    public static void write(DataFrame df, String path) throws IOException {
        write(df, path, TextOptions.defaults());
    }

    public static void write(DataFrame df, String path, TextOptions options) throws IOException {
        TextOptions opt = options == null ? TextOptions.defaults() : options;
        
        Charset charset = Charset.forName(opt.charset());
        Files.createDirectories(Path.of(path).getParent());
        
        try (BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(
                    opt.compress() ? new java.util.zip.GZIPOutputStream(
                        Files.newOutputStream(Path.of(path))) 
                        : Files.newOutputStream(Path.of(path)),
                    charset),
                opt.bufferSize())) {
            
            switch (opt.format()) {
                case LINE:
                    writeLineByLine(df, writer, opt);
                    break;
                case DELIMITED:
                    writeDelimited(df, writer, opt);
                    break;
                case JSONL:
                    writeJsonLines(df, writer, opt);
                    break;
                case HTML:
                    writeHtml(df, writer, opt);
                    break;
                case MARKDOWN:
                    writeMarkdown(df, writer, opt);
                    break;
                case COLUMN:
                    writeColumnFormat(df, writer, opt);
                    break;
            }
        }
    }

    private static void writeLineByLine(DataFrame df, BufferedWriter writer, TextOptions opt) throws IOException {
        String column = opt.column();
        
        for (int r = 0; r < df.rowCount(); r++) {
            Object val = column != null ? df.get(r, column) : df.get(r, 0);
            String line = val != null ? val.toString() : "";
            
            if (opt.trim()) {
                line = line.trim();
            }
            
            writer.write(line);
            writer.newLine();
        }
    }

    private static void writeDelimited(DataFrame df, BufferedWriter writer, TextOptions opt) throws IOException {
        String delim = opt.delimiter();
        boolean firstRow = opt.includeHeader();
        
        if (firstRow) {
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) writer.write(delim);
                writer.write(escape(df.column(c).name(), delim, opt));
            }
            writer.newLine();
        }
        
        for (int r = 0; r < df.rowCount(); r++) {
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) writer.write(delim);
                Object val = df.get(r, c);
                writer.write(escape(val != null ? val.toString() : "", delim, opt));
            }
            writer.newLine();
        }
    }

    private static void writeJsonLines(DataFrame df, BufferedWriter writer, TextOptions opt) throws IOException {
        for (int r = 0; r < df.rowCount(); r++) {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) sb.append(",");
                sb.append("\"").append(escapeJson(df.column(c).name())).append("\":");
                
                Object val = df.get(r, c);
                if (val == null) {
                    sb.append("null");
                } else if (val instanceof Number) {
                    sb.append(val);
                } else if (val instanceof Boolean) {
                    sb.append(val);
                } else {
                    sb.append("\"").append(escapeJson(val.toString())).append("\"");
                }
            }
            
            sb.append("}");
            writer.write(sb.toString());
            writer.newLine();
        }
    }

    private static void writeHtml(DataFrame df, BufferedWriter writer, TextOptions opt) throws IOException {
        writer.write("<!DOCTYPE html>\n");
        writer.write("<html><head>\n");
        writer.write("<meta charset=\"" + opt.charset() + "\">\n");
        writer.write("<style>\n");
        writer.write("table { border-collapse: collapse; width: 100%; }\n");
        writer.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        writer.write("th { background-color: #f2f2f2; }\n");
        writer.write("tr:nth-child(even) { background-color: #fafafa; }\n");
        writer.write("</style>\n");
        writer.write("</head><body>\n");
        writer.write("<table>\n");
        
        // Header
        if (opt.includeHeader()) {
            writer.write("<thead><tr>");
            for (int c = 0; c < df.columnCount(); c++) {
                writer.write("<th>");
                writer.write(escapeHtml(df.column(c).name()));
                writer.write("</th>");
            }
            writer.write("</tr></thead>\n");
        }
        
        writer.write("<tbody>\n");
        for (int r = 0; r < df.rowCount(); r++) {
            writer.write("<tr>");
            for (int c = 0; c < df.columnCount(); c++) {
                writer.write("<td>");
                Object val = df.get(r, c);
                writer.write(escapeHtml(val != null ? val.toString() : ""));
                writer.write("</td>");
            }
            writer.write("</tr>\n");
        }
        
        writer.write("</tbody>\n</table>\n</body></html>");
    }

    private static void writeMarkdown(DataFrame df, BufferedWriter writer, TextOptions opt) throws IOException {
        String delim = opt.delimiter();
        
        // Header
        if (opt.includeHeader()) {
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) writer.write(delim);
                writer.write("**");
                writer.write(df.column(c).name());
                writer.write("**");
            }
            writer.newLine();
            
            // Separator
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) writer.write(delim);
                writer.write("---");
            }
            writer.newLine();
        }
        
        // Data
        for (int r = 0; r < df.rowCount(); r++) {
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) writer.write(delim);
                Object val = df.get(r, c);
                writer.write(val != null ? val.toString() : "");
            }
            writer.newLine();
        }
    }

    private static void writeColumnFormat(DataFrame df, BufferedWriter writer, TextOptions opt) throws IOException {
        int maxColWidth = opt.maxColumnWidth();
        
        // Calculate column widths
        int[] widths = new int[df.columnCount()];
        for (int c = 0; c < df.columnCount(); c++) {
            widths[c] = Math.min(df.column(c).name().length(), maxColWidth);
            for (int r = 0; r < df.rowCount(); r++) {
                Object val = df.get(r, c);
                int len = val != null ? val.toString().length() : 4;
                widths[c] = Math.max(widths[c], Math.min(len, maxColWidth));
            }
        }
        
        String fmt = "|";
        String sep = "|";
        for (int c = 0; c < df.columnCount(); c++) {
            fmt += " %-" + widths[c] + "s |";
            sep += " " + "-".repeat(widths[c]) + " |";
        }
        fmt += "\n";
        sep += "\n";
        
        // Header
        if (opt.includeHeader()) {
            Object[] headers = new Object[df.columnCount()];
            for (int c = 0; c < df.columnCount(); c++) {
                headers[c] = df.column(c).name();
            }
            writer.write(String.format(fmt, headers));
            writer.write(sep);
        }
        
        // Data
        for (int r = 0; r < df.rowCount(); r++) {
            Object[] row = new Object[df.columnCount()];
            for (int c = 0; c < df.columnCount(); c++) {
                Object val = df.get(r, c);
                String str = val != null ? val.toString() : "null";
                if (str.length() > maxColWidth) {
                    str = str.substring(0, maxColWidth - 3) + "...";
                }
                row[c] = str;
            }
            writer.write(String.format(fmt, row));
        }
    }

    private static String escape(String s, String delim, TextOptions opt) {
        if (s == null) return "";
        if (!opt.quote().isEmpty() && (s.contains(delim) || s.contains(opt.quote()))) {
            return opt.quote() + s.replace(opt.quote(), opt.quote() + opt.quote()) + opt.quote();
        }
        return s;
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String escapeHtml(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\"", "&quot;");
    }

    // ====================== Options ======================

    public enum TextFormat { LINE, DELIMITED, JSONL, HTML, MARKDOWN, COLUMN }

    public static class TextOptions {
        private TextFormat format = TextFormat.LINE;
        private String charset = "UTF-8";
        private String delimiter = "\t";
        private String quote = "\"";
        private boolean includeHeader = true;
        private String column = null;  // For LINE format: which column to write
        private boolean trim = false;
        private int bufferSize = 65536;
        private int maxColumnWidth = 50;
        private boolean compress = false;

        public static TextOptions defaults() { return new TextOptions(); }
        
        public static TextOptions lines(String column) {
            return new TextOptions().column(column);
        }
        
        public static TextOptions delimited(char delim) {
            return new TextOptions().format(TextFormat.DELIMITED).delimiter(String.valueOf(delim));
        }
        
        public static TextOptions markdown() {
            return new TextOptions().format(TextFormat.MARKDOWN);
        }
        
        public static TextOptions html() {
            return new TextOptions().format(TextFormat.HTML);
        }

        public TextOptions format(TextFormat f) { this.format = f; return this; }
        public TextOptions charset(String c) { this.charset = c; return this; }
        public TextOptions delimiter(String d) { this.delimiter = d; return this; }
        public TextOptions quote(String q) { this.quote = q; return this; }
        public TextOptions includeHeader(boolean b) { this.includeHeader = b; return this; }
        public TextOptions column(String c) { this.column = c; return this; }
        public TextOptions trim(boolean b) { this.trim = b; return this; }
        public TextOptions bufferSize(int n) { this.bufferSize = n; return this; }
        public TextOptions maxColumnWidth(int n) { this.maxColumnWidth = n; return this; }
        public TextOptions compress(boolean b) { this.compress = b; return this; }

        public TextFormat format() { return format; }
        public String charset() { return charset; }
        public String delimiter() { return delimiter; }
        public String quote() { return quote; }
        public boolean includeHeader() { return includeHeader; }
        public String column() { return column; }
        public boolean trim() { return trim; }
        public int bufferSize() { return bufferSize; }
        public int maxColumnWidth() { return maxColumnWidth; }
        public boolean compress() { return compress; }
    }
}
