package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade YAML writer for configuration files.
 * 
 * <p>Writes DataFrames to YAML format with support for:
 * <ul>
 *   <li>Block scalar styles (literal, folded)</li>
 *   <li>Flow style for compact output</li>
 *   <li>Multi-document output</li>
 *   <li>Custom key formatting</li>
 * </ul>
 */
public class YamlWriter {

    private YamlWriter() {}

    /**
     * Write DataFrame to YAML file.
     */
    public static void write(DataFrame df, String path) throws IOException {
        write(df, path, YamlOptions.defaults());
    }

    public static void write(DataFrame df, String path, YamlOptions options) throws IOException {
        YamlOptions opt = options == null ? YamlOptions.defaults() : options;
        
        Charset charset = Charset.forName(opt.charset());
        Files.createDirectories(Path.of(path).getParent());
        
        try (BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(
                    opt.compress() ? new java.util.zip.GZIPOutputStream(
                        Files.newOutputStream(Path.of(path)))
                        : Files.newOutputStream(Path.of(path)),
                    charset))) {
            
            if (opt.multipleDocuments()) {
                writeMultipleDocuments(df, writer, opt);
            } else {
                writeSingleDocument(df, writer, opt);
            }
        }
    }

    private static void writeSingleDocument(DataFrame df, BufferedWriter writer, YamlOptions opt) throws IOException {
        if (opt.includeHeader()) {
            writer.write("# YAML written by DataFrame\n");
            writer.write("# Rows: " + df.rowCount() + "\n");
            writer.write("# Columns: " + df.columnCount() + "\n");
        }
        
        if (opt.style().equals(YamlOptions.Style.FLOW)) {
            writeFlowStyle(df, writer);
        } else {
            writeBlockStyle(df, writer, opt);
        }
    }

    private static void writeBlockStyle(DataFrame df, BufferedWriter writer, YamlOptions opt) throws IOException {
        writer.write("data:\n");
        
        if (opt.orient().equals(YamlOptions.Orient.ROWS)) {
            for (int r = 0; r < df.rowCount(); r++) {
                writer.write("  - ");
                boolean first = true;
                for (int c = 0; c < df.columnCount(); c++) {
                    if (!first) writer.write(", ");
                    first = false;
                    Object val = df.get(r, c);
                    writeYamlValue(writer, val, 2, opt);
                }
                writer.newLine();
            }
        } else {
            for (int c = 0; c < df.columnCount(); c++) {
                Column col = df.column(c);
                writer.write("  " + escapeKey(col.name()) + ":\n");
                for (int r = 0; r < df.rowCount(); r++) {
                    Object val = df.get(r, c);
                    writer.write("    - ");
                    writeYamlValue(writer, val, 4, opt);
                    writer.newLine();
                }
            }
        }
    }

    private static void writeFlowStyle(DataFrame df, BufferedWriter writer, YamlOptions opt) throws IOException {
        writer.write("data: ");
        
        if (opt.orient().equals(YamlOptions.Orient.ROWS)) {
            writer.write("[");
            for (int r = 0; r < df.rowCount(); r++) {
                if (r > 0) writer.write(", ");
                writer.write("{");
                for (int c = 0; c < df.columnCount(); c++) {
                    if (c > 0) writer.write(", ");
                    writer.write(escapeKey(df.column(c).name()) + ": ");
                    writeYamlValue(writer, df.get(r, c), 0, opt);
                }
                writer.write("}");
            }
            writer.write("\n");
        } else {
            writer.write("{");
            for (int c = 0; c < df.columnCount(); c++) {
                if (c > 0) writer.write(", ");
                writer.write(escapeKey(df.column(c).name()) + ": [");
                for (int r = 0; r < df.rowCount(); r++) {
                    if (r > 0) writer.write(", ");
                    writeYamlValue(writer, df.get(r, c), 0, opt);
                }
                writer.write("]");
            }
            writer.write("}\n");
        }
    }

    private static void writeMultipleDocuments(DataFrame df, BufferedWriter writer, YamlOptions opt) throws IOException {
        int rowsPerDoc = opt.rowsPerDocument() > 0 ? opt.rowsPerDocument() : df.rowCount();
        
        for (int start = 0; start < df.rowCount(); start += rowsPerDoc) {
            int end = Math.min(start + rowsPerDoc, df.rowCount());
            
            writer.write("---\n");
            writer.write("# Document " + ((start / rowsPerDoc) + 1) + "\n");
            writer.write("rows:\n");
            writer.write("  start: " + start + "\n");
            writer.write("  end: " + end + "\n");
            writer.write("data:\n");
            
            for (int r = start; r < end; r++) {
                writer.write("  - ");
                boolean first = true;
                for (int c = 0; c < df.columnCount(); c++) {
                    if (!first) writer.write(", ");
                    first = false;
                    writeYamlValue(writer, df.get(r, c), 2, opt);
                }
                writer.newLine();
            }
            
            if (start + rowsPerDoc < df.rowCount()) {
                writer.newLine();
            }
        }
    }

    private static void writeYamlValue(BufferedWriter writer, Object val, int indent, YamlOptions opt) throws IOException {
        if (val == null) {
            writer.write("null");
        } else if (val instanceof Number) {
            writer.write(val.toString());
        } else if (val instanceof Boolean) {
            writer.write(Boolean.TRUE.equals(val) ? "true" : "false");
        } else if (val instanceof String) {
            String s = (String) val;
            if (needsQuoting(s, opt)) {
                writer.write("\"" + escapeString(s) + "\"");
            } else {
                writer.write(s);
            }
        } else if (val instanceof List || val instanceof Object[]) {
            writer.write("[");
            Iterable<?> it = val instanceof List ? (List<?>)val : Arrays.asList((Object[])val);
            boolean first = true;
            for (Object item : it) {
                if (!first) writer.write(", ");
                first = false;
                writeYamlValue(writer, item, indent, opt);
            }
            writer.write("]");
        } else if (val instanceof Map) {
            writer.write("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>)val).entrySet()) {
                if (!first) writer.write(", ");
                first = false;
                writer.write(e.getKey() + ": ");
                writeYamlValue(writer, e.getValue(), indent, opt);
            }
            writer.write("}");
        } else if (val instanceof byte[]) {
            writer.write("!<binary> \"");
            writer.write(Base64.getEncoder().encodeToString((byte[]) val));
            writer.write("\"");
        } else {
            String s = val.toString();
            if (needsQuoting(s, opt)) {
                writer.write("\"" + escapeString(s) + "\"");
            } else {
                writer.write(s);
            }
        }
    }

    private static boolean needsQuoting(String s, YamlOptions opt) {
        if (s.isEmpty()) return true;
        if (s.contains("\n") || s.contains("\r") || s.contains(":") 
            || s.contains("#") || s.contains("\"") || s.contains("'")
            || s.contains("[") || s.contains("]") || s.contains("{") || s.contains("}")
            || s.contains(",") || s.contains("&") || s.contains("*")
            || s.contains("!") || s.contains("|") || s.contains(">")
            || s.equals("null") || s.equals("true") || s.equals("false")
            || s.equals("True") || s.equals("False") || s.equals("NULL")
            || s.equals("~") || s.equals("yes") || s.equals("no")
            || s.startsWith(" ") || s.endsWith(" ")
            || s.startsWith("-") || s.startsWith("?")
            || Character.isDigit(s.charAt(0)) && s.matches(".*[\\: #\\[\\]{}&*!|>\\'\"%@`].*")) {
            return true;
        }
        return false;
    }

    private static String escapeString(String s) {
        return s.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t")
                 .replace("\b", "\\b")
                 .replace("\f", "\\f");
    }

    private static String escapeKey(String key) {
        if (key == null || key.isEmpty()) return "\"\"";
        for (char c : key.toCharArray()) {
            if (Character.isWhitespace(c) || c == ':' || c == '#' || c == '"' || c == '\'') {
                return "\"" + escapeString(key) + "\"";
            }
        }
        return key;
    }

    // ====================== Options ======================

    public enum Style { BLOCK, FLOW }
    public enum Orient { ROWS, COLUMNS }

    public static class YamlOptions {
        private Style style = Style.BLOCK;
        private Orient orient = Orient.ROWS;
        private String charset = "UTF-8";
        private boolean includeHeader = true;
        private boolean compress = false;
        private boolean multipleDocuments = false;
        private int rowsPerDocument = 1000;

        public static YamlOptions defaults() { return new YamlOptions(); }

        public YamlOptions style(Style s) { this.style = s; return this; }
        public YamlOptions orient(Orient o) { this.orient = o; return this; }
        public YamlOptions charset(String c) { this.charset = c; return this; }
        public YamlOptions includeHeader(boolean b) { this.includeHeader = b; return this; }
        public YamlOptions compress(boolean b) { this.compress = b; return this; }
        public YamlOptions multipleDocuments(boolean b) { this.multipleDocuments = b; return this; }
        public YamlOptions rowsPerDocument(int n) { this.rowsPerDocument = n; return this; }

        public Style style() { return style; }
        public Orient orient() { return orient; }
        public String charset() { return charset; }
        public boolean includeHeader() { return includeHeader; }
        public boolean compress() { return compress; }
        public boolean multipleDocuments() { return multipleDocuments; }
        public int rowsPerDocument() { return rowsPerDocument; }
    }
}
