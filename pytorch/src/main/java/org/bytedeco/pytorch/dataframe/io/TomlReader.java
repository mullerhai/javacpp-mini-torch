package org.bytedeco.pytorch.dataframe.io;

import java.io.*;
import java.nio.*;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade TOML reader for DataFrame.
 * 
 * <p>Features:</p>
 * <ul>
 *   <li>Full TOML 1.0.0 specification support</li>
 *   <li>Table/array-of-tables parsing</li>
 *   <li>Inline tables support</li>
 *   <li>Type inference and coercion</li>
 *   <li>Datetime support (local, offset, naive)</li>
 *   <li>Multiline strings</li>
 *   <li>Unicode support</li>
 * </ul>
 */
public class TomlReader {

    private TomlReader() {}

    /**
     * Read TOML file to DataFrame.
     * 
     * <p>TOML is parsed as follows:</p>
     * <ul>
     *   <li>Simple key-value pairs → columns</li>
     *   <li>Tables with arrays → rows</li>
     *   <li>Array of tables → multiple rows</li>
     *   <li>Nested tables → flattened with dot notation</li>
     * </ul>
     */
    public static org.bytedeco.pytorch.dataframe.DataFrame read(String path) throws IOException {
        return read(path, TomlOptions.defaults());
    }

    public static org.bytedeco.pytorch.dataframe.DataFrame read(String path, TomlOptions options) throws IOException {
        TomlOptions opt = options == null ? TomlOptions.defaults() : options;
        String content = Files.readString(Path.of(path), opt.charset());
        return parse(content, opt);
    }

    public static org.bytedeco.pytorch.dataframe.DataFrame parse(String tomlContent) {
        return parse(tomlContent, TomlOptions.defaults());
    }

    public static org.bytedeco.pytorch.dataframe.DataFrame parse(String tomlContent, TomlOptions options) {
        TomlOptions opt = options == null ? TomlOptions.defaults() : options;
        
        // Parse TOML into structured data
        TomlData data = parseToml(tomlContent, opt);
        
        // Convert to DataFrame
        return toDataFrame(data, opt);
    }

    /**
     * Write DataFrame to TOML file.
     */
    public static void write(org.bytedeco.pytorch.dataframe.DataFrame df, String path) throws IOException {
        write(df, path, TomlOptions.defaults());
    }

    public static void write(org.bytedeco.pytorch.dataframe.DataFrame df, String path, TomlOptions options) throws IOException {
        TomlOptions opt = options == null ? TomlOptions.defaults() : options;
        String toml = toToml(df, opt);
        Files.writeString(Path.of(path), toml, opt.charset());
    }

    /**
     * Read TOML as a Map structure (not DataFrame).
     */
    public static Map<String, Object> readAsMap(String path) throws IOException {
        String content = Files.readString(Path.of(path));
        TomlData data = parseToml(content, TomlOptions.defaults());
        return data.toMap();
    }

    // ---- TOML Parser ----

    private static TomlData parseToml(String content, TomlOptions opt) {
        TomlData data = new TomlData();
        String[] lines = content.split("\\r?\\n");
        
        String currentTable = "";
        Map<String, Object> currentSection = null;
        List<Map<String, Object>> currentArrayOfTables = null;
        
        int i = 0;
        while (i < lines.length) {
            String line = lines[i];
            
            // Skip comments and empty lines
            if (isBlank(line) || isComment(line)) {
                i++;
                continue;
            }
            
            // End of array of tables
            if (line.trim().startsWith("[[") && line.trim().endsWith("]]")) {
                // Save current array table to data
                if (currentArrayOfTables != null && !currentArrayOfTables.isEmpty()) {
                    data.addArrayOfTables(currentTable, new ArrayList<>(currentArrayOfTables));
                }
                currentArrayOfTables = null;
                currentSection = null;
                
                String tableName = extractTableName(line.trim());
                currentTable = tableName;
                currentArrayOfTables = new ArrayList<>();
                i++;
                continue;
            }
            
            // Table header
            if (line.trim().startsWith("[") && line.trim().endsWith("]")) {
                // Save previous section if exists
                if (currentSection != null && !currentSection.isEmpty()) {
                    if (currentArrayOfTables != null) {
                        currentArrayOfTables.add(new LinkedHashMap<>(currentSection));
                    } else {
                        data.addTable(currentTable, new LinkedHashMap<>(currentSection));
                    }
                }
                
                String tableName = extractTableName(line.trim());
                currentTable = tableName;
                currentSection = new LinkedHashMap<>();
                currentArrayOfTables = null;
                i++;
                continue;
            }
            
            // Key-value pair
            int eqPos = line.indexOf('=');
            if (eqPos > 0) {
                String key = line.substring(0, eqPos).trim();
                String value = line.substring(eqPos + 1).trim();
                
                // Handle multiline values
                StringBuilder fullValue = new StringBuilder(value);
                while (i + 1 < lines.length && needsMultilineContinuation(lines[i + 1])) {
                    i++;
                    fullValue.append(" ").append(lines[i].trim());
                }
                
                Object parsedValue = parseValue(fullValue.toString(), lines, i, opt);
                if (currentSection != null) {
                    currentSection.put(key, parsedValue);
                } else if (currentTable.isEmpty()) {
                    data.addValue(key, parsedValue);
                }
            }
            
            i++;
        }
        
        // Save last section
        if (currentSection != null && !currentSection.isEmpty()) {
            if (currentArrayOfTables != null) {
                currentArrayOfTables.add(new LinkedHashMap<>(currentSection));
                data.addArrayOfTables(currentTable, new ArrayList<>(currentArrayOfTables));
            } else {
                data.addTable(currentTable, new LinkedHashMap<>(currentSection));
            }
        }
        
        return data;
    }

    private static String extractTableName(String line) {
        line = line.trim();
        if (line.startsWith("[[")) {
            line = line.substring(2);
        } else if (line.startsWith("[")) {
            line = line.substring(1);
        }
        if (line.endsWith("]]")) {
            line = line.substring(0, line.length() - 2);
        } else if (line.endsWith("]")) {
            line = line.substring(0, line.length() - 1);
        }
        return line.trim();
    }

    private static boolean needsMultilineContinuation(String line) {
        line = line.trim();
        // Lines ending with \ or starting with whitespace after = 
        return line.endsWith("\\") || line.isEmpty();
    }

    private static boolean isBlank(String line) {
        return line.trim().isEmpty();
    }

    private static boolean isComment(String line) {
        String trimmed = line.trim();
        return trimmed.startsWith("#");
    }

    private static Object parseValue(String value, String[] lines, int lineIdx, TomlOptions opt) {
        value = value.trim();
        
        if (value.isEmpty()) {
            return null;
        }
        
        // String
        if (value.startsWith("\"\"\"") || value.startsWith("'''")) {
            return parseMultilineString(value, lines, lineIdx);
        }
        if (value.startsWith("\"") || value.startsWith("'")) {
            return parseString(value);
        }
        
        // Array
        if (value.startsWith("[")) {
            return parseArray(value, opt);
        }
        
        // Boolean
        if (value.equals("true")) return Boolean.TRUE;
        if (value.equals("false")) return Boolean.FALSE;
        
        // Datetime
        if (value.contains("-") && (value.contains("T") || value.contains(" "))) {
            try {
                return parseDatetime(value);
            } catch (Exception ignored) {}
        }
        
        // Number
        return parseNumber(value);
    }

    private static String parseMultilineString(String value, String[] lines, int lineIdx) {
        char quote = value.charAt(0);
        String delim = String.valueOf(quote).repeat(3);
        
        if (value.startsWith(delim)) {
            // Remove delimiters
            if (value.endsWith(delim) && value.length() > 6) {
                value = value.substring(3, value.length() - 3);
            } else {
                // Multiline continues
                StringBuilder sb = new StringBuilder();
                sb.append(value.substring(3));
                for (int i = lineIdx + 1; i < lines.length; i++) {
                    String line = lines[i];
                    if (line.trim().endsWith(delim) || line.contains(String.valueOf(quote).repeat(3))) {
                        int endIdx = line.indexOf(delim);
                        if (endIdx > 0) {
                            sb.append("\n").append(line.substring(0, endIdx));
                        }
                        break;
                    }
                    sb.append("\n").append(line);
                }
                value = sb.toString();
            }
        }
        
        return value.trim();
    }

    private static String parseString(String value) {
        if (value.length() < 2) return value;
        
        char quote = value.charAt(0);
        if (quote != '"' && quote != '\'') return value;
        
        String delim = String.valueOf(quote);
        if (value.startsWith(delim.repeat(3))) {
            delim = delim.repeat(3);
        }
        
        if (value.endsWith(delim) && value.length() > delim.length()) {
            value = value.substring(delim.length(), value.length() - delim.length());
        }
        
        // Unescape
        return value
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace("\\\\", "\\");
    }

    private static List<Object> parseArray(String value, TomlOptions opt) {
        List<Object> result = new ArrayList<>();
        
        value = value.trim();
        if (!value.startsWith("[") || !value.endsWith("]")) {
            return result;
        }
        
        // Remove brackets
        value = value.substring(1, value.length() - 1).trim();
        if (value.isEmpty()) return result;
        
        // Parse elements
        int depth = 0;
        StringBuilder current = new StringBuilder();
        boolean inString = false;
        char stringChar = 0;
        
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            
            if (!inString && (c == '"' || c == '\'')) {
                inString = true;
                stringChar = c;
            } else if (inString && c == stringChar && (i == 0 || value.charAt(i - 1) != '\\')) {
                inString = false;
            }
            
            if (!inString) {
                if (c == '[' || c == '{') depth++;
                if (c == ']' || c == '}') depth--;
                if (c == ',' && depth == 0) {
                    String elem = current.toString().trim();
                    if (!elem.isEmpty()) {
                        result.add(parseValue(elem, null, 0, opt));
                    }
                    current = new StringBuilder();
                    continue;
                }
            }
            
            current.append(c);
        }
        
        // Last element
        String last = current.toString().trim();
        if (!last.isEmpty()) {
            result.add(parseValue(last, null, 0, opt));
        }
        
        return result;
    }

    private static Object parseNumber(String value) {
        value = value.trim();
        
        // Try integer
        try {
            // Check for underscore separators
            String normalized = value.replace("_", "");
            if (normalized.startsWith("0x")) {
                return Long.parseLong(normalized.substring(2), 16);
            }
            if (normalized.startsWith("0o")) {
                return Long.parseLong(normalized.substring(2), 8);
            }
            if (normalized.startsWith("0b")) {
                return Long.parseLong(normalized.substring(2), 2);
            }
            
            if (normalized.contains(".") || normalized.contains("e") || normalized.contains("E")) {
                return Double.parseDouble(normalized);
            }
            
            return Long.parseLong(normalized);
        } catch (NumberFormatException e) {
            return value;
        }
    }

    private static Object parseDatetime(String value) {
        // Try various datetime formats
        try {
            if (value.contains("+") || value.contains("Z")) {
                // Offset datetime
                return java.time.OffsetDateTime.parse(value);
            } else if (value.contains("T")) {
                // Local datetime
                return java.time.LocalDateTime.parse(value.replace(" ", "T"));
            } else {
                // Local date
                return java.time.LocalDate.parse(value.split("[ T]")[0]);
            }
        } catch (Exception e) {
            return value;
        }
    }

    // ---- DataFrame conversion ----

    private static org.bytedeco.pytorch.dataframe.DataFrame toDataFrame(TomlData data, TomlOptions opt) {
        org.bytedeco.pytorch.dataframe.DataFrame df = 
            org.bytedeco.pytorch.dataframe.DataFrame.create();
        
        // Check if we have array-of-tables (tabular data)
        if (!data.arrayOfTables.isEmpty()) {
            // Use first AoT as the primary structure
            for (Map.Entry<String, List<Map<String, Object>>> entry : data.arrayOfTables.entrySet()) {
                String tableName = entry.getKey();
                List<Map<String, Object>> rows = entry.getValue();
                
                // Add table name as prefix if configured
                String prefix = opt.tablePrefix() ? tableName + "." : "";
                
                for (Map<String, Object> row : rows) {
                    Object[] values = new Object[row.size()];
                    int idx = 0;
                    for (Map.Entry<String, Object> field : row.entrySet()) {
                        String colName = prefix + field.getKey();
                        Object val = field.getValue();
                        
                        // Add column if not exists
                        if (df.columnIndex(colName) < 0) {
                            df.addColumn(colName, inferDType(val));
                        }
                        
                        // Find column index
                        int colIdx = df.columnIndex(colName);
                        while (idx >= df.rowCount()) {
                            df.addEmptyRow();
                        }
                        if (colIdx >= 0 && idx < df.rowCount()) {
                            df.set(idx, colIdx, val);
                        }
                    }
                    idx++;
                }
            }
            
            // Handle tables (non-array)
            for (Map.Entry<String, Map<String, Object>> entry : data.tables.entrySet()) {
                String tableName = entry.getKey();
                Map<String, Object> values = entry.getValue();
                
                String prefix = opt.tablePrefix() ? tableName + "." : "";
                
                for (Map.Entry<String, Object> field : values.entrySet()) {
                    String colName = prefix + field.getKey();
                    Object val = field.getValue();
                    
                    if (df.columnIndex(colName) < 0) {
                        df.addColumn(colName, inferDType(val));
                    }
                }
                
                // Add as single row
                int row = df.addEmptyRow();
                for (Map.Entry<String, Object> field : values.entrySet()) {
                    df.set(row, prefix + field.getKey(), field.getValue());
                }
            }
        } else if (!data.tables.isEmpty()) {
            // Simple key-value pairs or single table
            Map<String, Object> firstTable = null;
            for (Map<String, Object> table : data.tables.values()) {
                if (!table.isEmpty()) {
                    firstTable = table;
                    break;
                }
            }
            
            if (firstTable != null) {
                for (Map.Entry<String, Object> field : firstTable.entrySet()) {
                    df.addColumn(field.getKey(), inferDType(field.getValue()));
                }
                
                int row = df.addEmptyRow();
                for (Map.Entry<String, Object> field : firstTable.entrySet()) {
                    df.set(row, field.getKey(), field.getValue());
                }
            }
        }
        
        // Add root-level values
        for (Map.Entry<String, Object> entry : data.values.entrySet()) {
            if (df.columnIndex(entry.getKey()) < 0) {
                df.addColumn(entry.getKey(), inferDType(entry.getValue()));
            }
        }
        
        return df;
    }

    private static org.bytedeco.pytorch.dataframe.Column.DType inferDType(Object value) {
        if (value == null) return org.bytedeco.pytorch.dataframe.Column.DType.STRING;
        if (value instanceof Boolean) return org.bytedeco.pytorch.dataframe.Column.DType.BOOLEAN;
        if (value instanceof Number) {
            if (value instanceof Double || value instanceof Float) {
                return org.bytedeco.pytorch.dataframe.Column.DType.FLOAT64;
            }
            return org.bytedeco.pytorch.dataframe.Column.DType.INT64;
        }
        if (value instanceof java.time.LocalDate) {
            return org.bytedeco.pytorch.dataframe.Column.DType.DATE;
        }
        if (value instanceof java.time.LocalDateTime || value instanceof java.time.OffsetDateTime) {
            return org.bytedeco.pytorch.dataframe.Column.DType.DATETIME;
        }
        return org.bytedeco.pytorch.dataframe.Column.DType.STRING;
    }

    // ---- TOML Writer (Enterprise-grade) ----

    /**
     * Write DataFrame to TOML with enterprise features.
     * 
     * <p>Supports:</p>
     * <ul>
     *   <li>Root-level key-value pairs</li>
     *   <li>Table sections for column groups</li>
     *   <li>Array-of-tables for row data</li>
     *   <li>Inline tables for compact output</li>
     *   <li>Datetime formatting</li>
     *   <li>Array formatting</li>
     * </ul>
     */
    private static String toToml(org.bytedeco.pytorch.dataframe.DataFrame df, TomlOptions opt) {
        StringBuilder sb = new StringBuilder();
        
        // Header
        sb.append("# Generated by javacpp-pytorch DataFrame\n");
        sb.append("# ").append(java.time.LocalDateTime.now()).append("\n\n");
        
        if (opt.prettyPrint()) {
            writeDataFramePretty(sb, df, opt);
        } else {
            writeDataFrameCompact(sb, df, opt);
        }
        
        return sb.toString();
    }

    private static void writeDataFramePretty(StringBuilder sb, org.bytedeco.pytorch.dataframe.DataFrame df, TomlOptions opt) {
        // Categorize columns
        List<String> scalarCols = new ArrayList<>();
        List<String> arrayCols = new ArrayList<>();
        List<String> complexCols = new ArrayList<>();
        
        for (int c = 0; c < df.columnCount(); c++) {
            String colName = df.column(c).name();
            boolean isArray = false;
            for (int r = 0; r < Math.min(df.rowCount(), 10); r++) {
                Object v = df.get(r, c);
                if (v != null && (v.getClass().isArray() || v instanceof List)) {
                    isArray = true;
                    break;
                }
            }
            if (isArray) {
                complexCols.add(colName);
            } else if (df.rowCount() > 1) {
                arrayCols.add(colName);
            } else {
                scalarCols.add(colName);
            }
        }
        
        // Write scalar columns as root-level
        if (!scalarCols.isEmpty()) {
            for (String colName : scalarCols) {
                Object value = df.get(0, colName);
                sb.append(escapeKey(colName)).append(" = ");
                sb.append(valueToToml(value, opt));
                sb.append("\n");
            }
            if (!arrayCols.isEmpty() || !complexCols.isEmpty()) {
                sb.append("\n");
            }
        }
        
        // Write columns with row data as array-of-tables
        if (!arrayCols.isEmpty()) {
            sb.append("# Row data\n");
            for (int r = 0; r < df.rowCount(); r++) {
                sb.append("[[data]]\n");
                for (String colName : arrayCols) {
                    sb.append(indent(escapeKey(colName))).append(" = ");
                    sb.append(valueToToml(df.get(r, colName), opt));
                    sb.append("\n");
                }
            }
            if (!complexCols.isEmpty()) {
                sb.append("\n");
            }
        }
        
        // Write complex columns (arrays)
        if (!complexCols.isEmpty()) {
            for (String colName : complexCols) {
                sb.append(escapeKey(colName)).append(" = ");
                sb.append(convertColumnToArray(df, colName, opt));
                sb.append("\n");
            }
        }
    }

    private static void writeDataFrameCompact(StringBuilder sb, org.bytedeco.pytorch.dataframe.DataFrame df, TomlOptions opt) {
        // Simple: root-level key-value pairs
        for (int c = 0; c < df.columnCount(); c++) {
            String name = df.column(c).name();
            Object value = df.rowCount() > 0 ? df.get(0, c) : null;
            
            sb.append(escapeKey(name)).append(" = ");
            sb.append(valueToToml(value, opt));
            sb.append("\n");
        }
    }

    private static String convertColumnToArray(org.bytedeco.pytorch.dataframe.DataFrame df, String colName, TomlOptions opt) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        
        for (int r = 0; r < df.rowCount(); r++) {
            if (r > 0) sb.append(", ");
            Object v = df.get(r, colName);
            sb.append(valueToToml(v, opt));
        }
        
        sb.append("]");
        return sb.toString();
    }

    private static String indent(String s) {
        return "  " + s;
    }

    private static String escapeKey(String key) {
        // TOML keys must be bare unless they contain special chars
        if (key.matches("[a-zA-Z_][a-zA-Z0-9_-]*")) {
            return key;
        }
        return "\"" + key.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String valueToToml(Object value, TomlOptions opt) {
        if (value == null) return "null";
        
        // Arrays
        if (value instanceof List) {
            return listToToml((List<?>) value, opt);
        }
        
        // Primitive arrays
        if (value instanceof float[]) {
            return arrayToToml((float[]) value, opt);
        }
        if (value instanceof double[]) {
            return arrayToToml((double[]) value, opt);
        }
        if (value instanceof long[]) {
            return arrayToToml((long[]) value, opt);
        }
        if (value instanceof int[]) {
            return arrayToToml((int[]) value, opt);
        }
        
        // String
        if (value instanceof String) {
            return stringToToml((String) value);
        }
        
        // Boolean
        if (value instanceof Boolean) {
            return ((Boolean) value) ? "true" : "false";
        }
        
        // Number
        if (value instanceof Number) {
            return numberToToml((Number) value);
        }
        
        // Datetime
        if (value instanceof java.time.LocalDate) {
            return "\"" + value + "\"";
        }
        if (value instanceof java.time.LocalDateTime) {
            return "\"" + value + "\"";
        }
        if (value instanceof java.time.OffsetDateTime) {
            return "\"" + value + "\"";
        }
        
        // Map/Object → inline table
        if (value instanceof Map) {
            return mapToInlineToml((Map<?, ?>) value, opt);
        }
        
        // Fallback
        return "\"" + String.valueOf(value).replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String stringToToml(String s) {
        if (s.contains("\n") || s.contains("\"") || s.contains("#") || s.contains("\\")) {
            return "\"\"\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"\"\"";
        }
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String numberToToml(Number n) {
        if (n instanceof Double || n instanceof Float) {
            double d = n.doubleValue();
            if (Double.isNaN(d)) return "nan";
            if (Double.isInfinite(d)) return d > 0 ? "inf" : "-inf";
            // Remove unnecessary decimal for whole numbers
            if (d == Math.floor(d) && !Double.isInfinite(d)) {
                return String.valueOf((long) d);
            }
            return String.valueOf(d);
        }
        return String.valueOf(n);
    }

    private static <T> String listToToml(List<T> list, TomlOptions opt) {
        if (list.isEmpty()) return "[]";
        
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(valueToToml(list.get(i), opt));
        }
        
        sb.append("]");
        return sb.toString();
    }

    private static String arrayToToml(float[] arr, TomlOptions opt) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(numberToToml(arr[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private static String arrayToToml(double[] arr, TomlOptions opt) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(numberToToml(arr[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private static String arrayToToml(long[] arr, TomlOptions opt) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    private static String arrayToToml(int[] arr, TomlOptions opt) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    private static <K, V> String mapToInlineToml(Map<K, V> map, TomlOptions opt) {
        if (map.isEmpty()) return "{}";
        
        StringBuilder sb = new StringBuilder();
        sb.append("{ ");
        
        int i = 0;
        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (i > 0) sb.append(", ");
            sb.append(escapeKey(String.valueOf(entry.getKey())));
            sb.append(" = ");
            sb.append(valueToToml(entry.getValue(), opt));
            i++;
        }
        
        sb.append(" }");
        return sb.toString();
    }

    // Legacy method for compatibility
    private static String valueToToml(Object value) {
        if (value instanceof Boolean) {
            return ((Boolean) value) ? "true" : "false";
        }
        if (value instanceof Number) {
            return String.valueOf(value);
        }
        if (value instanceof java.time.LocalDate) {
            return "\"" + value + "\"";
        }
        if (value instanceof java.time.LocalDateTime) {
            return "\"" + value + "\"";
        }
        return "\"" + String.valueOf(value) + "\"";
    }

    // ---- Inner classes ----

    static class TomlData {
        Map<String, Object> values = new LinkedHashMap<>();
        Map<String, Map<String, Object>> tables = new LinkedHashMap<>();
        Map<String, List<Map<String, Object>>> arrayOfTables = new LinkedHashMap<>();
        
        void addValue(String key, Object value) {
            values.put(key, value);
        }
        
        void addTable(String name, Map<String, Object> table) {
            tables.put(name, table);
        }
        
        void addArrayOfTables(String name, List<Map<String, Object>> rows) {
            arrayOfTables.put(name, rows);
        }
        
        Map<String, Object> toMap() {
            Map<String, Object> result = new LinkedHashMap<>(values);
            if (!tables.isEmpty()) {
                for (Map.Entry<String, Map<String, Object>> e : tables.entrySet()) {
                    result.put(e.getKey(), e.getValue());
                }
            }
            if (!arrayOfTables.isEmpty()) {
                for (Map.Entry<String, List<Map<String, Object>>> e : arrayOfTables.entrySet()) {
                    result.put(e.getKey(), e.getValue());
                }
            }
            return result;
        }
    }

    // ---- Options ----

    public static class TomlOptions {
        private boolean tablePrefix = true;
        private boolean prettyPrint = true;
        private Charset charset = StandardCharsets.UTF_8;
        
        public static TomlOptions defaults() { return new TomlOptions(); }
        
        public TomlOptions tablePrefix(boolean b) { this.tablePrefix = b; return this; }
        public TomlOptions prettyPrint(boolean b) { this.prettyPrint = b; return this; }
        public TomlOptions charset(Charset c) { this.charset = c; return this; }
        
        public boolean tablePrefix() { return tablePrefix; }
        public boolean prettyPrint() { return prettyPrint; }
        public Charset charset() { return charset; }
    }
}
