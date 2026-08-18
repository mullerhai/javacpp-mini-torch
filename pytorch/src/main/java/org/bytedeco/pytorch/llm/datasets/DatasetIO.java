/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.datasets;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tiny JSON serializer/deserializer used by the {@code llm.tunning} examples to round-trip
 * dataset rows through temp files. Kept here as a small utility rather than in
 * {@link HfDataset} to keep that class focused on the {@code datasets}-style row schema.
 */
public final class DatasetIO {

    private DatasetIO() {}

    public static String toJsonString(Object v) {
        if (v == null) return "null";
        if (v instanceof String) return "\"" + escapeJson((String) v) + "\"";
        if (v instanceof Number || v instanceof Boolean) return v.toString();
        if (v instanceof Map) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (!first) sb.append(",");
                first = false;
                sb.append("\"").append(escapeJson(String.valueOf(e.getKey()))).append("\":");
                sb.append(toJsonString(e.getValue()));
            }
            sb.append("}");
            return sb.toString();
        }
        if (v instanceof List) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object e : (List<?>) v) {
                if (!first) sb.append(",");
                first = false;
                sb.append(toJsonString(e));
            }
            sb.append("]");
            return sb.toString();
        }
        return "\"" + escapeJson(v.toString()) + "\"";
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> fromJsonString(String json) {
        JsonReader r = new JsonReader(json);
        Object o = r.parse();
        if (o == null) return new LinkedHashMap<>();
        return (Map<String, Object>) o;
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    /** Minimal recursive-descent JSON parser sufficient for dataset I/O. */
    private static final class JsonReader {
        private final String s;
        private int p;
        JsonReader(String s) { this.s = s; this.p = 0; }
        Object parse() { skip(); return readValue(); }
        private Object readValue() {
            skip();
            char c = s.charAt(p);
            if (c == '{') return readObject();
            if (c == '[') return readArray();
            if (c == '"') return readString();
            if (c == 't' || c == 'f') return readBool();
            if (c == 'n') { p += 4; return null; }
            return readNumber();
        }
        private Map<String, Object> readObject() {
            Map<String, Object> m = new LinkedHashMap<>();
            p++; skip();
            if (p < s.length() && s.charAt(p) == '}') { p++; return m; }
            while (true) {
                skip();
                String k = readString();
                skip();
                if (s.charAt(p) != ':') throw new RuntimeException("expected ':' at " + p);
                p++;
                Object v = readValue();
                m.put(k, v);
                skip();
                char c = s.charAt(p);
                if (c == ',') { p++; continue; }
                if (c == '}') { p++; return m; }
                throw new RuntimeException("expected ',' or '}' at " + p);
            }
        }
        private List<Object> readArray() {
            List<Object> a = new ArrayList<>();
            p++; skip();
            if (p < s.length() && s.charAt(p) == ']') { p++; return a; }
            while (true) {
                a.add(readValue());
                skip();
                char c = s.charAt(p);
                if (c == ',') { p++; skip(); continue; }
                if (c == ']') { p++; return a; }
                throw new RuntimeException("expected ',' or ']' at " + p);
            }
        }
        private String readString() {
            if (s.charAt(p) != '"') throw new RuntimeException("expected '\"' at " + p);
            p++;
            StringBuilder sb = new StringBuilder();
            while (p < s.length()) {
                char c = s.charAt(p);
                if (c == '"') { p++; return sb.toString(); }
                if (c == '\\') {
                    p++;
                    char n = s.charAt(p++);
                    switch (n) {
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        default: sb.append(n);
                    }
                } else { sb.append(c); p++; }
            }
            throw new RuntimeException("unterminated string");
        }
        private Boolean readBool() {
            if (s.startsWith("true", p)) { p += 4; return Boolean.TRUE; }
            if (s.startsWith("false", p)) { p += 5; return Boolean.FALSE; }
            throw new RuntimeException("bad bool at " + p);
        }
        private Object readNumber() {
            int start = p;
            if (s.charAt(p) == '-') p++;
            while (p < s.length() && "0123456789.eE+-".indexOf(s.charAt(p)) >= 0) p++;
            String n = s.substring(start, p);
            if (n.contains(".") || n.contains("e") || n.contains("E")) return Double.parseDouble(n);
            return Long.parseLong(n);
        }
        private void skip() {
            while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++;
        }
    }
}