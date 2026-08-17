/*
 * GraphX: JSON-Graph I/O.
 *
 * Inspired by networkx.readwrite.json_graph.node_link / adj_data.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.io;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.AttrMap;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class JSONGraph {
    private JSONGraph() {}

    // =========================================================================
    // Node-link format
    // =========================================================================

    /** Minimal JSON parser (string/number/bool/null/array/object). */
    public static Object parse(String s) {
        return new JsonParser(s).parseValue();
    }

    public static String toJson(Object o) {
        return new JsonWriter(o).toJson();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static Graph<Object> fromNodeLinkJson(String json) {
        Object parsed = parse(json);
        if (!(parsed instanceof Map)) throw new IllegalArgumentException("Expected JSON object at root");
        Map<String, Object> root = (Map<String, Object>) parsed;
        Object directed = root.get("directed");
        Graph<Object> g = (Boolean.TRUE.equals(directed)) ? new Graph<>() : new Graph<>();
        // nodes
        Object nodesObj = root.get("nodes");
        if (nodesObj instanceof List) {
            for (Object nodeEntry : (List<?>) nodesObj) {
                if (nodeEntry instanceof Map) {
                    Map<String, Object> ne = (Map<String, Object>) nodeEntry;
                    Object id = ne.get("id");
                    if (id == null) continue;
                    g.addNode(id);
                    // Copy all other keys as node attributes
                    for (Map.Entry<String, Object> e : ne.entrySet()) {
                        if (!e.getKey().equals("id")) {
                            g.setNodeAttribute(id, e.getKey(), e.getValue());
                        }
                    }
                }
            }
        }
        // links
        Object linksObj = root.get("links");
        if (linksObj instanceof List) {
            for (Object linkEntry : (List<?>) linksObj) {
                if (linkEntry instanceof Map) {
                    Map<String, Object> le = (Map<String, Object>) linkEntry;
                    Object source = le.get("source");
                    Object target = le.get("target");
                    if (source == null || target == null) continue;
                    Map<String, Object> attrs = new LinkedHashMap<>();
                    for (Map.Entry<String, Object> e : le.entrySet()) {
                        if (!e.getKey().equals("source") && !e.getKey().equals("target")) {
                            attrs.put(e.getKey(), e.getValue());
                        }
                    }
                    if (attrs.isEmpty()) g.addEdge(source, target);
                    else g.addEdge(source, target, attrs);
                }
            }
        }
        // graph-level attributes
        Object graphAttrs = root.get("graph");
        if (graphAttrs instanceof Map) {
            for (Map.Entry<String, Object> e : ((Map<String, Object>) graphAttrs).entrySet()) {
                // No graph-level attr setter; store as a phantom node attribute "graph"
                g.setNodeAttribute("__graph__" + e.getKey(), "graph", e.getValue());
            }
        }
        return g;
    }

    public static Graph<Object> readNodeLink(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
        }
        return fromNodeLinkJson(sb.toString());
    }

    public static <N> void writeNodeLink(Graph<N> g, String path) throws IOException {
        writeNodeLink(g, path, false);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static <N> void writeNodeLink(Graph<N> g, String path, boolean directed) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(toJson(toNodeLinkMap(g, directed)));
        }
    }

    @SuppressWarnings("rawtypes")
    static Map<String, Object> toNodeLinkMap(Graph g, boolean directed) {
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("directed", directed);
        root.put("multigraph", false);
        root.put("graph", new LinkedHashMap<String, Object>());
        List<Map<String, Object>> nodes = new ArrayList<>();
        for (Object n : g.nodes()) {
            Map<String, Object> ne = new LinkedHashMap<>();
            ne.put("id", n);
            AttrMap a = g.getNodeAttr(n);
            for (Map.Entry<String, Object> e : a.asMap().entrySet()) ne.put(e.getKey(), e.getValue());
            nodes.add(ne);
        }
        root.put("nodes", nodes);
        List<Map<String, Object>> links = new ArrayList<>();
        java.util.Set<Map.Entry<Object, Object>> seenEdges = new java.util.LinkedHashSet<>();
        for (Object oe : g.edges()) seenEdges.add((Map.Entry<Object, Object>) oe);
        for (Map.Entry<Object, Object> e : seenEdges) {
            Map<String, Object> le = new LinkedHashMap<>();
            le.put("source", e.getKey());
            le.put("target", e.getValue());
            AttrMap a = g.getEdgeAttr(e.getKey(), e.getValue());
            for (Map.Entry<String, Object> ea : a.asMap().entrySet()) le.put(ea.getKey(), ea.getValue());
            links.add(le);
        }
        root.put("links", links);
        return root;
    }

    // =========================================================================
    // Minimal JSON parser
    // =========================================================================

    static final class JsonParser {
        private final String s;
        private int p;
        JsonParser(String s) { this.s = s; this.p = 0; }

        Object parseValue() {
            skipWs();
            if (p >= s.length()) throw new RuntimeException("Unexpected end");
            char c = s.charAt(p);
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBool();
            if (c == 'n') { expectKeyword("null"); return null; }
            return parseNumber();
        }

        Map<String, Object> parseObject() {
            Map<String, Object> m = new LinkedHashMap<>();
            expect('{');
            skipWs();
            if (peek() == '}') { p++; return m; }
            while (true) {
                skipWs();
                String key = parseString();
                skipWs();
                expect(':');
                skipWs();
                Object value = parseValue();
                m.put(key, value);
                skipWs();
                if (peek() == ',') { p++; continue; }
                if (peek() == '}') { p++; return m; }
                throw new RuntimeException("Expected , or } at " + p);
            }
        }

        List<Object> parseArray() {
            List<Object> arr = new ArrayList<>();
            expect('[');
            skipWs();
            if (peek() == ']') { p++; return arr; }
            while (true) {
                skipWs();
                arr.add(parseValue());
                skipWs();
                if (peek() == ',') { p++; continue; }
                if (peek() == ']') { p++; return arr; }
                throw new RuntimeException("Expected , or ] at " + p);
            }
        }

        String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (p < s.length()) {
                char c = s.charAt(p);
                if (c == '"') { p++; return sb.toString(); }
                if (c == '\\') {
                    p++;
                    if (p >= s.length()) break;
                    char e = s.charAt(p++);
                    switch (e) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'n': sb.append('\n'); break;
                        case 't': sb.append('\t'); break;
                        case 'r': sb.append('\r'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        case 'u':
                            sb.append((char) Integer.parseInt(s.substring(p, p + 4), 16));
                            p += 4;
                            break;
                        default: sb.append(e);
                    }
                } else {
                    sb.append(c); p++;
                }
            }
            throw new RuntimeException("Unterminated string");
        }

        Object parseNumber() {
            int start = p;
            if (peek() == '-') p++;
            while (p < s.length() && "0123456789.eE+-".indexOf(s.charAt(p)) >= 0) p++;
            String num = s.substring(start, p);
            if (num.contains(".") || num.contains("e") || num.contains("E")) return Double.parseDouble(num);
            try { return Long.parseLong(num); } catch (NumberFormatException e) { return Double.parseDouble(num); }
        }

        Boolean parseBool() {
            if (s.startsWith("true", p)) { p += 4; return Boolean.TRUE; }
            if (s.startsWith("false", p)) { p += 5; return Boolean.FALSE; }
            throw new RuntimeException("Invalid bool at " + p);
        }

        void expect(char c) {
            skipWs();
            if (p < s.length() && s.charAt(p) == c) { p++; return; }
            throw new RuntimeException("Expected '" + c + "' at " + p);
        }

        void expectKeyword(String kw) {
            if (!s.startsWith(kw, p)) throw new RuntimeException("Expected " + kw + " at " + p);
            p += kw.length();
        }

        char peek() { return p < s.length() ? s.charAt(p) : '\0'; }

        void skipWs() {
            while (p < s.length() && Character.isWhitespace(s.charAt(p))) p++;
        }
    }

    // =========================================================================
    // Minimal JSON writer
    // =========================================================================

    static final class JsonWriter {
        private final Object obj;
        JsonWriter(Object obj) { this.obj = obj; }
        String toJson() { StringBuilder sb = new StringBuilder(); write(sb, obj); return sb.toString(); }
        static void write(StringBuilder sb, Object o) {
            if (o == null) sb.append("null");
            else if (o instanceof Boolean || o instanceof Number) sb.append(o);
            else if (o instanceof String) writeString(sb, (String) o);
            else if (o instanceof Map) { writeMap(sb, (Map<?, ?>) o); }
            else if (o instanceof List) { writeList(sb, (List<?>) o); }
            else writeString(sb, o.toString());
        }
        static void writeMap(StringBuilder sb, Map<?, ?> m) {
            sb.append('{'); boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                writeString(sb, String.valueOf(e.getKey()));
                sb.append(':');
                write(sb, e.getValue());
                first = false;
            }
            sb.append('}');
        }
        static void writeList(StringBuilder sb, List<?> l) {
            sb.append('['); boolean first = true;
            for (Object item : l) {
                if (!first) sb.append(',');
                write(sb, item);
                first = false;
            }
            sb.append(']');
        }
        static void writeString(StringBuilder sb, String s) {
            sb.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"': sb.append("\\\""); break;
                    case '\\': sb.append("\\\\"); break;
                    case '\n': sb.append("\\n"); break;
                    case '\t': sb.append("\\t"); break;
                    default:
                        if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                        else sb.append(c);
                }
            }
            sb.append('"');
        }
    }
}