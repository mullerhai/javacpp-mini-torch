/*
 * GraphX: GraphML I/O.
 *
 * GraphML is the standard XML format for graphs (http://graphml.graphdrawing.org/).
 * Inspired by networkx.readwrite.graphml. BSD 3-Clause license.
 *
 * Uses simple regex-based XML parsing to avoid pulling in a heavyweight
 * XML library. Suitable for typical graph files; for very large/complex files,
 * swap in a StAX or JAXB parser.
 */
package org.bytedeco.pytorch.graphx.io;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
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

public final class GraphML {
    private GraphML() {}

    private static final Pattern OPEN_TAG = Pattern.compile("<(graphml|graph|node|edge|data|key)([^>]*)(/?)>");
    private static final Pattern CLOSE_TAG = Pattern.compile("</(graphml|graph|node|edge|data|key)>");
    private static final Pattern ATTR = Pattern.compile("(\\w+)=\"([^\"]*)\"");

    // =========================================================================
    // Reading
    // =========================================================================

    public static Graph<Object> read(String path) throws IOException {
        return read(path, false);
    }

    public static Graph<Object> read(String path, boolean directed) throws IOException {
        StringBuilder xml = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) xml.append(line).append('\n');
        }
        return parseGraphML(xml.toString(), directed);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static Graph<Object> parseGraphML(String xml, boolean directed) {
        // Parse <key> definitions for attribute names → id, type
        Map<String, String> keyIds = new LinkedHashMap<>();
        Map<String, String> keyTypes = new LinkedHashMap<>();
        Map<String, String> keyFor = new LinkedHashMap<>(); // attr name → key id
        Matcher keyM = Pattern.compile("<key\\s+([^>]+)/?>", Pattern.DOTALL).matcher(xml);
        while (keyM.find()) {
            String attrs = keyM.group(1);
            String id = extractAttr(attrs, "id");
            String name = extractAttr(attrs, "attr.name");
            String type = extractAttr(attrs, "attr.type");
            String forAttr = extractAttr(attrs, "for");
            if (id != null) {
                keyIds.put(id, name == null ? id : name);
                keyTypes.put(id, type == null ? "string" : type);
                if (name != null) keyFor.put(name, id);
            }
        }

        Graph<Object> g = directed ? new Graph<>() : new Graph<>();
        // Parse <node> elements
        Matcher nodeM = Pattern.compile("<node\\s+([^>]+)>([\\s\\S]*?)</node>", Pattern.DOTALL).matcher(xml);
        while (nodeM.find()) {
            String attrs = nodeM.group(1);
            String inner = nodeM.group(2);
            String id = extractAttr(attrs, "id");
            if (id == null) continue;
            Object nodeId = coerce(id);
            g.addNode(nodeId);
            Map<String, Object> nodeAttrs = parseDataElements(inner, keyIds, keyTypes);
            for (Map.Entry<String, Object> e : nodeAttrs.entrySet()) {
                g.setNodeAttribute(nodeId, e.getKey(), e.getValue());
            }
        }
        // Parse <edge> elements
        Matcher edgeM = Pattern.compile("<edge\\s+([^>]+)>([\\s\\S]*?)</edge>", Pattern.DOTALL).matcher(xml);
        while (edgeM.find()) {
            String attrs = edgeM.group(1);
            String inner = edgeM.group(2);
            String source = extractAttr(attrs, "source");
            String target = extractAttr(attrs, "target");
            if (source == null || target == null) continue;
            Object u = coerce(source);
            Object v = coerce(target);
            Map<String, Object> edgeAttrs = parseDataElements(inner, keyIds, keyTypes);
            if (edgeAttrs.isEmpty()) g.addEdge(u, v);
            else g.addEdge(u, v, edgeAttrs);
        }
        return g;
    }

    static Map<String, Object> parseDataElements(String xml, Map<String, String> keyIds, Map<String, String> keyTypes) {
        Map<String, Object> result = new LinkedHashMap<>();
        Matcher m = Pattern.compile("<data\\s+([^>]*)>([\\s\\S]*?)</data>", Pattern.DOTALL).matcher(xml);
        while (m.find()) {
            String attrs = m.group(1);
            String value = m.group(2).trim();
            String keyId = extractAttr(attrs, "key");
            String attrName = keyIds.getOrDefault(keyId, keyId);
            String type = keyTypes.getOrDefault(keyId, "string");
            result.put(attrName, convertValue(value, type));
        }
        return result;
    }

    static String extractAttr(String attrs, String name) {
        Matcher m = Pattern.compile(name + "=\"([^\"]*)\"").matcher(attrs);
        return m.find() ? m.group(1) : null;
    }

    static Object convertValue(String s, String type) {
        switch (type) {
            case "int": try { return Long.parseLong(s); } catch (NumberFormatException e) { return 0L; }
            case "long": try { return Long.parseLong(s); } catch (NumberFormatException e) { return 0L; }
            case "double": try { return Double.parseDouble(s); } catch (NumberFormatException e) { return 0.0; }
            case "float": try { return Double.parseDouble(s); } catch (NumberFormatException e) { return 0.0; }
            case "boolean": return Boolean.parseBoolean(s);
            default: return s;
        }
    }

    static Object coerce(String s) {
        try { return Long.parseLong(s); } catch (NumberFormatException ignored) {}
        try { return Double.parseDouble(s); } catch (NumberFormatException ignored) {}
        return s;
    }

    // =========================================================================
    // Writing
    // =========================================================================

    public static <N> void write(Graph<N> g, String path) throws IOException {
        write(g, path, false);
    }

    public static <N> void write(Graph<N> g, String path, boolean directed) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            bw.write("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n");
            bw.write("         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
            bw.write("         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n");
            bw.write("         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");
            int keyId = 0;
            List<Map.Entry<String, String>> nodeKeys = new ArrayList<>();
            List<Map.Entry<String, String>> edgeKeys = new ArrayList<>();
            for (N n : g.nodes()) {
                AttrMap a = g.getNodeAttr(n);
                for (String name : a.keys()) {
                    if (nodeKeys.stream().noneMatch(e -> e.getKey().equals(name))) {
                        String id = "d" + (keyId++);
                        nodeKeys.add(new java.util.AbstractMap.SimpleImmutableEntry<>(name, id));
                        bw.write("  <key id=\"" + id + "\" for=\"node\" attr.name=\"" + name + "\" attr.type=\"string\"/>\n");
                    }
                }
            }
            for (Map.Entry<N, N> e : g.edges()) {
                AttrMap a = g.getEdgeAttr(e.getKey(), e.getValue());
                for (String name : a.keys()) {
                    if (edgeKeys.stream().noneMatch(en -> en.getKey().equals(name))) {
                        String id = "d" + (keyId++);
                        edgeKeys.add(new java.util.AbstractMap.SimpleImmutableEntry<>(name, id));
                        bw.write("  <key id=\"" + id + "\" for=\"edge\" attr.name=\"" + name + "\" attr.type=\"string\"/>\n");
                    }
                }
            }
            bw.write("  <graph id=\"G\" edgedefault=\"" + (directed ? "directed" : "undirected") + "\">\n");
            for (N n : g.nodes()) {
                bw.write("    <node id=\"" + escape(n.toString()) + "\">\n");
                AttrMap a = g.getNodeAttr(n);
                for (Map.Entry<String, String> k : nodeKeys) {
                    Object v = a.get(k.getKey());
                    if (v != null) bw.write("      <data key=\"" + k.getValue() + "\">" + escape(v.toString()) + "</data>\n");
                }
                bw.write("    </node>\n");
            }
            int edgeIdx = 0;
            for (Map.Entry<N, N> e : g.edges()) {
                bw.write("    <edge id=\"e" + (edgeIdx++) + "\" source=\"" + escape(e.getKey().toString()) + "\" target=\"" + escape(e.getValue().toString()) + "\">\n");
                AttrMap a = g.getEdgeAttr(e.getKey(), e.getValue());
                for (Map.Entry<String, String> k : edgeKeys) {
                    Object v = a.get(k.getKey());
                    if (v != null) bw.write("      <data key=\"" + k.getValue() + "\">" + escape(v.toString()) + "</data>\n");
                }
                bw.write("    </edge>\n");
            }
            bw.write("  </graph>\n</graphml>\n");
        }
    }

    static String escape(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;");
    }
}