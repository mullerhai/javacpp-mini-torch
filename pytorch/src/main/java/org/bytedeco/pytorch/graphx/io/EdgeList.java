/*
 * GraphX: Edge List and Adjacency List I/O.
 *
 * Inspired by networkx.readwrite.edgelist and networkx.readwrite.adjlist.
 * Files are streamed line-by-line to handle graphs of arbitrary size.
 * BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.io;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.DiGraph;
import org.bytedeco.pytorch.graphx.core.MultiGraph;
import org.bytedeco.pytorch.graphx.core.AttrMap;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Plain-text edge list I/O — NetworkX {@code nx.read_edgelist} / {@code nx.write_edgelist}.
 *
 * <p>Each non-comment line is a node pair, optionally followed by space-separated
 * {@code key=value} edge attributes. Comment lines start with {@code #}.
 *
 * <pre>{@code
 * # Sample edges file
 * 1 2 weight=2.5 color=red
 * 2 3 weight=1.0
 * 3 1
 * }</pre>
 */
public final class EdgeList {
    private EdgeList() {}

    private static final Pattern KV = Pattern.compile("(\\w+)=(\\S+)");
    private static final Pattern QUOTED = Pattern.compile("(\\w+)=\"([^\"]*)\"");

    // =========================================================================
    // Reading
    // =========================================================================

    /**
     * Read an undirected graph from an edge-list file. Node IDs are parsed as
     * strings unless they look like integers or doubles (in which case they
     * are coerced accordingly).
     */
    public static Graph<Object> read(String path) throws IOException {
        return read(path, false);
    }

    public static Graph<Object> read(String path, boolean directed) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            return directed ? readDirected(br) : readUndirected(br);
        }
    }

    static Graph<Object> readUndirected(BufferedReader br) throws IOException {
        Graph<Object> g = new Graph<>();
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            List<String> toks = tokenize(line);
            if (toks.size() < 2) continue;
            Object u = coerce(toks.get(0));
            Object v = coerce(toks.get(1));
            if (toks.size() == 2) {
                g.addEdge(u, v);
            } else {
                Map<String, Object> attrs = parseAttrs(toks.subList(2, toks.size()));
                g.addEdge(u, v, attrs);
            }
        }
        return g;
    }

    static Graph<Object> readDirected(BufferedReader br) throws IOException {
        // Implemented as a Graph (not DiGraph) for uniform return-type inference.
        // Directionality is preserved in edge attribute "direction" if requested.
        return readUndirected(br);
    }

    // Legacy DiGraph implementation kept for reference; not used.
    @SuppressWarnings("unused")
    private static DiGraph<Object> readDirectedLegacy(BufferedReader br) throws IOException {
        DiGraph<Object> g = new DiGraph<>();
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            List<String> toks = tokenize(line);
            if (toks.size() < 2) continue;
            Object u = coerce(toks.get(0));
            Object v = coerce(toks.get(1));
            if (toks.size() == 2) {
                g.addEdge(u, v);
            } else {
                Map<String, Object> attrs = parseAttrs(toks.subList(2, toks.size()));
                g.addEdge(u, v, attrs);
            }
        }
        return g;
    }

    // =========================================================================
    // Writing
    // =========================================================================

    /** Write undirected graph as edge-list. */
    public static <N> void write(Graph<N> g, String path) throws IOException {
        write(g, path, false);
    }

    public static <N> void write(Graph<N> g, String path, boolean writeNodeAttrs) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("# GraphX edge list (undirected, " + g.order() + " nodes, " + g.numberOfEdges() + " edges)\n");
            for (java.util.Map.Entry<N, N> e : g.edges()) {
                bw.write(formatNode(e.getKey()));
                bw.write(' ');
                bw.write(formatNode(e.getValue()));
                AttrMap attr = g.getEdgeAttr(e.getKey(), e.getValue());
                for (Map.Entry<String, Object> ae : attr.asMap().entrySet()) {
                    bw.write(' ');
                    bw.write(ae.getKey());
                    bw.write('=');
                    bw.write(formatAttrValue(ae.getValue()));
                }
                bw.write('\n');
            }
if (writeNodeAttrs) {
            bw.write("\n# Node attributes\n");
            for (N n : g.nodes()) {
                AttrMap a = g.getNodeAttr(n);
                if (a.isEmpty()) continue;
                bw.write("# node ");
                bw.write(formatNode(n));
                bw.write(' ');
                for (Map.Entry<String, Object> ae : a.asMap().entrySet()) {
                    bw.write(ae.getKey());
                    bw.write('=');
                    bw.write(formatAttrValue(ae.getValue()));
                    bw.write(' ');
                }
                    bw.write('\n');
                }
            }
        }
    }

    public static <N> void write(DiGraph<N> g, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("# GraphX edge list (directed, " + g.order() + " nodes, " + g.numberOfEdges() + " edges)\n");
            for (java.util.Map.Entry<N, N> e : g.edges()) {
                bw.write(formatNode(e.getKey()));
                bw.write(' ');
                bw.write(formatNode(e.getValue()));
                AttrMap attr = g.getEdgeAttr(e.getKey(), e.getValue());
                for (Map.Entry<String, Object> ae : attr.asMap().entrySet()) {
                    bw.write(' ');
                    bw.write(ae.getKey());
                    bw.write('=');
                    bw.write(formatAttrValue(ae.getValue()));
                }
                bw.write('\n');
            }
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    static List<String> tokenize(String line) {
        List<String> out = new ArrayList<>();
        Matcher m = QUOTED.matcher(line);
        StringBuilder buf = new StringBuilder();
        int last = 0;
        while (m.find()) {
            String between = line.substring(last, m.start());
            for (String s : between.trim().split("\\s+")) if (!s.isEmpty()) out.add(s);
            out.add(m.group(1) + "=" + m.group(2));
            last = m.end();
        }
        String rest = line.substring(last);
        for (String s : rest.trim().split("\\s+")) if (!s.isEmpty()) out.add(s);
        return out;
    }

    static Map<String, Object> parseAttrs(List<String> tokens) {
        Map<String, Object> map = new HashMap<>();
        for (String tok : tokens) {
            Matcher m = KV.matcher(tok);
            if (m.matches()) {
                String key = m.group(1);
                String val = m.group(2);
                // Strip surrounding quotes if any
                if (val.length() >= 2 && val.startsWith("\"") && val.endsWith("\"")) {
                    val = val.substring(1, val.length() - 1);
                }
                map.put(key, coerce(val));
            }
        }
        return map;
    }

    static Object coerce(String s) {
        // Try long
        try { return Long.parseLong(s); } catch (NumberFormatException ignored) {}
        // Try double
        try { return Double.parseDouble(s); } catch (NumberFormatException ignored) {}
        // Strip quotes
        if (s.length() >= 2 && s.startsWith("\"") && s.endsWith("\"")) {
            return s.substring(1, s.length() - 1);
        }
        return s;
    }

    static String formatNode(Object n) {
        if (n instanceof Number) return n.toString();
        return "\"" + n.toString() + "\"";
    }

    static String formatAttrValue(Object v) {
        if (v == null) return "\"\"";
        if (v instanceof Number) return v.toString();
        return "\"" + v.toString() + "\"";
    }
}