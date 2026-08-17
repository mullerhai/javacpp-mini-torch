/*
 * GraphX: Adjacency List I/O.
 *
 * Inspired by networkx.readwrite.adjlist.
 * Format:
 *   # header line (number of nodes, then edges)
 *   node1
 *     neighbor1 [k1=v1 k2=v2 ...]
 *     neighbor2 [...]
 *   node2
 *     ...
 *
 * BSD 3-Clause license.
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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class AdjList {
    private AdjList() {}
    private static final Pattern KV = Pattern.compile("(\\w+)=(\\S+)");

    public static Graph<Object> read(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            return readUndirected(br);
        }
    }

    public static DiGraph<Object> readDirected(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            return readDirected(br);
        }
    }

    static Graph<Object> readUndirected(BufferedReader br) throws IOException {
        Graph<Object> g = new Graph<>();
        String line;
        Object currentNode = null;
        Map<String, Object> currentNodeAttrs = null;
        int indent = 0;
        while ((line = br.readLine()) != null) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;
            int lead = 0;
            while (lead < line.length() && line.charAt(lead) == ' ') lead++;
            if (lead == 0) {
                // New node
                String[] parts = trimmed.split("\\s+", 2);
                currentNode = EdgeList.coerce(parts[0]);
                g.addNode(currentNode);
                currentNodeAttrs = new LinkedHashMap<>();
                if (parts.length > 1) {
                    Map<String, Object> parsed = EdgeList.parseAttrs(java.util.Arrays.asList(parts[1].split("\\s+")));
                    for (Map.Entry<String, Object> e : parsed.entrySet()) {
                        currentNodeAttrs.put(e.getKey(), e.getValue());
                    }
                    // Set node attrs
                    for (Map.Entry<String, Object> e : currentNodeAttrs.entrySet()) {
                        g.setNodeAttribute(currentNode, e.getKey(), e.getValue());
                    }
                }
                indent = lead;
            } else {
                // Neighbor line — must be indented relative to current node
                if (lead > indent && currentNode != null) {
                    String[] parts = trimmed.split("\\s+", 2);
                    Object neighbor = EdgeList.coerce(parts[0]);
                    if (parts.length > 1) {
                        Map<String, Object> attrs = EdgeList.parseAttrs(java.util.Arrays.asList(parts[1].split("\\s+")));
                        g.addEdge(currentNode, neighbor, attrs);
                    } else {
                        g.addEdge(currentNode, neighbor);
                    }
                }
            }
        }
        return g;
    }

    static DiGraph<Object> readDirected(BufferedReader br) throws IOException {
        DiGraph<Object> g = new DiGraph<>();
        String line;
        Object currentNode = null;
        int indent = 0;
        while ((line = br.readLine()) != null) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;
            int lead = 0;
            while (lead < line.length() && line.charAt(lead) == ' ') lead++;
            if (lead == 0) {
                String[] parts = trimmed.split("\\s+", 2);
                currentNode = EdgeList.coerce(parts[0]);
                g.addNode(currentNode);
                if (parts.length > 1) {
                    Map<String, Object> attrs = EdgeList.parseAttrs(java.util.Arrays.asList(parts[1].split("\\s+")));
                    for (Map.Entry<String, Object> e : attrs.entrySet()) {
                        g.setNodeAttribute(currentNode, e.getKey(), e.getValue());
                    }
                }
                indent = lead;
            } else if (lead > indent && currentNode != null) {
                String[] parts = trimmed.split("\\s+", 2);
                Object neighbor = EdgeList.coerce(parts[0]);
                if (parts.length > 1) {
                    Map<String, Object> attrs = EdgeList.parseAttrs(java.util.Arrays.asList(parts[1].split("\\s+")));
                    g.addEdge(currentNode, neighbor, attrs);
                } else {
                    g.addEdge(currentNode, neighbor);
                }
            }
        }
        return g;
    }

    public static <N> void write(Graph<N> g, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("# GraphX adjacency list (undirected)\n");
            bw.write("# " + g.order() + " nodes, " + g.numberOfEdges() + " edges\n");
            for (N n : g.nodes()) {
                bw.write(EdgeList.formatNode(n));
                AttrMap na = g.getNodeAttr(n);
                if (!na.isEmpty()) {
                    bw.write(' ');
                    for (Map.Entry<String, Object> e : na.asMap().entrySet()) {
                        bw.write(e.getKey());
                        bw.write('=');
                        bw.write(EdgeList.formatAttrValue(e.getValue()));
                        bw.write(' ');
                    }
                }
                bw.write('\n');
                for (N m : g.neighbors(n)) {
                    bw.write("  ");
                    bw.write(EdgeList.formatNode(m));
                    AttrMap ea = g.getEdgeAttr(n, m);
                    if (!ea.isEmpty()) {
                        bw.write(' ');
                        for (Map.Entry<String, Object> e : ea.asMap().entrySet()) {
                            bw.write(e.getKey());
                            bw.write('=');
                            bw.write(EdgeList.formatAttrValue(e.getValue()));
                            bw.write(' ');
                        }
                    }
                    bw.write('\n');
                }
            }
        }
    }

    public static <N> void write(DiGraph<N> g, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("# GraphX adjacency list (directed)\n");
            bw.write("# " + g.order() + " nodes, " + g.numberOfEdges() + " edges\n");
            for (N n : g.nodes()) {
                bw.write(EdgeList.formatNode(n));
                AttrMap na = g.getNodeAttr(n);
                if (!na.isEmpty()) {
                    bw.write(' ');
                for (Map.Entry<String, Object> e : na.asMap().entrySet()) {
                    bw.write(e.getKey());
                    bw.write('=');
                    bw.write(EdgeList.formatAttrValue(e.getValue()));
                    bw.write(' ');
                }
                }
                bw.write('\n');
                for (N m : g.successors(n)) {
                    bw.write("  ");
                    bw.write(EdgeList.formatNode(m));
                    AttrMap ea = g.getEdgeAttr(n, m);
                    if (!ea.isEmpty()) {
                        bw.write(' ');
                        for (Map.Entry<String, Object> e : ea.asMap().entrySet()) {
                            bw.write(e.getKey());
                            bw.write('=');
                            bw.write(EdgeList.formatAttrValue(e.getValue()));
                            bw.write(' ');
                        }
                    }
                    bw.write('\n');
                }
            }
        }
    }
}