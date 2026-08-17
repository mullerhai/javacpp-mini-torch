/*
 * GraphX: GEXF I/O.
 *
 * GEXF (Graph Exchange XML Format) — used by Gephi.
 * Inspired by networkx.readwrite.gexf. BSD 3-Clause license.
 */
package org.bytedeco.pytorch.graphx.io;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.AttrMap;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class GEXF {
    private GEXF() {}

    public static Graph<Object> read(String path) throws IOException {
        return parseGEXF(readFile(path));
    }

    static String readFile(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line).append('\n');
        }
        return sb.toString();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static Graph<Object> parseGEXF(String xml) {
        Graph<Object> g = new Graph<>();
        // Parse <nodes>
        Matcher nodeM = Pattern.compile("<node\\s+([^>]+)/?>", Pattern.DOTALL).matcher(xml);
        while (nodeM.find()) {
            String attrs = nodeM.group(1);
            String id = GraphML.extractAttr(attrs, "id");
            String label = GraphML.extractAttr(attrs, "label");
            if (id == null) continue;
            Object nodeId = GraphML.coerce(id);
            g.addNode(nodeId);
            if (label != null) g.setNodeAttribute(nodeId, "label", label);
        }
        // Parse <edges>
        Matcher edgeM = Pattern.compile("<edge\\s+([^/>]+)/?>([\\s\\S]*?)</edge>", Pattern.DOTALL).matcher(xml);
        while (edgeM.find()) {
            String attrs = edgeM.group(1);
            String inner = edgeM.group(2);
            String source = GraphML.extractAttr(attrs, "source");
            String target = GraphML.extractAttr(attrs, "target");
            String weight = GraphML.extractAttr(attrs, "weight");
            if (source == null || target == null) continue;
            Object u = GraphML.coerce(source);
            Object v = GraphML.coerce(target);
            Map<String, Object> attrs2 = new LinkedHashMap<>();
            if (weight != null) attrs2.put("weight", GraphML.coerce(weight));
            if (!attrs2.isEmpty()) g.addEdge(u, v, attrs2);
            else g.addEdge(u, v);
        }
        return g;
    }

    public static <N> void write(Graph<N> g, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            bw.write("<gexf xmlns=\"http://www.gexf.net/1.2draft\" version=\"1.2\">\n");
            bw.write("  <graph mode=\"static\" defaultedgetype=\"undirected\">\n");
            bw.write("    <nodes>\n");
            for (N n : g.nodes()) {
                bw.write("      <node id=\"" + escape(n.toString()) + "\" label=\"" + escape(n.toString()) + "\">\n");
                AttrMap a = g.getNodeAttr(n);
                if (!a.isEmpty()) {
                    bw.write("        <attvalues>\n");
                    for (Map.Entry<String, Object> e : a.asMap().entrySet()) {
                        bw.write("          <attvalue for=\"" + escape(e.getKey()) + "\" value=\"" + escape(String.valueOf(e.getValue())) + "\"/>\n");
                    }
                    bw.write("        </attvalues>\n");
                }
                bw.write("      </node>\n");
            }
            bw.write("    </nodes>\n");
            bw.write("    <edges>\n");
            int idx = 0;
            for (Map.Entry<N, N> e : g.edges()) {
                AttrMap a = g.getEdgeAttr(e.getKey(), e.getValue());
                String weight = "1.0";
                if (a.contains("weight")) weight = String.valueOf(a.get("weight"));
                bw.write("      <edge id=\"" + idx++ + "\" source=\"" + escape(e.getKey().toString())
                    + "\" target=\"" + escape(e.getValue().toString()) + "\" weight=\"" + weight + "\"/>\n");
            }
            bw.write("    </edges>\n");
            bw.write("  </graph>\n</gexf>\n");
        }
    }

    static String escape(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;");
    }
}