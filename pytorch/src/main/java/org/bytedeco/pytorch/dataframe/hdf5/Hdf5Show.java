package org.bytedeco.pytorch.dataframe.hdf5;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.hdf5.internal.Hdf5ReaderCore;
import org.bytedeco.pytorch.dataframe.hdf5.internal.Hdf5WriterCore;

import java.nio.file.Path;
import java.util.*;

/**
 * HDF5 file inspector with schema display and metadata inspection.
 * 
 * <p>Provides enterprise-grade HDF5 analysis capabilities:</p>
 * <ul>
 *   <li>Schema display with column types and shapes</li>
 *   <li>Dataset hierarchy exploration</li>
 *   <li>Metadata inspection</li>
 *   <li>Size statistics</li>
 * </ul>
 */
public final class Hdf5Show {

    public static final int DEFAULT_PREVIEW_ROWS = 5;
    public static final int DEFAULT_COL_WIDTH = 25;

    private final int previewRows;
    private final int colWidth;

    public Hdf5Show() {
        this(DEFAULT_PREVIEW_ROWS, DEFAULT_COL_WIDTH);
    }

    public Hdf5Show(int previewRows, int colWidth) {
        this.previewRows = Math.max(1, previewRows);
        this.colWidth = Math.max(10, colWidth);
    }

    // ====================== Public API ======================

    /**
     * Create a new Hdf5Show instance with default settings.
     */
    public static Hdf5Show show() {
        return new Hdf5Show();
    }

    /**
     * Print schema to stdout.
     */
    public void printSchema(String path) throws Exception {
        System.out.println(schema(path));
    }

    /**
     * Print schema for a specific key.
     */
    public void printSchema(String path, String key) throws Exception {
        System.out.println(schema(path, key));
    }

    /**
     * Get schema string for HDF5 file.
     */
    public String schema(String path) throws Exception {
        return schema(path, null);
    }

    /**
     * Get schema string for HDF5 file at specific key.
     */
    public String schema(String path, String key) throws Exception {
        Hdf5ReaderCore.Node root = Hdf5ReaderCore.open(Path.of(path));
        Hdf5ReaderCore.Node node = key != null ? Hdf5ReaderCore.resolve(root, key) : root;
        if (node == null) {
            return "Key not found: " + key;
        }
        return schemaNode(node, path, key);
    }

    /**
     * Get full file structure as tree string.
     */
    public String tree(String path) throws Exception {
        Hdf5ReaderCore.Node root = Hdf5ReaderCore.open(Path.of(path));
        StringBuilder sb = new StringBuilder();
        sb.append("HDF5: ").append(path).append("\n");
        sb.append("├── format: ").append(getAttr(root, "format", "unknown")).append("\n");
        sb.append("├── version: ").append(getAttr(root, "version", "1")).append("\n");
        sb.append("└── structure:\n");
        treeNode(sb, root, "    ", true);
        return sb.toString();
    }

    /**
     * Print file structure tree.
     */
    public void printTree(String path) throws Exception {
        System.out.println(tree(path));
    }

    /**
     * Get metadata as map.
     */
    public Map<String, Object> metadata(String path) throws Exception {
        Hdf5ReaderCore.Node root = Hdf5ReaderCore.open(Path.of(path));
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("format", getAttr(root, "format", "unknown"));
        meta.put("version", getAttr(root, "version", "1"));
        meta.put("created_by", getAttr(root, "created_by", "unknown"));
        meta.put("nrows", getAttr(root, "nrows", "0"));
        
        // Collect dataset info
        List<Map<String, Object>> datasets = new ArrayList<>();
        collectDatasets(root, datasets, "");
        meta.put("datasets", datasets);
        meta.put("dataset_count", datasets.size());
        
        return meta;
    }

    // ====================== Schema Display ======================

    private String schemaNode(Hdf5ReaderCore.Node node, String path, String key) {
        StringBuilder sb = new StringBuilder();
        String title = key != null ? "HDF5 Schema: " + key : "HDF5 Schema";
        sb.append("╔").append("═".repeat(70)).append("╗\n");
        sb.append(String.format("║ %-68s ║\n", title));
        sb.append("╠").append("═".repeat(70)).append("╣\n");

        if (node.group) {
            // Group view - show columns
            String[] columnNames = getStringArrayAttr(node, "column_names");
            String[] dtypes = getStringArrayAttr(node, "dtypes");
            int nrows = (int) getLongAttr(node, "nrows", 0);
            String format = getAttr(node, "format", "columnar");

            sb.append(String.format("║ Format: %-60s ║\n", format));
            sb.append(String.format("║ Rows: %-62s ║\n", nrows));
            sb.append("╠").append("═".repeat(70)).append("╣\n");

            if (columnNames != null && columnNames.length > 0) {
                // Header
                sb.append(String.format("║ %-" + (colWidth - 2) + "s │ %-12s │ %-15s ║\n",
                        "column", "dtype", "size"));
                sb.append("║ ").append("─".repeat(colWidth - 2)).append("─┼─")
                  .append("─".repeat(12)).append("─┼─").append("─".repeat(15)).append(" ║\n");

                // Rows
                for (int i = 0; i < columnNames.length; i++) {
                    String colName = truncate(columnNames[i], colWidth - 2);
                    String dtype = dtypes != null && i < dtypes.length ? dtypes[i] : "unknown";
                    String size = getColumnSize(node, columnNames[i], dtype);
                    
                    sb.append(String.format("║ %-" + (colWidth - 2) + "s │ %-12s │ %-15s ║\n",
                            colName, truncate(dtype, 12), truncate(size, 15)));
                }
            } else {
                // No column metadata - list children
                sb.append(String.format("║ %-65s ║\n", "Datasets:"));
                for (String name : node.children.keySet()) {
                    Hdf5ReaderCore.Node child = node.children.get(name);
                    if (!child.group) {
                        String info = getDatasetInfo(child.dataset);
                        sb.append(String.format("║   %-62s ║\n", name + " (" + info + ")"));
                    } else {
                        sb.append(String.format("║   [group] %-56s ║\n", name));
                    }
                }
            }
        } else {
            // Single dataset
            sb.append(String.format("║ Dataset: %-59s ║\n", key != null ? key : "root"));
            sb.append("╠").append("═".repeat(70)).append("╣\n");
            sb.append(String.format("║ Shape: %-62s ║\n", getDatasetShape(node.dataset)));
            sb.append(String.format("║ Dtype: %-62s ║\n", getDatasetDtype(node.dataset)));
        }

        sb.append("╚").append("═".repeat(70)).append("╝\n");
        return sb.toString();
    }

    private void treeNode(StringBuilder sb, Hdf5ReaderCore.Node node, String indent, boolean last) {
        String prefix = last ? "└── " : "├── ";
        String newIndent = indent + (last ? "    " : "│   ");

        if (!node.group) {
            String info = getDatasetInfo(node.dataset);
            sb.append(indent).append(prefix).append(node.name)
              .append(" (").append(info).append(")\n");
        } else {
            sb.append(indent).append(prefix).append(node.name).append("/\n");
            List<String> keys = new ArrayList<>(node.children.keySet());
            for (int i = 0; i < keys.size(); i++) {
                treeNode(sb, node.children.get(keys.get(i)), newIndent, i == keys.size() - 1);
            }
        }
    }

    // ====================== Helpers ======================

    private String getDatasetInfo(Hdf5WriterCore.EncodedData ds) {
        if (ds == null) return "null";
        long elements = ds.dim0 * (ds.rank > 1 ? ds.dim1 : 1);
        String dtype = dtypeName(ds.dtypeCode);
        long bytes = ds.raw != null ? ds.raw.length : 0;
        return String.format("%s, %d elements, %s", 
                formatShape(ds.rank, ds.dim0, ds.dim1), elements, formatBytes(bytes));
    }

    private String getDatasetShape(Hdf5WriterCore.EncodedData ds) {
        if (ds == null) return "[]";
        return formatShape(ds.rank, ds.dim0, ds.dim1);
    }

    private String getDatasetDtype(Hdf5WriterCore.EncodedData ds) {
        if (ds == null) return "unknown";
        return dtypeName(ds.dtypeCode);
    }

    private String formatShape(int rank, long d0, long d1) {
        if (rank <= 1) return "[" + d0 + "]";
        return "[" + d0 + ", " + d1 + "]";
    }

    private String getColumnSize(Hdf5ReaderCore.Node group, String name, String dtype) {
        Hdf5ReaderCore.Node child = group.children.get(name);
        if (child == null || child.dataset == null) return "unknown";
        long bytes = child.dataset.raw != null ? child.dataset.raw.length : 0;
        return formatBytes(bytes);
    }

    private String dtypeName(int code) {
        switch (code) {
            case 1: return "int32";
            case 2: return "int64";
            case 3: return "float32";
            case 4: return "float64";
            case 5: return "bool";
            case 6: return "string";
            default: return "unknown";
        }
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max - 3) + "...";
    }

    private void collectDatasets(Hdf5ReaderCore.Node node, List<Map<String, Object>> datasets, String prefix) {
        for (Map.Entry<String, Hdf5ReaderCore.Node> e : node.children.entrySet()) {
            String name = prefix.isEmpty() ? e.getKey() : prefix + "/" + e.getKey();
            Hdf5ReaderCore.Node child = e.getValue();
            if (!child.group) {
                Map<String, Object> info = new LinkedHashMap<>();
                info.put("name", name);
                info.put("shape", getDatasetShape(child.dataset));
                info.put("dtype", getDatasetDtype(child.dataset));
                info.put("size_bytes", child.dataset.raw != null ? child.dataset.raw.length : 0);
                datasets.add(info);
            } else {
                collectDatasets(child, datasets, name);
            }
        }
    }

    private String getAttr(Hdf5ReaderCore.Node node, String name, String defaultVal) {
        Object v = node.attrs.get(name);
        return v != null ? String.valueOf(v) : defaultVal;
    }

    private long getLongAttr(Hdf5ReaderCore.Node node, String name, long defaultVal) {
        Object v = node.attrs.get(name);
        if (v instanceof Long) return (Long) v;
        if (v instanceof Number) return ((Number) v).longValue();
        if (v instanceof String) {
            try { return Long.parseLong((String) v); } catch (Exception ignored) {}
        }
        return defaultVal;
    }

    private String[] getStringArrayAttr(Hdf5ReaderCore.Node node, String name) {
        Object v = node.attrs.get(name);
        if (v instanceof String[]) return (String[]) v;
        return null;
    }
}
