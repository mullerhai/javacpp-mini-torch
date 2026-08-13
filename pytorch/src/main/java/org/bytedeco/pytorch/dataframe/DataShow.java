package org.bytedeco.pytorch.dataframe;

import org.bytedeco.pytorch.dataframe.io.DataFrameReader;
import org.bytedeco.pytorch.dataframe.io.ImdbShow;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;

/**
 * Unified data display for all supported formats (NPY, Pickle, PT, SafeTensors).
 * 
 * <p>Provides pandas-like show() functionality with:</p>
 * <ul>
 *   <li>Formatted table output</li>
 *   <li>Column statistics (min, max, mean, null count)</li>
 *   <li>Search/filter capabilities</li>
 *   <li>Head/tail display</li>
 * </ul>
 */
public class DataShow {

    private DataShow() {}

    // ---- Public API ----

    /**
     * Show a file in any supported format.
     */
    public static String show(String path) throws Exception {
        return show(path, ShowOptions.defaults());
    }

    public static String show(String path, ShowOptions opts) throws Exception {
        Path p = Paths.get(path);
        String ext = getExtension(p.getFileName().toString()).toLowerCase();
        
        switch (ext) {
            case "csv":
            case "tsv":
            case "json":
            case "jsonl":
            case "ndjson":
                return showTextLike(path, opts);
            case "npy":
                return showNpy(path, opts);
            case "npz":
                return showNpz(path, opts);
            case "pt":
            case "pth":
                return showPT(path, opts);
            case "pkl":
            case "pickle":
            case "imdb":
                return showPickle(path, opts);
            case "safetensors":
                return showSafeTensors(path, opts);
            case "toml":
                return showToml(path, opts);
            case "bin":
                return showBin(path, opts);
            case "hdf5":
            case "hdf":
                return showHdf5(path, opts);
            case "parquet":
            case "pq":
                return showParquet(path, opts);
            case "arrow":
            case "ipc":
            case "feather":
                return showArrow(path, opts);
            case "xlsx":
            case "xls":
            case "xlsm":
                return showExcel(path, opts);
            case "avro":
                return showAvro(path, opts);
            case "orc":
                return showOrc(path, opts);
            case "gguf":
                return showGguf(path, opts);
            case "lance":
                return showLance(path, opts);
            case "lmdb":
                return showLmdb(path, opts);
            default:
                return showRaw(path, opts);
        }
    }

    /**
     * Show a DataFrame with full UI capabilities.
     */
    public static String show(DataFrame df) {
        return show(df, ShowOptions.defaults());
    }

    public static String show(DataFrame df, ShowOptions opts) {
        StringBuilder sb = new StringBuilder();
        
        // Header
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ DataFrame: %d rows × %d columns                                           ║\n", 
                df.rowCount(), df.columnCount()));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Column info
        int colWidth = Math.min(opts.maxColWidth(), 20);
        sb.append("║ Columns: ");
        for (int i = 0; i < df.columnCount(); i++) {
            Column c = df.column(i);
            String name = truncate(c.name(), colWidth);
            String dtype = c.dtype().name();
            sb.append(String.format("%s(%s)", name, dtype));
            if (i < df.columnCount() - 1) sb.append(", ");
        }
        sb.append(" ║\n");
        
        // Data preview
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        int displayRows = Math.min(opts.maxRows(), df.rowCount());
        int start = opts.startRow();
        
        // Header row
        sb.append("║ ");
        for (int i = 0; i < df.columnCount(); i++) {
            String name = truncate(df.column(i).name(), colWidth);
            sb.append(String.format("%-" + colWidth + "s", name));
            if (i < df.columnCount() - 1) sb.append(" │ ");
        }
        sb.append(" ║\n");
        
        // Separator
        sb.append("║ ");
        for (int i = 0; i < df.columnCount(); i++) {
            for (int j = 0; j < colWidth; j++) sb.append("─");
            if (i < df.columnCount() - 1) sb.append("─┼─");
        }
        sb.append(" ║\n");
        
        // Data rows
        for (int r = start; r < start + displayRows && r < df.rowCount(); r++) {
            sb.append("║ ");
            for (int c = 0; c < df.columnCount(); c++) {
                String val = formatValue(df.get(r, c), colWidth);
                sb.append(String.format("%-" + colWidth + "s", val));
                if (c < df.columnCount() - 1) sb.append(" │ ");
            }
            sb.append(" ║\n");
        }
        
        // Footer
        if (df.rowCount() > displayRows) {
            sb.append("║ ... ").append(df.rowCount() - displayRows).append(" more rows").append(" ".repeat(Math.max(0, 70 - String.valueOf(df.rowCount() - displayRows).length()))).append(" ║\n");
        }
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    // ---- NPY ----

    private static String showNpy(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.data.numpy.NDArray arr = org.bytedeco.pytorch.data.numpy.NP.load(path);
        
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ NumPy array: shape=%s, dtype=%s                                      ║\n",
                Arrays.toString(arr.shape), arr.dtype));
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");
        
        long total = arr.size;
        int display = (int) Math.min(opts.maxRows(), 10);
        
        sb.append("First ").append(display).append(" values:\n");
        for (int i = 0; i < display; i++) {
            double val = arr.getDouble(i);
            sb.append(String.format("  [%5d] = %.6f\n", i, val));
        }
        
        if (total > display) {
            sb.append("  ... (").append(total - display).append(" more values)\n");
        }
        
        return sb.toString();
    }

    private static String showNpz(String path, ShowOptions opts) throws Exception {
        // NPZ contains multiple arrays
        StringBuilder sb = new StringBuilder();
        sb.append("NPZ file: ").append(path).append("\n\n");
        
        // Load and display each array
        org.bytedeco.pytorch.data.numpy.NDArray[] arrays = 
            org.bytedeco.pytorch.data.numpy.NP.loadNpz(path);
        
        for (int i = 0; i < arrays.length; i++) {
            org.bytedeco.pytorch.data.numpy.NDArray arr = arrays[i];
            sb.append("Array ").append(i).append(": shape=")
              .append(Arrays.toString(arr.shape))
              .append(", dtype=").append(arr.dtype).append("\n");
            
            int display = (int) Math.min(5, arr.size);
            for (int j = 0; j < display; j++) {
                sb.append(String.format("  [%d] = %.4f\n", j, arr.getDouble(j)));
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }

    // ---- PT (PyTorch) ----

    private static String showPT(String path, ShowOptions opts) throws Exception {
        File file = new File(path);
        Map<String, org.bytedeco.pytorch.data.pt.PT.TensorData> tensors = 
            org.bytedeco.pytorch.data.pt.PT.load(file);
        
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ PyTorch checkpoint: %d tensors                                             ║\n",
                tensors.size()));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        for (Map.Entry<String, org.bytedeco.pytorch.data.pt.PT.TensorData> e : tensors.entrySet()) {
            String name = truncate(e.getKey(), 30);
            org.bytedeco.pytorch.data.pt.PT.TensorData td = e.getValue();
            
            sb.append("║ ").append(String.format("%-30s", name)).append(" │ ");
            sb.append(String.format("%-20s", formatShape(td.shape)));
            sb.append(" │ ").append(td.dtype.name()).append(" ║\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");
        
        // Show details for first tensor
        if (!tensors.isEmpty() && opts.showDetails()) {
            var entry = tensors.entrySet().iterator().next();
            sb.append("Tensor '").append(entry.getKey()).append("' preview:\n");
            sb.append(entry.getValue().preview());
            sb.append("\n");
        }
        
        return sb.toString();
    }

    // ---- Pickle / IMDB ----

    private static String showPickle(String path, ShowOptions opts) throws Exception {
        // Use ImdbShow for better display
        try {
            return ImdbShow.show(path, new ImdbShow.ShowOptions()
                .maxRows(opts.maxRows())
                .maxCols(opts.maxCols()));
        } catch (Exception e) {
            // Fallback to basic pickle display
            Object obj = org.bytedeco.pytorch.data.pickle.Pickle.load(new File(path));
            
            StringBuilder sb = new StringBuilder();
            File file = new File(path);
            sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
            sb.append(String.format("║ Pickle/IMDB: %s                                            ║\n",
                    obj == null ? "null" : obj.getClass().getSimpleName()));
            sb.append(String.format("║ File: %s (%s)                                   ║\n",
                    truncate(path, 55), formatBytes(file.length())));
            sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");
            
            if (obj instanceof Map) {
                Map<?, ?> map = (Map<?, ?>) obj;
                sb.append("Dict with ").append(map.size()).append(" entries:\n");
                int shown = 0;
                for (Map.Entry<?, ?> e : map.entrySet()) {
                    if (shown++ >= opts.maxRows()) break;
                    sb.append(String.format("  %s: %s\n", e.getKey(), truncate(String.valueOf(e.getValue()), 50)));
                }
            } else if (obj instanceof List) {
                List<?> list = (List<?>) obj;
                sb.append("List with ").append(list.size()).append(" entries:\n");
                int shown = 0;
                for (Object item : list) {
                    if (shown++ >= opts.maxRows()) break;
                    sb.append(String.format("  [%d] = %s\n", shown - 1, truncate(String.valueOf(item), 50)));
                }
            } else {
                sb.append(String.valueOf(obj));
            }
            
            return sb.toString();
        }
    }

    // ---- SafeTensors ----

    private static String showSafeTensors(String path, ShowOptions opts) throws Exception {
        File file = new File(path);
        Map<String, org.bytedeco.pytorch.Tensor> tensors = 
            org.bytedeco.pytorch.data.safetensors.SafeTensors.loadAsTensors(file, false);
        
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ SafeTensors: %d tensors                                              ║\n",
                tensors.size()));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        for (Map.Entry<String, org.bytedeco.pytorch.Tensor> e : tensors.entrySet()) {
            String name = truncate(e.getKey(), 30);
            org.bytedeco.pytorch.Tensor t = e.getValue();
            
            sb.append("║ ").append(String.format("%-30s", name)).append(" │ ");
            sb.append(String.format("%-20s", formatTensorShape(t)));
            sb.append(" │ ").append(t.scalar_type().toString()).append(" ║\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    // ---- TOML ----

    private static String showToml(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = 
            org.bytedeco.pytorch.dataframe.io.TomlReader.read(path);
        
        StringBuilder sb = new StringBuilder();
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ TOML: %d rows × %d columns                                            ║\n",
                df.rowCount(), df.columnCount()));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Show columns
        for (int i = 0; i < df.columnCount(); i++) {
            Column c = df.column(i);
            sb.append("║ ").append(String.format("%-20s", truncate(c.name(), 20)));
            sb.append(" │ ").append(String.format("%-12s", c.dtype().name()));
            sb.append(" │ rows=").append(c.size()).append(" ║\n");
        }
        
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Show sample data
        int rows = Math.min(opts.maxRows(), df.rowCount());
        for (int r = 0; r < rows; r++) {
            sb.append("║ Row ").append(r).append(": ");
            for (int c = 0; c < Math.min(df.columnCount(), 5); c++) {
                Object v = df.get(r, c);
                String str = truncate(String.valueOf(v), 15);
                sb.append(str);
                if (c < df.columnCount() - 1) sb.append(", ");
            }
            if (df.columnCount() > 5) sb.append("...");
            sb.append(" ║\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    // ---- Binary ----

    private static String showBin(String path, ShowOptions opts) throws Exception {
        // Use BinReader schema for better display
        org.bytedeco.pytorch.dataframe.io.BinReader.BinSchema schema = 
            org.bytedeco.pytorch.dataframe.io.BinReader.schema(path);
        
        org.bytedeco.pytorch.dataframe.DataFrame df = 
            org.bytedeco.pytorch.dataframe.io.BinReader.read(path);
        
        StringBuilder sb = new StringBuilder();
        File file = new File(path);
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ Binary (.bin): %s                                             ║\n", 
                schema.format != null ? schema.format : "MicroLens"));
        sb.append(String.format("║ File: %-62s║\n", truncate(path, 62)));
        sb.append(String.format("║ Size: %-59s║\n", formatBytes(file.length())));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Show schema table
        sb.append(String.format("║ %-3s │ %-20s │ %-10s │ %-15s ║\n", "#", "field", "dtype", "shape"));
        sb.append("╠═════╪═══════════════════════╪════════════╪═══════════════════╣\n");
        
        int idx = 0;
        for (org.bytedeco.pytorch.dataframe.io.BinReader.BinSchema.FieldInfo f : schema.fields) {
            sb.append(String.format("║ %3d │ %-20s │ %-10s │ %-15s ║\n",
                    idx++, truncate(f.name, 20), f.dtype, f.shape));
        }
        
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        // Special handling for matrix data (like MicroLens embeddings)
        boolean isMatrixData = schema.fields.size() > 0 && 
            schema.fields.get(0).cols > 100;  // Likely matrix if many columns
        
        if (isMatrixData) {
            sb.append("║                        Matrix Data Preview                             ║\n");
            sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
            sb.append(showMatrixPreview(df, schema, opts));
        } else {
            sb.append("║                         Data Preview                                   ║\n");
            sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
            
            // Show columns header
            int maxCols = Math.min(df.columnCount(), 5);
            sb.append("║  idx │");
            for (int c = 0; c < maxCols; c++) {
                sb.append(String.format(" %-18s│", truncate(df.column(c).name(), 18)));
            }
            if (df.columnCount() > maxCols) sb.append("     ... │");
            sb.append("\n");
            
            sb.append("╟──────┼");
            for (int c = 0; c < maxCols; c++) {
                sb.append("─────────────────────┼");
            }
            if (df.columnCount() > maxCols) sb.append("───────────┤");
            sb.append("\n");
            
            // Show sample data rows
            int rows = Math.min(opts.maxRows(), df.rowCount());
            for (int r = 0; r < rows; r++) {
                sb.append(String.format("║ %4d │", r));
                for (int c = 0; c < maxCols; c++) {
                    Object v = df.get(r, c);
                    String str = formatValueCompact(v, 18);
                    sb.append(" ").append(str).append("│");
                }
                if (df.columnCount() > maxCols) sb.append("     ... │");
                sb.append("\n");
            }
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    /**
     * Show matrix-style data with proper row/column grid.
     */
    private static String showMatrixPreview(DataFrame df, 
            org.bytedeco.pytorch.dataframe.io.BinReader.BinSchema schema,
            ShowOptions opts) {
        StringBuilder sb = new StringBuilder();
        
        if (df.columnCount() == 0) return "║ (empty)                                                                    ║\n";
        
        Column col = df.column(0);
        int totalCols = (int) col.size();
        int displayCols = Math.min(totalCols, opts.maxCols());
        int rows = df.rowCount();
        
        // Show column indices
        sb.append("║        │");
        for (int c = 0; c < displayCols; c++) {
            if (c < displayCols - 1) {
                sb.append(String.format("    col_%5d│", c));
            } else {
                sb.append(String.format("    col_%5d│", c));
            }
        }
        if (totalCols > displayCols) sb.append("      ...│");
        sb.append("\n");
        
        // Separator
        sb.append("║────────┼");
        for (int c = 0; c < displayCols; c++) {
            sb.append("─────────────┼");
        }
        if (totalCols > displayCols) sb.append("─────────┤");
        sb.append("\n");
        
        // Data rows
        int displayRows = Math.min(rows, opts.maxRows());
        for (int r = 0; r < displayRows; r++) {
            sb.append(String.format("║ row_%3d │", r));
            for (int c = 0; c < displayCols; c++) {
                double val = toDouble(col.get(r * totalCols + c));
                if (c < displayCols - 1) {
                    sb.append(String.format(" %11.4f│", val));
                } else {
                    sb.append(String.format(" %11.4f│", val));
                }
            }
            if (totalCols > displayCols) sb.append("      ...│");
            sb.append("\n");
        }
        
        // Trailing rows indicator
        if (rows > displayRows) {
            sb.append("║   ...  │");
            for (int c = 0; c < displayCols; c++) sb.append("           ...│");
            if (totalCols > displayCols) sb.append("─────────┤");
            sb.append("\n");
        }
        
        return sb.toString();
    }

    private static double toDouble(Object v) {
        if (v == null) return 0.0;
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(String.valueOf(v)); }
        catch (Exception e) { return 0.0; }
    }

    private static String formatValueCompact(Object v, int maxLen) {
        if (v == null) return padRight("null", maxLen);
        String str;
        if (v instanceof Number n) {
            if (n instanceof Double || n instanceof Float) {
                str = String.format("%.4f", n.doubleValue());
            } else {
                str = String.valueOf(n.longValue());
            }
        } else {
            str = String.valueOf(v);
        }
        return padRight(truncate(str, maxLen), maxLen);
    }

    private static String padRight(String s, int len) {
        if (s.length() >= len) return s.substring(0, len);
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < len) sb.append(' ');
        return sb.toString();
    }

    // ---- HDF5 ----

    private static String showHdf5(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = 
            org.bytedeco.pytorch.dataframe.hdf5.Hdf5Reader.read(path, "/df");
        
        StringBuilder sb = new StringBuilder();
        File file = new File(path);
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ HDF5 (.h5): %d rows × %d columns                                        ║\n",
                df.rowCount(), df.columnCount()));
        sb.append(String.format("║ File: %s (%s)                                   ║\n",
                truncate(path, 55), formatBytes(file.length())));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        int maxCols = Math.min(df.columnCount(), 5);
        sb.append("║  idx │");
        for (int c = 0; c < maxCols; c++) {
            sb.append(String.format(" %-18s│", truncate(df.column(c).name(), 18)));
        }
        if (df.columnCount() > maxCols) sb.append("       ... │");
        sb.append("\n");
        
        sb.append("╟──────┼");
        for (int c = 0; c < maxCols; c++) {
            sb.append("──────────────────────┼");
        }
        if (df.columnCount() > maxCols) sb.append("───────────┤");
        sb.append("\n");
        
        int rows = Math.min(opts.maxRows(), df.rowCount());
        for (int r = 0; r < rows; r++) {
            sb.append(String.format("║ %4d │", r));
            for (int c = 0; c < maxCols; c++) {
                Object v = df.get(r, c);
                String str = formatValueCompact(v, 18);
                sb.append(" ").append(str).append("│");
            }
            if (df.columnCount() > maxCols) sb.append("       ... │");
            sb.append("\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        return sb.toString();
    }

    // ---- CSV/TSV/JSON ----

    private static String showTextLike(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.read(path);
        return formatGenericShow(getExtension(path).toUpperCase(), path, df, opts);
    }

    // ---- Parquet ----

    private static String showParquet(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readParquet(path);
        return formatGenericShow("Parquet", path, df, opts);
    }

    // ---- Arrow/Feather ----

    private static String showArrow(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readArrow(path);
        return formatGenericShow("Arrow/Feather", path, df, opts);
    }

    // ---- Excel ----

    private static String showExcel(String path, ShowOptions opts) throws Exception {
        Map<String, org.bytedeco.pytorch.dataframe.DataFrame> sheets = DataFrame.readExcelAll(path);
        StringBuilder sb = new StringBuilder();
        File file = new File(path);
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ Excel (.xlsx): %d sheets                                               ║\n",
                sheets.size()));
        sb.append(String.format("║ File: %s (%s)                                   ║\n",
                truncate(path, 55), formatBytes(file.length())));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append(String.format("║ %-20s │ %-10s │ %-10s ║\n", "sheet_name", "rows", "columns"));
        sb.append("╠════════════════════════╪════════════╪═════════════╣\n");
        
        for (Map.Entry<String, org.bytedeco.pytorch.dataframe.DataFrame> e : sheets.entrySet()) {
            org.bytedeco.pytorch.dataframe.DataFrame df = e.getValue();
            sb.append(String.format("║ %-20s │ %10d │ %10d ║\n",
                    truncate(e.getKey(), 20), df.rowCount(), df.columnCount()));
        }
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    // ---- Avro ----

    private static String showAvro(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readAvro(path);
        return formatGenericShow("Avro", path, df, opts);
    }

    // ---- ORC ----

    private static String showOrc(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readOrc(path);
        return formatGenericShow("ORC", path, df, opts);
    }

    // ---- LMDB ----

    private static String showLmdb(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readLmdb(path);
        return formatGenericShow("LMDB", path, df, opts);
    }

    // ---- GGUF ----

    private static String showGguf(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readGguf(path);
        return formatGenericShow("GGUF", path, df, opts);
    }

    // ---- Lance ----

    private static String showLance(String path, ShowOptions opts) throws Exception {
        org.bytedeco.pytorch.dataframe.DataFrame df = DataFrame.readLance(path);
        return formatGenericShow("Lance", path, df, opts);
    }

    // ---- Generic format show helper ----

    private static String formatGenericShow(String formatName, String path, 
            org.bytedeco.pytorch.dataframe.DataFrame df, ShowOptions opts) throws Exception {
        StringBuilder sb = new StringBuilder();
        File file = new File(path);
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║ %s: %d rows × %d columns                                        ║\n",
                formatName, df.rowCount(), df.columnCount()));
        sb.append(String.format("║ File: %s (%s)                                   ║\n",
                truncate(path, 55), formatBytes(file.length())));
        sb.append("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        
        int maxCols = Math.min(df.columnCount(), 5);
        sb.append("║  idx │");
        for (int c = 0; c < maxCols; c++) {
            sb.append(String.format(" %-18s│", truncate(df.column(c).name(), 18)));
        }
        if (df.columnCount() > maxCols) sb.append("       ... │");
        sb.append("\n");
        
        sb.append("╟──────┼");
        for (int c = 0; c < maxCols; c++) {
            sb.append("──────────────────────┼");
        }
        if (df.columnCount() > maxCols) sb.append("───────────┤");
        sb.append("\n");
        
        int rows = Math.min(opts.maxRows(), df.rowCount());
        for (int r = 0; r < rows; r++) {
            sb.append(String.format("║ %4d │", r));
            for (int c = 0; c < maxCols; c++) {
                Object v = df.get(r, c);
                String str = formatValueCompact(v, 18);
                sb.append(" ").append(str).append("│");
            }
            if (df.columnCount() > maxCols) sb.append("       ... │");
            sb.append("\n");
        }
        
        if (df.rowCount() > rows) {
            sb.append("║  ... │");
            for (int c = 0; c < maxCols; c++) sb.append("                ... │");
            if (df.columnCount() > maxCols) sb.append("           ... │");
            sb.append("\n");
        }
        
        sb.append("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        return sb.toString();
    }

    private static String formatValue(Object v) {
        if (v == null) return "null";
        if (v instanceof Number) {
            if (v instanceof Double || v instanceof Float) {
                return String.format("%.4f", ((Number) v).doubleValue());
            }
            return String.valueOf(((Number) v).longValue());
        }
        if (v instanceof float[]) {
            float[] arr = (float[]) v;
            if (arr.length <= 4) {
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(String.format("%.3f", arr[i]));
                }
                return sb.append("]").toString();
            }
            return String.format("[%d floats: %.3f, ...]", arr.length, arr[0]);
        }
        if (v instanceof double[]) {
            double[] arr = (double[]) v;
            if (arr.length <= 4) {
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) {
                    if (i > 0) sb.append(",");
                    sb.append(String.format("%.3f", arr[i]));
                }
                return sb.append("]").toString();
            }
            return String.format("[%d doubles: %.3f, ...]", arr.length, arr[0]);
        }
        return String.valueOf(v);
    }

    private static String showRaw(String path, ShowOptions opts) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("Raw file: ").append(path).append("\n");
        
        try (BufferedReader br = Files.newBufferedReader(Paths.get(path))) {
            String line;
            int count = 0;
            while ((line = br.readLine()) != null && count++ < opts.maxRows()) {
                sb.append(line).append("\n");
            }
        }
        
        return sb.toString();
    }

    // ---- Search / Filter ----

    /**
     * Filter DataFrame by search query.
     */
    public static DataFrame filter(DataFrame df, String query) {
        return filter(df, query, true);
    }

    public static DataFrame filter(DataFrame df, String query, boolean caseSensitive) {
        Pattern pattern = Pattern.compile(
            caseSensitive ? query : query.toLowerCase(),
            Pattern.CASE_INSENSITIVE
        );
        
        DataFrame result = DataFrame.create(df.columns());
        
        for (int r = 0; r < df.rowCount(); r++) {
            boolean match = false;
            for (int c = 0; c < df.columnCount(); c++) {
                String val = String.valueOf(df.get(r, c));
                if (pattern.matcher(val).find()) {
                    match = true;
                    break;
                }
            }
            if (match) {
                int newRow = result.addEmptyRow();
                for (int c = 0; c < df.columnCount(); c++) {
                    result.set(newRow, c, df.get(r, c));
                }
            }
        }
        
        return result;
    }

    /**
     * Select columns by regex pattern.
     */
    public static DataFrame select(DataFrame df, String pattern) {
        Pattern p = Pattern.compile(pattern, Pattern.CASE_INSENSITIVE);
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < df.columnCount(); i++) {
            if (p.matcher(df.column(i).name()).find()) {
                indices.add(i);
            }
        }
        
        DataFrame result = DataFrame.create();
        for (int idx : indices) {
            result.addColumn(df.column(idx).copy());
        }
        
        return result;
    }

    // ---- Pivot ----

    /**
     * Create a pivot table (透视表).
     */
    public static DataFrame pivot(DataFrame df, String index, String column, String value) {
        // Get unique values
        Set<Object> indexVals = new LinkedHashSet<>();
        Set<Object> colVals = new LinkedHashSet<>();
        Map<String, List<Object>> data = new HashMap<>();
        
        int idxCol = df.columnIndex(index);
        int colCol = df.columnIndex(column);
        int valCol = df.columnIndex(value);
        
        for (int r = 0; r < df.rowCount(); r++) {
            Object iv = df.get(r, idxCol);
            Object cv = df.get(r, colCol);
            Object vv = df.get(r, valCol);
            
            indexVals.add(iv);
            colVals.add(cv);
            data.put(iv + "|" + cv, Collections.singletonList(vv));
        }
        
        // Build result DataFrame
        DataFrame result = DataFrame.create();
        result.addColumn(new Column(index, df.column(idxCol).dtype()));
        
        for (Object cv : colVals) {
            String colName = String.valueOf(cv);
            result.addColumn(new Column(colName, df.column(valCol).dtype()));
        }
        
        // Fill data
        for (Object iv : indexVals) {
            int newRow = result.addEmptyRow();
            result.set(newRow, 0, iv);
            
            int col = 1;
            for (Object cv : colVals) {
                List<Object> vals = data.get(iv + "|" + cv);
                if (vals != null && !vals.isEmpty()) {
                    result.set(newRow, col, vals.get(0));
                }
                col++;
            }
        }
        
        return result;
    }

    // ---- Stats ----

    /**
     * Get basic statistics for numeric columns.
     */
    public static Map<String, ColumnStats> describe(DataFrame df) {
        Map<String, ColumnStats> stats = new LinkedHashMap<>();
        
        for (int c = 0; c < df.columnCount(); c++) {
            Column col = df.column(c);
            if (col.isNumeric()) {
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                double sum = 0;
                int count = 0;
                int nullCount = 0;
                
                for (int r = 0; r < df.rowCount(); r++) {
                    Object v = df.get(r, c);
                    if (v == null) {
                        nullCount++;
                    } else {
                        double d = ((Number) v).doubleValue();
                        min = Math.min(min, d);
                        max = Math.max(max, d);
                        sum += d;
                        count++;
                    }
                }
                
                if (count > 0) {
                    stats.put(col.name(), new ColumnStats(
                        col.name(), count, nullCount, min, max, sum / count
                    ));
                }
            }
        }
        
        return stats;
    }

    public static String describeAsString(DataFrame df) {
        StringBuilder sb = new StringBuilder();
        sb.append("═══════════════════════════════════════════════════════════════\n");
        sb.append("                     DataFrame Statistics                      \n");
        sb.append("═══════════════════════════════════════════════════════════════\n\n");
        
        Map<String, ColumnStats> stats = describe(df);
        
        sb.append(String.format("%-15s %8s %8s %12s %12s %12s\n",
                "column", "count", "null", "min", "max", "mean"));
        sb.append(String.format("%-15s %8s %8s %12s %12s %12s\n",
                "------", "-----", "----", "---", "---", "----"));
        
        for (ColumnStats s : stats.values()) {
            sb.append(String.format("%-15s %8d %8d %12.4f %12.4f %12.4f\n",
                    truncate(s.name, 15), s.count, s.nullCount, s.min, s.max, s.mean));
        }
        
        return sb.toString();
    }

    // ---- Helpers ----

    private static String getExtension(String name) {
        int idx = name.lastIndexOf('.');
        return idx >= 0 ? name.substring(idx + 1) : name;
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "";
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }

    private static String formatValue(Object v, int width) {
        if (v == null) return "null";
        String s = v instanceof Number 
            ? String.format("%.4g", ((Number) v).doubleValue())
            : String.valueOf(v);
        return truncate(s, width);
    }

    private static String formatShape(long[] shape) {
        if (shape == null || shape.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    private static String formatTensorShape(org.bytedeco.pytorch.Tensor t) {
        if (t == null || !t.defined()) return "null";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(t.sizes().get(i));
        }
        sb.append("]");
        return sb.toString();
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    // ---- Inner classes ----

    public static class ShowOptions {
        private int maxRows = 10;
        private int maxCols = 6;
        private int maxColWidth = 20;
        private int startRow = 0;
        private boolean showDetails = false;

        public static ShowOptions defaults() {
            return new ShowOptions();
        }

        public ShowOptions maxRows(int n) { this.maxRows = n; return this; }
        public ShowOptions maxCols(int n) { this.maxCols = n; return this; }
        public ShowOptions maxColWidth(int w) { this.maxColWidth = w; return this; }
        public ShowOptions startRow(int r) { this.startRow = r; return this; }
        public ShowOptions showDetails(boolean b) { this.showDetails = b; return this; }

        public int maxRows() { return maxRows; }
        public int maxCols() { return maxCols; }
        public int maxColWidth() { return maxColWidth; }
        public int startRow() { return startRow; }
        public boolean showDetails() { return showDetails; }
    }

    public static class ColumnStats {
        public final String name;
        public final int count;
        public final int nullCount;
        public final double min;
        public final double max;
        public final double mean;

        public ColumnStats(String name, int count, int nullCount, 
                          double min, double max, double mean) {
            this.name = name;
            this.count = count;
            this.nullCount = nullCount;
            this.min = min;
            this.max = max;
            this.mean = mean;
        }
    }
}
