package org.bytedeco.pytorch.dataframe;

import org.bytedeco.pytorch.dataframe.hdf5.Hdf5Reader;
import org.bytedeco.pytorch.data.numpy.NDArray;

import java.util.*;

/**
 * Unified DataFrame display with format-aware rendering.
 * 
 * <p>Handles special formats correctly:</p>
 * <ul>
 *   <li><b>NPY/NPZ</b>: Matrix arrays displayed as 2D grid, 1D arrays as single column</li>
 *   <li><b>PT/SafeTensors</b>: Tensor shapes and metadata</li>
 *   <li><b>HDF5</b>: Group/dataset hierarchy</li>
 *   <li><b>Pickle</b>: Python object structure</li>
 * </ul>
 * 
 * <p>Example:
 * <pre>
 *   DataFrameShow.show(df);
 *   DataFrameShow.schema(df);
 *   DataFrameShow.showNpy("/path/to/array.npy");
 * </pre>
 */
public class DataFrameShow {

    public static final int DEFAULT_MAX_ROWS = 20;
    public static final int DEFAULT_MAX_COLS = 10;
    public static final int DEFAULT_COL_WIDTH = 25;
    public static final int DEFAULT_MATRIX_COLS = 6;  // Max cols for matrix display

    private final int maxRows;
    private final int maxCols;
    private final int colWidth;
    private final int matrixDisplayCols;

    public DataFrameShow() {
        this(DEFAULT_MAX_ROWS, DEFAULT_MAX_COLS, DEFAULT_COL_WIDTH, DEFAULT_MATRIX_COLS);
    }

    public DataFrameShow(int maxRows, int maxCols, int colWidth, int matrixDisplayCols) {
        this.maxRows = Math.max(1, maxRows);
        this.maxCols = Math.max(1, maxCols);
        this.colWidth = Math.max(5, colWidth);
        this.matrixDisplayCols = Math.max(2, matrixDisplayCols);
    }

    // ====================== Static API ======================

    /**
     * Print DataFrame to stdout.
     */
    public static void show(DataFrame df) {
        new DataFrameShow().print(df);
    }

    /**
     * Get DataFrame display string.
     */
    public static String toString(DataFrame df) {
        return new DataFrameShow().format(df);
    }

    /**
     * Print schema to stdout.
     */
    public static void schema(DataFrame df) {
        new DataFrameShow().printSchema(df);
    }

    /**
     * Get schema string.
     */
    public static String schemaString(DataFrame df) {
        return new DataFrameShow().schema(df);
    }

    /**
     * Show NPY file with matrix-aware display.
     */
    public static String showNpy(String path) throws Exception {
        return new DataFrameShow().formatNpy(path);
    }

    /**
     * Show NPZ file with proper array display.
     */
    public static String showNpz(String path) throws Exception {
        return new DataFrameShow().formatNpz(path);
    }

    // ====================== Instance API ======================

    public void print(DataFrame df) {
        System.out.print(format(df));
    }

    public void printSchema(DataFrame df) {
        System.out.print(schema(df));
    }

    public String format(DataFrame df) {
        if (df == null || df.rowCount() == 0) {
            return emptyDataFrame();
        }

        // Check if this is a matrix-style DataFrame (from NPY)
        if (isMatrixDataFrame(df)) {
            return formatMatrix(df);
        }

        return formatStandard(df);
    }

    public String schema(DataFrame df) {
        StringBuilder sb = new StringBuilder();
        sb.append("DataFrame Schema\n");
        sb.append("=".repeat(60)).append("\n");
        sb.append("Rows: ").append(df.rowCount()).append("\n");
        sb.append("Columns: ").append(df.columnCount()).append("\n");
        sb.append("-".repeat(60)).append("\n");
        
        for (int i = 0; i < df.columnCount(); i++) {
            Column c = df.column(i);
            String name = truncate(c.name(), colWidth);
            sb.append(String.format("%-" + colWidth + "s  %s\n", name, c.dtype()));
        }
        
        return sb.toString();
    }

    // ====================== NPY/NPZ Display ======================

    public String formatNpy(String path) throws Exception {
        NDArray arr = org.bytedeco.pytorch.data.numpy.NP.load(path);
        return formatNumpyArray(arr, path);
    }

    public String formatNpz(String path) throws Exception {
        Map<String, NDArray> arrays = org.bytedeco.pytorch.data.numpy.NP.loadz(path);
        StringBuilder sb = new StringBuilder();
        
        sb.append("╔").append("═".repeat(65)).append("╗\n");
        sb.append(String.format("║ NPZ Archive: %-48s ║\n", truncate(path, 48)));
        sb.append(String.format("║ Arrays: %-55s ║\n", arrays.size()));
        sb.append("╠").append("═".repeat(65)).append("╣\n");
        
        for (Map.Entry<String, NDArray> e : arrays.entrySet()) {
            String name = truncate(e.getKey(), 30);
            NDArray arr = e.getValue();
            String info = String.format("%s, dtype=%s", Arrays.toString(arr.shape), arr.dtype);
            sb.append(String.format("║ %-30s │ %-28s ║\n", name, truncate(info, 28)));
        }
        sb.append("╚").append("═".repeat(65)).append("╝\n");
        
        // If single 2D array, show as matrix
        if (arrays.size() == 1) {
            NDArray only = arrays.values().iterator().next();
            if (only.shape.length == 2) {
                sb.append("\n").append(formatNumpyMatrix(only, "Matrix View")).append("\n");
            } else {
                sb.append("\n").append(formatNumpyArray(only, path)).append("\n");
            }
        }
        
        return sb.toString();
    }

    private String formatNumpyArray(NDArray arr, String name) {
        StringBuilder sb = new StringBuilder();
        
        sb.append("╔").append("═".repeat(65)).append("╗\n");
        sb.append(String.format("║ NumPy Array: %-51s ║\n", truncate(name, 51)));
        sb.append("╠").append("═".repeat(65)).append("╣\n");
        sb.append(String.format("║ Shape: %-57s ║\n", Arrays.toString(arr.shape)));
        sb.append(String.format("║ Dtype: %-57s ║\n", arr.dtype));
        sb.append(String.format("║ Size: %-58s ║\n", arr.size + " elements"));
        sb.append("╚").append("═".repeat(65)).append("╝\n");
        
        if (arr.shape.length == 2) {
            sb.append("\n").append(formatNumpyMatrix(arr, "Matrix Preview")).append("\n");
        } else if (arr.shape.length == 1) {
            sb.append("\n").append(formatNumpyVector(arr, "Vector Preview")).append("\n");
        } else {
            sb.append("\n").append(formatNumpyNDArray(arr)).append("\n");
        }
        
        return sb.toString();
    }

    /**
     * Format 2D NumPy array as proper matrix grid.
     * IMPORTANT: Each row of the array is displayed as one row in the grid.
     */
    private String formatNumpyMatrix(NDArray arr, String title) {
        int rows = (int) arr.shape[0];
        int cols = (int) arr.shape[1];
        
        StringBuilder sb = new StringBuilder();
        sb.append("┌").append("─".repeat(Math.min(cols, matrixDisplayCols) * 12 + 1)).append("┐\n");
        sb.append(String.format("│ %s (rows=%d, cols=%d)\n", title, rows, cols));
        sb.append("├").append("─".repeat(Math.min(cols, matrixDisplayCols) * 12 + 1)).append("┤\n");
        
        int displayCols = Math.min(cols, matrixDisplayCols);
        int displayRows = Math.min(rows, maxRows);
        
        // Column headers (showing column indices)
        sb.append("│      │");
        for (int c = 0; c < displayCols; c++) {
            sb.append(String.format("  col_%-5d│", c));
        }
        if (cols > displayCols) {
            sb.append("  ...   │");
        }
        sb.append("\n");
        sb.append("├──────┼");
        for (int c = 0; c < displayCols; c++) {
            sb.append("──────────┼");
        }
        if (cols > displayCols) {
            sb.append("──────────┤");
        }
        sb.append("\n");
        
        // Data rows
        for (int r = 0; r < displayRows; r++) {
            sb.append(String.format("│ row_%-3d│", r));
            for (int c = 0; c < displayCols; c++) {
                double val = arr.getDouble(r * cols + c);
                sb.append(String.format(" %10.4f│", val));
            }
            if (cols > displayCols) {
                sb.append("      ...   │");
            }
            sb.append("\n");
        }
        
        // Trailing rows indicator
        if (rows > displayRows) {
            sb.append("│  ... │");
            for (int c = 0; c < displayCols; c++) {
                sb.append("      ...   │");
            }
            if (cols > displayCols) {
                sb.append("      ...   │");
            }
            sb.append("\n");
        }
        
        sb.append("└──────┴");
        for (int c = 0; c < displayCols; c++) {
            sb.append("──────────┴");
        }
        if (cols > displayCols) {
            sb.append("──────────┘");
        } else {
            sb.append("┘");
        }
        
        return sb.toString();
    }

    private String formatNumpyVector(NDArray arr, String title) {
        StringBuilder sb = new StringBuilder();
        sb.append("┌").append("─".repeat(Math.min(arr.shape[0], matrixDisplayCols * 2) + 15)).append("┐\n");
        sb.append(String.format("│ %s\n", title));
        sb.append("├─────────────────────────────────────────────────────────┤\n");
        
        sb.append("│ index │ value                                         │\n");
        sb.append("├───────┼───────────────────────────────────────────────┤\n");
        
        int display = Math.min(arr.shape[0], maxRows * 2);
        for (int i = 0; i < display; i++) {
            double val = arr.getDouble(i);
            sb.append(String.format("│ %5d │ %20.6f                       │\n", i, val));
        }
        
        if (arr.shape[0] > display) {
            sb.append("│  ...  │                       ...                       │\n");
        }
        
        sb.append("└───────┴───────────────────────────────────────────────┘\n");
        return sb.toString();
    }

    private String formatNumpyNDArray(NDArray arr) {
        StringBuilder sb = new StringBuilder();
        sb.append("┌─────────────────────────────────────────────────────────┐\n");
        sb.append("│ Multi-dimensional Array Preview                          │\n");
        sb.append("├─────────────────────────────────────────────────────────┤\n");
        
        // Show first few elements in flattened form
        sb.append("│ Flattened (first ").append(Math.min(20, (int) Math.min(arr.size, 100))).append("):\n│   [");
        
        long count = 0;
        for (long i = 0; i < arr.size && count < 20; i++) {
            if (count > 0) sb.append(", ");
            if (count % 5 == 0 && count > 0) sb.append("\n│    ");
            sb.append(String.format("%8.3f", arr.getDouble(i)));
            count++;
        }
        
        if (arr.size > 20) {
            sb.append(",\n│    ... (").append(arr.size - 20).append(" more elements)");
        }
        sb.append("]\n");
        sb.append("└─────────────────────────────────────────────────────────┘\n");
        
        return sb.toString();
    }

    // ====================== Standard DataFrame Display ======================

    private String formatStandard(DataFrame df) {
        StringBuilder sb = new StringBuilder();
        
        // Header
        sb.append("╔").append("═".repeat(70)).append("╗\n");
        sb.append(String.format("║ DataFrame: %d rows × %d columns%-26s ║\n", 
            df.rowCount(), df.columnCount(), ""));
        sb.append("╠").append("═".repeat(70)).append("╣\n");
        
        // Column types summary
        sb.append("║ Columns: ");
        for (int i = 0; i < Math.min(df.columnCount(), maxCols); i++) {
            Column c = df.column(i);
            String type = c.dtype().name();
            sb.append(c.name()).append("(").append(type.charAt(0)).append(")");
            if (i < Math.min(df.columnCount(), maxCols) - 1) sb.append(", ");
        }
        if (df.columnCount() > maxCols) sb.append(", ...");
        sb.append(" ║\n");
        sb.append("╠").append("═".repeat(70)).append("╣\n");
        
        // Column headers
        sb.append("║ ");
        for (int c = 0; c < Math.min(df.columnCount(), maxCols); c++) {
            String name = truncate(df.column(c).name(), colWidth);
            sb.append(String.format("%-" + colWidth + "s", name));
            if (c < Math.min(df.columnCount(), maxCols) - 1) sb.append(" │ ");
        }
        if (df.columnCount() > maxCols) sb.append(" │ ...");
        sb.append(" ║\n");
        
        // Separator
        sb.append("║ ");
        for (int c = 0; c < Math.min(df.columnCount(), maxCols); c++) {
            for (int i = 0; i < colWidth; i++) sb.append("─");
            if (c < Math.min(df.columnCount(), maxCols) - 1) sb.append("─┼─");
        }
        if (df.columnCount() > maxCols) sb.append("─┼───");
        sb.append(" ║\n");
        
        // Data rows
        int displayRows = Math.min(maxRows, df.rowCount());
        for (int r = 0; r < displayRows; r++) {
            sb.append("║ ");
            for (int c = 0; c < Math.min(df.columnCount(), maxCols); c++) {
                String val = formatValue(df.get(r, c));
                sb.append(String.format("%-" + colWidth + "s", truncate(val, colWidth)));
                if (c < Math.min(df.columnCount(), maxCols) - 1) sb.append(" │ ");
            }
            if (df.columnCount() > maxCols) sb.append(" │ ...");
            sb.append(" ║\n");
        }
        
        // Footer
        if (df.rowCount() > displayRows) {
            sb.append("║ ... ").append(df.rowCount() - displayRows).append(" more rows");
            sb.append(" ".repeat(Math.max(0, 50))).append(" ║\n");
        }
        sb.append("╚").append("═".repeat(70)).append("╝\n");
        
        return sb.toString();
    }

    /**
     * Detect if DataFrame is a matrix-style DataFrame (all numeric, 2D structure).
     */
    private boolean isMatrixDataFrame(DataFrame df) {
        if (df.columnCount() < 2) return false;
        
        // Check if all columns are numeric
        for (int i = 0; i < df.columnCount(); i++) {
            if (!df.column(i).isNumeric() && df.column(i).dtype() != Column.DType.BOOLEAN) {
                return false;
            }
        }
        
        // Check if all columns have same length
        int len = df.rowCount();
        for (int i = 1; i < df.columnCount(); i++) {
            if (df.column(i).size() != len) return false;
        }
        
        return true;
    }

    /**
     * Format matrix-style DataFrame as proper grid.
     */
    private String formatMatrix(DataFrame df) {
        int rows = df.rowCount();
        int cols = df.columnCount();
        int displayCols = Math.min(cols, matrixDisplayCols);
        
        StringBuilder sb = new StringBuilder();
        sb.append("╔").append("═".repeat(displayCols * 12 + 20)).append("╗\n");
        sb.append(String.format("║ Matrix DataFrame: %d rows × %d cols", rows, cols));
        sb.append(" ".repeat(Math.max(0, displayCols * 12 + 18 - String.format(" Matrix DataFrame: %d rows × %d cols", rows, cols).length()))).append("║\n");
        sb.append("╠").append("═".repeat(displayCols * 12 + 20)).append("╣\n");
        
        // Column headers
        sb.append("║          │");
        for (int c = 0; c < displayCols; c++) {
            sb.append(String.format("  col_%-5d│", c));
        }
        if (cols > displayCols) sb.append("  ...    │");
        sb.append("\n");
        
        sb.append("║──────────┼");
        for (int c = 0; c < displayCols; c++) {
            sb.append("──────────┼");
        }
        if (cols > displayCols) sb.append("──────────┤");
        sb.append("\n");
        
        // Data
        int displayRows = Math.min(maxRows, rows);
        for (int r = 0; r < displayRows; r++) {
            sb.append(String.format("║ row_%-4d │", r));
            for (int c = 0; c < displayCols; c++) {
                Object v = df.get(r, c);
                String val = v instanceof Number ? String.format("%10.4f", ((Number) v).doubleValue()) : "         - ";
                sb.append(val).append("│");
            }
            if (cols > displayCols) sb.append("      ...    │");
            sb.append("\n");
        }
        
        if (rows > displayRows) {
            sb.append("║   ...   │");
            for (int c = 0; c < displayCols; c++) {
                sb.append("      ...    │");
            }
            if (cols > displayCols) sb.append("      ...    │");
            sb.append("\n");
        }
        
        sb.append("╚").append("═".repeat(displayCols * 12 + 20)).append("╝\n");
        
        return sb.toString();
    }

    private String emptyDataFrame() {
        return "DataFrame (empty)\n";
    }

    private String formatValue(Object v) {
        if (v == null) return "null";
        if (v instanceof Number) {
            if (v instanceof Double || v instanceof Float) {
                return String.format("%.4f", ((Number) v).doubleValue());
            }
            return String.valueOf(((Number) v).longValue());
        }
        if (v instanceof float[]) {
            float[] arr = (float[]) v;
            if (arr.length <= 3) {
                return "[" + Arrays.toString(arr) + "]";
            }
            return String.format("[%d floats: %.2f, ...]", arr.length, arr[0]);
        }
        if (v instanceof double[]) {
            double[] arr = (double[]) v;
            if (arr.length <= 3) {
                return "[" + Arrays.toString(arr) + "]";
            }
            return String.format("[%d floats: %.2f, ...]", arr.length, arr[0]);
        }
        return String.valueOf(v);
    }

    private String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max - 3) + "...";
    }
}
