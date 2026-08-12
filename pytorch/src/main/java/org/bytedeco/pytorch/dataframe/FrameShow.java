package org.bytedeco.pytorch.dataframe;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.*;
import java.util.List;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import javax.swing.*;
import javax.swing.table.*;
import javax.swing.border.*;

/**
 * Enterprise-grade DataFrame viewer with search, filter, and pivot capabilities.
 * 
 * <p>Provides a popup dialog for viewing DataFrames with:</p>
 * <ul>
 *   <li>Virtualized table for large datasets</li>
 *   <li>Search box with regex support</li>
 *   <li>Column filter dropdown menus</li>
 *   <li>Column sorting</li>
 *   <li>Pivot table support</li>
 *   <li>Export options</li>
 * </ul>
 */
public class FrameShow {

    private static final int DEFAULT_WIDTH = 1200;
    private static final int DEFAULT_HEIGHT = 700;
    private static final int PAGE_SIZE = 100;

    private FrameShow() {}

    /**
     * Show DataFrame in a popup dialog.
     */
    public static void show(DataFrame df) {
        show(df, "DataFrame Viewer", DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }

    public static void show(DataFrame df, String title) {
        show(df, title, DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }

    public static void show(DataFrame df, String title, int width, int height) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame(title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setSize(width, height);
            frame.setLocationRelativeTo(null);
            
            DataFrameViewer viewer = new DataFrameViewer(df);
            frame.add(viewer.getContent());
            
            frame.setVisible(true);
        });
    }

    /**
     * Show file in popup dialog based on extension.
     */
    public static void show(String path) throws Exception {
        show(path, "Data Viewer", DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }

    public static void show(String path, String title, int width, int height) throws Exception {
        java.nio.file.Path p = java.nio.file.Paths.get(path);
        String ext = getExtension(p.getFileName().toString()).toLowerCase();
        
        switch (ext) {
            case "csv", "tsv" -> {
                DataFrame df = DataFrame.read().csv(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "json", "jsonl" -> {
                DataFrame df = DataFrame.read().json(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "parquet" -> {
                DataFrame df = DataFrame.read().parquet(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "pt", "pth" -> {
                DataFrame df = DataFrame.readPT(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "npy" -> {
                DataFrame df = DataFrame.readNpy(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "safetensors" -> {
                DataFrame df = DataFrame.readSafetensors(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "toml" -> {
                DataFrame df = org.bytedeco.pytorch.dataframe.io.TomlReader.read(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            case "bin" -> {
                DataFrame df = org.bytedeco.pytorch.dataframe.io.BinReader.read(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
            default -> {
                DataFrame df = DataFrame.read().load(path);
                show(df, title + " - " + p.getFileName(), width, height);
            }
        }
    }

    // ---- Viewer Component ----

    static class DataFrameViewer {
        private final DataFrame original;
        private DataFrame current;
        private JTable table;
        private JTextField searchField;
        private JLabel statusLabel;
        private JComboBox<String> columnFilter;
        private JTextField filterValue;
        private DefaultTableModel tableModel;
        private List<Integer> filteredRows;
        
        DataFrameViewer(DataFrame df) {
            this.original = df;
            this.current = df;
            this.filteredRows = null;
        }

        JPanel getContent() {
            JPanel main = new JPanel(new BorderLayout(5, 5));
            main.setBorder(new EmptyBorder(10, 10, 10, 10));
            
            // Top toolbar
            main.add(createToolbar(), BorderLayout.NORTH);
            
            // Center table
            main.add(createTable(), BorderLayout.CENTER);
            
            // Bottom status
            main.add(createStatusBar(), BorderLayout.SOUTH);
            
            return main;
        }

        private JPanel createToolbar() {
            JPanel toolbar = new JPanel(new BorderLayout(10, 5));
            toolbar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(Color.LIGHT_GRAY),
                new EmptyBorder(5, 5, 5, 5)
            ));
            
            // Left: Title and row count
            JLabel titleLabel = new JLabel(String.format(
                "<html><b>%d rows × %d columns</b></html>", 
                original.rowCount(), original.columnCount()
            ));
            toolbar.add(titleLabel, BorderLayout.WEST);
            
            // Center: Search and filter
            JPanel center = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
            
            center.add(new JLabel("Search:"));
            searchField = new JTextField(20);
            searchField.setToolTipText("Enter text to search (supports regex)");
            searchField.addActionListener(e -> applySearch());
            center.add(searchField);
            
            JButton searchBtn = new JButton("Find");
            searchBtn.addActionListener(e -> applySearch());
            center.add(searchBtn);
            
            JButton clearBtn = new JButton("Clear");
            clearBtn.addActionListener(e -> clearFilters());
            center.add(clearBtn);
            
            center.add(Box.createHorizontalStrut(20));
            
            center.add(new JLabel("Filter Column:"));
            columnFilter = new JComboBox<>();
            columnFilter.addItem("-- Select --");
            for (int i = 0; i < original.columnCount(); i++) {
                columnFilter.addItem(original.column(i).name());
            }
            center.add(columnFilter);
            
            center.add(new JLabel("="));
            filterValue = new JTextField(15);
            filterValue.setToolTipText("Enter value to filter");
            filterValue.addActionListener(e -> applyFilter());
            center.add(filterValue);
            
            JButton filterBtn = new JButton("Apply");
            filterBtn.addActionListener(e -> applyFilter());
            center.add(filterBtn);
            
            toolbar.add(center, BorderLayout.CENTER);
            
            // Right: Actions
            JPanel actions = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 0));
            
            JButton statsBtn = new JButton("Statistics");
            statsBtn.addActionListener(e -> showStatistics());
            actions.add(statsBtn);
            
            JButton pivotBtn = new JButton("Pivot");
            pivotBtn.addActionListener(e -> showPivotDialog());
            actions.add(pivotBtn);
            
            JButton exportBtn = new JButton("Export");
            exportBtn.addActionListener(e -> showExportDialog());
            actions.add(exportBtn);
            
            toolbar.add(actions, BorderLayout.EAST);
            
            return toolbar;
        }

        private JScrollPane createTable() {
            tableModel = new DefaultTableModel() {
                @Override
                public boolean isCellEditable(int row, int column) {
                    return false;
                }
            };
            
            // Add columns
            tableModel.addColumn("Row");
            for (int i = 0; i < current.columnCount(); i++) {
                String name = current.column(i).name();
                String dtype = current.column(i).dtype().name();
                tableModel.addColumn(name + " (" + dtype + ")");
            }
            
            // Add rows
            updateTableData();
            
            table = new JTable(tableModel);
            table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
            table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
            table.setRowHeight(22);
            table.getColumnModel().getColumn(0).setPreferredWidth(60);
            
            // Style header
            JTableHeader header = table.getTableHeader();
            header.setReorderingAllowed(true);
            header.setDefaultRenderer(new HeaderRenderer());
            
            // Column sorting
            table.setRowSorter(new TableRowSorter<>(tableModel) {
                @Override
                public boolean isSortable(int column) {
                    return column > 0; // Don't sort row index
                }
            });
            
            JScrollPane scroll = new JScrollPane(table);
            scroll.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
            scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
            
            return scroll;
        }

        private JPanel createStatusBar() {
            JPanel status = new JPanel(new BorderLayout());
            status.setBorder(BorderFactory.createEmptyBorder(5, 0, 0, 0));
            
            statusLabel = new JLabel(String.format(
                "Showing %d of %d rows", 
                Math.min(current.rowCount(), 1000), current.rowCount()
            ));
            status.add(statusLabel, BorderLayout.WEST);
            
            // Pagination
            JPanel paging = new JPanel(new FlowLayout(FlowLayout.RIGHT));
            paging.add(new JLabel("Rows per page:"));
            String[] pageSizes = {"100", "500", "1000", "5000"};
            JComboBox<String> pageCombo = new JComboBox<>(pageSizes);
            pageCombo.setSelectedItem("1000");
            pageCombo.addActionListener(e -> {
                // Re-render with new page size
            });
            paging.add(pageCombo);
            status.add(paging, BorderLayout.EAST);
            
            return status;
        }

        private void updateTableData() {
            tableModel.setRowCount(0);
            
            int displayRows = Math.min(current.rowCount(), 5000);
            for (int r = 0; r < displayRows; r++) {
                Object[] row = new Object[current.columnCount() + 1];
                row[0] = r;
                for (int c = 0; c < current.columnCount(); c++) {
                    row[c + 1] = formatValue(current.get(r, c));
                }
                tableModel.addRow(row);
            }
            
            if (current.rowCount() > displayRows) {
                statusLabel.setText(String.format(
                    "Showing %d of %d rows (use search to filter)",
                    displayRows, current.rowCount()
                ));
            }
        }

        private String formatValue(Object v) {
            if (v == null) return "null";
            if (v instanceof float[]) {
                float[] arr = (float[]) v;
                if (arr.length <= 5) {
                    StringBuilder sb = new StringBuilder("[");
                    for (int i = 0; i < arr.length; i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(String.format("%.4f", arr[i]));
                    }
                    sb.append("]");
                    return sb.toString();
                }
                return String.format("[%d floats]", arr.length);
            }
            if (v instanceof double[]) {
                double[] arr = (double[]) v;
                if (arr.length <= 5) {
                    StringBuilder sb = new StringBuilder("[");
                    for (int i = 0; i < arr.length; i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(String.format("%.4f", arr[i]));
                    }
                    sb.append("]");
                    return sb.toString();
                }
                return String.format("[%d doubles]", arr.length);
            }
            if (v instanceof long[]) {
                long[] arr = (long[]) v;
                if (arr.length <= 10) {
                    return Arrays.toString(arr);
                }
                return String.format("[%d longs]", arr.length);
            }
            if (v instanceof byte[]) {
                return String.format("[%d bytes]", ((byte[]) v).length);
            }
            if (v instanceof Number) {
                double d = ((Number) v).doubleValue();
                if (d == (long) d) {
                    return String.valueOf((long) d);
                }
                return String.format("%.6g", d);
            }
            String s = String.valueOf(v);
            if (s.length() > 100) {
                return s.substring(0, 97) + "...";
            }
            return s;
        }

        private void applySearch() {
            String query = searchField.getText().trim();
            if (query.isEmpty()) {
                clearFilters();
                return;
            }
            
            try {
                Pattern pattern = Pattern.compile(query, Pattern.CASE_INSENSITIVE);
                
                current = DataFrame.create();
                for (int i = 0; i < original.columnCount(); i++) {
                    current.addColumn(original.column(i).copy());
                }
                
                for (int r = 0; r < original.rowCount(); r++) {
                    boolean match = false;
                    for (int c = 0; c < original.columnCount(); c++) {
                        String val = String.valueOf(original.get(r, c));
                        if (pattern.matcher(val).find()) {
                            match = true;
                            break;
                        }
                    }
                    if (match) {
                        int newRow = current.addEmptyRow();
                        for (int c = 0; c < original.columnCount(); c++) {
                            current.set(newRow, c, original.get(r, c));
                        }
                    }
                }
                
                updateTableData();
                statusLabel.setText(String.format(
                    "Search '%s': found %d matches", query, current.rowCount()
                ));
            } catch (Exception e) {
                JOptionPane.showMessageDialog(table, 
                    "Invalid regex: " + e.getMessage(), 
                    "Search Error", JOptionPane.ERROR_MESSAGE);
            }
        }

        private void applyFilter() {
            int colIdx = columnFilter.getSelectedIndex() - 1;
            if (colIdx < 0) {
                JOptionPane.showMessageDialog(table,
                    "Please select a column to filter",
                    "Filter", JOptionPane.INFORMATION_MESSAGE);
                return;
            }
            
            String value = filterValue.getText().trim();
            if (value.isEmpty()) {
                clearFilters();
                return;
            }
            
            current = DataFrame.create();
            for (int i = 0; i < original.columnCount(); i++) {
                current.addColumn(original.column(i).copy());
            }
            
            for (int r = 0; r < original.rowCount(); r++) {
                String val = String.valueOf(original.get(r, colIdx));
                if (val.equals(value)) {
                    int newRow = current.addEmptyRow();
                    for (int c = 0; c < original.columnCount(); c++) {
                        current.set(newRow, c, original.get(r, c));
                    }
                }
            }
            
            updateTableData();
            statusLabel.setText(String.format(
                "Filter column '%s' = '%s': %d matches",
                original.column(colIdx).name(), value, current.rowCount()
            ));
        }

        private void clearFilters() {
            current = original;
            searchField.setText("");
            filterValue.setText("");
            columnFilter.setSelectedIndex(0);
            updateTableData();
            statusLabel.setText(String.format(
                "Showing %d rows", current.rowCount()
            ));
        }

        private void showStatistics() {
            StringBuilder sb = new StringBuilder();
            sb.append("═══════════════════════════════════════════════════\n");
            sb.append("                    Statistics                      \n");
            sb.append("═══════════════════════════════════════════════════\n\n");
            
            for (int c = 0; c < current.columnCount(); c++) {
                Column col = current.column(c);
                if (col.isNumeric()) {
                    double min = Double.MAX_VALUE;
                    double max = Double.MIN_VALUE;
                    double sum = 0;
                    int count = 0;
                    
                    for (int r = 0; r < current.rowCount(); r++) {
                        Object v = current.get(r, c);
                        if (v instanceof Number) {
                            double d = ((Number) v).doubleValue();
                            min = Math.min(min, d);
                            max = Math.max(max, d);
                            sum += d;
                            count++;
                        }
                    }
                    
                    if (count > 0) {
                        sb.append(String.format("%s (%s):\n", col.name(), col.dtype()));
                        sb.append(String.format("  count: %d\n", count));
                        sb.append(String.format("  mean:  %.6f\n", sum / count));
                        sb.append(String.format("  min:   %.6f\n", min));
                        sb.append(String.format("  max:   %.6f\n", max));
                        sb.append(String.format("  sum:   %.6f\n\n", sum));
                    }
                } else {
                    Set<Object> unique = new HashSet<>();
                    for (int r = 0; r < Math.min(current.rowCount(), 10000); r++) {
                        unique.add(current.get(r, c));
                    }
                    sb.append(String.format("%s (%s):\n", col.name(), col.dtype()));
                    sb.append(String.format("  unique: %d\n", unique.size()));
                    sb.append(String.format("  sample: %s\n\n", 
                        unique.stream().limit(5).toList()));
                }
            }
            
            JTextArea textArea = new JTextArea(sb.toString());
            textArea.setEditable(false);
            textArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            JScrollPane scroll = new JScrollPane(textArea);
            scroll.setPreferredSize(new Dimension(500, 400));
            
            JOptionPane.showMessageDialog(table, scroll, "Statistics", 
                JOptionPane.INFORMATION_MESSAGE);
        }

        private void showPivotDialog() {
            JPanel panel = new JPanel(new GridLayout(3, 2, 10, 10));
            panel.setBorder(new EmptyBorder(10, 10, 10, 10));
            
            JComboBox<String> indexCombo = new JComboBox<>();
            JComboBox<String> columnsCombo = new JComboBox<>();
            JComboBox<String> valuesCombo = new JComboBox<>();
            
            for (int i = 0; i < original.columnCount(); i++) {
                String name = original.column(i).name();
                indexCombo.addItem(name);
                columnsCombo.addItem(name);
                valuesCombo.addItem(name);
            }
            
            panel.add(new JLabel("Index (row):"));
            panel.add(indexCombo);
            panel.add(new JLabel("Columns:"));
            panel.add(columnsCombo);
            panel.add(new JLabel("Values:"));
            panel.add(valuesCombo);
            
            int result = JOptionPane.showConfirmDialog(table, panel,
                "Create Pivot Table", JOptionPane.OK_CANCEL_OPTION);
            
            if (result == JOptionPane.OK_OPTION) {
                String index = (String) indexCombo.getSelectedItem();
                String columns = (String) columnsCombo.getSelectedItem();
                String values = (String) valuesCombo.getSelectedItem();
                
                DataFrame pivoted = original.pivot(index, columns, values);
                show(pivoted, "Pivot Table - " + index + " vs " + columns);
            }
        }

        private void showExportDialog() {
            String[] formats = {"CSV", "Parquet", "JSON", "Pickle"};
            JComboBox<String> formatCombo = new JComboBox<>(formats);
            
            JTextField pathField = new JTextField(30);
            
            JPanel panel = new JPanel(new GridLayout(2, 2, 10, 10));
            panel.setBorder(new EmptyBorder(10, 10, 10, 10));
            panel.add(new JLabel("Format:"));
            panel.add(formatCombo);
            panel.add(new JLabel("Path:"));
            panel.add(pathField);
            
            int result = JOptionPane.showConfirmDialog(table, panel,
                "Export DataFrame", JOptionPane.OK_CANCEL_OPTION);
            
            if (result == JOptionPane.OK_OPTION) {
                String path = pathField.getText().trim();
                if (!path.isEmpty()) {
                    try {
                        String fmt = (String) formatCombo.getSelectedItem();
                        switch (fmt) {
                            case "CSV" -> current.write().csv(path);
                            case "Parquet" -> current.write().parquet(path);
                            case "JSON" -> current.write().json(path);
                            case "Pickle" -> current.toPickle(path);
                        }
                        JOptionPane.showMessageDialog(table,
                            "Exported successfully to:\n" + path,
                            "Export Complete", JOptionPane.INFORMATION_MESSAGE);
                    } catch (Exception e) {
                        JOptionPane.showMessageDialog(table,
                            "Export failed: " + e.getMessage(),
                            "Export Error", JOptionPane.ERROR_MESSAGE);
                    }
                }
            }
        }

        static class HeaderRenderer extends DefaultTableCellRenderer {
            @Override
            public Component getTableCellRendererComponent(JTable table, Object value,
                    boolean isSelected, boolean hasFocus, int row, int column) {
                JLabel label = (JLabel) super.getTableCellRendererComponent(
                    table, value, isSelected, hasFocus, row, column);
                label.setFont(label.getFont().deriveFont(Font.BOLD));
                label.setBackground(new Color(240, 240, 245));
                label.setBorder(BorderFactory.createMatteBorder(0, 0, 1, 1, Color.GRAY));
                return label;
            }
        }
    }

    // ---- Helpers ----

    private static String getExtension(String name) {
        int idx = name.lastIndexOf('.');
        return idx >= 0 ? name.substring(idx + 1) : name;
    }

    // ---- Convenience methods ----

    /**
     * Show PT file tensor viewer.
     */
    public static void showPT(String path) throws Exception {
        File f = new File(path);
        Map<String, org.bytedeco.pytorch.data.pt.PT.TensorData> tensors = 
            org.bytedeco.pytorch.data.pt.PT.load(f);
        
        DataFrame df = DataFrame.create();
        df.addColumn("name", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("elements", Column.DType.INT64);
        
        for (Map.Entry<String, org.bytedeco.pytorch.data.pt.PT.TensorData> e : tensors.entrySet()) {
            org.bytedeco.pytorch.data.pt.PT.TensorData td = e.getValue();
            long elements = 1;
            for (long d : td.shape) elements *= d;
            
            df.addRow(e.getKey(), 
                Arrays.toString(td.shape), 
                td.dtype.name(), 
                elements);
        }
        
        show(df, "PT Tensors - " + f.getName());
    }

    /**
     * Show SafeTensors file tensor viewer.
     */
    public static void showSafeTensors(String path) throws Exception {
        File f = new File(path);
        Map<String, org.bytedeco.pytorch.Tensor> tensors = 
            org.bytedeco.pytorch.data.safetensors.SafeTensors.loadAsTensors(f, false);
        
        DataFrame df = DataFrame.create();
        df.addColumn("name", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("elements", Column.DType.INT64);
        
        for (Map.Entry<String, org.bytedeco.pytorch.Tensor> e : tensors.entrySet()) {
            org.bytedeco.pytorch.Tensor t = e.getValue();
            long elements = 1;
            StringBuilder shape = new StringBuilder("[");
            for (int i = 0; i < t.dim(); i++) {
                if (i > 0) shape.append(", ");
                shape.append(t.sizes().get(i));
                elements *= t.sizes().get(i);
            }
            shape.append("]");
            
            df.addRow(e.getKey(), shape.toString(), t.scalar_type().toString(), elements);
        }
        
        show(df, "SafeTensors - " + f.getName());
    }
}
