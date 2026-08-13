package org.bytedeco.pytorch.data.pt;

import java.util.*;

/**
 * PT file formatter with NumPy/Pandas-style printing.
 * 
 * <p>Features:
 * <ul>
 *   <li>show(): PyTorch-style tensor printing with shape, dtype, preview</li>
 *   <li>schema(): DataFrame-style schema showing all tensor types and shapes</li>
 *   <li>printSchema(): Console output of schema</li>
 * </ul>
 * 
 * <p>Example output:
 * <pre>
 * schema():
 * name          dtype      shape              size    preview
 * item_id       int64      [1000000]          8 MB    [1000001, 1000002, ...]
 * category_id   int64      [1000000]          8 MB    [1, 2, 3, ...]
 * ...
 * 
 * show():
 * {
 *     'item_id': tensor([1000000], dtype=int64, [1000001, 1000002, 1000003, ...]),
 *     ...
 * }
 * </pre>
 */
public final class PTShow {

    public static final int DEFAULT_PREVIEW_ITEMS = 5;
    public static final int DEFAULT_LINE_WIDTH = 120;

    private final int previewItems;
    private final int lineWidth;

    public PTShow() {
        this(DEFAULT_PREVIEW_ITEMS, DEFAULT_LINE_WIDTH);
    }

    public PTShow(int previewItems, int lineWidth) {
        this.previewItems = Math.max(1, previewItems);
        this.lineWidth = Math.max(80, lineWidth);
    }

    // ====================== Public API ======================

    /**
     * Print tensors to stdout in PyTorch-style format.
     */
    public void show(Map<String, PT.TensorData> tensors) {
        System.out.println(showString(tensors));
    }

    /**
     * Get PyTorch-style string representation.
     */
    public String showString(Map<String, PT.TensorData> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            return "{}";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");

        int idx = 0;
        int total = tensors.size();
        for (Map.Entry<String, PT.TensorData> e : tensors.entrySet()) {
            String name = e.getKey();
            PT.TensorData td = e.getValue();

            sb.append("    '").append(name).append("': tensor(");
            sb.append(formatShape(td.shape));
            sb.append(", dtype=").append(td.dtype);
            sb.append(",\n        [");
            sb.append(formatPreview(td));
            sb.append("]");
            sb.append(")");
            if (++idx < total) sb.append(",");
            sb.append("\n");
        }

        sb.append("}\n");
        return sb.toString();
    }

    /**
     * Print schema to stdout.
     */
    public void printSchema(Map<String, PT.TensorData> tensors) {
        System.out.println(schema(tensors));
    }

    /**
     * Get DataFrame-style schema string.
     */
    public String schema(Map<String, PT.TensorData> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            return "PT Schema (0 tensors)\n";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("PT Schema (").append(tensors.size()).append(" tensors)\n");

        // Column headers
        String nameCol = "name";
        String dtypeCol = "dtype";
        String shapeCol = "shape";
        String sizeCol = "size";
        String previewCol = "preview";

        int nameWidth = Math.max(nameCol.length(), 20);
        int dtypeWidth = Math.max(dtypeCol.length(), 10);
        int shapeWidth = Math.max(shapeCol.length(), 25);
        int sizeWidth = Math.max(sizeCol.length(), 10);
        int previewWidth = Math.max(previewCol.length(), lineWidth - nameWidth - dtypeWidth - shapeWidth - sizeWidth - 10);

        // Header
        sb.append(padRight(nameCol, nameWidth))
          .append("  ")
          .append(padRight(dtypeCol, dtypeWidth))
          .append("  ")
          .append(padRight(shapeCol, shapeWidth))
          .append("  ")
          .append(padRight(sizeCol, sizeWidth))
          .append("  ")
          .append(previewCol)
          .append("\n");

        // Separator
        sb.append(padRight("", nameWidth, '-'))
          .append("  ")
          .append(padRight("", dtypeWidth, '-'))
          .append("  ")
          .append(padRight("", shapeWidth, '-'))
          .append("  ")
          .append(padRight("", sizeWidth, '-'))
          .append("  ")
          .append(padRight("", previewWidth, '-'))
          .append("\n");

        // Rows
        long totalBytes = 0;
        for (Map.Entry<String, PT.TensorData> e : tensors.entrySet()) {
            String name = e.getKey();
            PT.TensorData td = e.getValue();

            long bytes = td.dtype.sizeBytes() * td.elementCount();
            totalBytes += bytes;

            String preview = formatPreview(td);
            if (preview.length() > previewWidth) {
                preview = preview.substring(0, previewWidth - 3) + "...";
            }

            sb.append(padRight(truncate(name, nameWidth), nameWidth))
              .append("  ")
              .append(padRight(td.dtype.name(), dtypeWidth))
              .append("  ")
              .append(padRight(formatShape(td.shape), shapeWidth))
              .append("  ")
              .append(padRight(formatBytes(bytes), sizeWidth))
              .append("  ")
              .append(preview)
              .append("\n");
        }

        // Summary
        sb.append("\nTotal size: ").append(formatBytes(totalBytes));

        return sb.toString();
    }

    // ====================== Helpers ======================

    private String formatPreview(PT.TensorData td) {
        if (td.elementCount() == 0) return "";

        int show = Math.min(previewItems, (int) td.elementCount());
        StringBuilder sb = new StringBuilder();

        if (td.dtype.isFloat()) {
            float[] arr = td.asFloatArray();
            for (int i = 0; i < show; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatDouble(arr[i]));
            }
        } else {
            long[] arr = td.asLongArray();
            for (int i = 0; i < show; i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr[i]);
            }
        }

        if (td.elementCount() > show) {
            sb.append(", ...");
        }

        return sb.toString();
    }

    private String formatShape(long[] shape) {
        if (shape == null || shape.length == 0) return "[0]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private String formatDouble(double v) {
        if (Double.isNaN(v)) return "nan";
        if (Double.isInfinite(v)) return v > 0 ? "inf" : "-inf";
        if (v == Math.floor(v) && Math.abs(v) < 1e15) {
            return Long.toString((long) v);
        }
        return String.format("%.4f", v).replaceAll("0+$", "").replaceAll("\\.$", "");
    }

    private static String padRight(String s, int width) {
        return padRight(s, width, ' ');
    }

    private static String padRight(String s, int width, char pad) {
        if (s == null) s = "";
        if (s.length() >= width) return s;
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < width) sb.append(pad);
        return sb.toString();
    }

    private static String truncate(String s, int width) {
        if (s == null) return "";
        if (s.length() <= width) return s;
        return s.substring(0, width - 3) + "...";
    }
}
