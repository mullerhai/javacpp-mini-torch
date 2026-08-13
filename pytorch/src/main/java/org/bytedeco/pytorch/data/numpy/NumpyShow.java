/*
 * NumpyShow — formatter for NDArray / NpyHeader with NumPy/Pandas-style printing.
 *
 * <p>Behaviour:
 * <ul>
 *   <li>1-D arrays print as a row of values</li>
 *   <li>2-D arrays print as a table</li>
 *   <li>Higher-rank arrays show as bracketed hierarchical format with leading
 *       and trailing ellipses when truncated (NumPy default)</li>
 *   <li>Large 1-D arrays are head/tail truncated with summary line</li>
 *   <li>Complex types print as {@code (re + j im)} pairs</li>
 * </ul>
 */
package org.bytedeco.pytorch.data.numpy;

public final class NumpyShow {

    public static final int DEFAULT_THRESHOLD = 1000;
    public static final int DEFAULT_EDGEITEMS = 3;
    public static final int DEFAULT_LINEWIDTH = 75;

    private final int threshold;
    private final int edgeItems;
    private final int lineWidth;

    public NumpyShow() {
        this(DEFAULT_THRESHOLD, DEFAULT_EDGEITEMS, DEFAULT_LINEWIDTH);
    }

    public NumpyShow(int threshold, int edgeItems, int lineWidth) {
        this.threshold = Math.max(1, threshold);
        this.edgeItems = Math.max(1, edgeItems);
        this.lineWidth = Math.max(20, lineWidth);
    }

    /** Print to stdout using the configured threshold/edge/width. */
    public void show(NDArray arr) {
        System.out.println(format(arr));
    }

    public String format(NDArray arr) {
        if (arr == null) return "null";
        return formatArray(arr, 0);
    }

    public void showHeader(NpyHeader h) {
        System.out.println(formatHeader(h));
    }

    public String formatHeader(NpyHeader h) {
        if (h == null) return "null";
        return "NpyHeader{dtype=" + h.dtype.getDescriptor()
                + ", fortran_order=" + h.fortranOrder
                + ", shape=(" + shapeStr(h.shape) + ")"
                + ", numel=" + h.numel() + "}";
    }

    private String formatArray(NDArray arr, int level) {
        int ndim = arr.shape.length;
        if (ndim == 0) {
            return scalarString(arr, 0);
        }
        long size = arr.size;
        if (size > threshold && ndim > 1) {
            // NumPy prints the array as a multi-line nested form; we mimic the
            // head/tail truncation pattern.
            return truncatedRecursive(arr, level, new long[ndim]);
        }
        if (ndim == 1) {
            return formatFlat1D(arr, level);
        }
        return formatRecursive(arr, level, new long[ndim]);
    }

    private String formatFlat1D(NDArray arr, int level) {
        StringBuilder sb = new StringBuilder();
        sb.append(indent(level)).append('[');
        long n = arr.size;
        if (n <= 2 * edgeItems) {
            for (long i = 0; i < n; i++) {
                if (i > 0) sb.append(' ');
                sb.append(scalarString(arr, (int) i));
            }
        } else {
            for (int i = 0; i < edgeItems; i++) {
                if (i > 0) sb.append(' ');
                sb.append(scalarString(arr, i));
            }
            sb.append(" ... ");
            for (long i = n - edgeItems; i < n; i++) {
                sb.append(' ').append(scalarString(arr, (int) i));
            }
        }
        sb.append(']');
        return sb.toString();
    }

    private String formatRecursive(NDArray arr, int level, long[] cursor) {
        StringBuilder sb = new StringBuilder();
        long dim = arr.shape[level];
        sb.append(indent(level)).append('[');
        if (level == arr.shape.length - 2) {
            // innermost 2D rows
            for (long i = 0; i < dim; i++) {
                if (i > 0) sb.append('\n').append(indent(level + 1));
                cursor[level] = i;
                sb.append(formatRow(arr, cursor, level));
            }
        } else {
            for (long i = 0; i < dim; i++) {
                if (i > 0) sb.append('\n');
                cursor[level] = i;
                sb.append(formatRecursive(arr, level + 1, cursor));
            }
        }
        sb.append(indent(level)).append(']');
        return sb.toString();
    }

    private String truncatedRecursive(NDArray arr, int level, long[] cursor) {
        long dim = arr.shape[level];
        StringBuilder sb = new StringBuilder();
        sb.append(indent(level)).append('[');
        if (dim <= 2 * edgeItems) {
            for (long i = 0; i < dim; i++) {
                if (i > 0) sb.append('\n');
                cursor[level] = i;
                sb.append(level == arr.shape.length - 1
                        ? formatFlat1D(arr, level + 1)
                        : truncatedRecursive(arr, level + 1, cursor));
            }
        } else {
            for (long i = 0; i < edgeItems; i++) {
                if (i > 0) sb.append('\n');
                cursor[level] = i;
                sb.append(level == arr.shape.length - 1
                        ? formatFlat1D(arr, level + 1)
                        : truncatedRecursive(arr, level + 1, cursor));
            }
            sb.append('\n').append(indent(level + 1)).append("...");
            for (long i = dim - edgeItems; i < dim; i++) {
                sb.append('\n');
                cursor[level] = i;
                sb.append(level == arr.shape.length - 1
                        ? formatFlat1D(arr, level + 1)
                        : truncatedRecursive(arr, level + 1, cursor));
            }
        }
        sb.append(indent(level)).append(']');
        return sb.toString();
    }

    private String formatRow(NDArray arr, long[] cursor, int level) {
        long row = cursor[level];
        long inner = arr.shape[level + 1];
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (long j = 0; j < inner; j++) {
            if (j > 0) sb.append(' ');
            cursor[level + 1] = j;
            sb.append(scalarStringAt(arr, cursor));
        }
        sb.append(']');
        // row was stored in cursor[level]; no need to restore
        return sb.toString();
    }

    private String scalarString(NDArray arr, int flatIndex) {
        if (arr.dtype.isComplex()) {
            double re = arr.getReal(flatIndex);
            double im = arr.getImag(flatIndex);
            return formatComplex(re, im);
        }
        if (arr.dtype == DType.BOOL) {
            return arr.getLong(flatIndex) != 0 ? "True" : "False";
        }
        if (isFloatFamily(arr.dtype)) {
            return formatDouble(arr.getDouble(flatIndex));
        }
        long v = arr.getLong(flatIndex);
        return String.valueOf(v);
    }

    private String scalarStringAt(NDArray arr, long[] cursor) {
        long flat = 0;
        long stride = 1;
        for (int d = arr.shape.length - 1; d >= 0; d--) {
            flat += cursor[d] * stride;
            stride *= arr.shape[d];
        }
        return scalarString(arr, (int) flat);
    }

    private static boolean isFloatFamily(DType d) {
        return d == DType.FLOAT32 || d == DType.FLOAT64 || d == DType.FLOAT16;
    }

    private static String indent(int n) {
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; i++) sb.append(' ');
        return sb.toString();
    }

    private static String formatDouble(double v) {
        if (Double.isNaN(v)) return "nan";
        if (Double.isInfinite(v)) return v > 0 ? "inf" : "-inf";
        // NumPy prints short floats as 1.0, not 1.000000
        if (v == Math.floor(v) && Math.abs(v) < 1e16) {
            return Long.toString((long) v) + ".";
        }
        return String.format("%.6f", v).replaceAll("0+$", "").replaceAll("\\.$", ".0");
    }

    private static String formatComplex(double re, double im) {
        StringBuilder sb = new StringBuilder();
        sb.append('(').append(formatDouble(re)).append("+j").append(formatDouble(im)).append(')');
        return sb.toString();
    }

    private static String shapeStr(long[] shape) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        if (shape.length == 1) sb.append(',');
        return sb.toString();
    }
}
