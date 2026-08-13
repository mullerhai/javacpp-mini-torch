/*
 * NumpySchema — schema descriptor for NDArray / NpyHeader / npz archives.
 *
 * <p>Mirrors Spark/Polars printSchema() for the numpy family:
 * <pre>
 *   root
 *    |-- dtype: float64
 *    |-- shape: (1024, 768)
 *    |-- numel: 786432
 *    |-- byte_size: 6291456
 *    |-- fortran_order: false
 *    |-- contiguous: true
 *    |-- view: false
 * </pre>
 *
 * <p>For npz archives the schema reports each dataset as a column.
 */
package org.bytedeco.pytorch.data.numpy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class NumpySchema {

    public static final class Field {
        public final String name;
        public final String dtype;
        public final String shape;
        public final long numel;
        public final long byteSize;
        public final Map<String, String> extras;

        public Field(String name, String dtype, String shape, long numel, long byteSize,
                     Map<String, String> extras) {
            this.name = name;
            this.dtype = dtype;
            this.shape = shape;
            this.numel = numel;
            this.byteSize = byteSize;
            this.extras = extras == null ? Collections.emptyMap() : extras;
        }

        @Override
        public String toString() {
            return name + ": " + dtype + " " + shape + " (numel=" + numel + ", bytes=" + byteSize + ")";
        }
    }

    private final String title;
    private final List<Field> fields = new ArrayList<>();

    public NumpySchema(String title) {
        this.title = title == null ? "ndarray" : title;
    }

    public NumpySchema addField(Field f) { fields.add(f); return this; }
    public NumpySchema addField(String name, String dtype, String shape, long numel, long byteSize) {
        return addField(new Field(name, dtype, shape, numel, byteSize, Collections.emptyMap()));
    }

    public String title() { return title; }
    public List<Field> fields() { return fields; }

    /** Print Spark-style schema tree to stdout. */
    public void printSchema() {
        System.out.println(toSchemaString());
    }

    /** Spark-style schema tree as a string. */
    public String toSchemaString() {
        StringBuilder sb = new StringBuilder();
        sb.append(title).append('\n');
        for (int i = 0; i < fields.size(); i++) {
            Field f = fields.get(i);
            String branch = (i == fields.size() - 1) ? " |-- " : " |-- ";
            sb.append(branch).append(formatFieldLine(f)).append('\n');
        }
        return sb.toString();
    }

    /** Pandas describe-style summary as a string. */
    public String toDetailString() {
        StringBuilder sb = new StringBuilder();
        sb.append(title).append(" schema detail\n");
        sb.append(String.format("%-20s %-12s %-20s %12s %12s%n",
                "field", "dtype", "shape", "numel", "bytes"));
        sb.append(String.format("%-20s %-12s %-20s %12s %12s%n",
                dash(20), dash(12), dash(20), dash(12), dash(12)));
        for (Field f : fields) {
            sb.append(String.format("%-20s %-12s %-20s %12d %12d%n",
                    truncate(f.name, 20), f.dtype, truncate(f.shape, 20),
                    f.numel, f.byteSize));
        }
        return sb.toString();
    }

    private static String dash(int n) {
        StringBuilder s = new StringBuilder(n);
        for (int i = 0; i < n; i++) s.append('-');
        return s.toString();
    }

    private static String truncate(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(0, n - 1) + "~";
    }

    private static String formatFieldLine(Field f) {
        StringBuilder sb = new StringBuilder();
        sb.append(f.name).append(": ").append(f.dtype).append(' ').append(f.shape);
        sb.append(" (numel=").append(f.numel).append(", bytes=").append(f.byteSize).append(')');
        if (!f.extras.isEmpty()) {
            sb.append(' ');
            sb.append(f.extras);
        }
        return sb.toString();
    }

    /** Build a schema for a single NDArray. */
    public static NumpySchema forArray(NDArray arr) {
        NumpySchema s = new NumpySchema("ndarray");
        if (arr == null) return s;
        Map<String, String> extras = new LinkedHashMap<>();
        extras.put("contiguous", String.valueOf(arr.isContiguous()));
        extras.put("view", String.valueOf(arr.isView));
        long byteSize = arr.size * arr.dtype.getByteSize();
        s.addField(new Field("array", arr.dtype.name(),
                "(" + joinDims(arr.shape) + ")", arr.size, byteSize, extras));
        return s;
    }

    /** Build a schema for an NpyHeader. */
    public static NumpySchema forHeader(NpyHeader h) {
        NumpySchema s = new NumpySchema("npy_header");
        if (h == null) return s;
        Map<String, String> extras = new LinkedHashMap<>();
        extras.put("fortran_order", String.valueOf(h.fortranOrder));
        s.addField(new Field("array", h.dtype.name(),
                "(" + joinDims(h.shape) + ")", h.numel(),
                h.numel() * h.dtype.getByteSize(), extras));
        return s;
    }

    /** Build a schema for an npz archive's datasets. */
    public static NumpySchema forNpz(Map<String, NDArray> archive) {
        NumpySchema s = new NumpySchema("npz");
        if (archive == null) return s;
        for (Map.Entry<String, NDArray> e : archive.entrySet()) {
            NDArray a = e.getValue();
            if (a == null) continue;
            Map<String, String> extras = new LinkedHashMap<>();
            extras.put("contiguous", String.valueOf(a.isContiguous()));
            s.addField(new Field(e.getKey(), a.dtype.name(),
                    "(" + joinDims(a.shape) + ")", a.size,
                    a.size * a.dtype.getByteSize(), extras));
        }
        return s;
    }

    private static String joinDims(long[] shape) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        if (shape.length == 1) sb.append(',');
        return sb.toString();
    }
}
