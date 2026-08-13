/*
 * PickleShow — formatter for pickled Python objects with NumPy/Pandas-style printing.
 *
 * <p>Behaviour:
 * <ul>
 *   <li>Dict objects print as {@code {key: value, ...}} with indentation</li>
 *   <li>List objects print as {@code [item, ...]} with truncation for large lists</li>
 *   <li>Tuple objects print as {@code (item, ...)}</li>
 *   <li>PythonObject (class instances) show class name and dict</li>
 *   <li>Schema shows the type hierarchy and key structure</li>
 *   <li>Large collections are head/tail truncated with summary line</li>
 * </ul>
 */
package org.bytedeco.pytorch.data.pickle;

import java.util.Map;

public final class PickleShow {

    public static final int DEFAULT_THRESHOLD = 100;
    public static final int DEFAULT_EDGEITEMS = 3;
    public static final int DEFAULT_MAX_DEPTH = 5;
    public static final int DEFAULT_LINEWIDTH = 120;

    private final int threshold;
    private final int edgeItems;
    private final int maxDepth;
    private final int lineWidth;

    public PickleShow() {
        this(DEFAULT_THRESHOLD, DEFAULT_EDGEITEMS, DEFAULT_MAX_DEPTH, DEFAULT_LINEWIDTH);
    }

    public PickleShow(int threshold, int edgeItems, int maxDepth, int lineWidth) {
        this.threshold = Math.max(1, threshold);
        this.edgeItems = Math.max(1, edgeItems);
        this.maxDepth = Math.max(1, maxDepth);
        this.lineWidth = Math.max(20, lineWidth);
    }

    /** Print to stdout using the configured settings. */
    public void show(Object obj) {
        System.out.println(format(obj));
    }

    public String format(Object obj) {
        return format(obj, 0);
    }

    /** Print schema to stdout. */
    public void printSchema(Object obj) {
        System.out.println(schema(obj));
    }

    public String schema(Object obj) {
        return schema(obj, 0);
    }

    // ====================== Format ======================

    private String format(Object obj, int depth) {
        if (obj == null) return "null";
        if (depth > maxDepth) return "...";

        if (obj instanceof Pickle.PythonObject) {
            return formatPythonObject((Pickle.PythonObject) obj, depth);
        }
        if (obj instanceof Map) {
            return formatMap((Map<?, ?>) obj, depth);
        }
        if (obj instanceof java.util.List) {
            return formatList((java.util.List<?>) obj, depth);
        }
        if (obj instanceof Pickle.Tuple) {
            return formatTuple((Pickle.Tuple) obj, depth);
        }
        if (obj instanceof byte[]) {
            return formatBytes((byte[]) obj);
        }
        if (obj instanceof float[]) {
            return formatFloatArray((float[]) obj);
        }
        if (obj instanceof double[]) {
            return formatDoubleArray((double[]) obj);
        }
        if (obj instanceof long[]) {
            return formatLongArray((long[]) obj);
        }
        if (obj instanceof int[]) {
            return formatIntArray((int[]) obj);
        }
        if (obj instanceof Object[]) {
            return formatObjectArray((Object[]) obj, depth);
        }
        if (obj instanceof Number) {
            return formatNumber((Number) obj);
        }
        if (obj instanceof Boolean) {
            return obj.toString();
        }
        if (obj instanceof String) {
            return formatString((String) obj);
        }
        if (obj instanceof java.util.Set) {
            return formatSet((java.util.Set<?>) obj, depth);
        }

        return obj.toString();
    }

    private String formatPythonObject(Pickle.PythonObject obj, int depth) {
        String className = obj.className();
        Map<String, Object> dict = obj.dict();
        if (dict != null && !dict.isEmpty()) {
            return className + " {\n" + indent(1) + formatMap(dict, depth + 1).substring(1);
        }
        return className + "{}";
    }

    private String formatMap(Map<?, ?> map, int depth) {
        if (map.isEmpty()) return "{}";

        StringBuilder sb = new StringBuilder("{");
        int size = map.size();
        int count = 0;

        if (size > threshold) {
            // Truncate large maps
            int shown = 0;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (shown < edgeItems) {
                    sb.append("\n").append(indent(depth + 1));
                    sb.append(formatKey(e.getKey())).append(": ");
                    sb.append(format(e.getValue(), depth + 1));
                    sb.append(",");
                    shown++;
                }
                count++;
            }
            sb.append("\n").append(indent(depth + 1));
            sb.append("... (").append(size).append(" items)");
        } else {
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (count > 0) sb.append(",");
                sb.append("\n").append(indent(depth + 1));
                sb.append(formatKey(e.getKey())).append(": ");
                sb.append(format(e.getValue(), depth + 1));
                count++;
            }
        }
        sb.append("\n").append(indent(depth)).append("}");
        return sb.toString();
    }

    private String formatList(java.util.List<?> list, int depth) {
        if (list.isEmpty()) return "[]";

        StringBuilder sb = new StringBuilder("[");
        int size = list.size();

        if (size > threshold) {
            // Truncate large lists
            for (int i = 0; i < edgeItems; i++) {
                sb.append("\n").append(indent(depth + 1));
                sb.append(format(list.get(i), depth + 1));
                sb.append(",");
            }
            sb.append("\n").append(indent(depth + 1));
            sb.append("... (").append(size).append(" items)");
            for (int i = size - edgeItems; i < size; i++) {
                sb.append("\n").append(indent(depth + 1));
                sb.append(format(list.get(i), depth + 1));
                sb.append(",");
            }
        } else {
            for (int i = 0; i < size; i++) {
                if (i > 0) sb.append(",");
                sb.append("\n").append(indent(depth + 1));
                sb.append(format(list.get(i), depth + 1));
            }
        }
        sb.append("\n").append(indent(depth)).append("]");
        return sb.toString();
    }

    private String formatTuple(Pickle.Tuple tuple, int depth) {
        if (tuple.size() == 0) return "()";
        if (tuple.size() == 1) return "(" + format(tuple.get(0), depth) + ",)";

        StringBuilder sb = new StringBuilder("(");
        int size = tuple.size();
        int count = 0;
        for (Object item : tuple) {
            if (count > 0) sb.append(", ");
            sb.append(format(item, depth));
            count++;
        }
        if (tuple.size() == 1) sb.append(",");
        sb.append(")");
        return sb.toString();
    }

    private String formatSet(java.util.Set<?> set, int depth) {
        if (set.isEmpty()) return "set()";

        StringBuilder sb = new StringBuilder("{");
        int size = set.size();
        int count = 0;
        int shown = 0;

        for (Object item : set) {
            if (shown >= edgeItems && size > threshold) {
                sb.append("\n").append(indent(depth + 1));
                sb.append("... (").append(size).append(" items)");
                break;
            }
            if (count > 0) sb.append(", ");
            sb.append(format(item, depth));
            count++;
            shown++;
        }
        sb.append("}");
        return sb.toString();
    }

    private String formatBytes(byte[] bytes) {
        if (bytes.length <= 16) {
            StringBuilder sb = new StringBuilder("b'");
            for (byte b : bytes) {
                if (b >= 0x20 && b < 0x7f) sb.append((char) b);
                else sb.append(String.format("\\x%02x", b & 0xff));
            }
            sb.append("'");
            return sb.toString();
        }
        return "b'...(" + bytes.length + " bytes)'";
    }

    private String formatFloatArray(float[] arr) {
        if (arr.length <= 8) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatDouble(arr[i]));
            }
            sb.append("]");
            return sb.toString();
        }
        return "array([...], dtype=float32)  # " + arr.length + " items";
    }

    private String formatDoubleArray(double[] arr) {
        if (arr.length <= 8) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(formatDouble(arr[i]));
            }
            sb.append("]");
            return sb.toString();
        }
        return "array([...], dtype=float64)  # " + arr.length + " items";
    }

    private String formatLongArray(long[] arr) {
        if (arr.length <= 8) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr[i]);
            }
            sb.append("]");
            return sb.toString();
        }
        return "array([...], dtype=int64)  # " + arr.length + " items";
    }

    private String formatIntArray(int[] arr) {
        if (arr.length <= 8) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr[i]);
            }
            sb.append("]");
            return sb.toString();
        }
        return "array([...], dtype=int32)  # " + arr.length + " items";
    }

    private String formatObjectArray(Object[] arr, int depth) {
        if (arr.length == 0) return "[]";
        if (arr.length == 1) return "[" + format(arr[0], depth) + "]";

        StringBuilder sb = new StringBuilder("[");
        if (arr.length > threshold) {
            for (int i = 0; i < edgeItems; i++) {
                sb.append("\n").append(indent(depth + 1));
                sb.append(format(arr[i], depth + 1));
                sb.append(",");
            }
            sb.append("\n").append(indent(depth + 1));
            sb.append("... (").append(arr.length).append(" items)");
        } else {
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(",");
                sb.append("\n").append(indent(depth + 1));
                sb.append(format(arr[i], depth + 1));
            }
        }
        sb.append("\n").append(indent(depth)).append("]");
        return sb.toString();
    }

    private String formatNumber(Number n) {
        if (n instanceof Double || n instanceof Float) {
            double v = n.doubleValue();
            return formatDouble(v);
        }
        return n.toString();
    }

    private String formatString(String s) {
        if (s.length() > 200) {
            return "'" + s.substring(0, 100) + "...'";
        }
        return "'" + s + "'";
    }

    private String formatKey(Object key) {
        if (key instanceof String) return (String) key;
        return format(key, 0);
    }

    // ====================== Schema ======================

//    private String schema(Object obj) {
//        return schema(obj, 0);
//    }

    private String schema(Object obj, int depth) {
        if (obj == null) return "null";
        if (depth > maxDepth) return "...";

        if (obj instanceof Pickle.PythonObject) {
            return schemaPythonObject((Pickle.PythonObject) obj, depth);
        }
        if (obj instanceof Map) {
            return schemaMap((Map<?, ?>) obj, depth);
        }
        if (obj instanceof java.util.List) {
            return schemaList((java.util.List<?>) obj, depth);
        }
        if (obj instanceof Pickle.Tuple) {
            return schemaTuple((Pickle.Tuple) obj, depth);
        }
        if (obj instanceof java.util.Set) {
            return schemaSet((java.util.Set<?>) obj, depth);
        }
        if (obj instanceof byte[]) {
            return "bytes(" + ((byte[]) obj).length + ")";
        }
        if (obj instanceof float[]) {
            return "float32[" + ((float[]) obj).length + "]";
        }
        if (obj instanceof double[]) {
            return "float64[" + ((double[]) obj).length + "]";
        }
        if (obj instanceof long[]) {
            return "int64[" + ((long[]) obj).length + "]";
        }
        if (obj instanceof int[]) {
            return "int32[" + ((int[]) obj).length + "]";
        }
        if (obj instanceof Object[]) {
            return "object[" + ((Object[]) obj).length + "]";
        }

        return typeName(obj.getClass());
    }

    private String schemaPythonObject(Pickle.PythonObject obj, int depth) {
        String className = obj.className();
        Map<String, Object> dict = obj.dict();
        if (dict != null && !dict.isEmpty()) {
            return className + " {\n" + indent(depth + 1) + schemaMap(dict, depth + 1).substring(1);
        }
        return className + "{}";
    }

    private String schemaMap(Map<?, ?> map, int depth) {
        if (map.isEmpty()) return "{}";

        StringBuilder sb = new StringBuilder();
        int count = 0;
        int shown = 0;

        for (Map.Entry<?, ?> e : map.entrySet()) {
            if (count > 0) sb.append(",");
            sb.append("\n").append(indent(depth + 1));
            sb.append(formatKey(e.getKey())).append(": ");
            sb.append(schema(e.getValue(), depth + 1));
            count++;
            shown++;
        }
        sb.append("\n").append(indent(depth)).append("}");
        return sb.toString();
    }

    private String schemaList(java.util.List<?> list, int depth) {
        if (list.isEmpty()) return "List[?]";
        Object first = list.get(0);
        String elemType = schema(first, depth + 1);
        return "List[" + elemType + "](" + list.size() + ")";
    }

    private String schemaTuple(Pickle.Tuple tuple, int depth) {
        if (tuple.size() == 0) return "Tuple[](0)";
        StringBuilder sb = new StringBuilder("Tuple[");
        for (int i = 0; i < tuple.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(schema(tuple.get(i), depth + 1));
        }
        sb.append("](").append(tuple.size()).append(")");
        return sb.toString();
    }

    private String schemaSet(java.util.Set<?> set, int depth) {
        if (set.isEmpty()) return "Set[?]";
        Object first = set.iterator().next();
        String elemType = schema(first, depth + 1);
        return "Set[" + elemType + "](" + set.size() + ")";
    }

    private static String typeName(Class<?> c) {
        if (c == Boolean.class) return "bool";
        if (c == Byte.class) return "int8";
        if (c == Short.class) return "int16";
        if (c == Integer.class) return "int32";
        if (c == Long.class) return "int64";
        if (c == Float.class) return "float32";
        if (c == Double.class) return "float64";
        if (c == String.class) return "str";
        if (c == byte[].class) return "bytes";
        return c.getSimpleName();
    }

    // ====================== Utils ======================

    private static String indent(int n) {
        return "  ".repeat(Math.max(0, n));
    }

    private static String formatDouble(double v) {
        if (Double.isNaN(v)) return "nan";
        if (Double.isInfinite(v)) return v > 0 ? "inf" : "-inf";
        if (v == Math.floor(v) && Math.abs(v) < 1e16) {
            return Long.toString((long) v) + ".";
        }
        return String.format("%.6f", v).replaceAll("0+$", "").replaceAll("\\.$", ".0");
    }
}
