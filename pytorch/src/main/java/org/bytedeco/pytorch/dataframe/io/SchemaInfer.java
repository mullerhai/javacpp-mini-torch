package org.bytedeco.pytorch.dataframe.io;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.*;

import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Schema;
import org.bytedeco.pytorch.data.parquet.LocalParquetReader;

/**
 * Accurate multi-format schema inference for DataFrame loaders.
 *
 * <p>Combines extension detection ({@link FormatDetect}) with magic-byte
 * sniffing so misnamed files (e.g. {@code data.bin} that is actually parquet)
 * still load correctly. Nested LIST/MAP/STRUCT fields are preserved as
 * {@link Column.DType#LIST}/{@link Column.DType#MAP}/{@link Column.DType#STRUCT}.
 *
 * <pre>
 *   Schema s = SchemaInfer.infer("/path/to/valid.parquet");
 *   s.print(); // or DataFrame.read(path) which uses FormatDetect + this fallback
 * </pre>
 */
public final class SchemaInfer {
    private SchemaInfer() {}

    /** Infer schema without materializing all rows when possible. */
    public static Schema infer(String path) throws Exception {
        FormatDetect.Format fmt = FormatDetect.detect(path);
        if (fmt == FormatDetect.Format.UNKNOWN) {
            fmt = sniff(path);
        }
        return infer(path, fmt);
    }

    public static Schema infer(String path, FormatDetect.Format fmt) throws Exception {
        switch (fmt) {
            case PARQUET:
                return fromParquet(path);
            case CSV:
            case TSV:
            case JSON:
            case JSONL:
            case ARROW:
            case FEATHER:
            case PICKLE:
            case EXCEL:
            case HDF5:
            case AVRO:
            case ORC:
            case NPZ:
            case NPY:
            case SAFETENSORS:
            case GGUF:
                // Fall back: load (or head) via FormatDetect and take schema
                DataFrame df = FormatDetect.read(path);
                try {
                    return Schema.fromDataFrame(df);
                } finally {
                    try { df.close(); } catch (Exception ignored) {}
                }
            default:
                // last-chance sniff
                FormatDetect.Format sniffed = sniff(path);
                if (sniffed != FormatDetect.Format.UNKNOWN && sniffed != fmt) {
                    return infer(path, sniffed);
                }
                throw new IllegalArgumentException("Cannot infer schema for: " + path);
        }
    }

    /** Parquet schema only (no row materialization beyond footer). */
    public static Schema fromParquet(String path) throws IOException {
        try (LocalParquetReader r = LocalParquetReader.open(path)) {
            return fromParquetMessageType(r.getSchema());
        }
    }

    public static Schema fromParquetMessageType(MessageType mt) {
        Schema s = new Schema();
        for (Type field : mt.getFields()) {
            s.add(field.getName(), parquetTypeToDType(field));
        }
        return s;
    }

    /** Mirror of DataFrame.parquetTypeToDType for public schema peeking. */
    public static Column.DType parquetTypeToDType(Type ft) {
        if (ft.isPrimitive()) {
            org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName ptn =
                ft.asPrimitiveType().getPrimitiveTypeName();
            return switch (ptn) {
                case INT32 -> Column.DType.INT32;
                case INT64 -> Column.DType.INT64;
                case FLOAT -> Column.DType.FLOAT32;
                case DOUBLE -> Column.DType.FLOAT64;
                case BOOLEAN -> Column.DType.BOOLEAN;
                case BINARY, FIXED_LEN_BYTE_ARRAY -> {
                    var lta = ft.getLogicalTypeAnnotation();
                    if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.StringLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.EnumLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.JsonLogicalTypeAnnotation) {
                        yield Column.DType.STRING;
                    }
                    yield Column.DType.BINARY;
                }
                default -> Column.DType.STRING;
            };
        }
        var lta = ft.getLogicalTypeAnnotation();
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
            || (!ft.isPrimitive()
                && ft.getRepetition() == Type.Repetition.REPEATED)) {
            Column.DType elem = listElementDType(ft);
            if (elem == Column.DType.FLOAT32 || elem == Column.DType.FLOAT64) {
                return Column.DType.VECTOR;
            }
            return Column.DType.LIST;
        }
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapLogicalTypeAnnotation
            || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapKeyValueTypeAnnotation) {
            return Column.DType.MAP;
        }
        return Column.DType.STRUCT;
    }

    private static Column.DType listElementDType(Type ft) {
        try {
            org.apache.parquet.schema.GroupType gt = ft.asGroupType();
            if (gt.getFieldCount() == 0) return Column.DType.STRING;
            Type mid = gt.getType(0);
            if (!mid.isPrimitive()) {
                org.apache.parquet.schema.GroupType midG = mid.asGroupType();
                if (midG.getFieldCount() > 0) {
                    Type elem = midG.getType(0);
                    if (elem.isPrimitive()) return parquetTypeToDType(elem);
                    return Column.DType.LIST;
                }
            } else {
                return parquetTypeToDType(mid);
            }
        } catch (Exception ignored) { /* fall through */ }
        return Column.DType.STRING;
    }

    /**
     * Magic-byte sniff when extension is missing/wrong.
     * Recognises parquet (PAR1), npy, npz (PK), arrow IPC, gzip-ish jsonl, etc.
     */
    public static FormatDetect.Format sniff(String path) {
        Path p = Path.of(path);
        if (!Files.isRegularFile(p)) return FormatDetect.Format.UNKNOWN;
        try (InputStream in = new BufferedInputStream(Files.newInputStream(p))) {
            byte[] head = in.readNBytes(16);
            if (head.length < 4) return FormatDetect.Format.UNKNOWN;
            // Parquet: "PAR1"
            if (head[0] == 'P' && head[1] == 'A' && head[2] == 'R' && head[3] == '1')
                return FormatDetect.Format.PARQUET;
            // NPY: \x93NUMPY
            if (head.length >= 6 && (head[0] & 0xFF) == 0x93
                && head[1] == 'N' && head[2] == 'U' && head[3] == 'M'
                && head[4] == 'P' && head[5] == 'Y')
                return FormatDetect.Format.NPY;
            // ZIP-based: npz, xlsx, orc sometimes
            if (head[0] == 'P' && head[1] == 'K') {
                String lower = path.toLowerCase(Locale.ROOT);
                if (lower.endsWith(".npz")) return FormatDetect.Format.NPZ;
                if (lower.endsWith(".xlsx") || lower.endsWith(".xlsm")) return FormatDetect.Format.EXCEL;
                // default zip → try npz
                return FormatDetect.Format.NPZ;
            }
            // Arrow IPC magic "ARROW1" or feather V1 "FEA1"
            String ascii = new String(head, 0, Math.min(6, head.length), StandardCharsets.US_ASCII);
            if (ascii.startsWith("ARROW1")) return FormatDetect.Format.ARROW;
            if (ascii.startsWith("FEA1")) return FormatDetect.Format.FEATHER;
            // Avro Object Container File: Obj\x01
            if (head[0] == 'O' && head[1] == 'b' && head[2] == 'j' && head[3] == 0x01)
                return FormatDetect.Format.AVRO;
            // ORC: "ORC"
            if (head[0] == 'O' && head[1] == 'R' && head[2] == 'C')
                return FormatDetect.Format.ORC;
            // HDF5: \x89HDF
            if ((head[0] & 0xFF) == 0x89 && head[1] == 'H' && head[2] == 'D' && head[3] == 'F')
                return FormatDetect.Format.HDF5;
            // JSON start
            int i = 0;
            while (i < head.length && Character.isWhitespace((char) head[i])) i++;
            if (i < head.length && (head[i] == '{' || head[i] == '['))
                return FormatDetect.Format.JSON;
            // GGUF
            if (ascii.startsWith("GGUF")) return FormatDetect.Format.GGUF;
            // Safetensors is JSON header length LE u64 then JSON — hard to sniff; leave UNKNOWN
            return FormatDetect.Format.UNKNOWN;
        } catch (IOException e) {
            return FormatDetect.Format.UNKNOWN;
        }
    }

    /** Human-readable schema dump (Spark-style). */
    public static void print(Schema schema) {
        System.out.println("root");
        List<String> names = schema.fieldNames();
        List<Column.DType> types = schema.fieldTypes();
        for (int i = 0; i < names.size(); i++) {
            System.out.printf(" |-- %s: %s%n", names.get(i), types.get(i));
        }
    }

    /** Describe nested parquet schema with logical types for debugging. */
    public static Map<String, String> describeParquet(String path) throws IOException {
        Map<String, String> out = new LinkedHashMap<>();
        try (LocalParquetReader r = LocalParquetReader.open(path)) {
            MessageType mt = r.getSchema();
            for (Type f : mt.getFields()) {
                out.put(f.getName(), f.toString().replace('\n', ' '));
            }
        }
        return out;
    }

    /** List field names only. */
    public static List<String> fieldNames(String path) throws Exception {
        return new ArrayList<>(infer(path).fieldNames());
    }

    // ====================== Format-specific Schema as DataFrame ======================

    /**
     * Get schema as a preview DataFrame (with metadata, without loading full data).
     */
    public static DataFrame inferAsDataFrame(String path, CsvOptions opts) throws Exception {
        Schema s = SchemaInfer.infer(path);
        return schemaToPreviewDataFrame(s, path, "CSV");
    }

    public static DataFrame inferAsDataFrame(String path, JsonOptions opts) throws Exception {
        Schema s = SchemaInfer.infer(path);
        return schemaToPreviewDataFrame(s, path, "JSON");
    }

    public static DataFrame inferAsDataFrameParquet(String path) throws Exception {
        Schema s = fromParquet(path);
        return schemaToPreviewDataFrame(s, path, "Parquet");
    }

    public static DataFrame inferAsDataFrameNpy(String path) throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("#", Column.DType.INT32);
        df.addColumn("name", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("rows", Column.DType.INT64);
        df.addColumn("cols", Column.DType.INT64);
        df.addColumn("size", Column.DType.INT64);

        org.bytedeco.pytorch.data.numpy.NDArray arr = org.bytedeco.pytorch.data.numpy.NP.load(path);
        int ri = df.addEmptyRow();
        df.set(ri, "#", 0);
        df.set(ri, "name", inferFileName(path));
        df.set(ri, "shape", java.util.Arrays.toString(arr.shape));
        df.set(ri, "dtype", arr.dtype.toString());
        df.set(ri, "rows", arr.shape.length >= 1 ? arr.shape[0] : 1L);
        df.set(ri, "cols", arr.shape.length >= 2 ? arr.shape[1] : 1L);
        df.set(ri, "size", arr.size);

        return df;
    }

    public static DataFrame inferAsDataFrameNpz(String path) throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("#", Column.DType.INT32);
        df.addColumn("array_name", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("rows", Column.DType.INT64);
        df.addColumn("cols", Column.DType.INT64);
        df.addColumn("size", Column.DType.INT64);

        Map<String, org.bytedeco.pytorch.data.numpy.NDArray> arrays = 
            org.bytedeco.pytorch.data.numpy.NP.loadz(path);
        int idx = 0;
        for (Map.Entry<String, org.bytedeco.pytorch.data.numpy.NDArray> e : arrays.entrySet()) {
            org.bytedeco.pytorch.data.numpy.NDArray arr = e.getValue();
            int ri = df.addEmptyRow();
            df.set(ri, "#", idx);
            df.set(ri, "array_name", e.getKey());
            df.set(ri, "shape", java.util.Arrays.toString(arr.shape));
            df.set(ri, "dtype", arr.dtype.toString());
            df.set(ri, "rows", arr.shape.length >= 1 ? arr.shape[0] : 1L);
            df.set(ri, "cols", arr.shape.length >= 2 ? arr.shape[1] : 1L);
            df.set(ri, "size", arr.size);
            idx++;
        }

        return df;
    }

    public static DataFrame inferAsDataFrameHdf5(String path) throws Exception {
        DataFrame df = DataFrame.create();
        df.addColumn("#", Column.DType.INT32);
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("type", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("size_bytes", Column.DType.INT64);

        Map<String, Object> meta = Hdf5Reader.metadata(path);
        int idx = 0;

        // Add root level info
        int ri = df.addEmptyRow();
        df.set(ri, "#", idx++);
        df.set(ri, "path", "/");
        df.set(ri, "type", "group");
        df.set(ri, "dtype", "-");
        df.set(ri, "shape", "-");
        df.set(ri, "size_bytes", 0L);

        // Add datasets
        Object dsObj = meta.get("datasets");
        if (dsObj instanceof List) {
            for (Object o : (List<?>) dsObj) {
                if (o instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> ds = (Map<String, Object>) o;
                    ri = df.addEmptyRow();
                    df.set(ri, "#", idx++);
                    df.set(ri, "path", ds.get("name"));
                    df.set(ri, "type", "dataset");
                    df.set(ri, "dtype", ds.get("dtype"));
                    df.set(ri, "shape", ds.get("shape"));
                    df.set(ri, "size_bytes", ((Number) ds.get("size_bytes")).longValue());
                }
            }
        }

        return df;
    }

    public static DataFrame inferAsDataFramePickle(String path) throws Exception {
        // Load minimal data to get schema
        Object obj = org.bytedeco.pytorch.data.pickle.Pickle.load(new java.io.File(path));
        return objectToSchemaDataFrame(obj, path);
    }

    public static DataFrame inferAsDataFrameExcel(String path) throws Exception {
        // Fallback: load via Excel reader
        return DataFrame.readExcel(path, org.bytedeco.pytorch.dataframe.excel.ExcelOptions.builder()
            .maxRows(0).build());
    }

    public static DataFrame inferAsDataFrameArrow(String path) throws Exception {
        return DataFrame.readArrow(path);
    }

    // ====================== Helpers ======================

    private static DataFrame schemaToPreviewDataFrame(Schema s, String path, String format) {
        DataFrame df = DataFrame.create();
        df.addColumn("#", Column.DType.INT32);
        df.addColumn("column_name", Column.DType.STRING);
        df.addColumn("data_type", Column.DType.STRING);
        df.addColumn("nullable", Column.DType.BOOLEAN);

        List<String> names = s.fieldNames();
        List<Column.DType> types = s.fieldTypes();
        for (int i = 0; i < names.size(); i++) {
            int ri = df.addEmptyRow();
            df.set(ri, "#", i);
            df.set(ri, "column_name", names.get(i));
            df.set(ri, "data_type", types.get(i).name());
            df.set(ri, "nullable", true);
        }
        return df;
    }

    private static DataFrame objectToSchemaDataFrame(Object obj, String path) {
        DataFrame df = DataFrame.create();
        df.addColumn("key", Column.DType.STRING);
        df.addColumn("value_type", Column.DType.STRING);
        df.addColumn("sample_value", Column.DType.STRING);

        if (obj instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) obj;
            int idx = 0;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                int ri = df.addEmptyRow();
                df.set(ri, "key", String.valueOf(e.getKey()));
                df.set(ri, "value_type", e.getValue() != null ? e.getValue().getClass().getSimpleName() : "null");
                df.set(ri, "sample_value", truncate(String.valueOf(e.getValue()), 50));
                if (idx++ >= 100) break; // Limit to 100 entries
            }
        } else if (obj instanceof List) {
            List<?> list = (List<?>) obj;
            if (!list.isEmpty()) {
                Object first = list.get(0);
                int ri = df.addEmptyRow();
                df.set(ri, "key", "list_element");
                df.set(ri, "value_type", first != null ? first.getClass().getSimpleName() : "null");
                df.set(ri, "sample_value", first != null ? truncate(String.valueOf(first), 50) : "null");
            }
        }

        return df;
    }

    private static String inferFileName(String path) {
        if (path == null) return "array";
        int slash = path.lastIndexOf('/');
        String name = slash >= 0 ? path.substring(slash + 1) : path;
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, max - 3) + "...";
    }
}
