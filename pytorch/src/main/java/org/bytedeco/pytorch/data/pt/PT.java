package org.bytedeco.pytorch.data.pt;

import java.io.*;
import java.nio.*;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;
import java.util.regex.*;

/**
 * Enterprise-grade PyTorch .pt file reader with full pickle support.
 * 
 * <p>PyTorch .pt files are ZIP archives containing:</p>
 * <ul>
 *   <li>{name}/data.pkl: Protocol 2 pickle with tensor rebuild instructions</li>
 *   <li>{name}/data/{idx}: Binary tensor storage files</li>
 *   <li>{name}/byteorder: Endianness (usually "little")</li>
 *   <li>{name}/version: Format version (2 or 3)</li>
 * </ul>
 * 
 * <p>This module provides:</p>
 * <ul>
 *   <li>Full pickle protocol 2 parsing with persistent ID resolution</li>
 *   <li>PyTorch tensor reconstruction via torch._utils._rebuild_tensor_v2</li>
 *   <li>Conversion to/from SafeTensors format</li>
 *   <li>DataFrame integration</li>
 * </ul>
 */
public class PT {

    // PyTorch's pickle stream uses BINPERSID (0x51) with a 5-tuple
    // ('storage', dtype_class, '<file>', '<device>', size) to reference
    // external storage data. The classic 'MARK + persistent ID' description
    // for PID_TENSOR 0x94 was a misreading: 0x94 is the MEMOIZE opcode.
    // Actual constant kept here only as a marker for documentation.
    private static final byte PID_TENSOR = (byte) 0x51;
    
    private PT() {}

    // ---- Public API ----

    /**
     * Load a .pt file and return all tensors as a map.
     * Uses torch._utils._rebuild_tensor_v2 to reconstruct tensors.
     */
    public static Map<String, TensorData> load(File file) throws IOException {
        return load(file, true);
    }

    /**
     * Load tensors with optional fast mode (direct binary reading).
     * 
     * @param file the .pt file
     * @param parsePickle if true, parse pickle metadata for tensor names/shapes;
     *                     if false, use fast binary inference mode
     */
    public static Map<String, TensorData> load(File file, boolean parsePickle) throws IOException {
        if (!file.exists()) {
            throw new FileNotFoundException("PT file not found: " + file);
        }
        
        Map<String, byte[]> zipContents = readZipContents(file);
        String root = findRoot(zipContents.keySet());
        
        String byteorder = readMetadata(zipContents, root + "byteorder", "little");
        String version = readMetadata(zipContents, root + "version", "3");
        
        ByteOrder order = byteorder.equals("little") ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN;
        
        if (parsePickle) {
            return loadWithPickle(zipContents, root, order);
        } else {
            return loadFast(zipContents, root, order);
        }
    }

    /**
     * Load all tensors as float arrays.
     */
    public static Map<String, float[]> loadFloatTensors(File file) throws IOException {
        Map<String, TensorData> tensors = load(file);
        Map<String, float[]> result = new LinkedHashMap<>();
        for (Map.Entry<String, TensorData> e : tensors.entrySet()) {
            if (e.getValue().dtype.isFloat()) {
                result.put(e.getKey(), e.getValue().asFloatArray());
            }
        }
        return result;
    }

    /**
     * Load all tensors as long arrays.
     */
    public static Map<String, long[]> loadLongTensors(File file) throws IOException {
        Map<String, TensorData> tensors = load(file);
        Map<String, long[]> result = new LinkedHashMap<>();
        for (Map.Entry<String, TensorData> e : tensors.entrySet()) {
            if (e.getValue().dtype == DataType.INT64) {
                result.put(e.getKey(), e.getValue().asLongArray());
            }
        }
        return result;
    }

    /**
     * Convert .pt tensors to SafeTensors format.
     */
    public static void toSafeTensors(File ptFile, File ssFile) throws IOException {
        Map<String, TensorData> tensors = load(ptFile);
        org.bytedeco.pytorch.data.safetensors.SafeTensors.save(
            tensorsToTorchTensors(tensors), ssFile
        );
    }

    /**
     * Create a .pt file from a SafeTensors file.
     */
    public static void fromSafeTensors(File ssFile, File ptFile) throws IOException {
        Map<String, org.bytedeco.pytorch.Tensor> tensors = 
            org.bytedeco.pytorch.data.safetensors.SafeTensors.loadAsTensors(ssFile, false);
        
        saveTorchTensors(tensors, ptFile);
    }

    /**
     * Create a .pt file from tensor map.
     */
    public static void save(Map<String, org.bytedeco.pytorch.Tensor> tensors, File file) throws IOException {
        saveTorchTensors(tensors, file);
    }

    // ---- Pickle parsing with persistent IDs ----

    private static Map<String, TensorData> loadWithPickle(
            Map<String, byte[]> zipContents, String root, ByteOrder order) throws IOException {
        
        byte[] pklData = zipContents.get(root + "data.pkl");
        if (pklData == null) {
            throw new IOException("Missing data.pkl in PT file");
        }

        // Parse pickle with persistent ID resolution
        Map<String, StorageInfo> storages = new LinkedHashMap<>();
        Map<String, TensorBuildInfo> tensors = new LinkedHashMap<>();
        
        // First pass: extract storage data from ZIP and sort by numeric
        // filename so that data/2 < data/10.
        List<String> dataFiles = new ArrayList<>();
        for (String name : zipContents.keySet()) {
            if (name.startsWith(root + "data/") && !name.endsWith("/")) {
                dataFiles.add(name);
            }
        }
        dataFiles.sort((a, b) -> {
            String sa = a.substring(a.lastIndexOf('/') + 1);
            String sb = b.substring(b.lastIndexOf('/') + 1);
            try {
                int na = Integer.parseInt(sa);
                int nb = Integer.parseInt(sb);
                return Integer.compare(na, nb);
            } catch (NumberFormatException ex) {
                return sa.compareTo(sb);
            }
        });

        Map<Integer, byte[]> storageData = new HashMap<>();
        for (int i = 0; i < dataFiles.size(); i++) {
            storageData.put(i, zipContents.get(dataFiles.get(i)));
        }

        // Parse pickle protocol 2
        PickleParser parser = new PickleParser(pklData);
        parser.parse(storages, tensors, storageData, order);

        // Build result tensors
        Map<String, TensorData> result = new LinkedHashMap<>();
        for (Map.Entry<String, TensorBuildInfo> e : tensors.entrySet()) {
            TensorData td = buildTensor(e.getValue(), storages, order);
            if (td != null) {
                result.put(e.getKey(), td);
            }
        }
        
        return result;
    }

    private static TensorData buildTensor(TensorBuildInfo info,
            Map<String, StorageInfo> storages, ByteOrder order) {

        if (info.shape == null || info.shape.length == 0) return null;

        StorageInfo storage = findStorage(storages, info.storageKey);
        if (storage == null) return null;

        byte[] data = storage.data;
        if (data == null || data.length == 0) return null;

        DataType dtype = parseDataType(storage.dtype);
        int elemSize = dtype.sizeBytes();
        if (elemSize <= 0) return null;

        // Compute expected number of elements from shape.
        long expectedElems = 1;
        for (long dim : info.shape) expectedElems *= dim;

        long storageOffset = Math.max(0L, info.storageOffset);
        long startByte = storageOffset * elemSize;
        long needBytes = expectedElems * elemSize;
        if (startByte > data.length) return null;
        if (startByte + needBytes > data.length) {
            // Tolerate truncated trailing data.
            needBytes = data.length - startByte;
            expectedElems = needBytes / elemSize;
        }
        if (expectedElems <= 0) return null;

        ByteBuffer buf = ByteBuffer.wrap(data, (int) startByte, (int) needBytes)
                .order(order);

        Object array;
        if (dtype.isFloat()) {
            float[] floats = new float[(int) expectedElems];
            buf.asFloatBuffer().get(floats);
            array = floats;
        } else {
            long[] longs = new long[(int) expectedElems];
            buf.asLongBuffer().get(longs);
            array = longs;
        }

        return new TensorData(info.name, dtype, info.shape, array);
    }

    private static StorageInfo findStorage(Map<String, StorageInfo> storages, String key) {
        if (key == null) return null;
        StorageInfo s = storages.get(key);
        if (s != null) return s;
        // Fall back to matching by filename only, regardless of dtype prefix.
        for (Map.Entry<String, StorageInfo> e : storages.entrySet()) {
            if (e.getKey().endsWith(":" + key)) return e.getValue();
            if (e.getKey().equals(key)) return e.getValue();
        }
        return null;
    }

    private static DataType parseDataType(String dtype) {
        if (dtype == null) return DataType.FLOAT32;
        dtype = dtype.toLowerCase();
        if (dtype.contains("float")) {
            if (dtype.contains("64") || dtype.contains("double")) return DataType.FLOAT64;
            if (dtype.contains("16") || dtype.contains("half")) return DataType.FLOAT16;
            if (dtype.contains("8")) return DataType.FLOAT8;
        }
        if (dtype.contains("int")) {
            if (dtype.contains("64") || dtype.contains("long")) return DataType.INT64;
            if (dtype.contains("32") || dtype.contains("int")) return DataType.INT32;
            if (dtype.contains("16") || dtype.contains("short")) return DataType.INT16;
            if (dtype.contains("8")) return DataType.INT8;
        }
        if (dtype.contains("byte") || dtype.contains("uint8")) return DataType.UINT8;
        if (dtype.contains("bool")) return DataType.BOOL;
        return DataType.FLOAT32;
    }

    // ---- Fast binary loading ----

    private static Map<String, TensorData> loadFast(
            Map<String, byte[]> zipContents, String root, ByteOrder order) {
        
        List<String> dataFiles = new ArrayList<>();
        for (String name : zipContents.keySet()) {
            if (name.startsWith(root + "data/") && !name.endsWith("/")) {
                dataFiles.add(name);
            }
        }
        Collections.sort(dataFiles);
        
        Map<String, TensorData> result = new LinkedHashMap<>();
        int idx = 0;
        
        for (String fileName : dataFiles) {
            byte[] data = zipContents.get(fileName);
            String tensorName = "tensor_" + idx;
            
            DataType dtype = inferDataType(data, order);
            int elemSize = dtype.sizeBytes();
            long count = data.length / elemSize;
            long[] shape = inferShape(count);
            
            Object array;
            if (dtype.isFloat()) {
                float[] floats = new float[(int) count];
                ByteBuffer.wrap(data).order(order).asFloatBuffer().get(floats);
                array = floats;
            } else {
                long[] longs = new long[(int) count];
                ByteBuffer.wrap(data).order(order).asLongBuffer().get(longs);
                array = longs;
            }
            
            result.put(tensorName, new TensorData(tensorName, dtype, shape, array));
            idx++;
        }
        
        return result;
    }

    private static DataType inferDataType(byte[] data, ByteOrder order) {
        // Sample data to detect float vs int
        int samples = Math.min(20, data.length / 4);
        int floatCount = 0;
        
        for (int i = 0; i < samples; i++) {
            int bits = ByteBuffer.wrap(data, i * 4, 4).order(order).getInt();
            int exponent = (bits >> 23) & 0xff;
            int mantissa = bits & 0x7fffff;
            
            // Float has non-zero mantissa with reasonable exponent
            if (exponent > 0 && exponent < 255 && mantissa != 0) {
                floatCount++;
            }
        }
        
        // If >50% look like floats, treat as float32
        if (floatCount > samples / 2) {
            return DataType.FLOAT32;
        }
        
        // Check size pattern
        if (data.length % 8 == 0) return DataType.INT64;
        if (data.length % 4 == 0) return DataType.INT32;
        return DataType.INT64;
    }

    private static long[] inferShape(long elementCount) {
        // Try to find a reasonable shape
        // Common shapes: (N, 128), (N, 256), (N, 512), (N, 768), etc.
        long[][] candidates = {
            {elementCount},  // 1D
            {elementCount / 128, 128},  // common embedding dim
            {elementCount / 256, 256},
            {elementCount / 512, 512},
            {elementCount / 768, 768},
            {elementCount / 1024, 1024},
        };
        
        for (long[] shape : candidates) {
            boolean valid = true;
            for (long dim : shape) {
                if (dim <= 0 || dim > elementCount) {
                    valid = false;
                    break;
                }
            }
            if (valid) return shape;
        }
        
        return new long[]{elementCount};
    }

    // ---- SafeTensors conversion ----

    private static Map<String, org.bytedeco.pytorch.Tensor> tensorsToTorchTensors(
            Map<String, TensorData> tensors) throws IOException {
        
        Map<String, org.bytedeco.pytorch.Tensor> result = new LinkedHashMap<>();
        
        for (Map.Entry<String, TensorData> e : tensors.entrySet()) {
            String name = e.getKey();
            TensorData td = e.getValue();
            org.bytedeco.pytorch.Tensor t = createTorchTensor(td);
            if (t != null) {
                result.put(name, t);
            }
        }
        
        return result;
    }

    private static org.bytedeco.pytorch.Tensor createTorchTensor(TensorData td) {
        org.bytedeco.pytorch.Tensor t = null;
        long[] shape = td.shape;
        int nelem = (int) td.elementCount();
        
        if (td.dtype == DataType.FLOAT32 && td.array instanceof float[]) {
            float[] arr = (float[]) td.array;
            t = org.bytedeco.pytorch.global.torch.tensor(arr);
            if (shape.length > 1) t = t.reshape(shape);
        } else if (td.dtype == DataType.INT64 && td.array instanceof long[]) {
            long[] arr = (long[]) td.array;
            t = org.bytedeco.pytorch.global.torch.tensor(arr);
            if (shape.length > 1) t = t.reshape(shape);
        }
        
        return t;
    }

    private static void saveTorchTensors(
            Map<String, org.bytedeco.pytorch.Tensor> tensors, File file) throws IOException {
        
        Map<String, org.bytedeco.pytorch.Tensor> cpuTensors = new LinkedHashMap<>();
        for (Map.Entry<String, org.bytedeco.pytorch.Tensor> e : tensors.entrySet()) {
            if (e.getValue() != null && e.getValue().defined()) {
                cpuTensors.put(e.getKey(), e.getValue().cpu().contiguous());
            }
        }
        
        org.bytedeco.pytorch.data.safetensors.SafeTensors.save(cpuTensors, file);
    }

    // ---- ZIP reading ----

    private static Map<String, byte[]> readZipContents(File file) throws IOException {
        Map<String, byte[]> contents = new LinkedHashMap<>();
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(file))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory()) {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buffer = new byte[8192];
                    int read;
                    while ((read = zis.read(buffer)) != -1) {
                        baos.write(buffer, 0, read);
                    }
                    contents.put(entry.getName(), baos.toByteArray());
                }
                zis.closeEntry();
            }
        }
        return contents;
    }

    private static String findRoot(Set<String> names) {
        for (String name : names) {
            if (name.endsWith("/data.pkl")) {
                return name.substring(0, name.length() - "data.pkl".length());
            }
        }
        return "";
    }

    private static String readMetadata(Map<String, byte[]> contents, String key, String defaultValue) {
        byte[] data = contents.get(key);
        if (data == null) return defaultValue;
        return new String(data, StandardCharsets.US_ASCII).trim();
    }

    // ---- Show / display methods ----

    /**
     * Print tensors to stdout in PyTorch-style format.
     */
    public static void printSchema(File file) throws IOException {
        System.out.println(schema(file));
    }

    /**
     * Get DataFrame-style schema string showing all tensor metadata.
     */
    public static String schema(File file) throws IOException {
        return schema(load(file));
    }

    /**
     * Get DataFrame-style schema string.
     */
    public static String schema(Map<String, TensorData> tensors) {
        return new PTShow().schema(tensors);
    }

    /**
     * Get a string representation of the tensors (like Python print).
     */
    public static String show(File file) throws IOException {
        return show(load(file));
    }

    public static String show(Map<String, TensorData> tensors) {
        return new PTShow().showString(tensors);
    }

    private static String formatShape(long[] shape) {
        if (shape == null || shape.length == 0) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(shape[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    // ---- Inner classes ----

    public enum DataType {
        FLOAT64(8, true), FLOAT32(4, true), FLOAT16(2, true), FLOAT8(1, true),
        INT64(8, false), INT32(4, false), INT16(2, false), INT8(1, false),
        UINT8(1, false), BOOL(1, false);

        private final int size;
        private final boolean isFloat;

        DataType(int size, boolean isFloat) {
            this.size = size;
            this.isFloat = isFloat;
        }

        public int sizeBytes() { return size; }
        public boolean isFloat() { return isFloat; }
    }

    public static class TensorData {
        public final String name;
        public final DataType dtype;
        public final long[] shape;
        public final Object array; // float[] or long[]

        public TensorData(String name, DataType dtype, long[] shape, Object array) {
            this.name = name;
            this.dtype = dtype;
            this.shape = shape;
            this.array = array;
        }

        public long elementCount() {
            long count = 1;
            for (long d : shape) count *= d;
            return count;
        }

        public float[] asFloatArray() {
            if (array instanceof float[]) return (float[]) array;
            if (array instanceof long[]) {
                long[] src = (long[]) array;
                float[] dst = new float[src.length];
                for (int i = 0; i < src.length; i++) dst[i] = src[i];
                return dst;
            }
            return new float[0];
        }

        public long[] asLongArray() {
            if (array instanceof long[]) return (long[]) array;
            if (array instanceof float[]) {
                float[] src = (float[]) array;
                long[] dst = new long[src.length];
                for (int i = 0; i < src.length; i++) dst[i] = (long) src[i];
                return dst;
            }
            return new long[0];
        }

        public String preview() {
            int show = Math.min(5, (int) elementCount());
            if (dtype.isFloat) {
                float[] arr = asFloatArray();
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < show; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(String.format("%.4f", arr[i]));
                }
                if (elementCount() > show) sb.append(", ...");
                return sb.toString();
            } else {
                long[] arr = asLongArray();
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < show; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(arr[i]);
                }
                if (elementCount() > show) sb.append(", ...");
                return sb.toString();
            }
        }
    }

    static class StorageInfo {
        String dtype;
        String device;
        int size;
        byte[] data;
    }

    static class TensorBuildInfo {
        String name;
        String storageKey;
        long[] shape;
        long storageOffset;
        long[] stride;
        boolean requiresGrad;
    }

    // ---- Pickle Parser ----
    //
    // Pure-Java pickle unpickler that extracts PyTorch tensor rebuild metadata
    // from a .pt file's data.pkl without executing Python code.
    //
    // Strategy:
    //   - Run a real pickle dispatch (stack + memo + BINPERSID).
    //   - On GLOBAL/STACK_GLOBAL for torch.*Storage, capture a storage
    //     placeholder and remember its dtype / filename / device / size.
    //   - On REDUCE for torch._utils._rebuild_tensor_v2 / _rebuild_qtensor /
    //     _rebuild_parameter / _rebuild_sparse_tensor etc., capture
    //     (storage, offset, size, stride, requires_grad, ...).
    //   - Top-level DICT keys become tensor names; values become
    //     TensorBuildInfo entries. Nested dicts are flattened with
    //     dot-separated paths. Top-level non-dict objects fall back to
    //     positional names tensor_0, tensor_1, ...

    static class PickleParser {
        // Pickle opcodes we actually need.
        private static final int OP_PROTO       = 0x80;
        private static final int OP_FRAME       = 0x95;
        private static final int OP_MEMOIZE     = 0x94;
        private static final int OP_MARK        = '('; // 0x28
        private static final int OP_STOP        = '.'; // 0x2e
        private static final int OP_POP         = '0'; // 0x30
        private static final int OP_DUP         = '1'; // 0x31
        private static final int OP_EMPTY_TUPLE = ')'; // 0x29
        private static final int OP_EMPTY_LIST  = ']'; // 0x5d
        private static final int OP_EMPTY_DICT  = '}'; // 0x7d
        private static final int OP_TUPLE1      = 0x85;
        private static final int OP_TUPLE2      = 0x86;
        private static final int OP_TUPLE3      = 0x87;
        private static final int OP_TUPLE       = 't'; // 0x74
        private static final int OP_LIST        = 'l'; // 0x6c
        private static final int OP_DICT        = 'd'; // 0x75
        private static final int OP_SETITEM     = 's'; // 0x73
        private static final int OP_SETITEMS    = 0x75; // 'u'
        private static final int OP_APPEND      = 0x61; // 'a'
        private static final int OP_APPENDS     = 0x65; // 'e'
        private static final int OP_ADDITEMS    = 0x90;
        private static final int OP_FROZENSET   = 0x91;
        private static final int OP_GLOBAL      = 'c'; // 0x63
        private static final int OP_STACK_GLOBAL= 0x93;
        private static final int OP_INST        = 'i'; // 0x69
        private static final int OP_OBJ         = 'o'; // 0x6f
        private static final int OP_NEWOBJ      = 0x81;
        private static final int OP_NEWOBJ_EX   = 0x92;
        private static final int OP_REDUCE      = 'R'; // 0x52
        private static final int OP_BUILD       = 'b'; // 0x62
        private static final int OP_PERSID      = 'P'; // 0x50
        private static final int OP_BINPERSID   = 'Q'; // 0x51
        private static final int OP_PUT         = 'p'; // 0x70
        private static final int OP_BINPUT      = 'q'; // 0x71
        private static final int OP_LONG_BINPUT = 0x82;
        private static final int OP_GET         = 'g'; // 0x67
        private static final int OP_BINGET      = 'h'; // 0x68
        private static final int OP_LONG_BINGET = 0x6a;
        private static final int OP_NONE        = 'N'; // 0x4e
        private static final int OP_NEW_TRUE    = 0x88; // protocol 4
        private static final int OP_NEW_FALSE   = 0x89; // protocol 4
        // protocol 0/1/2 has no dedicated TRUE/FALSE opcodes; legacy code uses
        // INT with payload '01\n' / '00\n'. We don't define OP_TRUE / OP_FALSE.
        private static final int OP_INT         = 'I'; // 0x49
        private static final int OP_BININT      = 'J'; // 0x4a
        private static final int OP_BININT1     = 'K'; // 0x4b
        private static final int OP_BININT2     = 'M'; // 0x4d
        private static final int OP_LONG1       = 0x83;
        private static final int OP_LONG4       = 0x8b;
        private static final int OP_LONG        = 'L'; // 0x4c
        private static final int OP_FLOAT       = 'F'; // 0x46
        private static final int OP_BINFLOAT    = 'G'; // 0x47
        private static final int OP_STRING      = 'S'; // 0x53
        private static final int OP_BINSTRING   = 'T'; // 0x54
        private static final int OP_SHORT_BINSTRING = 'U'; // 0x55
        private static final int OP_BINBYTES    = 'B'; // 0x42
        private static final int OP_SHORT_BINBYTES = 'C'; // 0x43
        private static final int OP_BINBYTES8   = 0x8e;
        private static final int OP_BINUNICODE  = 'X'; // 0x58
        private static final int OP_SHORT_BINUNICODE = 0x8c;
        private static final int OP_BINUNICODE8 = 0x8d;
        private static final int OP_UNICODE     = 'V'; // 0x56

        private final byte[] data;
        private final Deque<Object> stack = new ArrayDeque<>();
        private final Map<Integer, Object> memo = new HashMap<>();
        private final Deque<Integer> markPositions = new ArrayDeque<>();

        // Pending tensor build info, keyed by REDUCE-call order. The final
        // names come from the surrounding container (dict keys, list index).
        private final Map<Integer, TensorBuildInfo> pendingTensors = new HashMap<>();
        // Filename seen in a persistent-id tuple -> StorageInfo placeholder.
        private final Map<String, StorageInfo> storageByFilename = new HashMap<>();
        // Filename -> integer index into the ZIP's data/ directory.
        private final Map<String, Integer> storageIndex = new HashMap<>();
        // Top-level object left on the stack after STOP.
        private Object rootObject;

        PickleParser(byte[] data) {
            this.data = data;
        }

        /**
         * Run the unpickler and collect tensor rebuild metadata.
         */
        void parse(Map<String, StorageInfo> outStorages,
                   Map<String, TensorBuildInfo> outTensors,
                   Map<Integer, byte[]> storageData,
                   ByteOrder order) throws IOException {

            int[] p = new int[]{0};
            dispatchLoop(this.data, p, data.length);
            finalizeResults(outStorages, outTensors, storageData);
        }

        // Single-pass dispatch loop. Reads bytes from `buf` starting at
        // `p[0]` and advances p[0] as it consumes each opcode. Stops on
        // STOP or end of buffer.
        private void dispatchLoop(byte[] buf, int[] p, int end) {
            while (p[0] < end) {
                int op = buf[p[0]++] & 0xff;
                if (op == OP_STOP) {
                    // Capture the top of the stack as the root object.
                    rootObject = stack.isEmpty() ? null : stack.peek();
                    return;
                }
                if (op == OP_PROTO) {
                    if (p[0] < end) p[0]++; // skip protocol byte
                    continue;
                }
                if (op == OP_FRAME) {
                    if (p[0] + 8 > end) return;
                    long frameLen = ByteBuffer.wrap(buf, p[0], 8)
                            .order(ByteOrder.LITTLE_ENDIAN).getLong();
                    p[0] += 8;
                    if (frameLen < 0 || p[0] + frameLen > end) return;
                    int frameStart = p[0];
                    int frameEnd = p[0] + (int) frameLen;
                    dispatchLoop(buf, p, frameEnd);
                    // any leftover bytes after the frame are ignored by design
                    return;
                }
                dispatchWith(op, buf, p);
            }
        }

        private void finalizeResults(Map<String, StorageInfo> outStorages,
                                     Map<String, TensorBuildInfo> outTensors,
                                     Map<Integer, byte[]> storageData) {
            // Resolve storages from collected info: map filename -> bytes.
            for (Map.Entry<String, StorageInfo> e : storageByFilename.entrySet()) {
                String fn = e.getKey();
                StorageInfo si = e.getValue();
                Integer idx = storageIndex.get(fn);
                if (idx != null && storageData.containsKey(idx)) {
                    si.data = storageData.get(idx);
                }
                outStorages.put(si.dtype + ":" + fn, si);
                outStorages.put(fn, si);
            }

            // If the top-level container was a dict, prefer those keys for
            // naming the tensors. Otherwise fall back to positional names.
            if (rootObject instanceof Map) {
                flattenDict((Map<?, ?>) rootObject, "", outTensors);
            } else if (rootObject instanceof List) {
                int idx = 0;
                for (Object v : (List<?>) rootObject) {
                    if (v instanceof TensorBuildInfo) {
                        TensorBuildInfo t = (TensorBuildInfo) v;
                        t.name = "tensor_" + idx;
                        outTensors.put(t.name, t);
                    }
                    idx++;
                }
            } else if (rootObject instanceof TensorBuildInfo) {
                TensorBuildInfo t = (TensorBuildInfo) rootObject;
                t.name = "tensor_0";
                outTensors.put(t.name, t);
            } else {
                // Last resort: emit all pending tensors positionally.
                int idx = 0;
                for (TensorBuildInfo t : pendingTensors.values()) {
                    if (t.name == null || t.name.isEmpty() || t.name.startsWith("tensor_")) {
                        // already named; keep
                    }
                    outTensors.put(t.name, t);
                    idx++;
                }
            }
        }

        @SuppressWarnings("unchecked")
        private void flattenDict(Map<?, ?> dict, String prefix,
                                 Map<String, TensorBuildInfo> outTensors) {
            for (Map.Entry<?, ?> e : dict.entrySet()) {
                String key = String.valueOf(e.getKey());
                String fullKey = prefix.isEmpty() ? key : prefix + "." + key;
                Object v = e.getValue();
                if (v instanceof TensorBuildInfo) {
                    TensorBuildInfo t = (TensorBuildInfo) v;
                    t.name = fullKey;
                    outTensors.put(fullKey, t);
                } else if (v instanceof Map) {
                    flattenDict((Map<?, ?>) v, fullKey, outTensors);
                } else if (v instanceof List) {
                    int idx = 0;
                    for (Object item : (List<?>) v) {
                        if (item instanceof TensorBuildInfo) {
                            TensorBuildInfo t = (TensorBuildInfo) item;
                            String name = fullKey + "." + idx;
                            t.name = name;
                            outTensors.put(name, t);
                        } else if (item instanceof Map) {
                            flattenDict((Map<?, ?>) item, fullKey + "." + idx, outTensors);
                        }
                        idx++;
                    }
                }
            }
        }

        private void dispatchWith(int op, byte[] buf, int[] idxHolder) {
            int p = idxHolder[0];
            try {
                switch (op) {
                    // ---- Stack manipulation ----
                    case OP_MARK: {
                        markPositions.push(stack.size());
                        break;
                    }
                    case OP_POP: {
                        if (!stack.isEmpty()) stack.pop();
                        break;
                    }
                    case OP_DUP: {
                        if (!stack.isEmpty()) stack.push(stack.peek());
                        break;
                    }
                    // ---- Containers ----
                    case OP_EMPTY_TUPLE: {
                        stack.push(Collections.emptyList());
                        break;
                    }
                    case OP_EMPTY_LIST: {
                        stack.push(new ArrayList<>());
                        break;
                    }
                    case OP_EMPTY_DICT: {
                        stack.push(new LinkedHashMap<>());
                        break;
                    }
                    case OP_TUPLE1: collectTuple(1, buf, idxHolder); return;
                    case OP_TUPLE2: collectTuple(2, buf, idxHolder); return;
                    case OP_TUPLE3: collectTuple(3, buf, idxHolder); return;
                    case OP_TUPLE: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        List<Object> items = new ArrayList<>();
                        if (mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i < arr.length; i++) items.add(arr[i]);
                            // remove those elements
                            while (stack.size() > mark) stack.pop();
                        }
                        stack.push(items);
                        break;
                    }
                    case OP_LIST: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        List<Object> items = new ArrayList<>();
                        if (mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i < arr.length; i++) items.add(arr[i]);
                            while (stack.size() > mark) stack.pop();
                        }
                        stack.push(items);
                        break;
                    }
                    case OP_DICT: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        Map<String, Object> m = new LinkedHashMap<>();
                        if (mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i + 1 < arr.length; i += 2) {
                                m.put(String.valueOf(arr[i]), arr[i + 1]);
                            }
                            while (stack.size() > mark) stack.pop();
                        }
                        stack.push(m);
                        break;
                    }
                    case OP_SETITEM: {
                        if (stack.size() >= 2) {
                            Object v = stack.pop();
                            Object k = stack.pop();
                            Object target = stack.peek();
                            if (target instanceof Map) {
                                ((Map<Object, Object>) target).put(k, v);
                            }
                        }
                        break;
                    }
                    case OP_SETITEMS: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        Object target = stack.isEmpty() ? null : stack.peek();
                        if (target instanceof Map && mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i + 1 < arr.length; i += 2) {
                                ((Map<Object, Object>) target).put(arr[i], arr[i + 1]);
                            }
                            while (stack.size() > mark + 1) stack.pop();
                        }
                        break;
                    }
                    case OP_APPEND: {
                        if (stack.size() >= 2) {
                            Object v = stack.pop();
                            Object target = stack.peek();
                            if (target instanceof List) {
                                ((List<Object>) target).add(v);
                            }
                        }
                        break;
                    }
                    case OP_APPENDS: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        Object target = stack.isEmpty() ? null : stack.peek();
                        if (target instanceof List && mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i < arr.length; i++) {
                                ((List<Object>) target).add(arr[i]);
                            }
                            while (stack.size() > mark + 1) stack.pop();
                        }
                        break;
                    }
                    case OP_ADDITEMS: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        Object target = stack.isEmpty() ? null : stack.peek();
                        if (target instanceof Set && mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i < arr.length; i++) {
                                ((Set<Object>) target).add(arr[i]);
                            }
                            while (stack.size() > mark + 1) stack.pop();
                        }
                        break;
                    }
                    case OP_FROZENSET: {
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        Set<Object> s = new LinkedHashSet<>();
                        if (mark >= 0) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i < arr.length; i++) s.add(arr[i]);
                            while (stack.size() > mark) stack.pop();
                        }
                        stack.push(s);
                        break;
                    }

                    // ---- Memo ----
                    case OP_MEMOIZE: {
                        memo.put(memo.size(), stack.isEmpty() ? null : stack.peek());
                        break;
                    }
                    case OP_PUT: {
                        int idx = readLineInt(buf, idxHolder);
                        Object top = stack.isEmpty() ? null : stack.peek();
                        memo.put(idx, top);
                        break;
                    }
                    case OP_BINPUT: {
                        if (idxHolder[0] >= buf.length) break;
                        int idx = buf[idxHolder[0]++] & 0xff;
                        Object top = stack.isEmpty() ? null : stack.peek();
                        memo.put(idx, top);
                        break;
                    }
                    case OP_LONG_BINPUT: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int idx = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        Object top = stack.isEmpty() ? null : stack.peek();
                        memo.put(idx, top);
                        break;
                    }
                    case OP_GET: {
                        int idx = readLineInt(buf, idxHolder);
                        stack.push(memo.get(idx));
                        break;
                    }
                    case OP_BINGET: {
                        if (idxHolder[0] >= buf.length) break;
                        int idx = buf[idxHolder[0]++] & 0xff;
                        stack.push(memo.get(idx));
                        break;
                    }
                    case OP_LONG_BINGET: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int idx = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        stack.push(memo.get(idx));
                        break;
                    }

                    // ---- Singletons ----
                    case OP_NONE: stack.push(null); break;
                    case OP_NEW_TRUE: stack.push(Boolean.TRUE); break;
                    case OP_NEW_FALSE: stack.push(Boolean.FALSE); break;
                    // protocol 0/1/2 has no dedicated TRUE/FALSE opcodes;
                    // legacy code uses INT with payload '01\n' / '00\n'.

                    // ---- Numbers ----
                    case OP_INT: {
                        String s = readLineAscii(buf, idxHolder);
                        stack.push(parseIntLike(s));
                        break;
                    }
                    case OP_BININT: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int v = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        stack.push(v);
                        break;
                    }
                    case OP_BININT1: {
                        if (idxHolder[0] >= buf.length) break;
                        int v = buf[idxHolder[0]++] & 0xff;
                        stack.push(v);
                        break;
                    }
                    case OP_BININT2: {
                        if (idxHolder[0] + 2 > buf.length) break;
                        short v = ByteBuffer.wrap(buf, idxHolder[0], 2)
                                .order(ByteOrder.LITTLE_ENDIAN).getShort();
                        idxHolder[0] += 2;
                        stack.push((int) v);
                        break;
                    }
                    case OP_LONG: {
                        String s = readLineAscii(buf, idxHolder);
                        try {
                            stack.push(Long.parseLong(s.trim()));
                        } catch (NumberFormatException ex) {
                            stack.push(0L);
                        }
                        break;
                    }
                    case OP_LONG1: {
                        if (idxHolder[0] >= buf.length) break;
                        int n = buf[idxHolder[0]++] & 0xff;
                        if (idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], n, StandardCharsets.US_ASCII);
                        idxHolder[0] += n;
                        stack.push(Long.parseLong(s));
                        break;
                    }
                    case OP_LONG4: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int n = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        if (idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], n, StandardCharsets.US_ASCII);
                        idxHolder[0] += n;
                        stack.push(Long.parseLong(s));
                        break;
                    }
                    case OP_BINFLOAT: {
                        if (idxHolder[0] + 8 > buf.length) break;
                        double v = ByteBuffer.wrap(buf, idxHolder[0], 8)
                                .order(ByteOrder.BIG_ENDIAN).getDouble();
                        idxHolder[0] += 8;
                        stack.push(v);
                        break;
                    }
                    case OP_FLOAT: {
                        String s = readLineAscii(buf, idxHolder);
                        try {
                            stack.push(Double.parseDouble(s));
                        } catch (NumberFormatException ex) {
                            stack.push(0.0);
                        }
                        break;
                    }

                    // ---- Strings / bytes ----
                    case OP_STRING: {
                        String s = readQuotedString(buf, idxHolder);
                        stack.push(s);
                        break;
                    }
                    case OP_BINSTRING: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int n = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        if (idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], n, StandardCharsets.UTF_8);
                        idxHolder[0] += n;
                        stack.push(s);
                        break;
                    }
                    case OP_SHORT_BINSTRING: {
                        if (idxHolder[0] >= buf.length) break;
                        int n = buf[idxHolder[0]++] & 0xff;
                        if (idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], n, StandardCharsets.UTF_8);
                        idxHolder[0] += n;
                        stack.push(s);
                        break;
                    }
                    case OP_UNICODE: {
                        String s = readQuotedString(buf, idxHolder);
                        stack.push(s);
                        break;
                    }
                    case OP_SHORT_BINUNICODE: {
                        if (idxHolder[0] >= buf.length) break;
                        int n = buf[idxHolder[0]++] & 0xff;
                        if (idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], n, StandardCharsets.UTF_8);
                        idxHolder[0] += n;
                        stack.push(s);
                        break;
                    }
                    case OP_BINUNICODE: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int n = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        if (idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], n, StandardCharsets.UTF_8);
                        idxHolder[0] += n;
                        stack.push(s);
                        break;
                    }
                    case OP_BINUNICODE8: {
                        if (idxHolder[0] + 8 > buf.length) break;
                        long n = ByteBuffer.wrap(buf, idxHolder[0], 8)
                                .order(ByteOrder.LITTLE_ENDIAN).getLong();
                        idxHolder[0] += 8;
                        if (n < 0 || n > Integer.MAX_VALUE
                                || idxHolder[0] + n > buf.length) break;
                        String s = new String(buf, idxHolder[0], (int) n, StandardCharsets.UTF_8);
                        idxHolder[0] += n;
                        stack.push(s);
                        break;
                    }
                    case OP_BINBYTES: {
                        if (idxHolder[0] + 4 > buf.length) break;
                        int n = ByteBuffer.wrap(buf, idxHolder[0], 4)
                                .order(ByteOrder.LITTLE_ENDIAN).getInt();
                        idxHolder[0] += 4;
                        if (idxHolder[0] + n > buf.length) break;
                        byte[] bytes = Arrays.copyOfRange(buf, idxHolder[0], idxHolder[0] + n);
                        idxHolder[0] += n;
                        stack.push(bytes);
                        break;
                    }
                    case OP_SHORT_BINBYTES: {
                        if (idxHolder[0] >= buf.length) break;
                        int n = buf[idxHolder[0]++] & 0xff;
                        if (idxHolder[0] + n > buf.length) break;
                        byte[] bytes = Arrays.copyOfRange(buf, idxHolder[0], idxHolder[0] + n);
                        idxHolder[0] += n;
                        stack.push(bytes);
                        break;
                    }
                    case OP_BINBYTES8: {
                        if (idxHolder[0] + 8 > buf.length) break;
                        long n = ByteBuffer.wrap(buf, idxHolder[0], 8)
                                .order(ByteOrder.LITTLE_ENDIAN).getLong();
                        idxHolder[0] += 8;
                        if (n < 0 || n > Integer.MAX_VALUE
                                || idxHolder[0] + n > buf.length) break;
                        byte[] bytes = Arrays.copyOfRange(buf, idxHolder[0], idxHolder[0] + (int) n);
                        idxHolder[0] += n;
                        stack.push(bytes);
                        break;
                    }

                    // ---- Class lookup ----
                    case OP_GLOBAL: {
                        String module = readLineAscii(buf, idxHolder);
                        String name = readLineAscii(buf, idxHolder);
                        stack.push(new GlobalRef(module, name));
                        break;
                    }
                    case OP_STACK_GLOBAL: {
                        Object name = stack.isEmpty() ? null : stack.pop();
                        Object module = stack.isEmpty() ? null : stack.pop();
                        String m = module instanceof String ? (String) module : String.valueOf(module);
                        String n = name instanceof String ? (String) name : String.valueOf(name);
                        stack.push(new GlobalRef(m, n));
                        break;
                    }
                    case OP_INST: {
                        // INST pops module and name from the stack, finds
                        // class, then pops args (collected from the last
                        // MARK) and pushes the constructed instance.
                        if (stack.size() < 2) break;
                        Object name = stack.pop();
                        Object module = stack.pop();
                        String m = module instanceof String ? (String) module : String.valueOf(module);
                        String n = name instanceof String ? (String) name : String.valueOf(name);
                        int mark = markPositions.isEmpty() ? -1 : markPositions.pop();
                        List<Object> args = new ArrayList<>();
                        if (mark >= 0 && mark <= stack.size()) {
                            Object[] arr = stack.toArray();
                            for (int i = mark; i < arr.length; i++) args.add(arr[i]);
                            while (stack.size() > mark) stack.pop();
                        }
                        stack.push(applyCallable(new GlobalRef(m, n), args));
                        break;
                    }
                    case OP_OBJ:
                    case OP_NEWOBJ: {
                        // Pop args tuple, pop callable (GlobalRef), invoke.
                        if (stack.size() < 2) break;
                        Object args = stack.pop();
                        Object callable = stack.pop();
                        stack.push(applyCallable(callable, args));
                        break;
                    }
                    case OP_NEWOBJ_EX: {
                        // args, kwargs -> cls(*args, **kwargs)
                        if (stack.size() < 3) break;
                        Object kwargs = stack.pop();
                        Object args = stack.pop();
                        Object callable = stack.pop();
                        stack.push(applyCallable(callable, args));
                        break;
                    }

                    case OP_REDUCE: {
                        if (stack.size() < 2) break;
                        Object args = stack.pop();
                        Object callable = stack.pop();
                        Object result = applyCallable(callable, args);
                        stack.push(result);
                        break;
                    }
                    case OP_BUILD: {
                        if (stack.size() < 2) break;
                        Object state = stack.pop();
                        Object target = stack.pop();
                        applyBuild(target, state);
                        stack.push(target);
                        break;
                    }

                    // ---- Persistent ID ----
                    // In Python pickle both PERSID and BINPERSID pop a
                    // (stack-built) pid tuple from the stack and pass it to
                    // the unpickler's persistent_load hook. We don't
                    // consume bytes for PERSID / BINPERSID; instead we read
                    // the pid from the stack.
                    case OP_PERSID:
                    case OP_BINPERSID: {
                        if (stack.isEmpty()) break;
                        Object pid = stack.pop();
                        stack.push(resolvePersistent(pid));
                        break;
                    }

                    default:
                        // Skip unknown opcodes conservatively: try to push a
                        // placeholder so subsequent STACK_GLOBAL / REDUCE
                        // still line up.
                        stack.push(new UnknownOp(op));
                        break;
                }
            } finally {
                // idxHolder[0] may have been advanced by helper; commit back.
                // (No-op when dispatchWith was called with this.data.)
            }
        }

        private void collectTuple(int n, byte[] buf, int[] idxHolder) {
            if (stack.size() < n) {
                stack.push(new ArrayList<>());
                return;
            }
            Object[] arr = stack.toArray();
            List<Object> items = new ArrayList<>(n);
            for (int i = arr.length - n; i < arr.length; i++) items.add(arr[i]);
            while (stack.size() > arr.length - n) stack.pop();
            stack.push(items);
        }

        private static int readLineInt(byte[] buf, int[] idxHolder) {
            String s = readLineAscii(buf, idxHolder);
            try {
                return Integer.parseInt(s.trim());
            } catch (NumberFormatException ex) {
                return 0;
            }
        }

        private static String readLineAscii(byte[] buf, int[] idxHolder) {
            int start = idxHolder[0];
            while (idxHolder[0] < buf.length && buf[idxHolder[0]] != '\n') {
                idxHolder[0]++;
            }
            int end = idxHolder[0];
            if (idxHolder[0] < buf.length) idxHolder[0]++; // consume '\n'
            return new String(buf, start, end - start, StandardCharsets.US_ASCII);
        }

        private static String readQuotedString(byte[] buf, int[] idxHolder) {
            int start = idxHolder[0];
            while (idxHolder[0] < buf.length && buf[idxHolder[0]] != '\n') {
                idxHolder[0]++;
            }
            int end = idxHolder[0];
            if (idxHolder[0] < buf.length) idxHolder[0]++;
            // Pickle STRING: '...'\\n with possible escape sequences. We
            // unescape minimally: \\n -> \n, \\t -> \t, \\\\ -> \\, \\' -> '.
            String raw = new String(buf, start, end - start, StandardCharsets.UTF_8);
            return unescapePickleString(raw);
        }

        private static String unescapePickleString(String raw) {
            if (raw.indexOf('\\') < 0) return raw;
            StringBuilder sb = new StringBuilder(raw.length());
            for (int i = 0; i < raw.length(); i++) {
                char c = raw.charAt(i);
                if (c != '\\' || i + 1 >= raw.length()) {
                    sb.append(c);
                    continue;
                }
                char n = raw.charAt(++i);
                switch (n) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case '\\': sb.append('\\'); break;
                    case '\'': sb.append('\''); break;
                    case '"': sb.append('"'); break;
                    case 'a': sb.append('\007'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'v': sb.append('\u000b'); break;
                    case '0': case '1': case '2': case '3':
                    case '4': case '5': case '6': case '7': {
                        int oct = n - '0';
                        for (int k = 0; k < 2 && i + 1 < raw.length(); k++) {
                            char d = raw.charAt(i + 1);
                            if (d < '0' || d > '7') break;
                            oct = oct * 8 + (d - '0');
                            i++;
                        }
                        sb.append((char) oct);
                        break;
                    }
                    case 'x': {
                        if (i + 2 < raw.length()) {
                            int hex = Integer.parseInt(raw.substring(i + 1, i + 3), 16);
                            sb.append((char) hex);
                            i += 2;
                        }
                        break;
                    }
                    default:
                        sb.append(n);
                }
            }
            return sb.toString();
        }

        private static Object parseIntLike(String s) {
            s = s.trim();
            if (s.isEmpty()) return 0;
            char c0 = s.charAt(0);
            if (c0 == '0' && s.length() > 1 && (s.charAt(1) == 'x' || s.charAt(1) == 'X')) {
                return Integer.parseInt(s.substring(2), 16);
            }
            // Pickle INT uses '01'/'00' as True/False in legacy protocol 0.
            if (s.equals("01")) return Boolean.TRUE;
            if (s.equals("00")) return Boolean.FALSE;
            try {
                return Long.parseLong(s);
            } catch (NumberFormatException ex) {
                return 0L;
            }
        }

        private Object applyCallable(Object callable, Object args) {
            if (!(callable instanceof GlobalRef)) {
                return new BuiltinResult(callable, args);
            }
            GlobalRef ref = (GlobalRef) callable;
            String key = ref.module + " " + ref.name;

            // PyTorch tensor rebuild functions: capture (storage, offset,
            // size, stride, requires_grad, ...) and produce a TensorBuildInfo.
            if (key.startsWith("torch._utils") &&
                    (ref.name.equals("_rebuild_tensor_v2")
                            || ref.name.equals("_rebuild_tensor")
                            || ref.name.equals("_rebuild_qtensor")
                            || ref.name.equals("_rebuild_sparse_tensor")
                            || ref.name.equals("_rebuild_meta_tensor_no_storage")
                            || ref.name.equals("_rebuild_parameter")
                            || ref.name.equals("_rebuild_storage_tensor")
                            || ref.name.equals("_rebuild_subclass"))) {
                TensorBuildInfo info = parseRebuildTensor(ref.name, args);
                if (info != null) {
                    pendingTensors.put(pendingTensors.size(), info);
                }
                return info;
            }

            // torch.*Storage class called as a constructor: persistent_load
            // will have already populated a StorageInfo; the args list usually
            // contains (size,). We just return the storage placeholder.
            if (key.startsWith("torch.") && ref.name.endsWith("Storage")
                    && args instanceof List) {
                List<?> a = (List<?>) args;
                if (!a.isEmpty() && a.get(0) instanceof StorageRef) {
                    return a.get(0);
                }
            }

            // torch.save / torch.* wrappers we don't care about.
            return new BuiltinResult(callable, args);
        }

        private void applyBuild(Object target, Object state) {
            // PyTorch sets backward_hooks via BUILD on the result of
            // _rebuild_tensor_v2; the state dict is typically empty
            // OrderedDict, but it can also be metadata dicts in newer
            // versions. Nothing meaningful to do for tensor objects here.
        }

        private TensorBuildInfo parseRebuildTensor(String fnName, Object args) {
            if (!(args instanceof List)) return null;
            List<?> a = (List<?>) args;
            TensorBuildInfo info = new TensorBuildInfo();
            info.name = "tensor";

            // storage (positional 0): either a StorageRef (BINPERSID) or a
            // memoized object that resolved to one.
            if (a.size() > 0 && a.get(0) instanceof StorageRef) {
                info.storageKey = ((StorageRef) a.get(0)).filename;
            }

            // storage_offset (1)
            if (a.size() > 1) {
                info.storageOffset = asLong(a.get(1));
            }

            // size (2) - tuple of ints
            if (a.size() > 2 && a.get(2) instanceof List) {
                info.shape = ((List<?>) a.get(2)).stream()
                        .mapToLong(o -> asLong(o)).toArray();
            }

            // stride (3) - tuple of ints
            if (a.size() > 3 && a.get(3) instanceof List) {
                info.stride = ((List<?>) a.get(3)).stream()
                        .mapToLong(o -> asLong(o)).toArray();
            }

            // requires_grad (4)
            if (a.size() > 4) {
                Object rg = a.get(4);
                if (rg instanceof Boolean) {
                    info.requiresGrad = (Boolean) rg;
                } else if (rg instanceof Number) {
                    info.requiresGrad = ((Number) rg).intValue() != 0;
                } else if (rg != null) {
                    info.requiresGrad = Boolean.parseBoolean(String.valueOf(rg));
                }
            }

            return info;
        }

        private static long asLong(Object o) {
            if (o == null) return 0;
            if (o instanceof Long) return (Long) o;
            if (o instanceof Integer) return ((Integer) o).longValue();
            if (o instanceof Number) return ((Number) o).longValue();
            if (o instanceof Boolean) return ((Boolean) o) ? 1L : 0L;
            try {
                return Long.parseLong(String.valueOf(o));
            } catch (NumberFormatException ex) {
                return 0L;
            }
        }

        private Object resolvePersistent(Object pidObj) {
            // pidObj is a stack-built tuple (or list) that PyTorch's
            // persistent_load receives. Its shape is:
            //   ('storage', dtype_class, '', '<device>', size)
            // dtype_class is normally a GlobalRef from the preceding
            // GLOBAL/STACK_GLOBAL opcode in the pickle stream.
            StorageInfo si = new StorageInfo();
            si.dtype = "Unknown";
            si.device = "cpu";
            si.size = 0;

            String filename = null;
            String dtypeClassName = null;

            try {
                if (pidObj instanceof List) {
                    List<?> pid = (List<?>) pidObj;
                    if (pid.size() >= 5) {
                        String tag = String.valueOf(pid.get(0));
                        if ("storage".equals(tag) || "Storage".equals(tag)) {
                            Object dtypeClass = pid.get(1);
                            if (dtypeClass instanceof GlobalRef) {
                                dtypeClassName = ((GlobalRef) dtypeClass).name;
                            } else {
                                dtypeClassName = String.valueOf(dtypeClass);
                            }
                            filename = String.valueOf(pid.get(2));
                            si.device = String.valueOf(pid.get(3));
                            Object sizeObj = pid.get(4);
                            if (sizeObj instanceof Number) {
                                si.size = ((Number) sizeObj).intValue();
                            } else {
                                try {
                                    si.size = Integer.parseInt(String.valueOf(sizeObj));
                                } catch (NumberFormatException ex) {
                                    si.size = 0;
                                }
                            }
                        }
                    }
                }
            } catch (RuntimeException ex) {
                // Best-effort; leave defaults.
            }

            if (dtypeClassName != null) {
                si.dtype = classNameToDtype(dtypeClassName);
            }
            if (filename == null || filename.isEmpty()) {
                filename = "_anon_" + storageIndex.size();
            }
            if (!storageByFilename.containsKey(filename)) {
                storageByFilename.put(filename, si);
                storageIndex.put(filename, storageIndex.size());
            }
            return new StorageRef(filename, si);
        }

        private static String classNameToDtype(String className) {
            // e.g. "torch.FloatStorage", "torch.cuda.LongStorage",
            //      "torch.BFloat16Storage", "torch.HalfStorage".
            String n = className;
            int dot = n.lastIndexOf('.');
            if (dot >= 0) n = n.substring(dot + 1);
            n = n.replace("Storage", "");
            if (n.endsWith("CUDA") || n.endsWith("Cuda")) {
                n = n.substring(0, n.length() - 4);
            }
            // Convert Python class name -> canonical dtype string used
            // elsewhere in PT.java's parseDataType.
            switch (n) {
                case "Float":   return "float32";
                case "Double":  return "float64";
                case "Half":    return "float16";
                case "BFloat16":return "float32"; // closest match in our enum
                case "Long":    return "int64";
                case "Int":     return "int32";
                case "Short":   return "int16";
                case "Char":    return "int8";
                case "Byte":    return "uint8";
                case "Bool":    return "bool";
                case "ComplexFloat": return "complex64";
                case "ComplexDouble": return "complex128";
                default:        return n;
            }
        }

        /** Tag class to remember a class+module from GLOBAL/STACK_GLOBAL. */
        static final class GlobalRef {
            final String module;
            final String name;
            GlobalRef(String module, String name) {
                this.module = module;
                this.name = name;
            }
            @Override
            public String toString() {
                return module + "." + name;
            }
        }

        /** Tag class for an unknown opcode pushed onto the stack. */
        static final class UnknownOp {
            final int op;
            UnknownOp(int op) { this.op = op; }
        }

        /** Tag class for results of REDUCE that we don't recognize. */
        static final class BuiltinResult {
            final Object callable;
            final Object args;
            BuiltinResult(Object callable, Object args) {
                this.callable = callable;
                this.args = args;
            }
        }

        /** Reference to a storage placeholder, captured from BINPERSID. */
        static final class StorageRef {
            final String filename;
            final StorageInfo info;
            StorageRef(String filename, StorageInfo info) {
                this.filename = filename;
                this.info = info;
            }
            @Override
            public String toString() {
                return "StorageRef(" + filename + ", " + info.dtype + ", " + info.size + ")";
            }
        }
    }
}
