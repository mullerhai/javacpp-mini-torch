package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Pure-Java unpickler for Python pickle protocol.
 * 
 * <p>This is a standalone implementation that handles:
 * <ul>
 *   <li>All protocol versions (0-5)</li>
 *   <li>Basic Python types: None, bool, int, float, str, bytes, list, tuple, dict, set</li>
 *   <li>torch-specific objects for model loading</li>
 * </ul>
 * </p>
 * 
 * <p>Unlike Razorvine, this does not execute arbitrary Python code.
 * Custom class instances are returned as Map&lt;String, Object&gt;.</p>
 */
public class UnpickleReader {

    /** Marker object for stack operations */
    public static final Object MARKER = new Object() {
        @Override public String toString() { return "MARKER"; }
    };

    private final InputStream input;
    private final List<Object> stack = new ArrayList<>();
    private final List<Integer> marks = new ArrayList<>();
    private final Map<Integer, Object> memo = new HashMap<>();
    private final Map<String, Tensor> tensors = new LinkedHashMap<>();
    
    private int protocol = 0;
    private ByteOrder byteOrder = ByteOrder.BIG_ENDIAN; // pickle floats are big-endian

    public UnpickleReader(byte[] data) {
        this(new ByteArrayInputStream(data));
    }

    public UnpickleReader(InputStream input) {
        this.input = input;
    }

    /**
     * Parse the pickle data and return the top-level object.
     */
    public Object load() throws IOException {
        while (true) {
            int opcode = input.read();
            if (opcode < 0) {
                throw new IOException("unexpected EOF in pickle stream");
            }
            
            Object result = dispatch(opcode);
            if (result != null) {
                return result;
            }
        }
    }

    /**
     * Parse and extract state_dict tensors.
     */
    public Map<String, Tensor> loadStateDict() throws IOException {
        load();
        return new LinkedHashMap<>(tensors);
    }

    private Object dispatch(int opcode) throws IOException {
        switch (opcode) {
            case PickleOpcodes.PROTO:
                loadProto();
                return null;
                
            case PickleOpcodes.FRAME:
                loadFrame();
                return null;
                
            case PickleOpcodes.STOP:
                if (stack.isEmpty()) return null;
                return stack.get(stack.size() - 1);
                
            case PickleOpcodes.MARK:
                marks.add(stack.size());
                return null;
                
            case PickleOpcodes.POP:
                if (!stack.isEmpty()) stack.remove(stack.size() - 1);
                return null;
                
            case PickleOpcodes.POP_MARK:
                popMark();
                return null;
                
            case PickleOpcodes.DUP:
                if (!stack.isEmpty()) {
                    stack.add(stack.get(stack.size() - 1));
                }
                return null;
                
            case PickleOpcodes.NONE:
                stack.add(null);
                return null;
                
            case PickleOpcodes.NEWTRUE:
                stack.add(Boolean.TRUE);
                return null;
                
            case PickleOpcodes.NEWFALSE:
                stack.add(Boolean.FALSE);
                return null;
                
            case PickleOpcodes.INT:
                loadInt();
                return null;
                
            case PickleOpcodes.BININT:
                loadBinInt();
                return null;
                
            case PickleOpcodes.BININT1:
                stack.add(PickleProtocolUtils.readUnsignedByte(input));
                return null;
                
            case PickleOpcodes.BININT2:
                loadBinInt2();
                return null;
                
            case PickleOpcodes.LONG:
            case PickleOpcodes.LONG1:
            case PickleOpcodes.LONG4:
                loadLong(opcode);
                return null;
                
            case PickleOpcodes.FLOAT:
                loadFloat();
                return null;
                
            case PickleOpcodes.BINFLOAT:
                loadBinFloat();
                return null;
                
            case PickleOpcodes.STRING:
                loadString();
                return null;
                
            case PickleOpcodes.SHORT_BINSTRING:
            case PickleOpcodes.BINSTRING:
                loadBinString(opcode);
                return null;
                
            case PickleOpcodes.UNICODE:
                loadUnicode();
                return null;
                
            case PickleOpcodes.SHORT_BINUNICODE:
            case PickleOpcodes.BINUNICODE:
            case PickleOpcodes.BINUNICODE8:
                loadBinUnicode(opcode);
                return null;
                
            case PickleOpcodes.SHORT_BINBYTES:
            case PickleOpcodes.BINBYTES:
            case PickleOpcodes.BINBYTES8:
                loadBinBytes(opcode);
                return null;
                
            case PickleOpcodes.BYTEARRAY8:
                loadByteArray();
                return null;
                
            case PickleOpcodes.EMPTY_LIST:
                stack.add(new ArrayList<>());
                return null;
                
            case PickleOpcodes.EMPTY_DICT:
                stack.add(new LinkedHashMap<>());
                return null;
                
            case PickleOpcodes.EMPTY_TUPLE:
                stack.add(new Object[0]);
                return null;
                
            case PickleOpcodes.EMPTY_SET:
                stack.add(new LinkedHashSet<>());
                return null;
                
            case PickleOpcodes.LIST:
                loadList();
                return null;
                
            case PickleOpcodes.DICT:
                loadDict();
                return null;
                
            case PickleOpcodes.TUPLE:
            case PickleOpcodes.TUPLE1:
            case PickleOpcodes.TUPLE2:
            case PickleOpcodes.TUPLE3:
                loadTuple(opcode);
                return null;
                
            case PickleOpcodes.FROZENSET:
                loadFrozenSet();
                return null;
                
            case PickleOpcodes.APPEND:
                loadAppend();
                return null;
                
            case PickleOpcodes.APPENDS:
                loadAppends();
                return null;
                
            case PickleOpcodes.SETITEM:
                loadSetItem();
                return null;
                
            case PickleOpcodes.SETITEMS:
                loadSetItems();
                return null;
                
            case PickleOpcodes.ADDITEMS:
                loadAddItems();
                return null;
                
            case PickleOpcodes.GLOBAL:
                loadGlobal();
                return null;
                
            case PickleOpcodes.STACK_GLOBAL:
                loadStackGlobal();
                return null;
                
            case PickleOpcodes.REDUCE:
            case PickleOpcodes.NEWOBJ:
            case PickleOpcodes.NEWOBJ_EX:
                loadReduce(opcode);
                return null;
                
            case PickleOpcodes.BUILD:
                loadBuild();
                return null;
                
            case PickleOpcodes.GET:
            case PickleOpcodes.BINGET:
            case PickleOpcodes.LONG_BINGET:
                loadGet(opcode);
                return null;
                
            case PickleOpcodes.PUT:
            case PickleOpcodes.BINPUT:
            case PickleOpcodes.LONG_BINPUT:
                loadPut(opcode);
                return null;
                
            case PickleOpcodes.MEMOIZE:
                memo.put(memo.size(), stack.isEmpty() ? null : stack.get(stack.size() - 1));
                return null;
                
            case PickleOpcodes.BINPERSID:
            case PickleOpcodes.PERSID:
                loadPersId(opcode);
                return null;
                
            default:
                throw new IOException(String.format("unknown pickle opcode 0x%02x", opcode));
        }
    }

    // ---- Protocol handling ----

    private void loadProto() throws IOException {
        protocol = PickleProtocolUtils.readUnsignedByte(input);
        if (protocol > 5) {
            throw new IOException("unsupported pickle protocol: " + protocol);
        }
    }

    private void loadFrame() throws IOException {
        // Skip 8-byte frame length
        byte[] len = PickleProtocolUtils.readBytes(input, 8);
        // Frame support: for simplicity, we just skip the frame header
        // A full implementation would buffer the frame
    }

    // ---- Integer handling ----

    private void loadInt() throws IOException {
        String line = PickleProtocolUtils.readline(input, true);
        if (line.endsWith("\n")) {
            line = line.substring(0, line.length() - 1);
        }
        if ("01".equals(line) || "00".equals(line)) {
            stack.add("01".equals(line));
        } else {
            try {
                stack.add(Integer.parseInt(line));
            } catch (NumberFormatException e) {
                stack.add(Long.parseLong(line));
            }
        }
    }

    private void loadBinInt() throws IOException {
        byte[] data = PickleProtocolUtils.readBytes(input, 4);
        stack.add(Integer.reverseBytes(data[0] << 24 | (data[1] & 0xff) << 16 
                | (data[2] & 0xff) << 8 | (data[3] & 0xff)));
    }

    private void loadBinInt2() throws IOException {
        int lo = PickleProtocolUtils.readUnsignedByte(input);
        int hi = PickleProtocolUtils.readUnsignedByte(input);
        stack.add(lo | (hi << 8));
    }

    private void loadLong(int opcode) throws IOException {
        if (opcode == PickleOpcodes.LONG) {
            String line = PickleProtocolUtils.readline(input);
            if (line.endsWith("L")) {
                line = line.substring(0, line.length() - 1);
            }
            stack.add(PickleProtocolUtils.decodeLong(line.getBytes(StandardCharsets.ISO_8859_1)));
        } else if (opcode == PickleOpcodes.LONG1) {
            int n = PickleProtocolUtils.readUnsignedByte(input);
            byte[] data = PickleProtocolUtils.readBytes(input, n);
            stack.add(PickleProtocolUtils.decodeLong(data));
        } else {
            byte[] lenBytes = PickleProtocolUtils.readBytes(input, 4);
            int n = PickleProtocolUtils.bytesToInt(lenBytes, 0);
            // Note: bytesToInt expects little-endian, but LONG4 uses little-endian
            byte[] data = PickleProtocolUtils.readBytes(input, n);
            stack.add(PickleProtocolUtils.decodeLong(data));
        }
    }

    // ---- Float handling ----

    private void loadFloat() throws IOException {
        String line = PickleProtocolUtils.readline(input, true);
        if (line.endsWith("\n")) {
            line = line.substring(0, line.length() - 1);
        }
        stack.add(Double.parseDouble(line));
    }

    private void loadBinFloat() throws IOException {
        byte[] data = PickleProtocolUtils.readBytes(input, 8);
        // Big-endian double
        long bits = PickleProtocolUtils.bytesToLongBE(data, 0);
        stack.add(Double.longBitsToDouble(bits));
    }

    // ---- String handling ----

    private void loadString() throws IOException {
        String rep = PickleProtocolUtils.readline(input);
        boolean hasQuotes = rep.startsWith("\"") || rep.startsWith("'");
        if (!hasQuotes) {
            throw new IOException("insecure string pickle: missing quotes");
        }
        char quote = rep.charAt(0);
        if (!rep.endsWith(String.valueOf(quote))) {
            throw new IOException("insecure string pickle: mismatched quotes");
        }
        rep = rep.substring(1, rep.length() - 1);
        stack.add(PickleProtocolUtils.decodeEscaped(rep));
    }

    private void loadBinString(int opcode) throws IOException {
        int n;
        if (opcode == PickleOpcodes.SHORT_BINSTRING) {
            n = PickleProtocolUtils.readUnsignedByte(input);
        } else {
            byte[] lenBytes = PickleProtocolUtils.readBytes(input, 4);
            n = PickleProtocolUtils.bytesToInt(lenBytes, 0);
            // Note: BINSTRING uses little-endian
            n = Integer.reverseBytes(n);
        }
        byte[] data = PickleProtocolUtils.readBytes(input, n);
        stack.add(PickleProtocolUtils.rawStringFromBytes(data));
    }

    private void loadUnicode() throws IOException {
        String str = PickleProtocolUtils.readline(input);
        stack.add(PickleProtocolUtils.decodeUnicodeEscaped(str));
    }

    private void loadBinUnicode(int opcode) throws IOException {
        int n;
        if (opcode == PickleOpcodes.SHORT_BINUNICODE) {
            n = PickleProtocolUtils.readUnsignedByte(input);
        } else if (opcode == PickleOpcodes.BINUNICODE) {
            byte[] lenBytes = PickleProtocolUtils.readBytes(input, 4);
            n = Integer.reverseBytes(PickleProtocolUtils.bytesToInt(lenBytes, 0));
        } else {
            byte[] lenBytes = PickleProtocolUtils.readBytes(input, 8);
            n = (int) PickleProtocolUtils.bytesToLong(lenBytes, 0);
            // Note: BINUNICODE8 uses little-endian
            n = (int) Long.reverseBytes(n);
        }
        byte[] data = PickleProtocolUtils.readBytes(input, n);
        stack.add(new String(data, StandardCharsets.UTF_8));
    }

    private void loadBinBytes(int opcode) throws IOException {
        int n;
        if (opcode == PickleOpcodes.SHORT_BINBYTES) {
            n = PickleProtocolUtils.readUnsignedByte(input);
        } else if (opcode == PickleOpcodes.BINBYTES) {
            byte[] lenBytes = PickleProtocolUtils.readBytes(input, 4);
            n = Integer.reverseBytes(PickleProtocolUtils.bytesToInt(lenBytes, 0));
        } else {
            byte[] lenBytes = PickleProtocolUtils.readBytes(input, 8);
            n = (int) Long.reverseBytes(PickleProtocolUtils.bytesToLong(lenBytes, 0));
        }
        byte[] data = PickleProtocolUtils.readBytes(input, n);
        stack.add(data);
    }

    private void loadByteArray() throws IOException {
        byte[] lenBytes = PickleProtocolUtils.readBytes(input, 8);
        long n = Long.reverseBytes(PickleProtocolUtils.bytesToLong(lenBytes, 0));
        byte[] data = PickleProtocolUtils.readBytes(input, (int) n);
        stack.add(data); // Treat as byte[] for simplicity
    }

    // ---- Container handling ----

    private void loadList() {
        int start = popMark();
        List<Object> list = new ArrayList<>(stack.subList(start, stack.size()));
        stack.subList(start, stack.size()).clear();
        stack.add(list);
    }

    private void loadDict() {
        int start = popMark();
        Map<Object, Object> dict = new LinkedHashMap<>();
        for (int i = start; i + 1 < stack.size(); i += 2) {
            dict.put(stack.get(i), stack.get(i + 1));
        }
        stack.subList(start, stack.size()).clear();
        stack.add(dict);
    }

    private void loadTuple(int opcode) {
        if (opcode == PickleOpcodes.TUPLE1) {
            Object a = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            stack.add(new Object[]{a});
        } else if (opcode == PickleOpcodes.TUPLE2) {
            Object b = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            Object a = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            stack.add(new Object[]{a, b});
        } else if (opcode == PickleOpcodes.TUPLE3) {
            Object c = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            Object b = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            Object a = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            stack.add(new Object[]{a, b, c});
        } else {
            int start = popMark();
            Object[] arr = stack.subList(start, stack.size()).toArray();
            stack.subList(start, stack.size()).clear();
            stack.add(arr);
        }
    }

    private void loadFrozenSet() {
        int start = popMark();
        Set<Object> set = new LinkedHashSet<>(stack.subList(start, stack.size()));
        stack.subList(start, stack.size()).clear();
        stack.add(set);
    }

    private void loadAppend() {
        if (stack.size() < 2) return;
        Object value = stack.remove(stack.size() - 1);
        Object listObj = stack.get(stack.size() - 1);
        if (listObj instanceof List) {
            @SuppressWarnings("unchecked")
            List<Object> list = (List<Object>) listObj;
            list.add(value);
        }
    }

    private void loadAppends() {
        int start = popMark();
        if (stack.size() > start) {
            Object listObj = stack.get(start - 1);
            if (listObj instanceof List) {
                @SuppressWarnings("unchecked")
                List<Object> list = (List<Object>) listObj;
                list.addAll(stack.subList(start, stack.size()));
            }
            stack.subList(start, stack.size()).clear();
        }
    }

    private void loadSetItem() {
        if (stack.size() < 3) return;
        Object value = stack.remove(stack.size() - 1);
        Object key = stack.remove(stack.size() - 1);
        Object dictObj = stack.get(stack.size() - 1);
        if (dictObj instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<Object, Object> dict = (Map<Object, Object>) dictObj;
            dict.put(key, value);
        }
    }

    private void loadSetItems() {
        int start = popMark();
        if (stack.size() > start) {
            Object dictObj = stack.get(start - 1);
            if (dictObj instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<Object, Object> dict = (Map<Object, Object>) dictObj;
                for (int i = start; i + 1 < stack.size(); i += 2) {
                    dict.put(stack.get(i), stack.get(i + 1));
                }
            }
            stack.subList(start, stack.size()).clear();
        }
    }

    private void loadAddItems() {
        int start = popMark();
        if (stack.size() > start) {
            Object setObj = stack.get(start - 1);
            if (setObj instanceof Set) {
                @SuppressWarnings("unchecked")
                Set<Object> set = (Set<Object>) setObj;
                set.addAll(stack.subList(start, stack.size()));
            }
            stack.subList(start, stack.size()).clear();
        }
    }

    // ---- Global/Reduce handling ----

    private void loadGlobal() throws IOException {
        String module = PickleProtocolUtils.readline(input);
        String name = PickleProtocolUtils.readline(input);
        stack.add(new GlobalRef(module, name));
    }

    private void loadStackGlobal() {
        if (stack.size() >= 2) {
            Object name = stack.remove(stack.size() - 1);
            Object module = stack.remove(stack.size() - 1);
            stack.add(new GlobalRef(String.valueOf(module), String.valueOf(name)));
        }
    }

    private void loadReduce(int opcode) throws IOException {
        if (stack.size() < 2) return;
        Object args = stack.remove(stack.size() - 1);
        Object callable = stack.remove(stack.size() - 1);
        
        if (callable instanceof GlobalRef) {
            GlobalRef ref = (GlobalRef) callable;
            Object result = applyReduce(ref, asArray(args), opcode);
            if (result != null) {
                stack.add(result);
            }
        } else {
            // Unknown callable - return as stub
            stack.add(new StubObject(String.valueOf(callable), args));
        }
    }

    private Object applyReduce(GlobalRef ref, Object[] args, int opcode) {
        // Handle torch._utils._rebuild_tensor_v2
        if (ref.is("torch._utils", "_rebuild_tensor_v2") 
                || ref.is("torch._utils", "_rebuild_tensor")
                || ref.is("torch._utils", "_rebuild_parameter")
                || ref.is("torch._utils", "_rebuild_qtensor")) {
            return buildTensor(args);
        }
        
        // Handle collections.OrderedDict
        if (ref.is("collections", "OrderedDict")) {
            Map<Object, Object> od = new LinkedHashMap<>();
            if (args.length == 1) {
                Object arg = args[0];
                if (arg instanceof List) {
                    for (Object item : (List<?>) arg) {
                        Object[] pair = asArray(item);
                        if (pair.length >= 2) od.put(pair[0], pair[1]);
                    }
                } else if (arg instanceof Object[]) {
                    for (Object item : (Object[]) arg) {
                        Object[] pair = asArray(item);
                        if (pair.length >= 2) od.put(pair[0], pair[1]);
                    }
                }
            }
            return od;
        }
        
        // Handle torch.storage._load_from_bytes
        if (ref.is("torch.storage", "_load_from_bytes")) {
            if (args.length >= 1) {
                Object arg = args[0];
                if (arg instanceof byte[]) {
                    // Recursively unpickle the embedded bytes
                    return new EmbeddedPickle((byte[]) arg);
                }
            }
        }
        
        // Handle _codecs.encode
        if (ref.is("_codecs", "encode")) {
            if (args.length >= 2 && args[1] instanceof byte[]) {
                return args[1];
            }
        }
        
        // Unknown global - return stub
        return new StubObject(ref.module + "." + ref.name, args);
    }

    private Object buildTensor(Object[] args) {
        if (args.length < 4) return null;
        
        long storageOffset = toLong(args[1]);
        long[] size = toLongArray(args[2]);
        
        // Check for embedded pickle in storage
        if (args[0] instanceof EmbeddedPickle) {
            EmbeddedPickle ep = (EmbeddedPickle) args[0];
            try {
                UnpickleReader reader = new UnpickleReader(ep.data);
                Object storage = reader.load();
                // Storage should be bytes or map
                if (storage instanceof byte[]) {
                    return bytesToTensor((byte[]) storage, storageOffset, size);
                }
            } catch (IOException e) {
                // Fall through
            }
        }
        
        return new StubObject("tensor", args);
    }

    private Tensor bytesToTensor(byte[] data, long offset, long[] size) {
        // Find actual tensor data start
        int dataStart = findDataStart(data);
        
        int elemSize = 4; // float32
        long numel = 1;
        for (long s : size) numel *= s;
        
        long byteStart = dataStart + (offset * elemSize);
        long byteLen = numel * elemSize;
        
        if (byteStart + byteLen > data.length) {
            byteStart = offset * elemSize;
            byteLen = Math.min(byteLen, data.length - byteStart);
        }
        
        byte[] tensorData = new byte[(int) byteLen];
        System.arraycopy(data, (int) byteStart, tensorData, 0, (int) byteLen);
        
        BytePointer ptr = new BytePointer(tensorData);
        Tensor t = torch.from_blob(ptr, size, new org.bytedeco.pytorch.TensorOptions(torch.ScalarType.Float));
        ptr.close();
        return t;
    }

    private int findDataStart(byte[] data) {
        for (int i = 0; i < data.length - 12; i++) {
            if (data[i] == 0x2e) { // STOP opcode
                return i + 1;
            }
        }
        return 0;
    }

    private void loadBuild() {
        if (stack.size() < 2) return;
        Object state = stack.remove(stack.size() - 1);
        Object target = stack.get(stack.size() - 1);
        
        if (target instanceof Map && state instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<Object, Object> m = (Map<Object, Object>) target;
            m.putAll((Map<?, ?>) state);
        }
    }

    // ---- Memo handling ----

    private void loadGet(int opcode) throws IOException {
        int idx;
        if (opcode == PickleOpcodes.GET) {
            String line = PickleProtocolUtils.readline(input);
            idx = Integer.parseInt(line);
        } else if (opcode == PickleOpcodes.BINGET) {
            idx = PickleProtocolUtils.readUnsignedByte(input);
        } else {
            byte[] data = PickleProtocolUtils.readBytes(input, 4);
            idx = PickleProtocolUtils.bytesToInt(data, 0);
        }
        Object val = memo.get(idx);
        if (val != null) {
            stack.add(val);
        }
    }

    private void loadPut(int opcode) throws IOException {
        int idx;
        if (opcode == PickleOpcodes.PUT) {
            String line = PickleProtocolUtils.readline(input);
            idx = Integer.parseInt(line);
        } else if (opcode == PickleOpcodes.BINPUT) {
            idx = PickleProtocolUtils.readUnsignedByte(input);
        } else {
            byte[] data = PickleProtocolUtils.readBytes(input, 4);
            idx = PickleProtocolUtils.bytesToInt(data, 0);
        }
        if (!stack.isEmpty()) {
            memo.put(idx, stack.get(stack.size() - 1));
        }
    }

    // ---- Persistent ID handling ----

    private void loadPersId(int opcode) throws IOException {
        if (opcode == PickleOpcodes.PERSID) {
            String pid = PickleProtocolUtils.readline(input);
            stack.add(persistentLoad(pid));
        } else {
            Object pid = stack.isEmpty() ? null : stack.remove(stack.size() - 1);
            stack.add(persistentLoad(String.valueOf(pid)));
        }
    }

    protected Object persistentLoad(String pid) {
        return new StubObject("persistent:" + pid, null);
    }

    // ---- Stack helpers ----

    private int popMark() {
        if (marks.isEmpty()) {
            stack.add(MARKER);
            return stack.size() - 1;
        }
        return marks.remove(marks.size() - 1);
    }

    private Object[] asArray(Object o) {
        if (o == null) return new Object[0];
        if (o instanceof Object[]) return (Object[]) o;
        if (o instanceof List) return ((List<?>) o).toArray();
        return new Object[]{o};
    }

    private long toLong(Object o) {
        if (o instanceof Number) return ((Number) o).longValue();
        if (o instanceof Boolean) return ((Boolean) o) ? 1L : 0L;
        try {
            return Long.parseLong(String.valueOf(o));
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private long[] toLongArray(Object o) {
        if (o == null) return new long[0];
        if (o instanceof long[]) return (long[]) o;
        if (o instanceof int[]) {
            int[] a = (int[]) o;
            long[] r = new long[a.length];
            for (int i = 0; i < a.length; i++) r[i] = a[i];
            return r;
        }
        if (o instanceof Object[]) {
            Object[] a = (Object[]) o;
            long[] r = new long[a.length];
            for (int i = 0; i < a.length; i++) r[i] = toLong(a[i]);
            return r;
        }
        if (o instanceof List) {
            List<?> l = (List<?>) o;
            long[] r = new long[l.size()];
            for (int i = 0; i < l.size(); i++) r[i] = toLong(l.get(i));
            return r;
        }
        return new long[]{toLong(o)};
    }

    // ---- Inner classes ----

    public static class GlobalRef {
        public final String module;
        public final String name;
        public final boolean stub;
        
        public GlobalRef(String module, String name) {
            this(module, name, false);
        }
        
        public GlobalRef(String module, String name, boolean stub) {
            this.module = module != null ? module.trim() : "";
            this.name = name != null ? name.trim() : "";
            this.stub = stub;
        }
        
        public boolean is(String mod, String n) {
            return module.equals(mod) && name.equals(n);
        }
        
        @Override
        public String toString() {
            return module + "." + name;
        }
    }

    public static class StubObject {
        public final String type;
        public final Object args;
        
        public StubObject(String type, Object args) {
            this.type = type;
            this.args = args;
        }
        
        @Override
        public String toString() {
            return "Stub(" + type + ")";
        }
    }

    /** Wrapper for embedded pickle data */
    public static class EmbeddedPickle {
        public final byte[] data;
        public EmbeddedPickle(byte[] data) {
            this.data = data;
        }
    }
}
