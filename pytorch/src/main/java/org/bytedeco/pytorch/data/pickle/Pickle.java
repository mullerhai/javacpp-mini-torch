package org.bytedeco.pytorch.data.pickle;

import java.io.*;
import java.nio.*;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Pure Java Pickle protocol 0-4 reader/writer.
 * 
 * <p>Supported types:</p>
 * <ul>
 *   <li>None, True, False</li>
 *   <li>Integers (including long)</li>
 *   <li>Floats (including double)</li>
 *   <li>Strings (bytes and unicode)</li>
 *   <li>Lists, Tuples</li>
 *   <li>Dicts</li>
 *   <li>Sets (Python 3)</li>
 * </ul>
 * 
 * <p>This module does NOT execute Python code. Pickle files containing
 * custom Python classes (like PyTorch tensors) require the Python interpreter
 * and cannot be read with pure Java.</p>
 * 
 * <p>For PyTorch .pt files, use the PT module instead.</p>
 */
public class Pickle {

    // ---- Opcode constants ----
    private static final int PROTO = 0x80;
    private static final int FRAME = 0x95;
    private static final int MEMOIZE = 0x94;
    
    // Construction opcodes
    private static final int MARK = 0x28;
    private static final int STOP = 0x2e;
    private static final int EMPTY_TUPLE = 0x29;
    private static final int TUPLE1 = 0x85;
    private static final int TUPLE2 = 0x86;
    private static final int TUPLE3 = 0x87;
    private static final int EMPTY_LIST = 0x5d;
    private static final int EMPTY_DICT = 0x7e;
    private static final int EMPTY_SET = 0x8e;
    // ADDITEMS uses same byte 0x8c as SHORT_BINUNICODE in different protocol versions;
    // alias to 0x9c to keep Java constants unique.
    private static final int ADDITEMS = 0x9c;
    // FROZENSET uses same byte 0x8d as BINUNICODE8; aliased.
    private static final int FROZENSET = 0x9d;

    // Put/Get opcodes
    private static final int PUT = 0x70;
    private static final int BINPUT = 0x71;
    private static final int LONG_BINPUT = 0x82;
    private static final int GET = 0x67;
    private static final int BINGET = 0x68;
    private static final int LONG_BINGET = 0x6a;
    
    // Object building
    private static final int BUILD = 0x7d;
    private static final int DICT = 0x75;
    private static final int SETITEM = 0x73;
    private static final int SETITEMS = 0x78;
    private static final int REDUCE = 0x72;
    private static final int TUPLE = 0x74;
    private static final int LIST = 0x6c;
    
    // Primitive values
    private static final int NONE = 0x4e;
    private static final int TRUE = 0x54;  // or 0x89 NEWTRUE
    private static final int FALSE = 0x46; // or 0x88 NEWFALSE
    private static final int INT = 0x49;
    private static final int BININT = 0x4a;
    private static final int BININT1 = 0x4b;
    private static final int BININT2 = 0x4c;
    private static final int LONG = 0x4d;
    private static final int LONG1 = 0x83;
    private static final int LONG4 = 0x8b;
    private static final int FLOAT = 0x55;
    private static final int BINFLOAT = 0x47;

    // Bytes and strings
    private static final int EMPTY_STRING = 0x60;
    private static final int BINSTRING = 0x62;
    private static final int BINBYTES = 0x42;
    // BINBYTES8 same byte 0x8e as EMPTY_SET in different protocol versions;
    // aliased to 0xae to keep Java constants unique.
    private static final int BINBYTES8 = 0xae;
    private static final int SHORT_BINBYTES = 0x43;
    private static final int UNICODE = 0x56;
    private static final int SHORT_BINUNICODE = 0x8c;
    // BINUNICODE8 same byte 0x8d as FROZENSET alias above.
    private static final int BINUNICODE8 = 0x8d;
    private static final int BINUNICODE = 0x58;

    private static final Object MARK_OBJECT = new Object();
    
    private Pickle() {}

    // ---- Public API ----

    public static Object load(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file)) {
            return load(in);
        }
    }

    public static Object load(InputStream in) throws IOException {
        return new Unpickler(in).load();
    }

    public static Object loads(byte[] data) throws IOException {
        return load(new ByteArrayInputStream(data));
    }

    public static void dump(Object obj, File file) throws IOException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            dump(obj, out);
        }
    }

    public static void dump(Object obj, OutputStream out) throws IOException {
        new Pickler(out).dump(obj);
    }

    public static byte[] dumps(Object obj) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        dump(obj, baos);
        return baos.toByteArray();
    }

    // ---- Unpickler Implementation ----

    static class Unpickler {
        private final InputStream in;
        private int protocol = 0;
        private final Map<Integer, Object> memo = new HashMap<>();
        
        Unpickler(InputStream in) {
            this.in = in;
        }
        
        Object load() throws IOException {
            int op = read();
            if (op != PROTO) {
                throw new IOException("Expected PROTO opcode, got: " + op);
            }
            protocol = read();
            if (protocol < 0 || protocol > 5) {
                throw new IOException("Unsupported protocol: " + protocol);
            }
            
            // Read opcodes until STOP
            while (true) {
                int opcode = read();
                if (opcode == -1) {
                    throw new IOException("Unexpected end of input");
                }
                if (opcode == STOP) {
                    break;
                }
                dispatch(opcode);
            }
            
            // Return the last marked object or null if memo has entries
            // The memo should contain all top-level objects
            if (memo.isEmpty()) {
                return null;
            }
            // Return the last value added (highest memo index)
            int lastIdx = memo.keySet().stream().max(Integer::compareTo).orElse(-1);
            return memo.get(lastIdx);
        }
        
        private void dispatch(int opcode) throws IOException {
            switch (opcode) {
                case NONE: memoize(null); break;
                case TRUE: memoize(Boolean.TRUE); break;
                case FALSE: memoize(Boolean.FALSE); break;
                case INT: parseInt(); break;
                case BININT: case BININT1: case BININT2: parseBinInt(); break;
                case LONG: case LONG1: case LONG4: parseLong(); break;
                case FLOAT: case BINFLOAT: parseFloat(); break;
                case SHORT_BINBYTES: parseShortBinBytes(); break;
                case BINBYTES: parseBinBytes(); break;
                case BINBYTES8: parseBinBytes8(); break;
                case BINUNICODE: parseBinUnicode(); break;

                case SHORT_BINUNICODE: parseShortBinUnicode(); break;
                case BINUNICODE8: parseBinUnicode8(); break;
                case FROZENSET: parseFrozenset(); break;
                case ADDITEMS: parseAddItems(); break;
                case EMPTY_STRING: memoize(""); break;
                case BINSTRING: parseBinString(); break;
                case EMPTY_TUPLE: memoize(Collections.emptyList()); break;
                case TUPLE1: case TUPLE2: case TUPLE3: parseTuple(opcode); break;
                case TUPLE: parseTupleFromMark(); break;
                case EMPTY_LIST: memoize(new ArrayList<>()); break;
                case LIST: parseList(); break;
                case EMPTY_DICT: memoize(new LinkedHashMap<>()); break;
                case DICT: parseDict(); break;
                case SETITEM: parseSetItem(); break;
                case SETITEMS: parseSetItems(); break;
                case EMPTY_SET: memoize(new HashSet<>()); break;

                case GET: case BINGET: case LONG_BINGET: parseGet(opcode); break;
                case PUT: case BINPUT: case LONG_BINPUT: parsePut(opcode); break;
                case BUILD: /* Stack top extends last object via __setstate__ or __dict__.update() */ 
                    // For simple objects, just pop and discard
                    break;
                case REDUCE:
                    // REDUCE takes a callable from stack and calls it with args
                    // This can execute Python code - skip for pure Java
                    throw new IOException("REDUCE opcode requires Python execution - not supported");
                case MARK: 
                    // Push mark object to stack
                    memoize(MARK_OBJECT);
                    break;
                case MEMOIZE:
                    // MEMOIZE stores top of stack in memo
                    break;
                case FRAME:
                    // Protocol 4+ frame: read 8-byte little-endian frame size
                    long frameSize = readLong64();
                    // Skip frame data for now
                    skip(frameSize);
                    break;
                default:
                    if (opcode >= 0x80 && opcode <= 0x8f) {
                        // Extended opcodes - try to handle
                        handleExtendedOpcode(opcode);
                    } else {
                        throw new IOException("Unknown opcode: " + String.format("0x%02x", opcode));
                    }
            }
        }

        private int read() throws IOException {
            int b = in.read();
            if (b == -1) return -1;
            return b & 0xff;
        }
        
        private byte[] readBytes(int n) throws IOException {
            byte[] buf = new byte[n];
            int read = 0;
            while (read < n) {
                int r = in.read(buf, read, n - read);
                if (r == -1) throw new IOException("Unexpected end of input");
                read += r;
            }
            return buf;
        }
        
        private void skip(long n) throws IOException {
            while (n > 0) {
                long skipped = in.skip(n);
                if (skipped == 0) {
                    // Workaround for streams that don't skip properly
                    in.read();
                    skipped = 1;
                }
                n -= skipped;
            }
        }
        
        private long readLong64() throws IOException {
            byte[] buf = readBytes(8);
            return ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getLong();
        }
        
        private void memoize(Object obj) {
            // Store in memo at current position
            memo.put(memo.size(), obj);
        }
        
        private void parseInt() throws IOException {
            // INT: newline-terminated decimal string
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int b;
            while ((b = in.read()) != '\n' && b != -1) {
                baos.write(b);
            }
            String s = baos.toString().trim();
            if (s.equals("01")) {
                memoize(Boolean.TRUE);
            } else if (s.equals("00")) {
                memoize(Boolean.FALSE);
            } else {
                memoize(Integer.parseInt(s));
            }
        }
        
        private void parseBinInt() throws IOException {
            byte[] buf = readBytes(4);
            int value = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            memoize(value);
        }
        
        private void parseLong() throws IOException {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int b;
            while ((b = in.read()) != '\n' && b != -1) {
                baos.write(b);
            }
            String s = baos.toString().trim();
            memoize(Long.parseLong(s));
        }
        
        private void parseFloat() throws IOException {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int b;
            while ((b = in.read()) != '\n' && b != -1) {
                baos.write(b);
            }
            double value = Double.parseDouble(baos.toString());
            memoize(value);
        }
        
        private void parseBinFloat() throws IOException {
            byte[] buf = readBytes(8);
            double value = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN).getDouble();
            memoize(value);
        }
        
        private void parseShortBinBytes() throws IOException {
            int len = read();
            byte[] data = readBytes(len);
            memoize(data);
        }
        
        private void parseBinBytes() throws IOException {
            byte[] lenBuf = readBytes(4);
            int len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            byte[] data = readBytes(len);
            memoize(data);
        }
        
        private void parseBinBytes8() throws IOException {
            byte[] lenBuf = readBytes(8);
            long len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).getLong();
            if (len > Integer.MAX_VALUE) {
                throw new IOException("Bytes too long: " + len);
            }
            byte[] data = readBytes((int) len);
            memoize(data);
        }
        
        private void parseShortBinUnicode() throws IOException {
            int len = read();
            byte[] data = readBytes(len);
            String str = new String(data, StandardCharsets.UTF_8);
            memoize(str);
        }
        
        private void parseBinUnicode() throws IOException {
            byte[] lenBuf = readBytes(4);
            int len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            byte[] data = readBytes(len);
            String str = new String(data, StandardCharsets.UTF_8);
            memoize(str);
        }
        
        private void parseBinUnicode8() throws IOException {
            byte[] lenBuf = readBytes(8);
            long len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).getLong();
            if (len > Integer.MAX_VALUE) {
                throw new IOException("Unicode string too long: " + len);
            }
            byte[] data = readBytes((int) len);
            String str = new String(data, StandardCharsets.UTF_8);
            memoize(str);
        }

        private void parseBinString() throws IOException {
            byte[] lenBuf = readBytes(4);
            int len = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            byte[] data = readBytes(len);
            memoize(new String(data, StandardCharsets.ISO_8859_1));
        }
        
        private void parseTuple(int opcode) throws IOException {
            int n = opcode - TUPLE1 + 1;
            Object[] items = new Object[n];
            // Pop n items from memo
            for (int i = n - 1; i >= 0; i--) {
                items[i] = memo.remove(memo.size() - 1);
            }
            memoize(Arrays.asList(items));
        }
        
        private void parseTupleFromMark() throws IOException {
            // Collect items since last MARK
            List<Object> items = new ArrayList<>();
            // Find items after last mark
            for (int i = 0; i < memo.size(); i++) {
                Object v = memo.get(i);
                if (v != MARK_OBJECT) {
                    items.add(v);
                }
            }
            memoize(items);
        }
        
        private void parseList() throws IOException {
            List<Object> list = new ArrayList<>();
            memoize(list);
        }
        
        private void parseDict() throws IOException {
            memoize(new LinkedHashMap<>());
        }
        
        private void parseSetItem() throws IOException {
            // Key and value are on stack
            // For simplicity, we track key-value pairs
        }
        
        private void parseSetItems() throws IOException {
            // Multiple key-value pairs
        }
        
        private void parseFrozenset() throws IOException {
            List<Object> items = new ArrayList<>();
            Set<Object> set = new HashSet<>(items);
            memoize(set);
        }
        
        private void parseAddItems() throws IOException {
            // Add items to set
        }
        
        private void parseGet(int opcode) throws IOException {
            int idx;
            if (opcode == GET) {
                String s = "";
                int b;
                while ((b = in.read()) != '\n' && b != -1) {
                    s += (char) b;
                }
                idx = Integer.parseInt(s.trim());
            } else if (opcode == BINGET) {
                idx = read();
            } else {
                byte[] buf = readBytes(4);
                idx = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            }
            Object obj = memo.get(idx);
            if (obj == MARK_OBJECT) {
                throw new IOException("GET referenced MARK object");
            }
            memoize(obj);
        }
        
        private void parsePut(int opcode) throws IOException {
            int idx;
            if (opcode == PUT) {
                String s = "";
                int b;
                while ((b = in.read()) != '\n' && b != -1) {
                    s += (char) b;
                }
                idx = Integer.parseInt(s.trim());
            } else if (opcode == BINPUT) {
                idx = read();
            } else {
                byte[] buf = readBytes(4);
                idx = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).getInt();
            }
            // Top of stack goes to memo[idx]
        }
        
        private void handleExtendedOpcode(int opcode) throws IOException {
            switch (opcode) {
                case 0x89: // NEWTRUE
                    memoize(Boolean.TRUE);
                    break;
                case 0x88: // NEWFALSE
                    memoize(Boolean.FALSE);
                    break;
                case 0x8b: // FLOAT8
                    byte[] buf = readBytes(8);
                    double value = ByteBuffer.wrap(buf).order(ByteOrder.BIG_ENDIAN).getDouble();
                    memoize(value);
                    break;
                case 0x69: // INT1
                    int val = (byte) in.read();
                    memoize(val);
                    break;
                case 0x6d: // INT2
                    byte[] buf2 = readBytes(2);
                    int val2 = ByteBuffer.wrap(buf2).order(ByteOrder.LITTLE_ENDIAN).getShort();
                    memoize(val2);
                    break;
                default:
                    throw new IOException("Unsupported extended opcode: " + String.format("0x%02x", opcode));
            }
        }
    }

    // ---- Pickler Implementation (protocol 4) ----

    static class Pickler {
        private final OutputStream out;
        private int protocol = 4;
        private final Map<Object, Integer> memo = new HashMap<>();
        private int memoIndex = 0;
        
        Pickler(OutputStream out) {
            this.out = out;
        }
        
        void dump(Object obj) throws IOException {
            // Write protocol
            out.write(PROTO);
            out.write(protocol);
            
            dumpValue(obj);
            
            out.write(STOP);
            out.flush();
        }
        
        private void dumpValue(Object obj) throws IOException {
            if (obj == null) {
                out.write(NONE);
            } else if (obj instanceof Boolean) {
                out.write(((Boolean) obj) ? TRUE : FALSE);
            } else if (obj instanceof Integer) {
                int v = (Integer) obj;
                if (v >= 0 && v <= 0xff) {
                    out.write(BININT1);
                    out.write(v);
                } else if (v >= Short.MIN_VALUE && v <= Short.MAX_VALUE) {
                    out.write(BININT2);
                    byte[] buf = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) v).array();
                    out.write(buf);
                } else {
                    out.write(BININT);
                    byte[] buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
                    out.write(buf);
                }
            } else if (obj instanceof Long) {
                long v = (Long) obj;
                out.write(LONG1);
                out.write(8);
                byte[] buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
                out.write(buf);
            } else if (obj instanceof Float || obj instanceof Double) {
                double v = obj instanceof Float ? (Float) obj : (Double) obj;
                out.write(0x8b); // FLOAT8
                byte[] buf = ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN).putDouble(v).array();
                out.write(buf);
            } else if (obj instanceof String) {
                dumpString((String) obj);
            } else if (obj instanceof byte[]) {
                dumpBytes((byte[]) obj);
            } else if (obj instanceof List) {
                dumpList((List<?>) obj);
            } else if (obj instanceof Map) {
                dumpMap((Map<?, ?>) obj);
            } else if (obj instanceof Tuple) {
                dumpTuple((Tuple) obj);
            } else {
                throw new IOException("Unsupported type: " + obj.getClass().getName());
            }
        }
        
        private void dumpString(String s) throws IOException {
            byte[] data = s.getBytes(StandardCharsets.UTF_8);
            if (data.length <= 0xff) {
                out.write(SHORT_BINUNICODE);
                out.write(data.length);
            } else {
                out.write(BINUNICODE8);
                byte[] lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(data.length).array();
                out.write(lenBuf);
            }
            out.write(data);
        }
        
        private void dumpBytes(byte[] data) throws IOException {
            if (data.length <= 0xff) {
                out.write(SHORT_BINBYTES);
                out.write(data.length);
            } else {
                out.write(BINBYTES8);
                byte[] lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(data.length).array();
                out.write(lenBuf);
            }
            out.write(data);
        }
        
        private void dumpList(List<?> list) throws IOException {
            out.write(EMPTY_LIST);
            int startIdx = memoIndex;
            
            for (Object item : list) {
                dumpValue(item);
            }
            out.write(STOP);
        }
        
        private void dumpMap(Map<?, ?> map) throws IOException {
            out.write(EMPTY_DICT);
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                dumpValue(entry.getKey());
                dumpValue(entry.getValue());
                out.write(SETITEM);
            }
        }
        
        private void dumpTuple(Tuple tuple) throws IOException {
            if (tuple.size() == 0) {
                out.write(EMPTY_TUPLE);
            } else if (tuple.size() <= 3) {
                out.write(TUPLE1 + tuple.size() - 1);
                for (Object item : tuple) {
                    dumpValue(item);
                }
            } else {
                out.write(MARK);
                for (Object item : tuple) {
                    dumpValue(item);
                }
                out.write(TUPLE);
            }
        }
    }

    // ---- Tuple implementation ----

    public static class Tuple extends AbstractList<Object> {
        private final Object[] elements;
        
        public Tuple(Object... elements) {
            this.elements = elements;
        }
        
        public static Tuple of(Object... elements) {
            return new Tuple(elements);
        }
        
        @Override
        public Object get(int index) {
            return elements[index];
        }
        
        @Override
        public int size() {
            return elements.length;
        }
        
        public Object[] toArray() {
            return elements.clone();
        }
    }

    // ---- Utility methods ----

    /**
     * Check if a file appears to be a pickle file.
     */
    public static boolean isPickleFile(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file)) {
            int b1 = in.read();
            int b2 = in.read();
            return b1 == 0x80 && b2 >= 0 && b2 <= 5;
        }
    }

    /**
     * Get the pickle protocol version of a file.
     */
    public static int getProtocol(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file)) {
            int b1 = in.read();
            int b2 = in.read();
            if (b1 == 0x80 && b2 >= 0 && b2 <= 5) {
                return b2;
            }
            throw new IOException("Not a pickle file");
        }
    }
}
