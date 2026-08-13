package org.bytedeco.pytorch.data.pickle;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Pure Java implementation of Python's pickle protocol 0-5.
 * Based on CPython's pickle.py: https://github.com/python/cpython/blob/main/Lib/pickle.py
 */
public class Pickle {

    // Protocol 0-3
    private static final int MARK            = 0x28; // '('
    private static final int STOP           = 0x2e; // '.'
    private static final int POP            = 0x30; // '0'
    private static final int POP_MARK      = 0x31; // '1'
    private static final int DUP            = 0x32; // '2'
    private static final int FLOAT          = 0x46; // 'F'
    private static final int INT           = 0x49; // 'I'
    private static final int BININT         = 0x4a; // 'J'
    private static final int BININT1        = 0x4b; // 'K'
    private static final int BININT2       = 0x4d; // 'M'
    private static final int LONG          = 0x4c; // 'L'
    private static final int NONE          = 0x4e; // 'N'
    private static final int PERSID         = 0x50; // 'P'
    private static final int BINPERSID    = 0x51; // 'Q'
    private static final int REDUCE        = 0x52; // 'R'
    private static final int STRING        = 0x53; // 'S'
    private static final int BINSTRING    = 0x54; // 'T'
    private static final int SHORT_BINSTRING = 0x55; // 'U'
    private static final int UNICODE      = 0x56; // 'V'
    private static final int BINUNICODE   = 0x58; // 'X'
    private static final int APPEND       = 0x61; // 'a'
    private static final int BUILD        = 0x62; // 'b'
    private static final int GLOBAL        = 0x63; // 'c'
    private static final int DICT         = 0x64; // 'd'
    private static final int APPENDS      = 0x65; // 'e'
    private static final int GET          = 0x67; // 'g'
    private static final int BINGET       = 0x68; // 'h'
    private static final int INST         = 0x69; // 'i'
    private static final int LONG_BINGET  = 0x6a; // 'j'
    private static final int LIST         = 0x6c; // 'l'
    private static final int EMPTY_LIST   = 0x5d; // ']'
    private static final int OBJ          = 0x6f; // 'o'
    private static final int EMPTY_TUPLE  = 0x29; // ')'
    private static final int PUT          = 0x70; // 'p'
    private static final int BINPUT       = 0x71; // 'q'
    private static final int LONG_BINPUT  = 0x72; // 'r'
    private static final int SETITEM      = 0x73; // 's'
    private static final int TUPLE        = 0x74; // 't'
    private static final int SETITEMS     = 0x75; // 'u'
    private static final int EMPTY_DICT   = 0x7d; // '}'
    private static final int EMPTY_STRING = 0x60; // '`'
    private static final int BINFLOAT     = 0x47; // 'G'
    private static final int SHORT_BINBYTES = 0x43; // 'C'
    private static final int BINBYTES     = 0x42; // 'B'

    // Protocol 2
    private static final int PROTO        = 0x80; // '\x80'
    private static final int NEWOBJ       = 0x81; // '\x81'
    private static final int EXT1         = 0x82; // '\x82'
    private static final int EXT2         = 0x83; // '\x83'
    private static final int EXT4         = 0x84; // '\x84'
    private static final int TUPLE1       = 0x85; // '\x85'
    private static final int TUPLE2       = 0x86; // '\x86'
    private static final int TUPLE3       = 0x87; // '\x87'
    private static final int NEWTRUE      = 0x88; // '\x88'
    private static final int NEWFALSE     = 0x89; // '\x89'
    private static final int LONG1        = 0x8a; // '\x8a'
    private static final int LONG4        = 0x8b; // '\x8b'

    // Protocol 4
    private static final int SHORT_BINUNICODE = 0x8c; // '\x8c'
    private static final int BINUNICODE8   = 0x8d; // '\x8d'
    private static final int EMPTY_SET    = 0x8f;  // '\x8f'
    private static final int ADDITEMS     = 0x90;  // '\x90'
    private static final int FROZENSET   = 0x91;  // '\x91'
    private static final int NEWOBJ_EX   = 0x92;  // '\x92'
    private static final int STACK_GLOBAL = 0x93;  // '\x93'
    private static final int MEMOIZE     = 0x94;  // '\x94'
    private static final int FRAME       = 0x95;  // '\x95'

    // Protocol 5
    private static final int BINBYTES8    = 0x8e;  // '\x8e'
    private static final int BYTEARRAY8   = 0x96;  // '\x96'
    private static final int NEXT_BUFFER  = 0x97;  // '\x97'
    private static final int READONLY_BUFFER = 0x98; // '\x98'

    private Pickle() {}

    // ---- Public API ----

    public static Object load(File file) throws IOException {
        try (InputStream in = new FileInputStream(file)) {
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
        try (OutputStream out = new FileOutputStream(file)) {
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

    // ---- MARK sentinel ----
    static final Object MARK_SENTINEL = new Object() {
        @Override public String toString() { return "MARK"; }
    };

    // ---- Reader hierarchy ----
    static abstract class Reader {
        abstract int read() throws IOException;
        int read1() throws IOException {
            int b = read();
            if (b < 0) throw new IOException("Unexpected end of input");
            return b;
        }
        byte[] readBytes(int n) throws IOException {
            byte[] buf = new byte[n];
            for (int i = 0; i < n; i++) {
                int b = read();
                if (b < 0) throw new IOException("Unexpected end of input reading " + n + " bytes (got " + i + ")");
                buf[i] = (byte) b;
            }
            return buf;
        }
        String readDecimalLine() throws IOException {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int b;
            while ((b = read()) != '\n') {
                if (b < 0) throw new IOException("Unexpected end of input in decimal line");
                baos.write(b);
            }
            return baos.toString().trim();
        }
    }

    static final class TopReader extends Reader {
        PushbackInputStream in;
        @Override int read() throws IOException { return in.read(); }
    }

    static final class FrameReader extends Reader {
        private final PushbackInputStream in;
        private final int size;
        private int pos = 0;
        final Reader returnTo;
        FrameReader(PushbackInputStream in, int size, Reader returnTo) {
            this.in = in; this.size = size; this.returnTo = returnTo;
        }
        @Override int read() throws IOException {
            if (pos >= size) return -1;
            int b = in.read();
            if (b < 0) return -1;
            pos++;
            return b & 0xff;
        }
    }

    // ---- Unpickler ----
    static final class Unpickler {
        private final PushbackInputStream pin;
        // Use ArrayList instead of ArrayDeque to allow null values
        private final ArrayList<Object> stack = new ArrayList<>();
        private final List<Object> memo = new ArrayList<>();
        private int memoCursor = 0;
        private int protocol = 4;
        final TopReader top = new TopReader();

        Unpickler(InputStream in) {
            this.pin = (in instanceof PushbackInputStream) ? (PushbackInputStream) in : new PushbackInputStream(in, 16);
            top.in = this.pin;
        }

        Object load() throws IOException {
            int op = top.read1();
            if (op != PROTO) throw new IOException("Expected PROTO, got 0x" + Integer.toHexString(op));
            int proto = top.read1();
            if (proto < 0 || proto > 5) throw new IOException("Unsupported protocol: " + proto);
            this.protocol = proto;

            Reader r = top;
            boolean gotStop = false;
            while (true) {
                int next = r.read();
                if (next < 0) {
                    // End of frame or stream
                    if (r instanceof FrameReader) {
                        r = ((FrameReader) r).returnTo;
                        continue;
                    }
                    // EOF at top level
                    break;
                }
                if (next == FRAME) {
                    long size = readLong64(r);
                    if (size > Integer.MAX_VALUE) throw new IOException("FRAME too large: " + size);
                    r = new FrameReader(pin, (int) size, r);
                    continue;
                }
                if (next == STOP) {
                    gotStop = true;
                    if (r instanceof FrameReader) {
                        r = ((FrameReader) r).returnTo;
                        continue;
                    }
                    break;
                }
                if (next == PROTO) { r.read1(); continue; }
                dispatch(next, r);
            }
            if (stack.isEmpty()) return null;
            return stack.get(stack.size() - 1);
        }

        private long readLong64(Reader r) throws IOException {
            byte[] b = r.readBytes(8);
            return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).getLong();
        }

        private int findMark() {
            Object[] arr = stack.toArray();
            for (int i = arr.length - 1; i >= 0; i--) {
                if (arr[i] == MARK_SENTINEL) return i;
            }
            return -1;
        }

        private void dispatch(int op, Reader r) throws IOException {
            switch (op) {
                // ---- Primitives ----
                case NONE: push(null); break;
                case NEWTRUE: push(Boolean.TRUE); break;
                case NEWFALSE: push(Boolean.FALSE); break;

                case INT: {
                    String s = r.readDecimalLine();
                    if (s.equals("0")) push(Boolean.FALSE);
                    else if (s.equals("01")) push(Boolean.TRUE);
                    else push(Integer.parseInt(s));
                    break;
                }
                case BININT: push(ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt()); break;
                case BININT1: push(r.read1()); break;
                case BININT2: push(ByteBuffer.wrap(r.readBytes(2)).order(ByteOrder.LITTLE_ENDIAN).getShort() & 0xFFFF); break;
                case LONG: {
                    String s = r.readDecimalLine();
                    if (!s.isEmpty() && s.charAt(s.length() - 1) == 'L') s = s.substring(0, s.length() - 1);
                    push(Long.parseLong(s));
                    break;
                }
                case LONG1: {
                    int len = r.read1();
                    push(decodeLong(r.readBytes(len)));
                    break;
                }
                case LONG4: {
                    int len = ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    if (len < 0) throw new IOException("LONG4 negative: " + len);
                    push(decodeLong(r.readBytes(len)));
                    break;
                }
                case FLOAT: {
                    String s = r.readDecimalLine();
                    if (s.equals("0")) push(Boolean.FALSE);
                    else if (s.equals("01")) push(Boolean.TRUE);
                    else push(Double.parseDouble(s));
                    break;
                }
                case BINFLOAT: push(ByteBuffer.wrap(r.readBytes(8)).order(ByteOrder.BIG_ENDIAN).getDouble()); break;

                // ---- String / bytes ----
                case STRING: {
                    byte[] line = r.readBytes(r.read() == '\n' ? 0 : -1);
                    if (line.length >= 2 && line[0] == line[line.length-1] && (line[0] == '"' || line[0] == '\'')) {
                        line = Arrays.copyOfRange(line, 1, line.length - 1);
                    }
                    push(processEscape(new String(line, StandardCharsets.UTF_8)));
                    break;
                }
                case EMPTY_STRING: push(""); break;
                case BINSTRING: {
                    int len = ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    if (len < 0) throw new IOException("BINSTRING negative: " + len);
                    push(new String(r.readBytes(len), StandardCharsets.ISO_8859_1));
                    break;
                }
                case SHORT_BINSTRING: {
                    int len = r.read1();
                    push(new String(r.readBytes(len), StandardCharsets.ISO_8859_1));
                    break;
                }
                case BINBYTES: {
                    int len = ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    if (len < 0) throw new IOException("BINBYTES negative: " + len);
                    push(r.readBytes(len));
                    break;
                }
                case SHORT_BINBYTES: {
                    int len = r.read1();
                    push(r.readBytes(len));
                    break;
                }
                case BINBYTES8: {
                    long len = readLong64(r);
                    if (len < 0 || len > Integer.MAX_VALUE) throw new IOException("BINBYTES8 invalid: " + len);
                    push(r.readBytes((int) len));
                    break;
                }
                case UNICODE: push(new String(r.readBytes(r.read() == '\n' ? 0 : -1), StandardCharsets.UTF_8)); break;
                case SHORT_BINUNICODE: {
                    int len = r.read1();
                    push(new String(r.readBytes(len), StandardCharsets.UTF_8));
                    break;
                }
                case BINUNICODE: {
                    int len = ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    push(new String(r.readBytes(len), StandardCharsets.UTF_8));
                    break;
                }
                case BINUNICODE8: {
                    long len = readLong64(r);
                    if (len < 0 || len > Integer.MAX_VALUE) throw new IOException("BINUNICODE8 invalid: " + len);
                    push(new String(r.readBytes((int) len), StandardCharsets.UTF_8));
                    break;
                }

                // ---- Collections ----
                case EMPTY_TUPLE: push(new Tuple()); break;
                case EMPTY_LIST: push(new ArrayList<>()); break;
                case EMPTY_DICT: push(new LinkedHashMap<>()); break;
                case EMPTY_SET: push(new LinkedHashSet<>()); break;

                case TUPLE1: { Object a = pop(); push(new Tuple(a)); break; }
                case TUPLE2: { Object b = pop(); Object a = pop(); push(new Tuple(a, b)); break; }
                case TUPLE3: { Object c = pop(); Object b = pop(); Object a = pop(); push(new Tuple(a, b, c)); break; }
                case TUPLE: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("TUPLE: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] arr = new Object[n];
                    for (int i = n - 1; i >= 0; i--) arr[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    push(new Tuple(arr));
                    break;
                }
                case LIST: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("LIST: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] arr = new Object[n];
                    for (int i = n - 1; i >= 0; i--) arr[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    push(new ArrayList<>(Arrays.asList(arr)));
                    break;
                }
                case DICT: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("DICT: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] arr = new Object[n];
                    for (int i = n - 1; i >= 0; i--) arr[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    if ((arr.length & 1) != 0) throw new IOException("DICT odd count");
                    Map<Object, Object> m = new LinkedHashMap<>();
                    for (int i = 0; i < arr.length; i += 2) m.put(arr[i], arr[i + 1]);
                    push(m);
                    break;
                }
                case FROZENSET: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("FROZENSET: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] arr = new Object[n];
                    for (int i = n - 1; i >= 0; i--) arr[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    push(Collections.unmodifiableSet(new LinkedHashSet<>(Arrays.asList(arr))));
                    break;
                }

                case APPEND: {
                    Object v = pop();
                    Object lst = pop();
                    if (!(lst instanceof List)) throw new IOException("APPEND on non-list");
                    ((List<Object>) lst).add(v);
                    push(lst);
                    break;
                }
                case APPENDS: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("APPENDS: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] items = new Object[n];
                    for (int i = n - 1; i >= 0; i--) items[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    if (stack.isEmpty()) throw new IOException("APPENDS: stack empty");
                    Object lst = pop();
                    if (!(lst instanceof List)) throw new IOException("APPENDS on non-list: " + lst.getClass().getName());
                    for (Object item : items) ((List<Object>) lst).add(item);
                    push(lst);
                    break;
                }
                case SETITEM: {
                    Object v = pop();
                    Object k = pop();
                    Object d = pop();
                    if (!(d instanceof Map)) throw new IOException("SETITEM on non-dict");
                    ((Map<Object, Object>) d).put(k, v);
                    push(d);
                    break;
                }
                case SETITEMS: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("SETITEMS: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] items = new Object[n];
                    for (int i = n - 1; i >= 0; i--) items[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    if (stack.isEmpty()) throw new IOException("SETITEMS: stack empty");
                    Object d = peekt();
                    if (d instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<Object, Object> m = (Map<Object, Object>) d;
                        if ((items.length & 1) != 0) throw new IOException("SETITEMS odd count");
                        for (int i = 0; i < items.length; i += 2) m.put(items[i], items[i + 1]);
                    } else if (d instanceof PythonObject) {
                        // PythonObject: store items as a list for later __setstate__
                        if ((items.length & 1) != 0) throw new IOException("SETITEMS odd count");
                        Map<Object, Object> dict = new LinkedHashMap<>();
                        for (int i = 0; i < items.length; i += 2) dict.put(items[i], items[i + 1]);
                        ((PythonObject) d).setState(dict);
                    }
                    break;
                }
                case ADDITEMS: {
                    int mi = findMark();
                    if (mi < 0) throw new IOException("ADDITEMS: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] items = new Object[n];
                    for (int i = n - 1; i >= 0; i--) items[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    if (stack.isEmpty()) throw new IOException("ADDITEMS: stack empty");
                    Object s = peekt();
                    if (!(s instanceof Set)) throw new IOException("ADDITEMS on non-set");
                    for (Object item : items) ((Set<Object>) s).add(item);
                    break;
                }

                case MARK: stack.add(MARK_SENTINEL); break;

                // ---- Memo ----
                case MEMOIZE: {
                    Object top = peekt();
                    if (top != null) memoize(memoCursor++, top);
                    break;
                }
                case PUT: {
                    int idx = Integer.parseInt(r.readDecimalLine());
                    Object top = peekt();
                    if (top != null) memoize(idx, top);
                    memoCursor = Math.max(memoCursor, idx + 1);
                    break;
                }
                case BINPUT: {
                    int idx = r.read1();
                    Object top = peekt();
                    if (top != null) memoize(idx, top);
                    memoCursor = Math.max(memoCursor, idx + 1);
                    break;
                }
                case LONG_BINPUT: {
                    int idx = ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt();
                    Object top = peekt();
                    if (top != null) memoize(idx, top);
                    memoCursor = Math.max(memoCursor, idx + 1);
                    break;
                }
                case GET: push(memoRef(Integer.parseInt(r.readDecimalLine()))); break;
                case BINGET: push(memoRef(r.read1())); break;
                case LONG_BINGET: push(memoRef(ByteBuffer.wrap(r.readBytes(4)).order(ByteOrder.LITTLE_ENDIAN).getInt())); break;

                // ---- Object construction ----
                case GLOBAL: {
                    String module = r.readDecimalLine();
                    String name = r.readDecimalLine();
                    push(new QualifiedClass(module, name));
                    break;
                }
                case STACK_GLOBAL: {
                    String name = (String) pop();
                    String module = (String) pop();
                    push(new QualifiedClass(module, name));
                    break;
                }
                case INST: {
                    String module = r.readDecimalLine();
                    String name = r.readDecimalLine();
                    int mi = findMark();
                    if (mi < 0) throw new IOException("INST: no MARK");
                    int n = stack.size() - mi - 1;
                    Object[] arr = new Object[n];
                    for (int i = n - 1; i >= 0; i--) arr[i] = stack.remove(stack.size() - 1);
                    stack.remove(stack.size() - 1);
                    push(new PythonObject(new QualifiedClass(module, name), new Tuple(arr)));
                    break;
                }
                case REDUCE: {
                    Object args = pop();
                    Object callable = pop();
                    push(new PythonObject(callable, args));
                    break;
                }
                case NEWOBJ: {
                    Object args = pop();
                    Object cls = pop();
                    push(new PythonObject(cls, args));
                    break;
                }
                case OBJ: {
                    Object args = pop();
                    Object cls = pop();
                    push(new PythonObject(cls, args));
                    break;
                }
                case BUILD: {
                    Object state = pop();
                    Object target = peekt();
                    if (target == null) {
                        push(state);
                    } else if (target instanceof PythonObject) {
                        ((PythonObject) target).setState(state);
                    } else if (target instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<Object, Object> m = (Map<Object, Object>) target;
                        if (state instanceof Map) {
                            @SuppressWarnings("unchecked")
                            Map<Object, Object> s = (Map<Object, Object>) state;
                            m.putAll(s);
                        }
                    }
                    break;
                }

                case STOP: return;

                default: throw new IOException("Unsupported opcode: 0x" + String.format("%02x", op));
            }
        }

        private void push(Object o) { stack.add(o); }
        private Object pop() throws IOException {
            if (stack.isEmpty()) throw new IOException("Stack underflow");
            return stack.remove(stack.size() - 1);
        }
        private Object peekt() { return stack.isEmpty() ? null : stack.get(stack.size() - 1); }
        private int stackSize() { return stack.size(); }

        private void memoize(int idx, Object value) {
            while (memo.size() <= idx) memo.add(null);
            memo.set(idx, value);
        }
        private Object memoRef(int idx) throws IOException {
            if (idx < 0 || idx >= memo.size()) return new PythonObject(new QualifiedClass("", "<memo-missing>"), null);
            Object v = memo.get(idx);
            return v == null ? new PythonObject(new QualifiedClass("", "<memo-unset>"), null) : v;
        }

        private static long decodeLong(byte[] b) {
            long v = 0;
            for (int i = 0; i < b.length; i++) v |= ((long) (b[i] & 0xff)) << (8 * i);
            return v;
        }

        private static String processEscape(String s) {
            StringBuilder sb = new StringBuilder();
            int i = 0;
            while (i < s.length()) {
                char c = s.charAt(i++);
                if (c == '\\' && i < s.length()) {
                    char e = s.charAt(i++);
                    switch (e) {
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case '\\': sb.append('\\'); break;
                        case 'x': if (i + 1 < s.length()) { try { sb.append((char) Integer.parseInt(s.substring(i, i + 2), 16)); i += 2; } catch (NumberFormatException ex) { sb.append('x'); } } else { sb.append('x'); } break;
                        default: sb.append(e); break;
                    }
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }
    }

    // ---- Public types ----

    public static final class QualifiedClass {
        public final String module, name;
        public QualifiedClass(String module, String name) {
            this.module = module == null ? "" : module;
            this.name = name == null ? "" : name;
        }
        public String qualifiedName() { return module.isEmpty() ? name : (module + "." + name); }
        @Override public String toString() { return "<class '" + qualifiedName() + "'>"; }
        @Override public boolean equals(Object o) {
            if (!(o instanceof QualifiedClass)) return false;
            QualifiedClass q = (QualifiedClass) o;
            return module.equals(q.module) && name.equals(q.name);
        }
        @Override public int hashCode() { return module.hashCode() * 31 + name.hashCode(); }
    }

    public static final class PythonObject {
        private final Object callable;
        private final Object args;
        private Object state;
        PythonObject(Object callable, Object args) { this.callable = callable; this.args = args; }
        public Object callable() { return callable; }
        public Object args() { return args; }
        public Object state() { return state; }
        void setState(Object state) { this.state = state; }
        public Map<String, Object> dict() {
            Object s = state;
            if (s == null) return null;
            if (s instanceof Map) return toStringKeyed((Map<Object, Object>) s);
            if (s instanceof Tuple) { Tuple t = (Tuple) s; if (t.size() >= 1 && t.get(0) instanceof Map) return toStringKeyed((Map<Object, Object>) t.get(0)); }
            if (s instanceof List) { List<?> list = (List<?>) s; if (!list.isEmpty() && list.get(0) instanceof Map) return toStringKeyed((Map<Object, Object>) list.get(0)); }
            return null;
        }
        public String className() {
            if (callable instanceof QualifiedClass) return ((QualifiedClass) callable).qualifiedName();
            return String.valueOf(callable);
        }
        @Override public String toString() { return className() + "{state=" + state + "}"; }
        private static Map<String, Object> toStringKeyed(Map<Object, Object> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<Object, Object> e : m.entrySet()) out.put(String.valueOf(e.getKey()), e.getValue());
            return out;
        }
    }

    public static class Tuple extends AbstractList<Object> {
        private final Object[] elements;
        public Tuple(Object... elements) { this.elements = elements == null ? new Object[0] : elements; }
        public static Tuple of(Object... elements) { return new Tuple(elements); }
        @Override public Object get(int index) { return elements[index]; }
        @Override public int size() { return elements.length; }
        public Object[] toArray() { return elements.clone(); }
        @Override public int hashCode() { return Arrays.hashCode(elements); }
        @Override public boolean equals(Object o) { return o instanceof Tuple && Arrays.equals(elements, ((Tuple) o).elements); }
        @Override public String toString() {
            StringBuilder sb = new StringBuilder("(");
            for (int i = 0; i < elements.length; i++) { if (i > 0) sb.append(", "); sb.append(elements[i]); }
            if (elements.length == 1) sb.append(",");
            sb.append(")");
            return sb.toString();
        }
    }

    // ---- Pickler (protocol 4) ----
    static class Pickler {
        private final OutputStream out;
        private int protocol = 4;

        Pickler(OutputStream out) { this.out = out; }

        void dump(Object obj) throws IOException {
            out.write(PROTO);
            out.write(protocol);
            dumpValue(obj);
            out.write(STOP);
            out.flush();
        }

        private void dumpValue(Object obj) throws IOException {
            if (obj == null) out.write(NONE);
            else if (obj instanceof Boolean) out.write(((Boolean) obj) ? NEWTRUE : NEWFALSE);
            else if (obj instanceof Integer) {
                int v = (Integer) obj;
                if (v >= 0 && v <= 0xff) { out.write(BININT1); out.write(v); }
                else { out.write(BININT); out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array()); }
            } else if (obj instanceof Long) {
                byte[] data = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong((Long) obj).array();
                out.write(LONG1); out.write(data.length); out.write(data);
            } else if (obj instanceof Float || obj instanceof Double) {
                double v = obj instanceof Float ? (Float) obj : (Double) obj;
                out.write(BINFLOAT); out.write(ByteBuffer.allocate(8).order(ByteOrder.BIG_ENDIAN).putDouble(v).array());
            } else if (obj instanceof String) dumpString((String) obj);
            else if (obj instanceof byte[]) dumpBytes((byte[]) obj);
            else if (obj instanceof List) dumpList((List<?>) obj);
            else if (obj instanceof Map) dumpMap((Map<?, ?>) obj);
            else if (obj instanceof Tuple) dumpTuple((Tuple) obj);
            else throw new IOException("Unsupported type: " + obj.getClass().getName());
        }

        private void dumpString(String s) throws IOException {
            byte[] data = s.getBytes(StandardCharsets.UTF_8);
            out.write(SHORT_BINUNICODE); out.write(data.length); out.write(data);
        }

        private void dumpBytes(byte[] data) throws IOException {
            if (data.length < 256) { out.write(SHORT_BINBYTES); out.write(data.length); }
            else { out.write(BINBYTES); out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(data.length).array()); }
            out.write(data);
        }

        private void dumpList(List<?> list) throws IOException {
            out.write(MARK); out.write(EMPTY_LIST);
            for (Object item : list) dumpValue(item);
            out.write(APPENDS);
        }

        private void dumpMap(Map<?, ?> map) throws IOException {
            out.write(MARK); out.write(EMPTY_DICT);
            for (Map.Entry<?, ?> entry : map.entrySet()) { dumpValue(entry.getKey()); dumpValue(entry.getValue()); }
            out.write(SETITEMS);
        }

        private void dumpTuple(Tuple tuple) throws IOException {
            if (tuple.size() == 0) out.write(EMPTY_TUPLE);
            else if (tuple.size() <= 3) { out.write(TUPLE1 + tuple.size() - 1); for (Object item : tuple) dumpValue(item); }
            else { out.write(MARK); for (Object item : tuple) dumpValue(item); out.write(TUPLE); }
        }
    }

    public static boolean isPickleFile(File file) throws IOException {
        try (InputStream in = new FileInputStream(file)) {
            int b1 = in.read(); int b2 = in.read();
            return b1 == 0x80 && b2 >= 0 && b2 <= 5;
        }
    }

    public static int getProtocol(File file) throws IOException {
        try (InputStream in = new FileInputStream(file)) {
            int b1 = in.read(); int b2 = in.read();
            if (b1 == 0x80 && b2 >= 0 && b2 <= 5) return b2;
            throw new IOException("Not a pickle file");
        }
    }
}
