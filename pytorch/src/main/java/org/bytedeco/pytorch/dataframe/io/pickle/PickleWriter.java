package org.bytedeco.pytorch.dataframe.io.pickle;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Enterprise-grade Pickle writer for Python pickle format.
 * 
 * <p>Writes DataFrames to Python pickle format with support for:
 * <ul>
 *   <li>Multiple protocol versions (0-5)</li>
 *   <li>DataFrame as list of dicts</li>
 *   <li>DataFrame as pandas DataFrame format</li>
 *   <li>Compressed output (gzip)</li>
 * </ul>
 */
public class PickleWriter {

    private PickleWriter() {}

    /**
     * Write DataFrame to pickle format.
     */
    public static void write(DataFrame df, String path) throws IOException {
        write(df, path, PickleOptions.defaults());
    }

    public static void write(DataFrame df, String path, PickleOptions options) throws IOException {
        PickleOptions opt = options == null ? PickleOptions.defaults() : options;
        
        byte[] data;
        
        if (opt.compress()) {
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
                 OutputStream gzip = new java.util.zip.GZIPOutputStream(baos)) {
                writePickleData(df, gzip, opt);
                data = baos.toByteArray();
            }
        } else {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            writePickleData(df, baos, opt);
            data = baos.toByteArray();
        }
        
        Files.write(Path.of(path), data);
    }

    private static void writePickleData(DataFrame df, OutputStream out, PickleOptions opt) throws IOException {
        // Write pickle opcodes based on protocol
        int protocol = opt.protocol();
        
        // Protocol header
        writeOp(out, protocol >= 2 ? 0x80 : 0);  // PROTO
        writeInt(out, protocol, 1);
        
        if (opt.dataFrameFormat().equals(PickleOptions.DataFrameFormat.LIST_OF_DICTS)) {
            writeListOfDicts(df, out, protocol);
        } else if (opt.dataFrameFormat().equals(PickleOptions.DataFrameFormat.PANDAS)) {
            writePandasFormat(df, out, protocol);
        } else {
            writeTupleList(df, out, protocol);
        }
        
        // MARKER + STOP
        writeOp(out, 0x2e);  // STOP
    }

    private static void writeListOfDicts(DataFrame df, OutputStream out, int protocol) throws IOException {
        // Build list of dictionaries
        int nrows = df.rowCount();
        int ncols = df.columnCount();
        
        // Start empty list
        writeOp(out, 0x5d);  // EMPTY_LIST
        writeOp(out, 0x61);  // APPENDS
        
        for (int r = 0; r < nrows; r++) {
            // Push dict marker
            writeOp(out, 0x7d);  // EMPTY_DICT
            
            for (int c = 0; c < ncols; c++) {
                Column col = df.column(c);
                Object val = df.get(r, c);
                
                // Set item: key (MARKER + string) + value
                writeOp(out, 0x7c);  // MARKER
                writeString(out, col.name(), protocol);
                writeValue(out, val, col.dtype(), protocol);
                writeOp(out, 0x29);  // SETITEM
            }
            
            writeOp(out, 0x61);  // APPEND
        }
        
        writeOp(out, 0x5d);  // LIST terminator
    }

    private static void writeTupleList(DataFrame df, OutputStream out, int protocol) throws IOException {
        int nrows = df.rowCount();
        int ncols = df.columnCount();
        
        // Create empty list
        writeOp(out, 0x5d);  // EMPTY_LIST
        writeOp(out, 0x61);  // APPENDS
        
        for (int r = 0; r < nrows; r++) {
            // Create tuple for row
            if (protocol >= 2) {
                writeOp(out, 0x94);  // MEMOIZE (protocol 2+)
                // Tuple building with MARKER
            }
            
            for (int c = 0; c < ncols; c++) {
                Column col = df.column(c);
                Object val = df.get(r, c);
                writeValue(out, val, col.dtype(), protocol);
            }
            
            // Tuple
            writeOp(out, 0x74);  // TUPLE
            
            writeOp(out, 0x61);  // APPEND
        }
        
        writeOp(out, 0x5d);  // LIST terminator
    }

    private static void writePandasFormat(DataFrame df, OutputStream out, int protocol) throws IOException {
        // Simplified pandas format - stores as dict with columns
        // This mimics pandas.DataFrame.from_dict(orient='columns')
        
        writeOp(out, protocol >= 2 ? 0x80 : 0);
        writeInt(out, protocol, 1);
        
        // EMPTY_DICT
        writeOp(out, 0x7d);
        
        // Store columns
        writeOp(out, 0x7c);  // MARKER
        writeString(out, "columns", protocol);
        
        // List of column names
        writeOp(out, 0x5d);
        for (int c = 0; c < df.columnCount(); c++) {
            writeString(out, df.column(c).name(), protocol);
        }
        writeOp(out, 0x5d);
        writeOp(out, 0x71);  // SETITEM
        
        // Store data as dict of arrays
        writeOp(out, 0x7c);
        writeString(out, "data", protocol);
        
        writeOp(out, 0x7d);  // EMPTY_DICT
        for (int c = 0; c < df.columnCount(); c++) {
            Column col = df.column(c);
            writeOp(out, 0x7c);
            writeString(out, col.name(), protocol);
            
            // List of values
            writeOp(out, 0x5d);
            for (int r = 0; r < df.rowCount(); r++) {
                writeValue(out, df.get(r, c), col.dtype(), protocol);
            }
            writeOp(out, 0x5d);
            
            writeOp(out, 0x71);  // SETITEM
        }
        
        writeOp(out, 0x71);  // SETITEM
        
        writeOp(out, 0x2e);  // STOP
    }

    private static void writeValue(OutputStream out, Object val, Column.DType dtype, int protocol) throws IOException {
        if (val == null) {
            writeOp(out, 0x4e);  // NONE
            return;
        }
        
        switch (dtype) {
            case BOOLEAN:
                writeBoolean(out, val);
                break;
            case INT8:
            case INT16:
            case INT32:
                writeInt(out, ((Number) val).intValue(), 4);
                break;
            case UINT8:
            case UINT16:
            case UINT32:
                writeInt(out, ((Number) val).intValue(), 4);
                break;
            case INT64:
            case UINT64:
                writeInt(out, ((Number) val).longValue(), 8);
                break;
            case FLOAT32:
                writeFloat(out, ((Number) val).floatValue());
                break;
            case FLOAT64:
                writeDouble(out, ((Number) val).doubleValue());
                break;
            case STRING:
                writeString(out, val.toString(), protocol);
                break;
            case BINARY:
                writeBytes(out, (byte[]) val);
                break;
            default:
                writeString(out, val.toString(), protocol);
        }
    }

    private static void writeOp(OutputStream out, int opcode) throws IOException {
        out.write(opcode);
    }

    private static void writeInt(OutputStream out, long val, int size) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(size);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        if (size == 1) buf.put((byte) val);
        else if (size == 2) buf.putShort((short) val);
        else if (size == 4) buf.putInt((int) val);
        else if (size == 8) buf.putLong(val);
        out.write(buf.array());
    }

    private static void writeFloat(OutputStream out, float val) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(4);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putFloat(val);
        out.write(buf.array());
    }

    private static void writeDouble(OutputStream out, double val) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putDouble(val);
        out.write(buf.array());
    }

    private static void writeBoolean(OutputStream out, Object val) throws IOException {
        if (Boolean.TRUE.equals(val)) {
            writeOp(out, 0x54);  // TRUE
        } else {
            writeOp(out, 0x46);  // FALSE
        }
    }

    private static void writeString(OutputStream out, String s, int protocol) throws IOException {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        if (protocol >= 3) {
            writeOp(out, 0x60);  // BINSTRING
            writeInt(out, bytes.length, 4);
            out.write(bytes);
        } else {
            writeOp(out, 0x53);  // STRING
            writeBytes(out, bytes);
            writeOp(out, 0x74);  // TSTRING
        }
    }

    private static void writeBytes(OutputStream out, byte[] bytes) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(4);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(bytes.length);
        out.write(buf.array());
        out.write(bytes);
    }

    // ====================== Options ======================

    public enum DataFrameFormat { LIST_OF_DICTS, TUPLE_LIST, PANDAS }

    public static class PickleOptions {
        private int protocol = 4;
        private boolean compress = false;
        private DataFrameFormat dataFrameFormat = DataFrameFormat.LIST_OF_DICTS;

        public static PickleOptions defaults() { return new PickleOptions(); }
        
        public static PickleOptions records() {
            return new PickleOptions().dataFrameFormat(DataFrameFormat.LIST_OF_DICTS);
        }
        
        public static PickleOptions pandas() {
            return new PickleOptions().dataFrameFormat(DataFrameFormat.PANDAS);
        }

        public PickleOptions protocol(int p) { this.protocol = Math.max(0, Math.min(5, p)); return this; }
        public PickleOptions compress(boolean c) { this.compress = c; return this; }
        public PickleOptions dataFrameFormat(DataFrameFormat f) { this.dataFrameFormat = f; return this; }

        public int protocol() { return protocol; }
        public boolean compress() { return compress; }
        public DataFrameFormat dataFrameFormat() { return dataFrameFormat; }
    }
}
