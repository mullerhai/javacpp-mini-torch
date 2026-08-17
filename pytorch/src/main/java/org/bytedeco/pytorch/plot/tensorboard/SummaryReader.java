/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.plot.tensorboard;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Predicate;

import static org.bytedeco.pytorch.plot.tensorboard.SummaryReader.ProtoReader.WIRE_LEN;

/**
 * Reads the TensorBoard event files written by {@link SummaryWriter}.
 *
 * <p>Provides a pull-based iterator over {@link Event} records and
 * convenience collectors per tag. Strictly mirrors the symmetric surface
 * of {@code SummaryWriter} so any tag written via
 * {@code add_scalar / add_histogram / add_image / add_audio / add_text /
 * add_pr_curve / add_mesh / add_tensor} can be read back.</p>
 *
 * <p>The on-disk record format is the same as the TensorFlow RecordWriter:
 * </p>
 * <pre>
 *   uint64 little-endian length
 *   uint32 little-endian masked-crc32c(length)
 *   length bytes of payload
 *   uint32 little-endian masked-crc32c(payload)
 * </pre>
 * <p>Payload is an {@code Event} message:</p>
 * <ul>
 *   <li>1 wall_time (double)</li>
 *   <li>2 step (int64)</li>
 *   <li>3 file_version (string)</li>
 *   <li>5 summary (Summary)</li>
 *   <li>10 session_log (SessionLog)</li>
 * </ul>
 * <p>Each {@code Summary.value} contains a tagged {@code Value} with at most
 * one of: simple_value(4 double), histo(7 message), image(8 message),
 * audio(9 message), tensor(10 message), mesh(13 message).</p>
 *
 * <pre>{@code
 * try (SummaryReader r = SummaryReader.open(logDir)) {
 *     r.scalars("train/loss").forEach((step, v) -> System.out.println(step + " " + v));
 *     r.histograms("layer1/weights").forEach((step, h) -> ...);
 *     r.images("val/sample").forEach((step, img) -> ...);
 * }
 * }</pre>
 */
public final class SummaryReader implements AutoCloseable {

    /** Parsed Event record. */
    public static final class Event {
        public final double wallTime;
        public final long step;
        public final String fileVersion;
        public final List<TaggedValue> values;

        public Event(double wallTime, long step, String fileVersion, List<TaggedValue> values) {
            this.wallTime = wallTime;
            this.step = step;
            this.fileVersion = fileVersion;
            this.values = values == null ? Collections.emptyList() : values;
        }
    }

    /** A single tagged Summary value — exactly one of {@code payload*} will be non-null. */
    public static final class TaggedValue {
        public final String tag;
        public final Double simpleValue;     // field 4
        public final Histogram histogram;    // field 7
        public final ImageProto image;        // field 8
        public final AudioProto audio;        // field 9
        public final TensorData tensor;       // field 10
        public final MeshData mesh;           // field 13
        public final Metadata metadata;       // field 9 within Value (SummaryMetadata)

        TaggedValue(String tag, Double s, Histogram h, ImageProto img, AudioProto au,
                    TensorData t, MeshData m, Metadata md) {
            this.tag = tag;
            this.simpleValue = s;
            this.histogram = h;
            this.image = img;
            this.audio = au;
            this.tensor = t;
            this.mesh = m;
            this.metadata = md;
        }
    }

    /** Tagged value plus its enclosing event step (returned by reader iterators). */
    public static final class SteppedValue {
        public final String tag;
        public final long step;
        public final Double simpleValue;
        public final Histogram histogram;
        public final ImageProto image;
        public final AudioProto audio;
        public final TensorData tensor;
        public final MeshData mesh;
        public final Metadata metadata;

        public SteppedValue(String tag, long step, Double s, Histogram h, ImageProto img,
                            AudioProto au, TensorData t, MeshData m, Metadata md) {
            this.tag = tag;
            this.step = step;
            this.simpleValue = s;
            this.histogram = h;
            this.image = img;
            this.audio = au;
            this.tensor = t;
            this.mesh = m;
            this.metadata = md;
        }
    }

    /** SummaryMetadata { plugin_data { plugin_name, content }, display_name, summary_description, data_class }. */
    public static final class Metadata {
        public final String pluginName;
        public final byte[] pluginContent;
        public final String displayName;
        public final String summaryDescription;
        public final int dataClass;

        Metadata(String pluginName, byte[] pluginContent, String displayName,
                 String summaryDescription, int dataClass) {
            this.pluginName = pluginName;
            this.pluginContent = pluginContent;
            this.displayName = displayName;
            this.summaryDescription = summaryDescription;
            this.dataClass = dataClass;
        }
    }

    /** Decoded histogram (matches SummaryWriter.add_histogram payload). */
    public static final class Histogram {
        public final double min;
        public final double max;
        public final long num;
        public final double sum;
        public final double sumSquares;
        /** Bucket limits — TensorBoard proto uses right-edge convention (n+1 edges for n buckets). */
        public final double[] bucketLimits;
        /** Bucket counts. */
        public final double[] bucketCounts;

        public Histogram(double min, double max, long num, double sum, double sumSquares,
                         double[] bucketLimits, double[] bucketCounts) {
            this.min = min;
            this.max = max;
            this.num = num;
            this.sum = sum;
            this.sumSquares = sumSquares;
            this.bucketLimits = bucketLimits;
            this.bucketCounts = bucketCounts;
        }
    }

    /** Decoded image. Width / Height / Channels reflect HWC interpretation; raw tensor preserved. */
    public static final class ImageProto {
        public final int height;
        public final int width;
        public final int channels;
        /** Decoded tensor shape (e.g. {H, W, C}). */
        public final long[] tensorShape;
        public final byte[] encodedImage;     // PNG-encoded preview if present
        public final float[] floatData;       // raw float values (length = H*W*C if available)
        public final double[] doubleData;
        public final int[] intData;

        public ImageProto(int h, int w, int c, long[] shape, byte[] enc, float[] f, double[] d, int[] i) {
            this.height = h;
            this.width = w;
            this.channels = c;
            this.tensorShape = shape;
            this.encodedImage = enc;
            this.floatData = f;
            this.doubleData = d;
            this.intData = i;
        }
    }

    /** Decoded audio clip. */
    public static final class AudioProto {
        public final byte[] encodedAudio;     // WAV (typically) bytes
        public final float[] floatData;
        public final long[] tensorShape;

        public AudioProto(byte[] enc, float[] data, long[] shape) {
            this.encodedAudio = enc;
            this.floatData = data;
            this.tensorShape = shape;
        }
    }

    /** Decoded tensor. */
    public static final class TensorData {
        public final int dtype;          // matches ProtoWire.DT_*
        public final long[] shape;
        public final float[] floats;
        public final double[] doubles;
        public final int[] ints;
        public final long[] longs;
        public final byte[] stringData;
        public final byte[] tensorContent;

        public TensorData(int dtype, long[] shape, float[] f, double[] d, int[] i, long[] l,
                          byte[] s, byte[] content) {
            this.dtype = dtype;
            this.shape = shape;
            this.floats = f;
            this.doubles = d;
            this.ints = i;
            this.longs = l;
            this.stringData = s;
            this.tensorContent = content;
        }
    }

    /** Decoded mesh component (one per Value; combine VERTEX/FACE/COLOR by tag suffix). */
    public static final class MeshData {
        /** "VERTEX", "FACE", "COLOR" or null when unknown. */
        public final String contentType;
        public final String tagSuffix;
        public final long[] shape;
        public final float[] data;
        public final int components;
        public final String configJson;

        public MeshData(String contentType, String tagSuffix, long[] shape, float[] data,
                        int components, String configJson) {
            this.contentType = contentType;
            this.tagSuffix = tagSuffix;
            this.shape = shape;
            this.data = data;
            this.components = components;
            this.configJson = configJson;
        }
    }

    /** Strategy for files containing invalid CRCs. */
    public enum ErrorMode { STRICT, LENIENT, BEST_EFFORT }

    private final InputStream in;
    private final Path path;
    private final boolean ownStream;
    private final ErrorMode errorMode;
    private final boolean verifyCrc;
    private long bytesRead;
    private long eventsRead;
    private boolean closed;
    // Eagerly buffered events for multi-pass iteration.
    private final List<Event> buffered;

    private SummaryReader(InputStream in, Path path, boolean ownStream,
                          ErrorMode errorMode, boolean verifyCrc) throws IOException {
        this.in = in;
        this.path = path;
        this.ownStream = ownStream;
        this.errorMode = errorMode;
        this.verifyCrc = verifyCrc;
        // Eagerly drain the stream so the reader supports multiple iteration passes.
        this.buffered = new ArrayList<>();
        if (in != null) {
            Event e;
            while ((e = readNextEvent()) != null) {
                buffered.add(e);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------------

    /** Open a single event file (any {@code events.out.tfevents.*}). */
    public static SummaryReader open(Path eventFile) throws IOException {
        return open(eventFile, ErrorMode.LENIENT, true);
    }

    public static SummaryReader open(Path eventFile, ErrorMode errorMode, boolean verifyCrc) throws IOException {
        Objects.requireNonNull(eventFile, "eventFile");
        if (!Files.exists(eventFile)) {
            throw new IOException("Event file not found: " + eventFile);
        }
        InputStream in = new BufferedInputStream(new FileInputStream(eventFile.toFile()), 64 * 1024);
        return new SummaryReader(in, eventFile, true, errorMode, verifyCrc);
    }

    /** Wrap a user-supplied stream. Length-prefixed TFRecord format expected. The stream is
     *  fully drained at open time so the reader supports multiple iteration passes. */
    public static SummaryReader of(InputStream in) throws IOException {
        return new SummaryReader(Objects.requireNonNull(in, "in"), null, false,
                ErrorMode.LENIENT, true);
    }

    // -----------------------------------------------------------------------
    // State accessors
    // -----------------------------------------------------------------------

    public Path filePath() { return path; }
    public long bytesRead() { return bytesRead; }
    public long eventsRead() { return buffered.size(); }
    public ErrorMode errorMode() { return errorMode; }
    public boolean isClosed() { return closed; }
    /** Read-only view of the in-memory event buffer. */
    public List<Event> bufferedEvents() { return Collections.unmodifiableList(buffered); }

    // -----------------------------------------------------------------------
    // Iteration
    // -----------------------------------------------------------------------

    /** Iterate every Event in the file. */
    public Iterator<Event> events() {
        return new java.util.AbstractList<Event>() {
            @Override public Event get(int index) { return buffered.get(index); }
            @Override public int size() { return buffered.size(); }
        }.iterator();
    }

    /** Iterate only events whose Summary contains a value matching {@code tagFilter}. */
    public Iterator<Event> events(Predicate<String> tagFilter) {
        List<Event> filtered = new ArrayList<>();
        for (Event e : buffered) {
            for (TaggedValue v : e.values) {
                if (tagFilter == null || tagFilter.test(v.tag)) { filtered.add(e); break; }
            }
        }
        return filtered.iterator();
    }

    /** Iterate {@link TaggedValue}s across all events. */
    public Iterator<SteppedValue> values() {
        return new ValueIterator(events(), null);
    }

    public Iterator<SteppedValue> values(String tagPrefix) {
        return new ValueIterator(events(), tagPrefix == null ? null : tagPrefix::startsWith);
    }

    public Iterator<SteppedValue> values(Predicate<String> tagFilter) {
        return new ValueIterator(events(), tagFilter);
    }

    // -----------------------------------------------------------------------
    // Collectors
    // -----------------------------------------------------------------------

    /** All scalars for the given exact tag, as parallel {@code [step, value]} arrays in step order. */
    public List<double[]> scalars(String tag) {
        List<double[]> out = new ArrayList<>();
        forEachScalar(tag, (step, v) -> out.add(new double[]{step, v}));
        Collections.sort(out, (a, b) -> Double.compare(a[0], b[0]));
        return out;
    }

    /** Streamed callback API for large logs (step, value). */
    public void forEachScalar(String tag, java.util.function.BiConsumer<Long, Double> cb) {
        Objects.requireNonNull(cb, "cb");
        Iterator<SteppedValue> it = values(tag);
        while (it.hasNext()) {
            SteppedValue v = it.next();
            if (v.simpleValue != null) cb.accept(v.step, v.simpleValue);
        }
    }

    public List<long[]> histogramSteps(String tag) {
        List<long[]> out = new ArrayList<>();
        forEachHistogram(tag, (step, h) -> out.add(new long[]{step}));
        return out;
    }

    public void forEachHistogram(String tag, java.util.function.BiConsumer<Long, Histogram> cb) {
        Objects.requireNonNull(cb, "cb");
        Iterator<SteppedValue> it = values(tag);
        while (it.hasNext()) {
            SteppedValue v = it.next();
            if (v.histogram != null) cb.accept(v.step, v.histogram);
        }
    }

    public List<ImageProto> images(String tag) {
        List<ImageProto> out = new ArrayList<>();
        for (Iterator<SteppedValue> it = values(tag); it.hasNext(); ) {
            SteppedValue v = it.next();
            if (v.image != null) out.add(v.image);
        }
        return out;
    }

    public List<AudioProto> audios(String tag) {
        List<AudioProto> out = new ArrayList<>();
        for (Iterator<SteppedValue> it = values(tag); it.hasNext(); ) {
            SteppedValue v = it.next();
            if (v.audio != null) out.add(v.audio);
        }
        return out;
    }

    public List<TensorData> tensors(String tag) {
        List<TensorData> out = new ArrayList<>();
        for (Iterator<SteppedValue> it = values(tag); it.hasNext(); ) {
            SteppedValue v = it.next();
            if (v.tensor != null) out.add(v.tensor);
        }
        return out;
    }

    public List<MeshData> meshes(String tag) {
        List<MeshData> out = new ArrayList<>();
        for (Iterator<SteppedValue> it = values(tag); it.hasNext(); ) {
            SteppedValue v = it.next();
            if (v.mesh != null) out.add(v.mesh);
        }
        return out;
    }

    /** Distinct tags that have been written. */
    public List<String> tags() {
        java.util.LinkedHashSet<String> set = new java.util.LinkedHashSet<>();
        Iterator<SteppedValue> it = values();
        while (it.hasNext()) set.add(it.next().tag);
        return new ArrayList<>(set);
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            if (ownStream) in.close();
        } catch (IOException ignored) {
        }
    }

    // -----------------------------------------------------------------------
    // Internals: streaming read (used at open time only)
    // -----------------------------------------------------------------------

    Event readNextEvent() throws IOException {
        if (closed) throw new IOException("Reader closed");
        // 8 bytes length LE
        byte[] lenBytes = readFully(8);
        if (lenBytes == null) return null; // EOF
        long length = ByteBuffer.wrap(lenBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
        if (length < 0 || length > 1L << 30) {
            throw new IOException("Invalid record length: " + length);
        }
        int crcLen = readIntLE();
        if (verifyCrc) {
            int expected = Crc32C.maskedCrc32c(lenBytes);
            if (expected != crcLen) handleError("CRC mismatch on length prefix (got " + crcLen + ", expected " + expected + ")");
        }
        byte[] payload = readFully((int) length);
        if (payload == null) throw new IOException("Truncated record payload");
        int crcData = readIntLE();
        if (verifyCrc) {
            int expected = Crc32C.maskedCrc32c(payload);
            if (expected != crcData) handleError("CRC mismatch on payload (got " + crcData + ", expected " + expected + ")");
        }
        Event e = parseEvent(payload);
        if (e != null) eventsRead++;
        return e;
    }

    private void handleError(String msg) throws IOException {
        switch (errorMode) {
            case STRICT:
                throw new IOException(msg);
            case LENIENT:
                System.err.println("[SummaryReader] " + msg);
                break;
            case BEST_EFFORT:
            default:
                // silent
                break;
        }
    }

    private byte[] readFully(int n) throws IOException {
        byte[] buf = new byte[n];
        int off = 0;
        while (off < n) {
            int r = in.read(buf, off, n - off);
            if (r < 0) {
                if (off == 0) return null;
                throw new IOException("Truncated read: got " + off + "/" + n);
            }
            off += r;
        }
        bytesRead += n;
        return buf;
    }

    private int readIntLE() throws IOException {
        byte[] b = readFully(4);
        if (b == null) return -1;
        return (b[0] & 0xff) | ((b[1] & 0xff) << 8) | ((b[2] & 0xff) << 16) | ((b[3] & 0xff) << 24);
    }

    // -----------------------------------------------------------------------
    // Event parsing
    // -----------------------------------------------------------------------

    static Event parseEvent(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        double wall = 0;
        long step = 0;
        String fileVersion = null;
        List<TaggedValue> values = null;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 1:
                    wall = r.readDouble();
                    break;
                case 2:
                    step = r.readVarintLong();
                    break;
                case 3:
                    fileVersion = r.readString();
                    break;
                case 5: {
                    byte[] sumBytes = r.readBytes();
                    values = parseSummary(sumBytes);
                    break;
                }
                default:
                    r.skipField();
            }
        }
        return new Event(wall, step, fileVersion, values == null ? Collections.emptyList() : values);
    }

    static List<TaggedValue> parseSummary(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        List<TaggedValue> out = new ArrayList<>();
        while (r.nextField()) {
            if (r.fieldNumber == 1) {
                byte[] vBytes = r.readBytes();
                out.add(parseValue(vBytes));
            } else {
                r.skipField();
            }
        }
        return out;
    }

    static TaggedValue parseValue(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        String tag = null;
        Double simple = null;
        Histogram histo = null;
        ImageProto image = null;
        AudioProto audio = null;
        TensorData tensor = null;
        MeshData mesh = null;
        Metadata meta = null;
        // Buffer field-8 (tensor vs mesh) until we see field-9 metadata.
        byte[] field8 = null;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 1: tag = r.readString(); break;
                case 2: simple = r.readDouble(); break;  // simple_value (double, wire 1)
                case 4: image = parseImage(r.readBytes()); break;
                case 5: histo = parseHistogram(r.readBytes()); break;
                case 6: audio = parseAudio(r.readBytes()); break;
                case 8: field8 = r.readBytes(); break;
                case 9: meta = parseMetadata(r.readBytes()); break;
                default: r.skipField();
            }
        }
        if (field8 != null) {
            if (meta != null && "mesh".equalsIgnoreCase(meta.pluginName)) {
                mesh = parseMesh(field8, meta);
            } else {
                tensor = parseTensor(field8);
            }
        }
        return new TaggedValue(tag == null ? "" : tag, simple, histo, image, audio, tensor, mesh, meta);
    }

    static Metadata parseMetadata(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        String pluginName = null;
        byte[] pluginContent = null;
        String displayName = null;
        String desc = null;
        int dataClass = 0;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 1:
                    // Could be display_name (string, wire 2) or plugin_data (message, wire 2).
                    // Try as plugin_data first when wire is LEN.
                    byte[] b = r.readBytes();
                    if (b.length > 0) {
                        // Heuristic: plugin_data has plugin_name (string) at field 1; metadata has display_name at field 2.
                        ProtoReader rr = new ProtoReader(b);
                        while (rr.nextField()) {
                            if (rr.fieldNumber == 1 && rr.wireType == WIRE_LEN) {
                                pluginName = rr.readString();
                            } else if (rr.fieldNumber == 2 && rr.wireType == WIRE_LEN) {
                                pluginContent = rr.readBytes();
                            } else {
                                rr.skipField();
                            }
                        }
                    }
                    break;
                case 2: displayName = r.readString(); break;
                case 3: desc = r.readString(); break;
                case 4: dataClass = (int) r.readVarintLong(); break;
                default: r.skipField();
            }
        }
        return new Metadata(pluginName, pluginContent, displayName, desc, dataClass);
    }

    static Histogram parseHistogram(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        double min = Double.NaN, max = Double.NaN, sum = 0, sumSq = 0;
        long num = 0;
        java.util.List<Double> lim = new java.util.ArrayList<>();
        java.util.List<Double> cnt = new java.util.ArrayList<>();
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 1: min = r.readDouble(); break;
                case 2: max = r.readDouble(); break;
                case 3: num = r.readVarintLong(); break;
                case 4: sum = r.readDouble(); break;
                case 5: sumSq = r.readDouble(); break;
                case 6: lim.add(r.readDouble()); break;  // non-packed repeated double
                case 7: cnt.add(r.readDouble()); break;
                default: r.skipField();
            }
        }
        double[] limA = new double[lim.size()];
        for (int i = 0; i < limA.length; i++) limA[i] = lim.get(i);
        double[] cntA = new double[cnt.size()];
        for (int i = 0; i < cntA.length; i++) cntA[i] = cnt.get(i);
        return new Histogram(min, max, num, sum, sumSq, limA, cntA);
    }

    static ImageProto parseImage(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        int h = 0, w = 0, c = 0;
        long[] shape = null;
        byte[] enc = null;
        float[] fvals = null;
        double[] dvals = null;
        int[] ivals = null;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 1: h = (int) r.readVarintLong(); break;
                case 2: w = (int) r.readVarintLong(); break;
                case 3: c = (int) r.readVarintLong(); break;
                case 4: enc = r.readBytes(); break;
                default: r.skipField();
            }
        }
        return new ImageProto(h, w, c, shape, enc, fvals, dvals, ivals);
    }

    static AudioProto parseAudio(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        byte[] enc = null;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 4: enc = r.readBytes(); break; // encoded_audio
                default: r.skipField();
            }
        }
        return new AudioProto(enc, null, null);
    }

    static TensorData parseTensor(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        int dtype = 0;
        long[] shape = null;
        float[] f = null;
        double[] d = null;
        int[] i = null;
        long[] l = null;
        java.util.List<byte[]> strings = null;
        byte[] content = null;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 1: dtype = (int) r.readVarintLong(); break;
                case 2: shape = parseTensorShape(r.readBytes()); break;
                case 4: content = r.readBytes(); break;
                case 5: f = r.readPackedFloat(); break;
                case 6: d = r.readPackedDouble(); break;
                case 7: i = r.readPackedInt32(); break;
                case 8: {
                    if (strings == null) strings = new java.util.ArrayList<>();
                    strings.add(r.readBytes());
                    break;
                }
                case 10: l = r.readPackedInt64(); break;
                default: r.skipField();
            }
        }
        byte[] s = null;
        if (strings != null && !strings.isEmpty()) {
            // Concatenate with newlines as a textual fallback (matches Python TB).
            int total = 0;
            for (byte[] bs : strings) total += bs.length + 1;
            byte[] all = new byte[Math.max(0, total - 1)];
            int off = 0;
            for (int idx = 0; idx < strings.size(); idx++) {
                byte[] bs = strings.get(idx);
                System.arraycopy(bs, 0, all, off, bs.length);
                off += bs.length;
                if (idx < strings.size() - 1 && off < all.length) all[off++] = '\n';
            }
            s = all;
        }
        return new TensorData(dtype, shape, f, d, i, l, s, content);
    }

    static MeshData parseMesh(byte[] payload, Metadata meta) {
        ProtoReader r = new ProtoReader(payload);
        long[] shape = null;
        float[] data = null;
        while (r.nextField()) {
            switch (r.fieldNumber) {
                case 2: shape = parseTensorShape(r.readBytes()); break;
                case 5: data = r.readPackedFloat(); break;
                default: r.skipField();
            }
        }
        String ct = null;
        int components = 0;
        String json = null;
        if (meta != null && meta.pluginContent != null) {
            ProtoReader pr = new ProtoReader(meta.pluginContent);
            while (pr.nextField()) {
                switch (pr.fieldNumber) {
                    case 3: ct = contentTypeName((int) pr.readVarintLong()); break;
                    case 5: json = pr.readString(); break;
                    case 7: components = (int) pr.readVarintLong(); break;
                    default: pr.skipField();
                }
            }
        }
        return new MeshData(ct, null, shape, data, components, json);
    }

    private static String contentTypeName(int n) {
        switch (n) {
            case 1: return "VERTEX";
            case 2: return "FACE";
            case 3: return "COLOR";
            default: return "TYPE_" + n;
        }
    }

    static long[] parseTensorShape(byte[] payload) {
        ProtoReader r = new ProtoReader(payload);
        java.util.List<Long> dims = new ArrayList<>();
        while (r.nextField()) {
            if (r.fieldNumber == 2) {
                // Dim.size field 1
                byte[] dimBytes = r.readBytes();
                ProtoReader rr = new ProtoReader(dimBytes);
                while (rr.nextField()) {
                    if (rr.fieldNumber == 1) dims.add(rr.readVarintLong());
                    else rr.skipField();
                }
            } else r.skipField();
        }
        long[] out = new long[dims.size()];
        for (int i = 0; i < out.length; i++) out[i] = dims.get(i);
        return out;
    }

    // -----------------------------------------------------------------------
    // Mini protobuf decoder (subset)
    // -----------------------------------------------------------------------

    static final class ProtoReader {
        public static final int WIRE_VARINT = 0;
        public static final int WIRE_64 = 1;
        public static final int WIRE_LEN = 2;
        public static final int WIRE_32 = 5;
        final byte[] buf;
        int pos;
        int fieldNumber;
        int wireType;
        ProtoReader(byte[] buf) {
            this.buf = buf;
            this.pos = 0;
        }

        boolean hasMore() {
            return pos < buf.length;
        }

        boolean nextField() {
            if (pos >= buf.length) return false;
            long tag = readVarintLong();
            wireType = (int) (tag & 0x7);
            fieldNumber = (int) (tag >>> 3);
            return true;
        }

        long readVarintLong() {
            long result = 0;
            int shift = 0;
            while (pos < buf.length) {
                int b = buf[pos++] & 0xff;
                result |= ((long) (b & 0x7f)) << shift;
                if ((b & 0x80) == 0) return result;
                shift += 7;
                if (shift >= 64) throw new RuntimeException("Varint too long");
            }
            throw new RuntimeException("Truncated varint");
        }

        int readVarintInt() {
            return (int) readVarintLong();
        }

        double readDouble() {
            ensure(8);
            long bits = readLongLE();
            return Double.longBitsToDouble(bits);
        }

        float readFloat() {
            ensure(4);
            int bits = readIntLE();
            return Float.intBitsToFloat(bits);
        }

        int readFixed32() {
            ensure(4);
            return readIntLE();
        }

        long readFixed64() {
            ensure(8);
            return readLongLE();
        }

        private void ensure(int n) {
            if (pos + n > buf.length) throw new RuntimeException("Truncated fixed field");
        }

        private int readIntLE() {
            int b0 = buf[pos++] & 0xff;
            int b1 = buf[pos++] & 0xff;
            int b2 = buf[pos++] & 0xff;
            int b3 = buf[pos++] & 0xff;
            return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }

        private long readLongLE() {
            long b0 = buf[pos++] & 0xff;
            long b1 = buf[pos++] & 0xff;
            long b2 = buf[pos++] & 0xff;
            long b3 = buf[pos++] & 0xff;
            long b4 = buf[pos++] & 0xff;
            long b5 = buf[pos++] & 0xff;
            long b6 = buf[pos++] & 0xff;
            long b7 = buf[pos++] & 0xff;
            return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                    | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56);
        }

        byte[] readBytes() {
            int len = readVarintInt();
            if (len < 0 || pos + len > buf.length) throw new RuntimeException("Bad length: " + len);
            byte[] out = new byte[len];
            System.arraycopy(buf, pos, out, 0, len);
            pos += len;
            return out;
        }

        String readString() {
            byte[] b = readBytes();
            return new String(b, StandardCharsets.UTF_8);
        }

        float[] readPackedFloat() {
            byte[] raw = readBytes();
            float[] out = new float[raw.length / 4];
            for (int i = 0; i < out.length; i++) {
                int b0 = raw[i * 4] & 0xff;
                int b1 = raw[i * 4 + 1] & 0xff;
                int b2 = raw[i * 4 + 2] & 0xff;
                int b3 = raw[i * 4 + 3] & 0xff;
                out[i] = Float.intBitsToFloat(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
            }
            return out;
        }

        double[] readPackedDouble() {
            byte[] raw = readBytes();
            double[] out = new double[raw.length / 8];
            for (int i = 0; i < out.length; i++) {
                long bits = 0;
                for (int b = 0; b < 8; b++) {
                    bits |= ((long) (raw[i * 8 + b] & 0xff)) << (b * 8);
                }
                out[i] = Double.longBitsToDouble(bits);
            }
            return out;
        }

        int[] readPackedInt32() {
            byte[] raw = readBytes();
            java.io.ByteArrayInputStream is = new java.io.ByteArrayInputStream(raw);
            java.util.List<Integer> list = new ArrayList<>();
            while (is.available() > 0) {
                long v = 0;
                int shift = 0;
                while (true) {
                    int b = is.read();
                    if (b < 0) break;
                    v |= ((long) (b & 0x7f)) << shift;
                    if ((b & 0x80) == 0) break;
                    shift += 7;
                }
                list.add((int) v);
            }
            int[] out = new int[list.size()];
            for (int i = 0; i < out.length; i++) out[i] = list.get(i);
            return out;
        }

        long[] readPackedInt64() {
            byte[] raw = readBytes();
            java.io.ByteArrayInputStream is = new java.io.ByteArrayInputStream(raw);
            java.util.List<Long> list = new ArrayList<>();
            while (is.available() > 0) {
                long v = 0;
                int shift = 0;
                while (true) {
                    int b = is.read();
                    if (b < 0) break;
                    v |= ((long) (b & 0x7f)) << shift;
                    if ((b & 0x80) == 0) break;
                    shift += 7;
                }
                list.add(v);
            }
            long[] out = new long[list.size()];
            for (int i = 0; i < out.length; i++) out[i] = list.get(i);
            return out;
        }

        void skipField() {
            switch (wireType) {
                case 0: readVarintLong(); break;
                case 1: pos += 8; break;
                case 2: readBytes(); break;
                case 5: pos += 4; break;
                default: throw new RuntimeException("Unknown wire type: " + wireType);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Value iterator (over in-memory event buffer)
    // -----------------------------------------------------------------------

    private static final class ValueIterator implements Iterator<SteppedValue> {
        private final Iterator<Event> events;
        private final Predicate<String> tagFilter;
        private Iterator<TaggedValue> current;
        private SteppedValue next;
        private long attachedStep = -1;
        private ValueIterator(Iterator<Event> events, Predicate<String> tagFilter) {
            this.events = events;
            this.tagFilter = tagFilter;
        }

        @Override public boolean hasNext() {
            while (next == null) {
                while (current == null || !current.hasNext()) {
                    if (!events.hasNext()) return false;
                    Event e = events.next();
                    attachedStep = e.step;
                    current = e.values.iterator();
                }
                TaggedValue v = current.next();
                if (tagFilter == null || tagFilter.test(v.tag)) {
                    next = new SteppedValue(v.tag, attachedStep, v.simpleValue, v.histogram,
                            v.image, v.audio, v.tensor, v.mesh, v.metadata);
                }
            }
            return true;
        }

        @Override public SteppedValue next() {
            if (!hasNext()) throw new NoSuchElementException();
            SteppedValue v = next;
            next = null;
            return v;
        }
    }
}