package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

import org.bytedeco.pytorch.dataframe.io.config.MediaBackend;
import org.bytedeco.pytorch.dataframe.io.config.MediaBackend.Backend;

/**
 * Enterprise-grade reader for video folder datasets (UCF-101 / Kinetics-400 / AVA /
 * ActivityNet / generic {@code root/&lt;class&gt;/&lt;clip&gt;.&lt;ext&gt;}).
 *
 * <p>Expected directory layouts:</p>
 *
 * <pre>
 * 1. ImageFolder convention (default):
 *    root/
 *    ├── class_a/
 *    │   ├── clip_001.mp4
 *    │   ├── clip_002.mov
 *    │   └── ...
 *    ├── class_b/
 *    │   └── ...
 *    └── ...
 *
 * 2. CSV label map (VideoFolderOptions.labelMode(CSV)):
 *    root/
 *    ├── clips/*.mp4
 *    └── labels.csv           # columns: path,label[,class_index]
 *
 * 3. JSON metadata (VideoFolderOptions.labelMode(JSON)):
 *    root/
 *    ├── clips/*.mp4
 *    └── metadata.jsonl       # one row per clip, with at least "path" and "label"
 * </pre>
 *
 * <p>The reader reports per-file metadata (duration, fps, frame_count, width, height,
 * codec, audio flag) <b>best-effort</b> by sniffing the container header (MP4 / MOV /
 * MKV / WebM). It does <b>not</b> decode frames. Consumers (training loops, JVM-based
 * TorchVideo replacements, etc.) can pipe {@code relative_path} → frame decoder of
 * their choice.</p>
 *
 * <p>Example:</p>
 * <pre>
 *   DataFrame df = VideoFolderReader.read("/path/to/kinetics");
 *
 *   // With options
 *   VideoFolderOptions opts = VideoFolderOptions.defaults()
 *       .recursive(true)
 *       .includePath(true)
 *       .includeSize(true)
 *       .maxDurationSeconds(60.0)
 *       .frameMode(VideoFolderOptions.FrameMode.MIDDLE);
 *   DataFrame df = VideoFolderReader.read("/path/to/videos", opts);
 *
 *   // Via DataFrameReader
 *   DataFrame df = DataFrame.read().videofolder("/path/to/videos");
 * </pre>
 */
public class VideoFolderReader {

    private VideoFolderReader() {}

    // Common video container extensions (lower-case + upper-case).
    private static final Set<String> VIDEO_EXTENSIONS = Set.of(
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv",
        ".mpg", ".mpeg", ".mpe", ".m1v", ".m2v", ".mp2", ".mpv",
        ".ogv", ".ogg", ".3gp", ".3g2", ".ts", ".m2ts", ".mts",
        ".tsv", ".f4v", ".f4p", ".f4a", ".f4b",
        ".jpg", ".jpeg", ".png",  // video can ship as a frame sequence (image folders)
        ".MP4", ".MOV", ".MKV", ".WEBM", ".AVI", ".M4V", ".FLV", ".WMV",
        ".MPG", ".MPEG", ".OGV", ".OGG", ".3GP", ".TS", ".M2TS",
        ".JPG", ".JPEG", ".PNG"
    );

    // ---- public API -------------------------------------------------------

    public static DataFrame read(String path) throws IOException {
        return read(path, VideoFolderOptions.defaults());
    }

    public static DataFrame read(String path, VideoFolderOptions options) throws IOException {
        VideoFolderOptions opts = options == null ? VideoFolderOptions.defaults() : options;
        Path rootPath = Paths.get(path);
        if (!Files.exists(rootPath)) {
            throw new IOException("VideoFolder path does not exist: " + path);
        }
        if (!Files.isDirectory(rootPath)) {
            throw new IOException("VideoFolder path must be a directory: " + path);
        }

        // Resolve class labels ahead of traversal.
        ClassMap classMap = resolveClassMap(rootPath, opts);

        DataFrame df = DataFrame.create();
        df.addColumn("relative_path", Column.DType.STRING);
        df.addColumn("class_name", Column.DType.STRING);
        df.addColumn("class_index", Column.DType.INT32);
        if (opts.includePath())            df.addColumn("full_path", Column.DType.STRING);
        if (opts.includeSize())            df.addColumn("file_size", Column.DType.INT64);
        if (opts.includeExtension())       df.addColumn("extension", Column.DType.STRING);
        if (opts.includeModifiedTime())    df.addColumn("modified_time", Column.DType.INT64);
        if (opts.includeName())            df.addColumn("file_name", Column.DType.STRING);
        if (opts.includeMetadata()) {
            df.addColumn("duration_seconds", Column.DType.FLOAT64);
            df.addColumn("fps", Column.DType.FLOAT64);
            df.addColumn("frame_count", Column.DType.INT64);
            df.addColumn("width", Column.DType.INT32);
            df.addColumn("height", Column.DType.INT32);
            df.addColumn("codec", Column.DType.STRING);
            df.addColumn("has_audio", Column.DType.BOOLEAN);
        }
        if (opts.includeOrdinal())         df.addColumn("ordinal", Column.DType.INT64);
        if (opts.frameMode() != VideoFolderOptions.FrameMode.NONE) {
            df.addColumn("frame_mode", Column.DType.STRING);
        }
        if (opts.eagerDecode()) {
            if (opts.thumbnailCol() != null && !opts.thumbnailCol().isBlank()) {
                df.addColumn(opts.thumbnailCol(), Column.DType.TENSOR);
            }
            if (opts.framesCol() != null && !opts.framesCol().isBlank()) {
                df.addColumn(opts.framesCol(), Column.DType.TENSOR);
            }
        }

        // Discover classes (subdirectories) if labelMode == FOLDER.
        List<Path> classDirs = new ArrayList<>();
        if (opts.labelMode() == VideoFolderOptions.LabelMode.FOLDER) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootPath)) {
                for (Path entry : stream) {
                    if (Files.isDirectory(entry) && !entry.getFileName().toString().startsWith(".")) {
                        classDirs.add(entry);
                    }
                }
            }
            classDirs.sort(Comparator.comparing(p -> p.getFileName().toString()));
        }

        // Build class name → index map.
        Map<String, Integer> classIndexMap = new LinkedHashMap<>();
        if (opts.labelMode() == VideoFolderOptions.LabelMode.FOLDER) {
            for (int i = 0; i < classDirs.size(); i++) {
                classIndexMap.put(classDirs.get(i).getFileName().toString(), i);
            }
        } else {
            // Build from the resolved classMap so labels are consistent.
            for (String c : classMap.names()) {
                if (!classIndexMap.containsKey(c)) classIndexMap.put(c, classIndexMap.size());
            }
        }

        SeenSet seen = opts.unique() ? new SeenSet() : null;
        long ordinal = 0;

        // Walk: FOLDER mode walks each class dir; CSV/JSON mode walks the root.
        if (opts.labelMode() == VideoFolderOptions.LabelMode.FOLDER) {
            for (Map.Entry<String, Integer> entry : classIndexMap.entrySet()) {
                String className = entry.getKey();
                int classIdx = entry.getValue();
                Path classDir = classDirs.stream()
                        .filter(p -> p.getFileName().toString().equals(className))
                        .findFirst().orElse(null);
                if (classDir == null) continue;
                ordinal = collectVideos(df, classDir, rootPath, className, classIdx,
                        opts, ordinal, seen);
            }
        } else {
            ordinal = collectVideos(df, rootPath, rootPath, /*className*/ null, /*classIdx*/ -1,
                    opts, ordinal, seen);
        }

        return df;
    }

    /**
     * Get class names and counts without loading all video metadata.
     */
    public static Map<String, Long> getClassStats(String path) throws IOException {
        return getClassStats(path, VideoFolderOptions.defaults());
    }

    public static Map<String, Long> getClassStats(String path, VideoFolderOptions options)
            throws IOException {
        VideoFolderOptions opts = options == null ? VideoFolderOptions.defaults() : options;
        Path rootPath = Paths.get(path);
        Map<String, Long> stats = new LinkedHashMap<>();
        if (!Files.exists(rootPath)) return stats;

        if (opts.labelMode() == VideoFolderOptions.LabelMode.FOLDER) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootPath)) {
                for (Path entry : stream) {
                    if (Files.isDirectory(entry)) {
                        String className = entry.getFileName().toString();
                        long count = countVideos(entry, opts);
                        stats.put(className, count);
                    }
                }
            }
        } else {
            // Single flat namespace: count by class label derived from the label file.
            ClassMap classMap = resolveClassMap(rootPath, opts);
            Map<String, Long> byClass = new LinkedHashMap<>();
            try (java.util.stream.Stream<Path> walk = Files.walk(rootPath)) {
                java.util.Iterator<Path> it = walk.filter(Files::isRegularFile)
                        .filter(VideoFolderReader::isVideoFile)
                        .iterator();
                while (it.hasNext()) {
                    Path p = it.next();
                    if (opts.recursive() == false && rootPath.relativize(p).getNameCount() > 1) {
                        continue;
                    }
                    String rel = rootPath.relativize(p).toString();
                    String label = classMap.lookup(rel);
                    if (label == null) label = "<unknown>";
                    byClass.merge(label, 1L, Long::sum);
                }
            }
            stats.putAll(byClass);
        }
        return stats;
    }

    // ---- internals --------------------------------------------------------

    /** {@code relPath -> (label, classIndex)} where classIndex may be -1. */
    private static final class ClassMap {
        final Map<String, String> labels = new LinkedHashMap<>();
        final Map<String, Integer> indices = new LinkedHashMap<>();
        // Union of all distinct labels, used to build the class index.
        final Map<String, Integer> allNames = new LinkedHashMap<>();

        void put(String rel, String label, Integer idx) {
            labels.put(rel, label);
            if (idx != null) indices.put(rel, idx);
            if (label != null && !allNames.containsKey(label)) {
                allNames.put(label, allNames.size());
            }
        }

        String lookup(String rel) {
            return labels.get(rel);
        }

        int indexOf(String label) {
            Integer i = allNames.get(label);
            return i == null ? -1 : i;
        }

        List<String> names() {
            return new ArrayList<>(allNames.keySet());
        }

        boolean isEmpty() {
            return labels.isEmpty();
        }
    }

    private static ClassMap resolveClassMap(Path root, VideoFolderOptions opts) throws IOException {
        ClassMap m = new ClassMap();
        if (opts.labelMode() == VideoFolderOptions.LabelMode.FOLDER) {
            return m; // subdir scan will fill in
        }
        Path labelFile = opts.labelFile();
        if (labelFile == null) {
            labelFile = findDefaultLabelFile(root);
        }
        if (labelFile == null || !Files.isRegularFile(labelFile)) {
            throw new IOException("VideoFolder labelMode=" + opts.labelMode()
                    + " but no label file found at " + root
                    + " (expected labels.csv, metadata.jsonl, or metadata.json; "
                    + "use VideoFolderOptions.labelFile(Path) to override).");
        }
        String name = labelFile.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".csv")) {
            readCsvLabels(labelFile, m);
        } else {
            readJsonLabels(labelFile, m);
        }
        return m;
    }

    private static Path findDefaultLabelFile(Path root) {
        String[] candidates = {"labels.csv", "metadata.jsonl", "metadata.json", "annotations.json"};
        for (String c : candidates) {
            Path p = root.resolve(c);
            if (Files.isRegularFile(p)) return p;
        }
        return null;
    }

    private static void readCsvLabels(Path file, ClassMap m) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(file)) {
            String header = br.readLine();
            if (header == null) return;
            List<String> cols = splitCsv(header);
            int pathIdx = indexOf(cols, "path", "relpath", "relative_path", "file");
            int labelIdx = indexOf(cols, "label", "class", "class_name", "tag");
            int indexIdx = indexOf(cols, "class_index", "label_id", "index", "idx");
            if (pathIdx < 0 || labelIdx < 0) {
                throw new IOException("labels.csv " + file + " must have 'path' and 'label' columns (got: " + cols + ")");
            }
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty() || line.trim().isEmpty()) continue;
                List<String> cells = splitCsv(line);
                if (pathIdx >= cells.size() || labelIdx >= cells.size()) continue;
                String rel = cells.get(pathIdx).trim();
                String label = cells.get(labelIdx).trim();
                Integer idx = null;
                if (indexIdx >= 0 && indexIdx < cells.size()) {
                    try { idx = Integer.parseInt(cells.get(indexIdx).trim()); }
                    catch (NumberFormatException ignored) {}
                }
                m.put(rel, label, idx);
            }
        }
    }

    private static void readJsonLabels(Path file, ClassMap m) throws IOException {
        // Support both NDJSON (one {"path":...,"label":...} per line) and a single
        // { "rel/path": "...label...", ... } JSON object.
        String text = Files.readString(file).trim();
        if (text.startsWith("[")) {
            // JSON array of objects.
            int i = 0;
            while (i < text.length()) {
                while (i < text.length() && (Character.isWhitespace(text.charAt(i)) || text.charAt(i) == ',')) i++;
                if (i >= text.length() || text.charAt(i) == ']') break;
                if (text.charAt(i) != '{') { i++; continue; }
                int depth = 0, start = i;
                for (; i < text.length(); i++) {
                    char c = text.charAt(i);
                    if (c == '{') depth++;
                    else if (c == '}') { depth--; if (depth == 0) { i++; break; } }
                }
                Map<String, Object> obj = MiniJson.parseObject(text.substring(start, i));
                String rel = stringField(obj, "path", "relative_path", "file");
                String label = stringField(obj, "label", "class", "class_name");
                Integer idx = intField(obj, "class_index", "label_id", "index");
                if (rel != null) m.put(rel, label, idx);
            }
        } else if (text.startsWith("{")) {
            // NDJSON? multiple lines starting with '{'.
            if (text.contains("\n") && MiniJson.countTopLevels(text) > 1) {
                for (String line : text.split("\n")) {
                    String t = line.trim();
                    if (t.isEmpty() || !t.startsWith("{")) continue;
                    Map<String, Object> obj = MiniJson.parseObject(t);
                    String rel = stringField(obj, "path", "relative_path", "file");
                    String label = stringField(obj, "label", "class", "class_name");
                    Integer idx = intField(obj, "class_index", "label_id", "index");
                    if (rel != null) m.put(rel, label, idx);
                }
            } else {
                // Single object: keys are relative paths, values are labels (or nested objects).
                Map<String, Object> obj = MiniJson.parseObject(text);
                for (Map.Entry<String, Object> e : obj.entrySet()) {
                    Object v = e.getValue();
                    if (v instanceof String s) {
                        m.put(e.getKey(), s, null);
                    } else if (v instanceof Map<?, ?> vm) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> sub = (Map<String, Object>) vm;
                        m.put(e.getKey(), stringField(sub, "label", "class", "class_name"),
                                intField(sub, "class_index", "label_id", "index"));
                    }
                }
            }
        } else {
            throw new IOException("Unsupported label file format: " + file);
        }
    }

    private static String stringField(Map<String, Object> obj, String... keys) {
        for (String k : keys) {
            Object v = obj.get(k);
            if (v != null) return v.toString();
        }
        return null;
    }

    private static Integer intField(Map<String, Object> obj, String... keys) {
        for (String k : keys) {
            Object v = obj.get(k);
            if (v instanceof Number n) return n.intValue();
            if (v instanceof String s) {
                try { return Integer.parseInt(s.trim()); } catch (NumberFormatException ignored) {}
            }
        }
        return null;
    }

    private static int indexOf(List<String> cols, String... names) {
        for (int i = 0; i < cols.size(); i++) {
            String c = cols.get(i).trim().toLowerCase(Locale.ROOT);
            for (String n : names) if (c.equals(n)) return i;
        }
        return -1;
    }

    private static List<String> splitCsv(String line) {
        List<String> cells = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (inQ && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    cur.append('"'); i++;
                } else inQ = !inQ;
            } else if (c == ',' && !inQ) {
                cells.add(cur.toString()); cur.setLength(0);
            } else cur.append(c);
        }
        cells.add(cur.toString());
        return cells;
    }

    private static long collectVideos(DataFrame df, Path dir, Path rootPath,
                                      String className, int classIdx,
                                      VideoFolderOptions opts, long ordinal,
                                      SeenSet seen) throws IOException {
        ClassMap classMap = null;
        if (opts.labelMode() != VideoFolderOptions.LabelMode.FOLDER) {
            classMap = resolveClassMap(rootPath, opts);
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (opts.recursive()) {
                        ordinal = collectVideos(df, entry, rootPath, className, classIdx,
                                opts, ordinal, seen);
                    }
                } else if (isVideoFile(entry)) {
                    String rel = rootPath.relativize(entry).toString();
                    if (seen != null && !seen.add(rel)) continue;

                    String thisClass = className;
                    int thisIdx = classIdx;
                    if (classMap != null) {
                        String label = classMap.lookup(rel);
                        if (label != null) {
                            thisClass = label;
                            thisIdx = classMap.indexOf(label);
                        }
                    }
                    if (thisClass == null) thisClass = "<unknown>";

                    // maxBytes filter
                    if (opts.maxBytes() > 0) {
                        try {
                            long sz = Files.size(entry);
                            if (sz > opts.maxBytes()) continue;
                        } catch (IOException ignored) {}
                    }

                    int ri = df.addEmptyRow();
                    df.set(ri, "relative_path", rel);
                    df.set(ri, "class_name", thisClass);
                    df.set(ri, "class_index", thisIdx);
                    if (opts.includePath())         df.set(ri, "full_path", entry.toAbsolutePath().toString());
                    if (opts.includeSize())         df.set(ri, "file_size", Files.size(entry));
                    if (opts.includeExtension()) {
                        String n = entry.getFileName().toString();
                        int dot = n.lastIndexOf('.');
                        df.set(ri, "extension", dot > 0 ? n.substring(dot).toLowerCase(Locale.ROOT) : "");
                    }
                    if (opts.includeModifiedTime()) {
                        try {
                            df.set(ri, "modified_time",
                                    Files.getLastModifiedTime(entry).toMillis());
                        } catch (IOException e) {
                            df.set(ri, "modified_time", 0L);
                        }
                    }
                    if (opts.includeName())         df.set(ri, "file_name", entry.getFileName().toString());

                    if (opts.includeMetadata()) {
                        // Probe via FFmpeg when requested & available; fall back to pure-Java sniff.
                        VideoMeta m = probeVideo(entry, opts.backend());
                        df.set(ri, "duration_seconds", m.durationSeconds);
                        df.set(ri, "fps", m.fps);
                        df.set(ri, "frame_count", m.frameCount);
                        df.set(ri, "width", m.width);
                        df.set(ri, "height", m.height);
                        df.set(ri, "codec", m.codec);
                        df.set(ri, "has_audio", m.hasAudio);
                        if (opts.maxDurationSeconds() > 0 && m.durationSeconds > opts.maxDurationSeconds()) {
                            // remove the row we just added
                            removeRow(df, ri);
                            continue;
                        }
                    }
                    if (opts.includeOrdinal())      df.set(ri, "ordinal", ordinal);
                    if (opts.frameMode() != VideoFolderOptions.FrameMode.NONE) {
                        df.set(ri, "frame_mode", opts.frameMode().name());
                    }
                    if (opts.eagerDecode() && MediaBackend.resolve(opts.backend()) == Backend.FFMPEG_OPENCV) {
                        decodeIntoRow(df, ri, entry, opts);
                    }
                    ordinal++;
                }
            }
        }
        return ordinal;
    }

    private static boolean isVideoFile(Path p) {
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        return VIDEO_EXTENSIONS.stream().anyMatch(name::endsWith);
    }

    private static long countVideos(Path dir, VideoFolderOptions opts) throws IOException {
        long count = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (opts.recursive()) count += countVideos(entry, opts);
                } else if (isVideoFile(entry)) {
                    count++;
                }
            }
        }
        return count;
    }

    /** Best-effort row deletion that does not require a public DataFrame.removeRow(). */
    private static void removeRow(DataFrame df, int rowIndex) {
        try {
            // DataFrame stores columns as Column objects; clearing the cell at the
            // last index is sufficient for downstream consumers that only read up to
            // a rowCount() that we deliberately do not shrink here. To avoid leaking
            // an empty row, we leave the cells as null and let the caller drop
            // any analysis on this row.
            for (String c : df.getColumnNames()) {
                try { df.set(rowIndex, c, null); } catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {}
    }

    /** Tracks dedup of relative paths. */
    private static final class SeenSet {
        private final Set<String> seen = new HashSet<>();
        boolean add(String rel) { return seen.add(rel); }
    }

    // ---- header sniffing --------------------------------------------------

    /** Best-effort metadata extracted from container headers. */
    static final class VideoMeta {
        double durationSeconds = -1.0;
        double fps = -1.0;
        long frameCount = -1;
        int width = -1;
        int height = -1;
        String codec = null;
        boolean hasAudio = false;
        long bitRate = 0L;
    }

    static VideoMeta sniffVideo(Path path) {
        VideoMeta m = new VideoMeta();
        try {
            BasicFileAttributes attrs = Files.readAttributes(path, BasicFileAttributes.class);
            // Skip sniffing for very large files (>2 GiB) to avoid scanning the entire container.
            if (attrs.size() > 2L * 1024 * 1024 * 1024) return m;
        } catch (IOException ignored) {}

        try (java.io.InputStream in = Files.newInputStream(path)) {
            byte[] head = new byte[Math.min(2 * 1024 * 1024, (int) Math.min(2L * 1024 * 1024, safeSize(path)))];
            int total = 0;
            int n;
            while (total < head.length && (n = in.read(head, total, head.length - total)) > 0) {
                total += n;
            }
            if (total <= 0) return m;
            Sniff sniff = sniffContainer(head, total);
            m.codec = sniff.codec;
            m.width = sniff.width;
            m.height = sniff.height;
            m.durationSeconds = sniff.durationSeconds;
            m.fps = sniff.fps;
            m.frameCount = sniff.frameCount;
            m.hasAudio = sniff.hasAudio;
        } catch (IOException ignored) {}
        return m;
    }

    private static long safeSize(Path p) {
        try { return Files.size(p); } catch (IOException e) { return 0L; }
    }

    /** Probe video metadata. When the resolved backend is {@code FFMPEG_OPENCV},
     *  delegates to {@link MediaBackend#probeVideo(String)} for true width/height
     *  / fps / duration / codec / bit rate; otherwise falls back to the
     *  pure-Java {@link #sniffVideo(Path)} header parser. */
    static VideoMeta probeVideo(Path path, Backend backendOpt) {
        Backend backend = MediaBackend.resolve(backendOpt);
        if (backend == Backend.FFMPEG_OPENCV) {
            MediaBackend.VideoMeta m = MediaBackend.probeVideo(path.toString());
            if (m != null) {
                VideoMeta v = new VideoMeta();
                v.durationSeconds = m.durationSec;
                v.fps = m.fps;
                v.frameCount = m.numFrames;
                v.width = m.width;
                v.height = m.height;
                v.codec = m.codecName;
                v.hasAudio = false;
                v.bitRate = m.bitRate;
                return v;
            }
        }
        return sniffVideo(path);
    }

    /** Populate eager-decode columns for a row using FFmpeg. */
    private static void decodeIntoRow(DataFrame df, int ri, Path entry, VideoFolderOptions opts) {
        try (MediaBackend.AC vf = MediaBackend.openVideoHandle(entry.toString())) {
            if (vf == null) return;
            if (opts.thumbnailCol() != null && !opts.thumbnailCol().isBlank()) {
                Object thumb = reflectArgs(vf.handle, "thumbnail", null, null);
                if (thumb != null) df.set(ri, opts.thumbnailCol(), thumb);
            }
            if (opts.framesCol() != null && !opts.framesCol().isBlank()) {
                Object frames;
                if (opts.decodeMaxFrames() > 0) {
                    Object list = reflectArgs(vf.handle, "readFrames",
                            new Class<?>[]{int.class}, new Object[]{opts.decodeMaxFrames()});
                    frames = (list == null) ? null
                            : reflectStaticArgs("org.bytedeco.pytorch.vision.ffmpeg.VideoFile",
                                    "stackFrames", new Class<?>[]{List.class}, new Object[]{list});
                } else {
                    frames = reflectArgs(vf.handle, "read", null, null);
                }
                if (frames != null) df.set(ri, opts.framesCol(), frames);
                // Optional re-encode to a different container.
                if (opts.reencodeTo() != null && !opts.reencodeTo().isBlank() && frames != null) {
                    reencodeRow((java.util.List<?>) null, frames, opts, entry, ri);
                }
            }
        } catch (Throwable t) {
            // decode failure leaves cells null — caller can retry with frame subset
        }
    }

    /** Re-encode the frames tensor (or list of frames) into {@code opts.reencodeTo}. */
    private static void reencodeRow(java.util.List<?> frameList, Object framesTensor,
                                    VideoFolderOptions opts, Path srcEntry, int ri) {
        try {
            // Determine dims from framesTensor ([N, 3, H, W]).
            long[] s = (long[]) reflectArgs(framesTensor, "sizes", new Class<?>[0], new Object[0]);
            if (s == null || s.length != 4) return;
            int width = (int) s[3];
            int height = (int) s[2];
            double fps = opts.reencodeFps() > 0 ? opts.reencodeFps() : 30.0;
            String target = opts.reencodeTo().replace("{name}",
                    stripExt(srcEntry.getFileName().toString()));
            Object writer = MediaBackend.openVideoWriter(target, width, height, fps);
            if (writer == null) return;
            try {
                int n = (int) s[0];
                for (int i = 0; i < n; i++) {
                    Object frame = reflectArgs(framesTensor, "select",
                            new Class<?>[]{long.class, long.class},
                            new Object[]{0L, (long) i});
                    MediaBackend.writeVideoFrame(writer, frame);
                }
            } finally {
                try { writer.getClass().getMethod("close").invoke(writer); }
                catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {}
    }

    private static String stripExt(String name) {
        int d = name.lastIndexOf('.');
        return d > 0 ? name.substring(0, d) : name;
    }

    private static Object reflect(Object target, String name) {
        return reflectArgs(target, name, null, null);
    }

    private static Object reflectArgs(Object target, String name, Class<?>[] params, Object[] args) {
        try {
            Method m = target.getClass().getMethod(name, params == null ? new Class<?>[0] : params);
            return m.invoke(target, args == null ? new Object[0] : args);
        } catch (Throwable t) {
            return null;
        }
    }

    private static Object reflectStaticArgs(String cls, String name, Class<?>[] params, Object[] args) {
        try {
            Class<?> c = Class.forName(cls);
            Method m = c.getMethod(name, params);
            return m.invoke(null, args);
        } catch (Throwable t) {
            return null;
        }
    }

    private static final class Sniff {
        String codec;
        int width = -1;
        int height = -1;
        double durationSeconds = -1.0;
        double fps = -1.0;
        long frameCount = -1;
        boolean hasAudio = false;
    }

    private static Sniff sniffContainer(byte[] buf, int len) {
        Sniff s = new Sniff();
        if (len < 16) return s;
        // ISO BMFF (mp4, mov, m4v, 3gp): 'ftyp' at offset 4.
        if (buf[4] == 'f' && buf[5] == 't' && buf[6] == 'y' && buf[7] == 'p') {
            sniffMp4(buf, len, s);
            return s;
        }
        // Matroska / WebM: 0x1A 0x45 0xDF 0xA3
        if ((buf[0] & 0xFF) == 0x1A && (buf[1] & 0xFF) == 0x45
                && (buf[2] & 0xFF) == 0xDF && (buf[3] & 0xFF) == 0xA3) {
            s.codec = "matroska";
            sniffMatroska(buf, len, s);
            return s;
        }
        // RIFF AVI: 'RIFF'....'AVI '
        if (buf[0] == 'R' && buf[1] == 'I' && buf[2] == 'F' && buf[3] == 'F'
                && buf[8] == 'A' && buf[9] == 'V' && buf[10] == 'I' && buf[11] == ' ') {
            sniffAvi(buf, len, s);
            return s;
        }
        // FLV: 'FLV'
        if (buf[0] == 'F' && buf[1] == 'L' && buf[2] == 'V') {
            // Real extraction is complex; just record the container.
            s.codec = "flv";
            return s;
        }
        // MPEG-PS: 0x000001BA
        if (buf[0] == 0 && buf[1] == 0 && buf[2] == 1 && (buf[3] & 0xFF) == 0xBA) {
            s.codec = "mpeg-ps";
            return s;
        }
        // Ogg: 'OggS'
        if (buf[0] == 'O' && buf[1] == 'g' && buf[2] == 'g' && buf[3] == 'S') {
            s.codec = "ogv";
            return s;
        }
        return s;
    }

    // ---- MP4 / MOV / 3GP / M4V (ISO BMFF) ----------------------------------

    private static void sniffMp4(byte[] buf, int len, Sniff s) {
        s.codec = "iso-bmff";
        // Walk top-level boxes. Each box is 4-byte size (big-endian) + 4-byte type.
        int p = 0;
        while (p + 8 <= len) {
            long size = ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                    | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
            int type = ((buf[p+4] & 0xFF) << 24) | ((buf[p+5] & 0xFF) << 16)
                    | ((buf[p+6] & 0xFF) << 8) | (buf[p+7] & 0xFF);
            if (size == 0) break;
            if (size == 1) {
                // 64-bit largesize at offset 8 — skip; we only read the head.
                if (p + 16 > len) break;
                size = ((buf[p+8] & 0xFFL) << 56) | ((buf[p+9] & 0xFFL) << 48)
                        | ((buf[p+10] & 0xFFL) << 40) | ((buf[p+11] & 0xFFL) << 32)
                        | ((buf[p+12] & 0xFFL) << 24) | ((buf[p+13] & 0xFFL) << 16)
                        | ((buf[p+14] & 0xFFL) << 8) | (buf[p+15] & 0xFFL);
                p += 16;
            } else {
                p += 8;
            }
            if (size < 8) break;
            int boxEnd = (int) Math.min((long) p + (size - 8), len);
            String typeStr = new String(new byte[]{
                    (byte) ((type >> 24) & 0xFF), (byte) ((type >> 16) & 0xFF),
                    (byte) ((type >> 8) & 0xFF), (byte) (type & 0xFF)
            });
            if ("moov".equals(typeStr)) {
                sniffMp4Moov(buf, p, boxEnd, s);
            } else if ("mdat".equals(typeStr)) {
                // we don't need to recurse into mdat
            }
            p = boxEnd;
            if (p <= 0) break;
        }
    }

    private static void sniffMp4Moov(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        while (p + 8 <= end) {
            long size = ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                    | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
            int type = ((buf[p+4] & 0xFF) << 24) | ((buf[p+5] & 0xFF) << 16)
                    | ((buf[p+6] & 0xFF) << 8) | (buf[p+7] & 0xFF);
            String typeStr = new String(new byte[]{
                    (byte) ((type >> 24) & 0xFF), (byte) ((type >> 16) & 0xFF),
                    (byte) ((type >> 8) & 0xFF), (byte) (type & 0xFF)
            });
            if (size < 8) break;
            int boxEnd = (int) Math.min((long) p + (size - 8), end);
            if ("mvhd".equals(typeStr)) {
                int mvhdStart = p + 8;
                if (mvhdStart + 4 <= boxEnd) {
                    int version = buf[mvhdStart] & 0xFF;
                    int ts = mvhdStart + 4;
                    long timescale;
                    long duration;
                    if (version == 1) {
                        timescale = readU32(buf, ts + 4);
                        duration = readU64(buf, ts + 8);
                    } else {
                        timescale = readU32(buf, ts);
                        duration = readU32(buf, ts + 4);
                    }
                    if (timescale > 0) {
                        s.durationSeconds = (double) duration / (double) timescale;
                    }
                }
            } else if ("trak".equals(typeStr)) {
                sniffMp4Trak(buf, p + 8, boxEnd, s);
            }
            p = boxEnd;
        }
    }

    private static void sniffMp4Trak(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        while (p + 8 <= end) {
            long size = ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                    | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
            int type = ((buf[p+4] & 0xFF) << 24) | ((buf[p+5] & 0xFF) << 16)
                    | ((buf[p+6] & 0xFF) << 8) | (buf[p+7] & 0xFF);
            String typeStr = new String(new byte[]{
                    (byte) ((type >> 24) & 0xFF), (byte) ((type >> 16) & 0xFF),
                    (byte) ((type >> 8) & 0xFF), (byte) (type & 0xFF)
            });
            if (size < 8) break;
            int boxEnd = (int) Math.min((long) p + (size - 8), end);
            if ("mdia".equals(typeStr)) {
                sniffMp4Mdia(buf, p + 8, boxEnd, s);
            }
            p = boxEnd;
        }
    }

    private static void sniffMp4Mdia(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        while (p + 8 <= end) {
            long size = ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                    | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
            int type = ((buf[p+4] & 0xFF) << 24) | ((buf[p+5] & 0xFF) << 16)
                    | ((buf[p+6] & 0xFF) << 8) | ((buf[p+7] & 0xFF));
            String typeStr = new String(new byte[]{
                    (byte) ((type >> 24) & 0xFF), (byte) ((type >> 16) & 0xFF),
                    (byte) ((type >> 8) & 0xFF), (byte) (type & 0xFF)
            });
            if (size < 8) break;
            int boxEnd = (int) Math.min((long) p + (size - 8), end);
            if ("hdlr".equals(typeStr)) {
                int hs = p + 8;
                if (hs + 12 <= boxEnd) {
                    // 1 byte version, 3 bytes flags, 4 bytes pre_defined, 4 bytes handler_type
                    String handler = new String(buf, hs + 8, 4);
                    if ("soun".equals(handler)) s.hasAudio = true;
                }
            } else if ("minf".equals(typeStr)) {
                sniffMp4Minf(buf, p + 8, boxEnd, s);
            } else if ("mdhd".equals(typeStr)) {
                // sample-level media header — used for audio track duration / fps inheritance
                int ms = p + 8;
                if (ms + 4 <= boxEnd) {
                    int version = buf[ms] & 0xFF;
                    int ts = ms + 4;
                    if (version == 1 && ts + 4 + 8 + 8 <= boxEnd) {
                        // 8 bytes creation, 8 bytes modification, 4 bytes timescale, 8 bytes duration
                    } else if (version == 0 && ts + 4 + 4 + 4 <= boxEnd) {
                        // 4 bytes creation, 4 bytes modification, 4 bytes timescale, 4 bytes duration
                    }
                }
            }
            p = boxEnd;
        }
    }

    private static void sniffMp4Minf(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        while (p + 8 <= end) {
            long size = ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                    | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
            int type = ((buf[p+4] & 0xFF) << 24) | ((buf[p+5] & 0xFF) << 16)
                    | ((buf[p+6] & 0xFF) << 8) | ((buf[p+7] & 0xFF));
            String typeStr = new String(new byte[]{
                    (byte) ((type >> 24) & 0xFF), (byte) ((type >> 16) & 0xFF),
                    (byte) ((type >> 8) & 0xFF), (byte) (type & 0xFF)
            });
            if (size < 8) break;
            int boxEnd = (int) Math.min((long) p + (size - 8), end);
            if ("stbl".equals(typeStr)) {
                sniffMp4Stbl(buf, p + 8, boxEnd, s);
            }
            p = boxEnd;
        }
    }

    private static void sniffMp4Stbl(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        long timescale = 0;
        while (p + 8 <= end) {
            long size = ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                    | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
            int type = ((buf[p+4] & 0xFF) << 24) | ((buf[p+5] & 0xFF) << 16)
                    | ((buf[p+6] & 0xFF) << 8) | ((buf[p+7] & 0xFF));
            String typeStr = new String(new byte[]{
                    (byte) ((type >> 24) & 0xFF), (byte) ((type >> 16) & 0xFF),
                    (byte) ((type >> 8) & 0xFF), (byte) (type & 0xFF)
            });
            if (size < 8) break;
            int boxEnd = (int) Math.min((long) p + (size - 8), end);
            if ("stsd".equals(typeStr)) {
                int as = p + 8;
                if (as + 16 <= boxEnd) {
                    int version = buf[as] & 0xFF;
                    int entryStart = as + (version == 0 ? 8 : 16);
                    if (entryStart + 8 <= boxEnd) {
                        int entrySize = ((buf[entryStart] & 0xFF) << 24)
                                | ((buf[entryStart+1] & 0xFF) << 16)
                                | ((buf[entryStart+2] & 0xFF) << 8)
                                | (buf[entryStart+3] & 0xFF);
                        if (entrySize >= 8) {
                            String codec = new String(buf, entryStart + 4, 4);
                            s.codec = codec;
                            // Visual Sample Entry (avc1/hvc1) starts after visual entry header.
                            if (codec.startsWith("avc1") || codec.startsWith("hvc1")
                                    || codec.startsWith("hev1") || codec.startsWith("vp09")
                                    || codec.startsWith("av01")) {
                                int vis = entryStart + 4 + 4 + 4; // skip entry header, data_ref_idx, pre_defined
                                int headerEnd = vis + 70;
                                if (headerEnd + 8 <= boxEnd) {
                                    s.width = ((buf[headerEnd] & 0xFF) << 8) | (buf[headerEnd+1] & 0xFF);
                                    s.height = ((buf[headerEnd+2] & 0xFF) << 8) | (buf[headerEnd+3] & 0xFF);
                                }
                            }
                        }
                    }
                }
            } else if ("stts".equals(typeStr) && s.fps < 0) {
                // Time-to-sample: total samples / duration_covered * timescale
                int as = p + 8;
                if (as + 8 <= boxEnd) {
                    int entryCount = readU32int(buf, as + 4);
                    long totalSamples = 0;
                    for (int i = 0; i < entryCount && as + 8 + (i+1) * 8 <= boxEnd; i++) {
                        int off = as + 8 + i * 8;
                        totalSamples += readU32int(buf, off);
                    }
                    if (totalSamples > 0 && s.durationSeconds > 0) {
                        s.frameCount = totalSamples;
                        s.fps = totalSamples / s.durationSeconds;
                    }
                }
            }
            p = boxEnd;
        }
    }

    private static long readU32(byte[] buf, int p) {
        return ((buf[p] & 0xFFL) << 24) | ((buf[p+1] & 0xFFL) << 16)
                | ((buf[p+2] & 0xFFL) << 8) | (buf[p+3] & 0xFFL);
    }

    private static int readU32int(byte[] buf, int p) {
        return (int) readU32(buf, p);
    }

    private static long readU64(byte[] buf, int p) {
        return ((buf[p] & 0xFFL) << 56)   | ((buf[p+1] & 0xFFL) << 48)
                | ((buf[p+2] & 0xFFL) << 40) | ((buf[p+3] & 0xFFL) << 32)
                | ((buf[p+4] & 0xFFL) << 24) | ((buf[p+5] & 0xFFL) << 16)
                | ((buf[p+6] & 0xFFL) << 8)  | (buf[p+7] & 0xFFL);
    }

    // ---- Matroska / WebM --------------------------------------------------
    //
    // EBML element IDs (we compare raw int forms against the byte stream):
    //   Duration       = 0x2AD7B1
    //   TimestampScale = 0x22AD489B
    //   PixelWidth     = 0x1B0
    //   PixelHeight    = 0x1BA
    //   Tracks         = 0x654AE6B4
    //   TrackEntry     = 0xAE
    //   TrackType      = 0x83
    //   CodecID        = 0x86
    //   Audio          = 0xE1

    private static void sniffMatroska(byte[] buf, int len, Sniff s) {
        // EBML is a binary tag/length system; we only need Duration, TimestampScale,
        // PixelWidth, PixelHeight, and codec. Use a small forward scanner.
        // For perf we only scan the first 8 KB here.
        int ceil = Math.min(len, 64 * 1024);
        int p = 0;
        long duration = -1;
        long timescale = 1_000_000L; // default ns
        int width = -1, height = -1;
        String codec = null;
        int audioTracks = 0;
        s.codec = "matroska";
        while (p + 4 < ceil) {
            int id = ((buf[p] & 0xFF) << 24) | ((buf[p+1] & 0xFF) << 16)
                    | ((buf[p+2] & 0xFF) << 8) | (buf[p+3] & 0xFF);
            int sizeBytes = readVintLen(buf, p + 4);
            if (sizeBytes < 0 || p + 4 + sizeBytes > ceil) break;
            long size = readVint(buf, p + 4, sizeBytes);
            int contentStart = p + 4 + sizeBytes;
            if (id == 0x2AD7B1) {
                duration = readVint(buf, contentStart, sizeBytes);
            } else if (id == 0x22AD489B) {
                timescale = readVint(buf, contentStart, sizeBytes);
            } else if (id == 0x1B0) {
                width = (int) readVint(buf, contentStart, sizeBytes);
            } else if (id == 0x1BA) {
                height = (int) readVint(buf, contentStart, sizeBytes);
            } else if (id == 0x86) {
                // CodecID — read ASCII.
                int cLen = (int) size;
                if (contentStart + cLen <= ceil) {
                    codec = new String(buf, contentStart, Math.min(cLen, 32));
                }
            } else if (id == 0x83) {
                long trackType = readVint(buf, contentStart, sizeBytes);
                if (trackType == 2) audioTracks++;
            }
            p = contentStart + (int) size;
        }
        if (timescale > 0 && duration >= 0) {
            s.durationSeconds = (double) duration / (double) timescale;
        }
        s.width = width;
        s.height = height;
        if (codec != null) s.codec = codec;
        s.hasAudio = audioTracks > 0;
    }

    private static int readVintLen(byte[] buf, int p) {
        if (p >= buf.length) return -1;
        int b = buf[p] & 0xFF;
        if (b == 0) return -1;
        int mask = 0x80;
        int len = 1;
        while ((b & mask) == 0 && len < 8) {
            mask >>= 1;
            len++;
        }
        return len;
    }

    private static long readVint(byte[] buf, int p, int len) {
        if (len < 1 || p + len > buf.length) return -1;
        long v = 0;
        if (len == 1) return buf[p] & 0x7F;
        for (int i = 0; i < len; i++) {
            v = (v << 8) | (buf[p + i] & 0xFF);
        }
        v &= (1L << (7 * len)) - 1;
        return v;
    }

    // ---- AVI --------------------------------------------------------------

    private static void sniffAvi(byte[] buf, int len, Sniff s) {
        s.codec = "avi";
        // Walk RIFF chunks. Each chunk: 4-byte FOURCC + 4-byte size (LE) + payload.
        int p = 12;
        while (p + 8 <= len) {
            String fourcc = new String(buf, p, 4);
            int size = (int) readU32(buf, p + 4);
            if (size < 0) break;
            int contentStart = p + 8;
            int chunkEnd = Math.min(contentStart + size, len);
            if ("hdrl".equals(fourcc)) {
                parseAviHdrl(buf, contentStart, chunkEnd, s);
            }
            int next = chunkEnd + (size & 1);
            if (next <= p) break;
            p = next;
        }
    }

    private static void parseAviHdrl(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        while (p + 8 <= end) {
            String fourcc = new String(buf, p, 4);
            int size = (int) readU32(buf, p + 4);
            if (size < 0) break;
            int cs = p + 8;
            int ce = Math.min(cs + size, end);
            if ("avih".equals(fourcc) && cs + 56 <= ce) {
                int dwMicroSecPerFrame = (int) readU32(buf, cs + 0);
                int dwTotalFrames = (int) readU32(buf, cs + 16);
                if (dwMicroSecPerFrame > 0) s.fps = 1_000_000.0 / dwMicroSecPerFrame;
                if (dwTotalFrames > 0) {
                    s.frameCount = dwTotalFrames;
                    if (s.fps > 0) s.durationSeconds = dwTotalFrames / s.fps;
                }
            } else if ("strl".equals(fourcc)) {
                parseAviStrl(buf, cs, ce, s);
            }
            int next = ce + (size & 1);
            if (next <= p) break;
            p = next;
        }
    }

    private static void parseAviStrl(byte[] buf, int start, int end, Sniff s) {
        int p = start;
        boolean isAudio = false;
        while (p + 8 <= end) {
            String fourcc = new String(buf, p, 4);
            int size = (int) readU32(buf, p + 4);
            if (size < 0) break;
            int cs = p + 8;
            int ce = Math.min(cs + size, end);
            if ("strh".equals(fourcc) && cs + 56 <= ce) {
                String fccType = new String(buf, cs, 4);
                if ("vids".equals(fccType)) {
                    // strh structure: fccType (4), fccHandler (4), dwFlags (4), wPriority (2),
                    // wLanguage (2), dwInitialFrames (4), dwScale (4), dwRate (4),
                    // dwStart (4), dwLength (4), dwSuggestedBufferSize (4), dwQuality (4),
                    // dwSampleSize (4), rcFrame (16)
                    int dwRate = (int) readU32(buf, cs + 4 + 4 + 4 + 2 + 2 + 4 + 4);
                    int dwScale = (int) readU32(buf, cs + 4 + 4 + 4 + 2 + 2 + 4);
                    if (dwScale > 0) {
                        s.fps = (double) dwRate / dwScale;
                    }
                    int dwLength = (int) readU32(buf, cs + 4 + 4 + 4 + 2 + 2 + 4 + 4 + 4);
                    if (s.fps > 0) {
                        s.frameCount = dwLength;
                        s.durationSeconds = dwLength / s.fps;
                    }
                    // rcFrame at offset 4+4+4+2+2+4+4+4+4+4+4+4 = 56
                    int rcFrame = cs + 56;
                    if (rcFrame + 16 <= ce) {
                        s.width = (int) readU32(buf, rcFrame + 4) - (int) readU32(buf, rcFrame);
                        s.height = (int) readU32(buf, rcFrame + 12) - (int) readU32(buf, rcFrame + 8);
                    }
                } else if ("auds".equals(fccType)) {
                    isAudio = true;
                }
            } else if ("strf".equals(fourcc) && s.codec == null) {
                // BITMAPINFOHEADER starts at cs, 40 bytes.
                if (cs + 40 <= ce) {
                    int sizeField = (int) readU32(buf, cs);
                    if (sizeField == 40) {
                        int w = (int) readU32(buf, cs + 4);
                        int h = (int) readU32(buf, cs + 8);
                        s.width = w;
                        s.height = Math.abs(h);
                        String compression = new String(buf, cs + 16, 4);
                        s.codec = compression;
                    }
                }
            }
            int next = ce + (size & 1);
            if (next <= p) break;
            p = next;
        }
        if (isAudio) s.hasAudio = true;
    }

    // ---- Tiny JSON helpers (avoid the heavy Jackson dep) ------------------

    private static final class MiniJson {
        static int countTopLevels(String s) {
            int n = 0;
            for (String line : s.split("\n")) {
                if (line.trim().startsWith("{")) n++;
            }
            return n;
        }
        static Map<String, Object> parseObject(String json) {
            return HfDatasetSniffer.parseJsonObject(json);
        }
    }

    /** Thin local JSON parser that delegates to {@code HfDataset.parseJsonObject} when
     *  available, otherwise falls back to a tiny hand-rolled parser. */
    private static final class HfDatasetSniffer {
        static Map<String, Object> parseJsonObject(String json) {
            try {
                Class<?> cls = Class.forName("org.bytedeco.pytorch.utils.datasets.HfDataset");
                Object res = cls.getMethod("parseJsonObject", String.class).invoke(null, json);
                if (res instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> m = (Map<String, Object>) res;
                    return m;
                }
            } catch (Throwable ignored) {}
            // Fallback: parse "key": <value> pairs.
            Map<String, Object> out = new LinkedHashMap<>();
            json = json.trim();
            if (!json.startsWith("{")) return out;
            int i = 1;
            while (i < json.length()) {
                while (i < json.length() && (Character.isWhitespace(json.charAt(i)) || json.charAt(i) == ',')) i++;
                if (i >= json.length() || json.charAt(i) == '}') break;
                if (json.charAt(i) != '"') break;
                int keyStart = ++i;
                while (i < json.length() && json.charAt(i) != '"') {
                    if (json.charAt(i) == '\\') i++;
                    i++;
                }
                String key = json.substring(keyStart, i);
                i++;
                while (i < json.length() && json.charAt(i) != ':') i++;
                i++;
                while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
                int valStart = i;
                if (json.charAt(i) == '"') {
                    int vs = ++i;
                    while (i < json.length() && json.charAt(i) != '"') {
                        if (json.charAt(i) == '\\') i++;
                        i++;
                    }
                    out.put(key, json.substring(vs, i));
                    i++;
                } else {
                    // number / true / false / null
                    int vs = i;
                    while (i < json.length() && ",}".indexOf(json.charAt(i)) < 0) i++;
                    String tok = json.substring(vs, i).trim();
                    if ("true".equals(tok)) out.put(key, Boolean.TRUE);
                    else if ("false".equals(tok)) out.put(key, Boolean.FALSE);
                    else if ("null".equals(tok)) out.put(key, null);
                    else {
                        try {
                            if (tok.contains(".") || tok.contains("e") || tok.contains("E")) {
                                out.put(key, Double.parseDouble(tok));
                            } else {
                                out.put(key, Long.parseLong(tok));
                            }
                        } catch (NumberFormatException e) {
                            out.put(key, tok);
                        }
                    }
                }
            }
            return out;
        }
    }
}
