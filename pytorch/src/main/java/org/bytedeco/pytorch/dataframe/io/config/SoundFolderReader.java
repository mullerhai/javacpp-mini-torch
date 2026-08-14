package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.io.config.MediaBackend.Backend;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade reader for audio/sound folder datasets.
 * 
 * <p>Similar to ImageFolder, but for audio files. Supports various audio formats
 * and can organize audio files by folder structure (e.g., speaker ID, emotion, etc.).</p>
 * 
 * <p>Expected directory structure:</p>
 * <pre>
 * root/
 * ├── category1/
 * │   ├── audio1.wav
 * │   ├── audio2.mp3
 * │   └── ...
 * ├── category2/
 * │   └── ...
 * └── ...
 * </pre>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   DataFrame df = SoundFolder.read("/path/to/audio_dataset");
 *   
 *   // With options
 *   SoundFolder.SoundFolderOptions opts = SoundFolder.options()
 *       .recursive(true)
 *       .includeMetadata(true);
 *   DataFrame df = SoundFolder.read("/path/to/audio_dataset", opts);
 *   
 *   // Via DataFrameReader
 *   DataFrame df = DataFrame.read().soundfolder("/path/to/audio_dataset");
 * </pre>
 */
public class SoundFolderReader {

    private SoundFolderReader() {}

    // Supported audio extensions
    private static final Set<String> AUDIO_EXTENSIONS = Set.of(
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif",
        ".opus", ".webm", ".mid", ".midi",
        ".WAV", ".MP3", ".FLAC", ".OGG", ".M4A", ".AAC", ".WMA", ".AIFF", ".AIF",
        ".OPUS", ".WEBM", ".MID", ".MIDI"
    );

    /**
     * Read a sound folder dataset into a DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, SoundFolderOptions.defaults());
    }

    public static DataFrame read(String path, SoundFolderOptions options) throws IOException {
        SoundFolderOptions opts = options == null ? SoundFolderOptions.defaults() : options;
        Path rootPath = Path.of(path);
        
        if (!Files.exists(rootPath)) {
            throw new IOException("SoundFolder path does not exist: " + path);
        }
        
        if (!Files.isDirectory(rootPath)) {
            throw new IOException("SoundFolder path must be a directory: " + path);
        }
        
        DataFrame df = DataFrame.create();

        Backend backend = MediaBackend.resolve(opts.backend());

        // Build schema
        df.addColumn("relative_path", Column.DType.STRING);
        df.addColumn("category", Column.DType.STRING);
        df.addColumn("category_index", Column.DType.INT32);

        if (opts.includePath()) {
            df.addColumn("full_path", Column.DType.STRING);
        }
        if (opts.includeSize()) {
            df.addColumn("file_size", Column.DType.INT64);
        }
        if (opts.includeExtension()) {
            df.addColumn("extension", Column.DType.STRING);
        }
        if (opts.includeModifiedTime()) {
            df.addColumn("modified_time", Column.DType.INT64);
        }
        if (opts.includeName()) {
            df.addColumn("file_name", Column.DType.STRING);
        }
        if (backend == Backend.FFMPEG_OPENCV) {
            df.addColumn("sample_rate", Column.DType.INT32);
            df.addColumn("channels", Column.DType.INT32);
            df.addColumn("num_samples", Column.DType.INT64);
            df.addColumn("duration_seconds", Column.DType.FLOAT64);
        }
        if (opts.eagerDecode() && backend == Backend.FFMPEG_OPENCV
                && opts.waveformCol() != null && !opts.waveformCol().isBlank()) {
            df.addColumn(opts.waveformCol(), Column.DType.TENSOR);
        }
        if (opts.storeBytes()) {
            df.addColumn(opts.bytesCol(), Column.DType.BINARY);
        }
        
        // Discover categories (subdirectories)
        List<Path> categoryDirs = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootPath)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry) && !entry.getFileName().toString().startsWith(".")) {
                    categoryDirs.add(entry);
                }
            }
        }
        
        // Sort categories for consistent ordering
        categoryDirs.sort(Comparator.comparing(p -> p.getFileName().toString()));
        
        // Build category name to index mapping
        Map<String, Integer> categoryIndexMap = new LinkedHashMap<>();
        for (int i = 0; i < categoryDirs.size(); i++) {
            String categoryName = categoryDirs.get(i).getFileName().toString();
            categoryIndexMap.put(categoryName, i);
        }
        
        // Read audio files from each category
        for (Map.Entry<String, Integer> entry : categoryIndexMap.entrySet()) {
            String categoryName = entry.getKey();
            int categoryIdx = entry.getValue();
            Path categoryDir = categoryDirs.stream()
                .filter(p -> p.getFileName().toString().equals(categoryName))
                .findFirst()
                .orElse(null);
            
            if (categoryDir == null) continue;
            
            collectAudioFiles(df, categoryDir, rootPath, categoryName, categoryIdx, opts);
        }
        
        return df;
    }

    private static void collectAudioFiles(DataFrame df, Path dir, Path rootPath,
                                        String category, int categoryIdx,
                                        SoundFolderOptions opts) throws IOException {
        Backend backend = MediaBackend.resolve(opts.backend());
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (opts.recursive()) {
                        collectAudioFiles(df, entry, rootPath, category, categoryIdx, opts);
                    }
                } else if (isAudioFile(entry)) {
                    int ri = df.addEmptyRow();

                    // Relative path from root
                    String relativePath = rootPath.relativize(entry).toString();
                    df.set(ri, "relative_path", relativePath);
                    df.set(ri, "category", category);
                    df.set(ri, "category_index", categoryIdx);

                    if (opts.includePath()) {
                        df.set(ri, "full_path", entry.toAbsolutePath().toString());
                    }
                    if (opts.includeSize()) {
                        df.set(ri, "file_size", Files.size(entry));
                    }
                    if (opts.includeExtension()) {
                        String name = entry.getFileName().toString();
                        int dotIdx = name.lastIndexOf('.');
                        df.set(ri, "extension", dotIdx > 0 ? name.substring(dotIdx).toLowerCase() : "");
                    }
                    if (opts.includeModifiedTime()) {
                        try {
                            df.set(ri, "modified_time", Files.getLastModifiedTime(entry).toMillis());
                        } catch (Exception e) {
                            df.set(ri, "modified_time", 0L);
                        }
                    }
                    if (opts.includeName()) {
                        df.set(ri, "file_name", entry.getFileName().toString());
                    }

                    // FFmpeg probe + eager decode.
                    if (backend == Backend.FFMPEG_OPENCV) {
                        boolean probeOk = false;
                        if (opts.maxBytes() <= 0 || Files.size(entry) <= opts.maxBytes()) {
                            try (MediaBackend.AC af = MediaBackend.openAudioHandle(entry.toString())) {
                                if (af != null) {
                                    int sr = ((Number) reflect(af.handle, "sampleRate")).intValue();
                                    int ch = ((Number) reflect(af.handle, "channels")).intValue();
                                    long ns = ((Number) reflect(af.handle, "numSamples")).longValue();
                                    double du = ((Number) reflect(af.handle, "durationSec")).doubleValue();
                                    df.set(ri, "sample_rate", sr);
                                    df.set(ri, "channels", ch);
                                    df.set(ri, "num_samples", ns);
                                    df.set(ri, "duration_seconds", du);
                                    if (opts.eagerDecode()
                                            && opts.waveformCol() != null && !opts.waveformCol().isBlank()) {
                                        Object wave = reflect(af.handle, "read");
                                        if (wave != null) df.set(ri, opts.waveformCol(), wave);
                                    }
                                    probeOk = true;
                                }
                            } catch (Throwable t) {
                                // native unavailable or decode failure: fall through,
                                // columns above stay null; caller can retry later.
                            }
                        }
                        if (!probeOk) {
                            df.set(ri, "sample_rate", 0);
                            df.set(ri, "channels", 0);
                            df.set(ri, "num_samples", 0L);
                            df.set(ri, "duration_seconds", 0.0);
                        }
                    }

                    if (opts.storeBytes() && opts.maxBytes() > 0
                            ? Files.size(entry) <= opts.maxBytes() : true) {
                        long sz;
                        try { sz = Files.size(entry); }
                        catch (IOException e) { sz = 0L; }
                        if (sz <= Integer.MAX_VALUE) {
                            df.set(ri, opts.bytesCol(), readAllBytes(entry));
                        }
                    }
                }
            }
        }
    }

    private static byte[] readAllBytes(Path p) throws IOException {
        long sz = Files.size(p);
        if (sz > Integer.MAX_VALUE) {
            throw new IOException("Audio too large for in-memory BINARY column: " + p);
        }
        try (InputStream is = Files.newInputStream(p)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream((int) sz);
            byte[] buf = new byte[8192];
            int n;
            while ((n = is.read(buf)) > 0) baos.write(buf, 0, n);
            return baos.toByteArray();
        }
    }

    private static boolean isAudioFile(Path path) {
        String name = path.getFileName().toString().toLowerCase();
        return AUDIO_EXTENSIONS.stream().anyMatch(name::endsWith);
    }

    /**
     * Get category names and counts without loading all files.
     */
    public static Map<String, Long> getCategoryStats(String path) throws IOException {
        return getCategoryStats(path, true);
    }

    public static Map<String, Long> getCategoryStats(String path, boolean recursive) throws IOException {
        Path rootPath = Path.of(path);
        Map<String, Long> stats = new LinkedHashMap<>();
        
        if (!Files.exists(rootPath)) {
            return stats;
        }
        
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootPath)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    String categoryName = entry.getFileName().toString();
                    long count = countAudioFiles(entry, recursive);
                    stats.put(categoryName, count);
                }
            }
        }
        
        return stats;
    }

    private static long countAudioFiles(Path dir, boolean recursive) throws IOException {
        long count = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (recursive) {
                        count += countAudioFiles(entry, true);
                    }
                } else if (isAudioFile(entry)) {
                    count++;
                }
            }
        }
        return count;
    }

    // ====================== Options ======================


    private static Object reflect(Object target, String name) {
        try {
            return target.getClass().getMethod(name).invoke(target);
        } catch (Throwable t) {
            return null;
        }
    }
}
