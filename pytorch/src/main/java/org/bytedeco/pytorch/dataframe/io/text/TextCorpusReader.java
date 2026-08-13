package org.bytedeco.pytorch.dataframe.io.text;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade Text Corpus reader for large document collections.
 * 
 * <p>Text corpus readers are essential for NLP tasks. This reader supports:</p>
 * <ul>
 *   <li>Large document collections with chunking</li>
 *   <li>Various text encodings (UTF-8, UTF-16, ISO-8859, etc.)</li>
 *   <li>Line-by-line reading for large files</li>
 *   <li>Document-level reading for smaller files</li>
 *   <li>Metadata extraction from filenames</li>
 *   <li>Streaming mode for memory-efficient processing</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   // Read text corpus
 *   DataFrame df = TextCorpusReader.read("/path/to/corpus/");
 *   
 *   // With chunking for long documents
 *   TextCorpusReader.TextOptions opts = TextCorpusReader.options()
 *       .chunkSize(512)  // tokens or characters
 *       .chunkOverlap(50)
 *       .includeMetadata(true);
 *   DataFrame df = TextCorpusReader.read("/path/to/corpus/", opts);
 *   
 *   // Streaming for very large files
 *   TextCorpusReader.stream("/path/to/large.txt", (chunk, meta) -> {
 *       process(chunk);
 *       return true;  // continue
 *   });
 * </pre>
 */
public class TextCorpusReader {

    private TextCorpusReader() {}

    // Common text file extensions
    private static final Set<String> TEXT_EXTENSIONS = Set.of(
        ".txt", ".text", ".log", ".csv", ".tsv",
        ".json", ".jsonl", ".ndjson",
        ".xml", ".html", ".htm", ".md", ".markdown",
        ".py", ".java", ".js", ".c", ".cpp", ".h",
        ".sh", ".bash", ".yaml", ".yml", ".toml",
        ".sql", ".r", ".scala", ".go"
    );

    /**
     * Read text corpus into DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, TextOptions.defaults());
    }

    public static DataFrame read(String path, TextOptions options) throws IOException {
        TextOptions opts = options == null ? TextOptions.defaults() : options;
        
        DataFrame df = DataFrame.create();
        
        if (opts.readMode() == TextOptions.ReadMode.LINE) {
            df.addColumn("line_number", Column.DType.INT64);
        } else {
            df.addColumn("document_id", Column.DType.INT32);
        }
        
        df.addColumn("text", Column.DType.STRING);
        df.addColumn("num_chars", Column.DType.INT32);
        df.addColumn("num_words", Column.DType.INT32);
        df.addColumn("num_lines", Column.DType.INT32);
        
        if (opts.includePath()) {
            df.addColumn("file_path", Column.DType.STRING);
        }
        if (opts.includeFilename()) {
            df.addColumn("file_name", Column.DType.STRING);
        }
        if (opts.includeMetadata()) {
            df.addColumn("file_size", Column.DType.INT64);
            df.addColumn("modified_time", Column.DType.INT64);
            df.addColumn("encoding", Column.DType.STRING);
        }
        if (opts.includeChunkInfo()) {
            df.addColumn("chunk_id", Column.DType.INT32);
            df.addColumn("total_chunks", Column.DType.INT32);
        }
        
        Path p = Path.of(path);
        
        if (Files.isDirectory(p)) {
            readDirectory(df, p, opts);
        } else {
            readFile(df, p, opts, 0);
        }
        
        return df;
    }

    private static void readDirectory(DataFrame df, Path dir, TextOptions opts) throws IOException {
        List<Path> files = new ArrayList<>();
        
        collectTextFiles(dir, files, opts);
        files.sort(Comparator.comparing(Path::toString));
        
        int docId = 0;
        for (Path file : files) {
            if (opts.maxDocuments() > 0 && docId >= opts.maxDocuments()) break;
            readFile(df, file, opts, docId++);
        }
    }

    private static void collectTextFiles(Path dir, List<Path> files, TextOptions opts) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (opts.recursive()) {
                        collectTextFiles(entry, files, opts);
                    }
                } else if (Files.isRegularFile(entry)) {
                    String ext = getExtension(entry.getFileName().toString()).toLowerCase();
                    if (opts.filterExtensions() == null || opts.filterExtensions().contains(ext)) {
                        files.add(entry);
                    }
                }
            }
        }
    }

    private static void readFile(DataFrame df, Path file, TextOptions opts, int docId) throws IOException {
        String fileName = file.getFileName().toString();
        String path = file.toAbsolutePath().toString();
        
        if (opts.readMode() == TextOptions.ReadMode.LINE) {
            readByLine(df, file, path, fileName, opts);
        } else {
            readByDocument(df, file, path, fileName, opts, docId);
        }
    }

    private static void readByLine(DataFrame df, Path file, String path, String fileName, TextOptions opts) throws IOException {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file.toFile()), detectEncoding(file)), 
                opts.bufferSize())) {
            
            String line;
            long lineNum = 1;
            
            while ((line = reader.readLine()) != null) {
                if (opts.skipEmpty() && line.trim().isEmpty()) {
                    lineNum++;
                    continue;
                }
                
                if (opts.minLength() > 0 && line.length() < opts.minLength()) {
                    lineNum++;
                    continue;
                }
                
                int ri = df.addEmptyRow();
                df.set(ri, "line_number", lineNum);
                df.set(ri, "text", line);
                df.set(ri, "num_chars", line.length());
                df.set(ri, "num_words", countWords(line));
                df.set(ri, "num_lines", 1);
                
                if (opts.includePath()) df.set(ri, "file_path", path);
                if (opts.includeFilename()) df.set(ri, "file_name", fileName);
                if (opts.includeMetadata()) setFileMetadata(df, ri, file);
                
                lineNum++;
                
                if (opts.maxLines() > 0 && lineNum > opts.maxLines()) break;
            }
        }
    }

    private static void readByDocument(DataFrame df, Path file, String path, String fileName, TextOptions opts, int docId) throws IOException {
        String encoding = detectEncoding(file);
        String content = Files.readString(file);
        
        if (opts.chunkSize() <= 0) {
            // No chunking - single document
            int ri = df.addEmptyRow();
            df.set(ri, "document_id", docId);
            df.set(ri, "text", content);
            df.set(ri, "num_chars", content.length());
            df.set(ri, "num_words", countWords(content));
            df.set(ri, "num_lines", content.split("\\r?\\n").length);
            
            if (opts.includePath()) df.set(ri, "file_path", path);
            if (opts.includeFilename()) df.set(ri, "file_name", fileName);
            if (opts.includeMetadata()) {
                setFileMetadata(df, ri, file);
                df.set(ri, "encoding", encoding);
            }
        } else {
            // Chunk the document
            List<String> chunks = chunkText(content, opts.chunkSize(), opts.chunkOverlap(), opts.chunkBy().toString());
            int totalChunks = chunks.size();
            
            for (int i = 0; i < chunks.size(); i++) {
                String chunk = chunks.get(i);
                
                int ri = df.addEmptyRow();
                df.set(ri, "document_id", docId);
                df.set(ri, "text", chunk);
                df.set(ri, "num_chars", chunk.length());
                df.set(ri, "num_words", countWords(chunk));
                df.set(ri, "num_lines", chunk.split("\\r?\\n").length);
                
                if (opts.includePath()) df.set(ri, "file_path", path);
                if (opts.includeFilename()) df.set(ri, "file_name", fileName);
                if (opts.includeMetadata()) {
                    setFileMetadata(df, ri, file);
                    df.set(ri, "encoding", encoding);
                }
                if (opts.includeChunkInfo()) {
                    df.set(ri, "chunk_id", i);
                    df.set(ri, "total_chunks", totalChunks);
                }
            }
        }
    }

    /**
     * Stream text file with callback.
     */
    public static void stream(String path, TextConsumer consumer) throws IOException {
        stream(path, TextOptions.defaults(), consumer);
    }

    public static void stream(String path, TextOptions options, TextConsumer consumer) throws IOException {
        TextOptions opts = options == null ? TextOptions.defaults() : options;
        Path p = Path.of(path);
        
        if (Files.isDirectory(p)) {
            List<Path> files = new ArrayList<>();
            collectTextFiles(p, files, opts);
            files.sort(Comparator.comparing(Path::toString));

            for (Path file : files) {
                streamFile(file, opts, consumer);
            }
        } else {
            streamFile(p, opts, consumer);
        }
    }

    private static void streamFile(Path file, TextOptions options, TextConsumer consumer) throws IOException {
        String encoding = detectEncoding(file);
        
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(file.toFile()), encoding),
                options.bufferSize())) {
            
            String line;
            long lineNum = 1;
            
            while ((line = reader.readLine()) != null) {
                if (options.skipEmpty() && line.trim().isEmpty()) {
                    lineNum++;
                    continue;
                }
                
                TextMetadata meta = new TextMetadata();
                meta.lineNumber = lineNum;
                meta.filePath = file.toAbsolutePath().toString();
                meta.fileName = file.getFileName().toString();
                meta.encoding = encoding;
                
                if (!consumer.accept(line, meta)) {
                    return;  // Stop streaming
                }
                
                lineNum++;
            }
        }
    }

    /**
     * Stream document chunks with callback.
     */
    public static void streamDocuments(String path, DocumentConsumer consumer) throws IOException {
        streamDocuments(path, TextOptions.defaults(), consumer);
    }

    public static void streamDocuments(String path, TextOptions options, DocumentConsumer consumer) throws IOException {
        TextOptions opts = options == null ? TextOptions.defaults() : options;
        
        List<Path> files = new ArrayList<>();
        if (Files.isDirectory(Path.of(path))) {
            collectTextFiles(Path.of(path), files, opts);
        } else {
            files.add(Path.of(path));
        }
        
        files.sort(Comparator.comparing(Path::toString));
        
        int docId = 0;
        for (Path file : files) {
            if (opts.maxDocuments() > 0 && docId >= opts.maxDocuments()) break;
            
            String encoding = detectEncoding(file);
            String content = Files.readString(file);
            
            DocumentMetadata meta = new DocumentMetadata();
            meta.documentId = docId;
            meta.filePath = file.toAbsolutePath().toString();
            meta.fileName = file.getFileName().toString();
            meta.encoding = encoding;
            meta.numChars = content.length();
            meta.numWords = countWords(content);
            meta.numLines = content.split("\\r?\\n").length;
            
            if (opts.chunkSize() <= 0) {
                TextChunkMetadata single = new TextChunkMetadata();
                single.documentId = meta.documentId;
                single.filePath = meta.filePath;
                single.fileName = meta.fileName;
                single.encoding = meta.encoding;
                single.numChars = meta.numChars;
                single.numWords = meta.numWords;
                single.numLines = meta.numLines;
                single.chunkId = 0;
                single.totalChunks = 1;
                single.text = content;
                consumer.accept(single, meta);
            } else {
                List<String> chunks = chunkText(content, opts.chunkSize(), opts.chunkOverlap(), opts.chunkBy().toString());
                for (int i = 0; i < chunks.size(); i++) {
                    TextChunkMetadata chunkMeta = new TextChunkMetadata();
                    chunkMeta.documentId = meta.documentId;
                    chunkMeta.filePath = meta.filePath;
                    chunkMeta.fileName = meta.fileName;
                    chunkMeta.encoding = meta.encoding;
                    chunkMeta.numChars = meta.numChars;
                    chunkMeta.numWords = meta.numWords;
                    chunkMeta.numLines = meta.numLines;
                    chunkMeta.chunkId = i;
                    chunkMeta.totalChunks = chunks.size();
                    chunkMeta.text = chunks.get(i);
                    consumer.accept(chunkMeta, meta);
                }
            }
            
            docId++;
        }
    }

    // ====================== Text Processing ======================

    private static List<String> chunkText(String text, int chunkSize, int overlap, String chunkBy) {
        List<String> chunks = new ArrayList<>();
        
        if (chunkSize <= 0) {
            chunks.add(text);
            return chunks;
        }
        
        String[] units = chunkBy.equals("word") ? text.split("\\s+") : text.split("");
        int step = chunkSize - overlap;
        step = Math.max(1, step);
        
        for (int i = 0; i < units.length; i += step) {
            int end = Math.min(i + chunkSize, units.length);
            
            if (chunkBy.equals("word")) {
                StringBuilder sb = new StringBuilder();
                for (int j = i; j < end; j++) {
                    if (j > i) sb.append(" ");
                    sb.append(units[j]);
                }
                chunks.add(sb.toString());
            } else {
                chunks.add(text.substring(i, end));
            }
        }
        
        return chunks;
    }

    private static int countWords(String text) {
        if (text == null || text.isEmpty()) return 0;
        return text.trim().split("\\s+").length;
    }

    private static String getExtension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        return dot > 0 ? fileName.substring(dot) : "";
    }

    private static String detectEncoding(Path file) {
        // Try to detect encoding from BOM
        try (InputStream is = Files.newInputStream(file)) {
            byte[] bom = new byte[4];
            int read = is.read(bom);
            
            if (read >= 3 && bom[0] == (byte)0xEF && bom[1] == (byte)0xBB && bom[2] == (byte)0xBF) {
                return "UTF-8";
            }
            if (read >= 2) {
                if (bom[0] == (byte)0xFE && bom[1] == (byte)0xFF) {
                    return "UTF-16BE";
                }
                if (bom[0] == (byte)0xFF && bom[1] == (byte)0xFE) {
                    return "UTF-16LE";
                }
            }
        } catch (Exception ignored) {}
        
        return "UTF-8";  // Default
    }

    private static void setFileMetadata(DataFrame df, int row, Path file) {
        try {
            df.set(row, "file_size", Files.size(file));
            df.set(row, "modified_time", Files.getLastModifiedTime(file).toMillis());
        } catch (Exception ignored) {}
    }

    // ====================== Consumer Interfaces ======================

    @FunctionalInterface
    public interface TextConsumer {
        boolean accept(String line, TextMetadata metadata);
    }

    @FunctionalInterface
    public interface DocumentConsumer {
        void accept(TextChunkMetadata chunk, DocumentMetadata metadata);
    }

    public static class TextMetadata {
        public long lineNumber;
        public String filePath;
        public String fileName;
        public String encoding;
    }

    public static class DocumentMetadata {
        public int documentId;
        public String filePath;
        public String fileName;
        public String encoding;
        public int numChars;
        public int numWords;
        public int numLines;
    }

    public static class TextChunkMetadata extends DocumentMetadata {
        public int chunkId;
        public int totalChunks;
        public String text;
    }

    // ====================== Options ======================

    public static class TextOptions {
        public enum ReadMode { LINE, DOCUMENT }
        public enum ChunkBy { CHARACTER, WORD }
        
        private ReadMode readMode = ReadMode.DOCUMENT;
        private boolean recursive = true;
        private boolean skipEmpty = false;
        private int minLength = 0;
        private int maxLines = 0;
        private int maxDocuments = 0;
        private int chunkSize = 0;  // 0 = no chunking
        private int chunkOverlap = 0;
        private ChunkBy chunkBy = ChunkBy.CHARACTER;
        private boolean includePath = true;
        private boolean includeFilename = true;
        private boolean includeMetadata = true;
        private boolean includeChunkInfo = false;
        private int bufferSize = 65536;
        private Set<String> filterExtensions = null;

        public static TextOptions defaults() {
            return new TextOptions();
        }

        public TextOptions readMode(ReadMode m) { this.readMode = m; return this; }
        public TextOptions recursive(boolean v) { this.recursive = v; return this; }
        public TextOptions skipEmpty(boolean v) { this.skipEmpty = v; return this; }
        public TextOptions minLength(int v) { this.minLength = v; return this; }
        public TextOptions maxLines(int v) { this.maxLines = v; return this; }
        public TextOptions maxDocuments(int v) { this.maxDocuments = v; return this; }
        public TextOptions chunkSize(int v) { this.chunkSize = v; return this; }
        public TextOptions chunkOverlap(int v) { this.chunkOverlap = v; return this; }
        public TextOptions chunkBy(ChunkBy v) { this.chunkBy = v; return this; }
        public TextOptions includePath(boolean v) { this.includePath = v; return this; }
        public TextOptions includeFilename(boolean v) { this.includeFilename = v; return this; }
        public TextOptions includeMetadata(boolean v) { this.includeMetadata = v; return this; }
        public TextOptions includeChunkInfo(boolean v) { this.includeChunkInfo = v; return this; }
        public TextOptions bufferSize(int v) { this.bufferSize = v; return this; }
        public TextOptions filterExtensions(Set<String> v) { this.filterExtensions = v; return this; }

        public ReadMode readMode() { return readMode; }
        public boolean recursive() { return recursive; }
        public boolean skipEmpty() { return skipEmpty; }
        public int minLength() { return minLength; }
        public int maxLines() { return maxLines; }
        public int maxDocuments() { return maxDocuments; }
        public int chunkSize() { return chunkSize; }
        public int chunkOverlap() { return chunkOverlap; }
        public ChunkBy chunkBy() { return chunkBy; }
        public boolean includePath() { return includePath; }
        public boolean includeFilename() { return includeFilename; }
        public boolean includeMetadata() { return includeMetadata; }
        public boolean includeChunkInfo() { return includeChunkInfo; }
        public int bufferSize() { return bufferSize; }
        public Set<String> filterExtensions() { return filterExtensions; }
    }
}
