package org.bytedeco.pytorch.dataframe.io.document;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade document reader for various document formats.
 * 
 * <p>Supported formats:</p>
 * <ul>
 *   <li>HTML files</li>
 *   <li>Plain text files</li>
 *   <li>Markdown files</li>
 *   <li>XML files</li>
 *   <li>RTF files (basic)</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   // Read multiple documents
 *   DataFrame df = DocumentReader.read("documents/");
 *   
 *   // Read with options
 *   DocumentReader.DocumentOptions opts = DocumentReader.options()
 *       .recursive(true)
 *       .extractLinks(true);
 *   DataFrame df = DocumentReader.read("documents/", opts);
 * </pre>
 */
public class DocumentReader {

    private DocumentReader() {}

    private static final Set<String> TEXT_EXTENSIONS = Set.of(
        ".txt", ".text", ".log"
    );
    
    private static final Set<String> HTML_EXTENSIONS = Set.of(
        ".html", ".htm", ".xhtml", ".xht"
    );
    
    private static final Set<String> MARKDOWN_EXTENSIONS = Set.of(
        ".md", ".markdown", ".mdown", ".mkd"
    );
    
    private static final Set<String> XML_EXTENSIONS = Set.of(
        ".xml", ".svg", ".xsl", ".xslt"
    );
    
    private static final Set<String> RTF_EXTENSIONS = Set.of(
        ".rtf"
    );

    /**
     * Read document(s) into DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, DocumentOptions.defaults());
    }

    public static DataFrame read(String path, DocumentOptions options) throws IOException {
        DocumentOptions opts = options == null ? DocumentOptions.defaults() : options;
        
        DataFrame df = DataFrame.create();
        df.addColumn("file_path", Column.DType.STRING);
        df.addColumn("file_name", Column.DType.STRING);
        df.addColumn("extension", Column.DType.STRING);
        df.addColumn("content", Column.DType.STRING);
        df.addColumn("content_type", Column.DType.STRING);
        df.addColumn("num_characters", Column.DType.INT32);
        df.addColumn("num_words", Column.DType.INT32);
        df.addColumn("num_lines", Column.DType.INT32);
        
        if (opts.extractMetadata()) {
            df.addColumn("file_size", Column.DType.INT64);
            df.addColumn("modified_time", Column.DType.INT64);
        }
        
        if (opts.extractLinks()) {
            df.addColumn("links", Column.DType.STRING);
        }
        
        if (opts.extractHeadings()) {
            df.addColumn("headings", Column.DType.STRING);
        }
        
        Path p = Path.of(path);
        if (Files.isDirectory(p)) {
            readDirectory(df, p, opts);
        } else {
            readFile(df, p, opts);
        }
        
        return df;
    }

    private static void readDirectory(DataFrame df, Path dir, DocumentOptions opts) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            List<Path> files = new ArrayList<>();
            for (Path entry : stream) {
                if (Files.isDirectory(entry) && opts.recursive()) {
                    readDirectory(df, entry, opts);
                } else if (Files.isRegularFile(entry)) {
                    files.add(entry);
                }
            }
            
            // Sort for consistent ordering
            files.sort(Comparator.comparing(Path::toString));
            
            for (Path file : files) {
                readFile(df, file, opts);
            }
        }
    }

    private static void readFile(DataFrame df, Path file, DocumentOptions opts) throws IOException {
        String fileName = file.getFileName().toString();
        String ext = getExtension(fileName).toLowerCase();
        
        // Skip non-document files
        if (!isDocumentFile(ext)) {
            return;
        }
        
        String contentType = detectContentType(ext);
        String content;
        List<String> links = new ArrayList<>();
        List<String> headings = new ArrayList<>();
        
        try {
            if (MARKDOWN_EXTENSIONS.contains(ext) || TEXT_EXTENSIONS.contains(ext)) {
                content = Files.readString(file);
                if (MARKDOWN_EXTENSIONS.contains(ext) && opts.extractHeadings()) {
                    headings = extractMarkdownHeadings(content);
                }
            } else if (HTML_EXTENSIONS.contains(ext)) {
                content = Files.readString(file);
                String plainText = stripHtml(content);
                if (opts.htmlAsText()) {
                    content = plainText;
                }
                if (opts.extractLinks()) {
                    links = extractHtmlLinks(content);
                }
                if (opts.extractHeadings()) {
                    headings = extractHtmlHeadings(content);
                }
            } else if (XML_EXTENSIONS.contains(ext)) {
                content = Files.readString(file);
                if (opts.xmlAsText()) {
                    content = stripXml(content);
                }
            } else if (RTF_EXTENSIONS.contains(ext)) {
                content = readRtf(file);
            } else {
                content = Files.readString(file);
            }
        } catch (Exception e) {
            content = "[Error reading file: " + e.getMessage() + "]";
        }
        
        int ri = df.addEmptyRow();
        df.set(ri, "file_path", file.toAbsolutePath().toString());
        df.set(ri, "file_name", fileName);
        df.set(ri, "extension", ext);
        df.set(ri, "content", content);
        df.set(ri, "content_type", contentType);
        df.set(ri, "num_characters", content.length());
        df.set(ri, "num_words", countWords(content));
        df.set(ri, "num_lines", content.split("\\r?\\n").length);
        
        if (opts.extractMetadata()) {
            try {
                df.set(ri, "file_size", Files.size(file));
                df.set(ri, "modified_time", Files.getLastModifiedTime(file).toMillis());
            } catch (Exception ignored) {}
        }
        
        if (opts.extractLinks() && !links.isEmpty()) {
            df.set(ri, "links", String.join(", ", links));
        }
        
        if (opts.extractHeadings() && !headings.isEmpty()) {
            df.set(ri, "headings", String.join(" | ", headings));
        }
    }

    private static boolean isDocumentFile(String ext) {
        return TEXT_EXTENSIONS.contains(ext)
            || HTML_EXTENSIONS.contains(ext)
            || MARKDOWN_EXTENSIONS.contains(ext)
            || XML_EXTENSIONS.contains(ext)
            || RTF_EXTENSIONS.contains(ext);
    }

    private static String detectContentType(String ext) {
        if (TEXT_EXTENSIONS.contains(ext)) return "text/plain";
        if (HTML_EXTENSIONS.contains(ext)) return "text/html";
        if (MARKDOWN_EXTENSIONS.contains(ext)) return "text/markdown";
        if (XML_EXTENSIONS.contains(ext)) return "text/xml";
        if (RTF_EXTENSIONS.contains(ext)) return "application/rtf";
        return "text/plain";
    }

    private static String getExtension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        return dot > 0 ? fileName.substring(dot) : "";
    }

    // ====================== HTML Processing ======================

    private static String stripHtml(String html) {
        // Remove script and style content
        html = html.replaceAll("(?s)<script[^>]*>.*?</script>", "");
        html = html.replaceAll("(?s)<style[^>]*>.*?</style>", "");
        
        // Remove HTML tags
        html = html.replaceAll("<[^>]+>", " ");
        
        // Decode entities
        html = html.replaceAll("&nbsp;", " ");
        html = html.replaceAll("&lt;", "<");
        html = html.replaceAll("&gt;", ">");
        html = html.replaceAll("&amp;", "&");
        html = html.replaceAll("&quot;", "\"");
        html = html.replaceAll("&#(\\d+);", match -> String.valueOf((char)Integer.parseInt(match.group(1))));
        
        // Clean up whitespace
        html = html.replaceAll("\\s+", " ").trim();
        
        return html;
    }

    private static List<String> extractHtmlLinks(String html) {
        List<String> links = new ArrayList<>();
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
            "<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>", java.util.regex.Pattern.CASE_INSENSITIVE);
        java.util.regex.Matcher matcher = pattern.matcher(html);
        while (matcher.find()) {
            links.add(matcher.group(1));
        }
        return links;
    }

    private static List<String> extractHtmlHeadings(String html) {
        List<String> headings = new ArrayList<>();
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
            "<h([1-6])[^>]*>(.*?)</h\\1>", java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.DOTALL);
        java.util.regex.Matcher matcher = pattern.matcher(html);
        while (matcher.find()) {
            String text = matcher.group(2).replaceAll("<[^>]+>", "").trim();
            if (!text.isEmpty()) {
                headings.add(text);
            }
        }
        return headings;
    }

    // ====================== Markdown Processing ======================

    private static List<String> extractMarkdownHeadings(String content) {
        List<String> headings = new ArrayList<>();
        for (String line : content.split("\\r?\\n")) {
            line = line.trim();
            if (line.startsWith("#")) {
                int level = 0;
                while (level < line.length() && line.charAt(level) == '#') {
                    level++;
                }
                if (level > 0 && level <= 6) {
                    String text = line.substring(level).trim();
                    if (text.startsWith(" ")) {
                        text = text.substring(1);
                    }
                    if (!text.isEmpty()) {
                        headings.add(text);
                    }
                }
            }
        }
        return headings;
    }

    // ====================== XML Processing ======================

    private static String stripXml(String xml) {
        // Remove XML declarations
        xml = xml.replaceAll("<\\?[^>]+\\?>", "");
        
        // Remove comments
        xml = xml.replaceAll("<!--.*?-->", "");
        
        // Remove tags
        xml = xml.replaceAll("<[^>]+>", " ");
        
        // Decode entities
        xml = xml.replaceAll("&lt;", "<");
        xml = xml.replaceAll("&gt;", ">");
        xml = xml.replaceAll("&amp;", "&");
        xml = xml.replaceAll("&quot;", "\"");
        
        // Clean up whitespace
        xml = xml.replaceAll("\\s+", " ").trim();
        
        return xml;
    }

    // ====================== RTF Processing ======================

    private static String readRtf(Path file) throws IOException {
        StringBuilder sb = new StringBuilder();
        List<String> lines = Files.readAllLines(file);
        
        for (String line : lines) {
            if (line.contains("{\\rtf")) continue;
            
            // Extract text from RTF
            StringBuilder word = new StringBuilder();
            boolean skipGroup = 0;
            
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                
                if (c == '{') {
                    skipGroup++;
                } else if (c == '}') {
                    skipGroup--;
                } else if (skipGroup == 0 && c != '\\') {
                    word.append(c);
                } else if (c == '\\') {
                    if (i + 1 < line.length()) {
                        char next = line.charAt(i + 1);
                        if (next == '\\' || next == '{' || next == '}') {
                            word.append(next);
                            i++;
                        } else if (next == '\n' || next == '\r') {
                            i++;
                        } else {
                            // Skip control word
                            while (i < line.length() && Character.isLetter(line.charAt(i))) {
                                i++;
                            }
                            if (i < line.length() && Character.isDigit(line.charAt(i))) {
                                while (i < line.length() && (Character.isDigit(line.charAt(i)) || line.charAt(i) == '-')) {
                                    i++;
                                }
                            }
                            while (i < line.length() && line.charAt(i) == ' ') {
                                i++;
                            }
                            if (i < line.length() && line.charAt(i) == '\\') {
                                continue;
                            }
                        }
                    }
                }
            }
            
            String text = word.toString().trim();
            if (!text.isEmpty()) {
                sb.append(text).append(" ");
            }
        }
        
        return sb.toString().trim();
    }

    private static int countWords(String text) {
        if (text == null || text.isEmpty()) return 0;
        return text.trim().split("\\s+").length;
    }

    // ====================== Options ======================

    public static class DocumentOptions {
        private boolean recursive = true;
        private boolean extractMetadata = true;
        private boolean extractLinks = false;
        private boolean extractHeadings = false;
        private boolean htmlAsText = true;
        private boolean xmlAsText = false;
        private Set<String> includeExtensions = null;

        public static DocumentOptions defaults() {
            return new DocumentOptions();
        }

        public DocumentOptions recursive(boolean v) { this.recursive = v; return this; }
        public DocumentOptions extractMetadata(boolean v) { this.extractMetadata = v; return this; }
        public DocumentOptions extractLinks(boolean v) { this.extractLinks = v; return this; }
        public DocumentOptions extractHeadings(boolean v) { this.extractHeadings = v; return this; }
        public DocumentOptions htmlAsText(boolean v) { this.htmlAsText = v; return this; }
        public DocumentOptions xmlAsText(boolean v) { this.xmlAsText = v; return this; }
        public DocumentOptions includeExtensions(Set<String> exts) { this.includeExtensions = exts; return this; }

        public boolean recursive() { return recursive; }
        public boolean extractMetadata() { return extractMetadata; }
        public boolean extractLinks() { return extractLinks; }
        public boolean extractHeadings() { return extractHeadings; }
        public boolean htmlAsText() { return htmlAsText; }
        public boolean xmlAsText() { return xmlAsText; }
        public Set<String> includeExtensions() { return includeExtensions; }
    }
}
