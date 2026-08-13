package org.bytedeco.pytorch.dataframe.io.folder;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade reader for torchvision-style ImageFolder datasets.
 * 
 * <p>ImageFolder expects the following directory structure:</p>
 * <pre>
 * root/
 * ├── class1/
 * │   ├── img1.jpg
 * │   ├── img2.png
 * │   └── ...
 * ├── class2/
 * │   ├── img1.jpg
 * │   └── ...
 * └── class3/
 *     └── ...
 * </pre>
 * 
 * <p>Each subdirectory name is treated as a class label, and all images
 * in that directory belong to that class.</p>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   DataFrame df = ImageFolder.read("/path/to/imagenet");
 *   
 *   // With options
 *   ImageFolder.ImageFolderOptions opts = ImageFolder.options()
 *       .recursive(true)
 *       .includePath(true)
 *       .maxImagesPerClass(1000);
 *   DataFrame df = ImageFolder.read("/path/to/imagenet", opts);
 *   
 *   // Via DataFrameReader
 *   DataFrame df = DataFrame.read().imagefolder("/path/to/imagenet");
 * </pre>
 */
public class ImageFolderReader {

    private ImageFolderReader() {}

    // Supported image extensions
    private static final Set<String> IMAGE_EXTENSIONS = Set.of(
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
        ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".WEBP", ".TIFF", ".TIF"
    );

    /**
     * Read an ImageFolder dataset into a DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, ImageFolderOptions.defaults());
    }

    public static DataFrame read(String path, ImageFolderOptions options) throws IOException {
        ImageFolderOptions opts = options == null ? ImageFolderOptions.defaults() : options;
        Path rootPath = Path.of(path);
        
        if (!Files.exists(rootPath)) {
            throw new IOException("ImageFolder path does not exist: " + path);
        }
        
        if (!Files.isDirectory(rootPath)) {
            throw new IOException("ImageFolder path must be a directory: " + path);
        }
        
        DataFrame df = DataFrame.create();
        
        // Build schema
        df.addColumn("relative_path", Column.DType.STRING);
        df.addColumn("class_name", Column.DType.STRING);
        df.addColumn("class_index", Column.DType.INT32);
        
        if (opts.includePath()) {
            df.addColumn("full_path", Column.DType.STRING);
        }
        if (opts.includeSize()) {
            df.addColumn("file_size", Column.DType.INT64);
        }
        if (opts.includeModifiedTime()) {
            df.addColumn("modified_time", Column.DType.INT64);
        }
        
        // Discover classes (subdirectories)
        List<Path> classDirs = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootPath)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry) && !entry.getFileName().toString().startsWith(".")) {
                    classDirs.add(entry);
                }
            }
        }
        
        // Sort classes for consistent ordering
        classDirs.sort(Comparator.comparing(p -> p.getFileName().toString()));
        
        // Build class name to index mapping
        Map<String, Integer> classIndexMap = new LinkedHashMap<>();
        for (int i = 0; i < classDirs.size(); i++) {
            String className = classDirs.get(i).getFileName().toString();
            classIndexMap.put(className, i);
        }
        
        // Collect all image files
        int totalImages = 0;
        for (Path classDir : classDirs) {
            totalImages += countImages(classDir, opts.recursive());
            if (opts.maxImagesPerClass() > 0 && totalImages > opts.maxImagesPerClass() * classDirs.size()) {
                break;
            }
        }
        
        // Pre-allocate rows if possible
        df.ensureCapacity(totalImages);
        
        // Read images from each class
        int rowIndex = 0;
        for (Map.Entry<String, Integer> entry : classIndexMap.entrySet()) {
            String className = entry.getKey();
            int classIdx = entry.getValue();
            Path classDir = classDirs.stream()
                .filter(p -> p.getFileName().toString().equals(className))
                .findFirst()
                .orElse(null);
            
            if (classDir == null) continue;
            
            int classImageCount = 0;
            collectImages(df, classDir, rootPath, className, classIdx, opts, rowIndex);
            rowIndex += df.rowCount() - rowIndex;
            
            classImageCount = df.rowCount() - (rowIndex - classImageCount);
            
            if (opts.maxImagesPerClass() > 0 && classImageCount >= opts.maxImagesPerClass()) {
                // Limit reached for this class
                while (df.rowCount() > rowIndex) {
                    df.removeLastRow();
                }
            }
        }
        
        return df;
    }

    private static int countImages(Path dir, boolean recursive) throws IOException {
        int count = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (recursive) {
                        count += countImages(entry, true);
                    }
                } else if (isImageFile(entry)) {
                    count++;
                }
            }
        }
        return count;
    }

    private static void collectImages(DataFrame df, Path dir, Path rootPath, 
                                      String className, int classIdx,
                                      ImageFolderOptions opts, int startRow) throws IOException {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (opts.recursive()) {
                        collectImages(df, entry, rootPath, className, classIdx, opts, startRow);
                    }
                } else if (isImageFile(entry)) {
                    int ri = df.addEmptyRow();
                    
                    // Relative path from root
                    String relativePath = rootPath.relativize(entry).toString();
                    df.set(ri, "relative_path", relativePath);
                    df.set(ri, "class_name", className);
                    df.set(ri, "class_index", classIdx);
                    
                    if (opts.includePath()) {
                        df.set(ri, "full_path", entry.toAbsolutePath().toString());
                    }
                    if (opts.includeSize()) {
                        df.set(ri, "file_size", Files.size(entry));
                    }
                    if (opts.includeModifiedTime()) {
                        try {
                            df.set(ri, "modified_time", Files.getLastModifiedTime(entry).toMillis());
                        } catch (Exception e) {
                            df.set(ri, "modified_time", 0L);
                        }
                    }
                }
            }
        }
    }

    private static boolean isImageFile(Path path) {
        String name = path.getFileName().toString().toLowerCase();
        return IMAGE_EXTENSIONS.stream().anyMatch(name::endsWith);
    }

    /**
     * Get class names and counts without loading all images.
     */
    public static Map<String, Long> getClassStats(String path) throws IOException {
        return getClassStats(path, true);
    }

    public static Map<String, Long> getClassStats(String path, boolean recursive) throws IOException {
        Path rootPath = Path.of(path);
        Map<String, Long> stats = new LinkedHashMap<>();
        
        if (!Files.exists(rootPath)) {
            return stats;
        }
        
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootPath)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    String className = entry.getFileName().toString();
                    long count = countImages(entry, recursive);
                    stats.put(className, count);
                }
            }
        }
        
        return stats;
    }

    // ====================== Options ======================

    public static class ImageFolderOptions {
        private boolean recursive = true;
        private boolean includePath = false;
        private boolean includeSize = false;
        private boolean includeModifiedTime = false;
        private int maxImagesPerClass = 0;  // 0 = no limit

        public static ImageFolderOptions defaults() {
            return new ImageFolderOptions();
        }

        public ImageFolderOptions recursive(boolean v) { this.recursive = v; return this; }
        public ImageFolderOptions includePath(boolean v) { this.includePath = v; return this; }
        public ImageFolderOptions includeSize(boolean v) { this.includeSize = v; return this; }
        public ImageFolderOptions includeModifiedTime(boolean v) { this.includeModifiedTime = v; return this; }
        public ImageFolderOptions maxImagesPerClass(int v) { this.maxImagesPerClass = v; return this; }

        public boolean recursive() { return recursive; }
        public boolean includePath() { return includePath; }
        public boolean includeSize() { return includeSize; }
        public boolean includeModifiedTime() { return includeModifiedTime; }
        public int maxImagesPerClass() { return maxImagesPerClass; }
    }
}
