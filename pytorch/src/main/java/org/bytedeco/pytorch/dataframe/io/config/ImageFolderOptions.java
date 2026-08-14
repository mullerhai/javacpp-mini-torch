package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.io.config.MediaBackend.Backend;

public class ImageFolderOptions {
    private boolean recursive = true;
    private boolean includePath = false;
    private boolean includeSize = false;
    private boolean includeModifiedTime = false;
    private int maxImagesPerClass = 0;  // 0 = no limit

    /**
     * Backend selection.
     * <ul>
     *   <li>{@link Backend#INPUT_STREAM} (default for legacy compatibility):
     *       scan and record raw file paths + sizes + header-derived metadata.
     *       Byte cells are not materialised.</li>
     *   <li>{@link Backend#FFMPEG_OPENCV}: use {@code vision/opencv.OpenCVIO}
     *       to load each image as a {@code [C,H,W]} tensor as it is scanned.
     *       Falls back to {@code INPUT_STREAM} when OpenCV natives are
     *       unavailable.</li>
     *   <li>{@link Backend#NATIVE_REQUIRED}: force the native path; throw
     *       on decode failures.</li>
     * </ul>
     */
    private Backend backend = Backend.FFMPEG_OPENCV;

    /** Eagerly decode each image into a Tensor column. Default {@code true} when
     *  backend is FFMPEG_OPENCV and natives are available; {@code false} for
     *  the InputStream legacy path so behaviour matches the historical default. */
    private boolean eagerDecode = false;

    /** When {@code eagerDecode=true}, name of the OBJECT column for tensor cells.
     *  {@code null} or blank disables the column. */
    private String imageTensorCol = "image";

    /** When {@code eagerDecode=true}, also store the raw bytes alongside the
     *  tensor (in a BINARY column named {@link #bytesCol}). {@code false} skips
     *  the byte copy. Default: {@code false}. */
    private boolean storeBytes = false;

    private String bytesCol = "image_bytes";

    public static ImageFolderOptions defaults() {
        return new ImageFolderOptions();
    }

    public ImageFolderOptions recursive(boolean v) { this.recursive = v; return this; }
    public ImageFolderOptions includePath(boolean v) { this.includePath = v; return this; }
    public ImageFolderOptions includeSize(boolean v) { this.includeSize = v; return this; }
    public ImageFolderOptions includeModifiedTime(boolean v) { this.includeModifiedTime = v; return this; }
    public ImageFolderOptions maxImagesPerClass(int v) { this.maxImagesPerClass = v; return this; }
    public ImageFolderOptions backend(Backend v) { this.backend = v == null ? Backend.FFMPEG_OPENCV : v; return this; }
    public ImageFolderOptions eagerDecode(boolean v) { this.eagerDecode = v; return this; }
    public ImageFolderOptions imageTensorCol(String v) { this.imageTensorCol = v; return this; }
    public ImageFolderOptions storeBytes(boolean v) { this.storeBytes = v; return this; }
    public ImageFolderOptions bytesCol(String v) { this.bytesCol = v; return this; }

    public boolean recursive() { return recursive; }
    public boolean includePath() { return includePath; }
    public boolean includeSize() { return includeSize; }
    public boolean includeModifiedTime() { return includeModifiedTime; }
    public int maxImagesPerClass() { return maxImagesPerClass; }
    public Backend backend() { return backend; }
    public boolean eagerDecode() { return eagerDecode; }
    public String imageTensorCol() { return imageTensorCol; }
    public boolean storeBytes() { return storeBytes; }
    public String bytesCol() { return bytesCol; }
}
