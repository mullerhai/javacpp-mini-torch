package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.io.config.MediaBackend.Backend;

/**
 * Options for {@link VideoFolderReader}.
 *
 * <p>Mirrors the structure of {@link ImageFolderOptions} / {@link SoundFolderOptions}
 * with video-specific extras:
 * <ul>
 *   <li>How a sample's class label is determined: {@code folder} (default, subdir name),
 *       {@code csv} (sibling {@code labels.csv} with {@code path,label} columns), or
 *       {@code json} (sibling {@code metadata.jsonl}/{@code metadata.json} keyed by
 *       relative path).</li>
 *   <li>Whether to attach compact per-video metadata ({@code duration_seconds},
 *       {@code fps}, {@code frame_count}, {@code width}, {@code height}, {@code codec},
 *       {@code audio}). Default is on because it is fully zero-copy (sniffs MP4/MKV/WebM
 *       EBML headers); pass {@code .includeMetadata(false)} to skip.</li>
 *   <li>Whether to record the row index in the source path ({@code includeOrdinal}).</li>
 *   <li>Whether to deduplicate identical paths ({@code unique}).</li>
 *   <li>Whether to follow symbolic links ({@code followSymlinks}).</li>
 *   <li>Maximum sample size in bytes ({@code maxBytes}). {@code 0} = no limit.</li>
 *   <li>Maximum sample duration in seconds ({@code maxDurationSeconds}). {@code 0} = no limit.</li>
 *   <li>Encoding target for downstream training: {@link #frameMode()} decides how many
 *       frames the consumer should take ({@code first}, {@code middle}, {@code all},
 *       {@code none}). Recorded as a column; no actual decoding happens here.</li>
 * </ul>
 *
 * <pre>{@code
 *   VideoFolderOptions opts = VideoFolderOptions.defaults()
 *       .recursive(true)
 *       .includePath(true)
 *       .labelMode(VideoFolderOptions.LabelMode.CSV)
 *       .frameMode(VideoFolderOptions.FrameMode.MIDDLE)
 *       .maxDurationSeconds(60.0);
 *   DataFrame df = VideoFolderReader.read("/path/to/videos", opts);
 * }</pre>
 */
public class VideoFolderOptions {

    /** How to infer the class label for each video file. */
    public enum LabelMode {
        /** Sub-directory name (ImageFolder convention). */
        FOLDER,
        /** Sibling {@code labels.csv} with columns {@code path,label[,class_index]}. */
        CSV,
        /** Sibling {@code metadata.jsonl} or {@code metadata.json} keyed by relative path. */
        JSON
    }

    /** Hint for downstream training consumers (recorded as a column, no decoding is done). */
    public enum FrameMode {
        NONE, FIRST, MIDDLE, ALL
    }

    private boolean recursive = true;
    private boolean includePath = false;
    private boolean includeSize = false;
    private boolean includeExtension = false;
    private boolean includeModifiedTime = false;
    private boolean includeName = false;
    private boolean includeMetadata = true;
    private boolean includeOrdinal = false;
    private boolean unique = false;
    private boolean followSymlinks = false;
    private int maxBytes = 0;
    private double maxDurationSeconds = 0.0;
    private LabelMode labelMode = LabelMode.FOLDER;
    private FrameMode frameMode = FrameMode.NONE;
    /** Optional explicit label file overriding the location convention. */
    private java.nio.file.Path labelFile = null;

    /**
     * Backend selection.
     * <ul>
     *   <li>{@link Backend#INPUT_STREAM} (legacy): pure-Java header sniffing for
     *       duration / fps / dimensions / codec. Used when FFmpeg natives are
     *       unavailable.</li>
     *   <li>{@link Backend#FFMPEG_OPENCV} (preferred): probe via
     *       {@code vision/ffmpeg.VideoFile.meta()} — true width/height, fps,
     *       duration, codec, bit rate. Decoded frames/tensors go into a TENSOR
     *       column when {@link #eagerDecode()} is true.</li>
     *   <li>{@link Backend#NATIVE_REQUIRED}: throw if natives cannot be loaded.</li>
     * </ul>
     */
    private Backend backend = Backend.FFMPEG_OPENCV;

    /** Probe-only by default; set true to also load frames. */
    private boolean eagerDecode = false;

    /** How many frames to capture when {@code eagerDecode=true}: {@code N=0} means
     *  all frames via {@code read()} (large memory cost). */
    private int decodeMaxFrames = 0;

    /** Column name for the decoded frames TENSOR cells. {@code null}/blank disables. */
    private String framesCol = "frames";

    /** Column name for a thumbnail (first frame) TENSOR cell. */
    private String thumbnailCol = "thumbnail";

    /** Re-encode decoded frames into a target container; {@code null} disables. */
    private String reencodeTo = null;

    /** Target bit rate (bps) for re-encoding. 0 = use codec default. */
    private int reencodeBitRate = 0;

    /** Fps for re-encoding; 0 = use original fps. */
    private double reencodeFps = 0.0;

    public static VideoFolderOptions defaults() {
        return new VideoFolderOptions();
    }

    public VideoFolderOptions recursive(boolean v) { this.recursive = v; return this; }
    public VideoFolderOptions includePath(boolean v) { this.includePath = v; return this; }
    public VideoFolderOptions includeSize(boolean v) { this.includeSize = v; return this; }
    public VideoFolderOptions includeExtension(boolean v) { this.includeExtension = v; return this; }
    public VideoFolderOptions includeModifiedTime(boolean v) { this.includeModifiedTime = v; return this; }
    public VideoFolderOptions includeName(boolean v) { this.includeName = v; return this; }
    public VideoFolderOptions includeMetadata(boolean v) { this.includeMetadata = v; return this; }
    public VideoFolderOptions includeOrdinal(boolean v) { this.includeOrdinal = v; return this; }
    public VideoFolderOptions unique(boolean v) { this.unique = v; return this; }
    public VideoFolderOptions followSymlinks(boolean v) { this.followSymlinks = v; return this; }
    public VideoFolderOptions maxBytes(int v) { this.maxBytes = v; return this; }
    public VideoFolderOptions maxDurationSeconds(double v) { this.maxDurationSeconds = v; return this; }
    public VideoFolderOptions labelMode(LabelMode v) { this.labelMode = v; return this; }
    public VideoFolderOptions frameMode(FrameMode v) { this.frameMode = v; return this; }
    public VideoFolderOptions labelFile(java.nio.file.Path v) { this.labelFile = v; return this; }
    public VideoFolderOptions backend(Backend v) { this.backend = v == null ? Backend.FFMPEG_OPENCV : v; return this; }
    public VideoFolderOptions eagerDecode(boolean v) { this.eagerDecode = v; return this; }
    public VideoFolderOptions decodeMaxFrames(int v) { this.decodeMaxFrames = v; return this; }
    public VideoFolderOptions framesCol(String v) { this.framesCol = v; return this; }
    public VideoFolderOptions thumbnailCol(String v) { this.thumbnailCol = v; return this; }
    public VideoFolderOptions reencodeTo(String v) { this.reencodeTo = v; return this; }
    public VideoFolderOptions reencodeBitRate(int v) { this.reencodeBitRate = v; return this; }
    public VideoFolderOptions reencodeFps(double v) { this.reencodeFps = v; return this; }

    public boolean recursive() { return recursive; }
    public boolean includePath() { return includePath; }
    public boolean includeSize() { return includeSize; }
    public boolean includeExtension() { return includeExtension; }
    public boolean includeModifiedTime() { return includeModifiedTime; }
    public boolean includeName() { return includeName; }
    public boolean includeMetadata() { return includeMetadata; }
    public boolean includeOrdinal() { return includeOrdinal; }
    public boolean unique() { return unique; }
    public boolean followSymlinks() { return followSymlinks; }
    public int maxBytes() { return maxBytes; }
    public double maxDurationSeconds() { return maxDurationSeconds; }
    public LabelMode labelMode() { return labelMode; }
    public FrameMode frameMode() { return frameMode; }
    public java.nio.file.Path labelFile() { return labelFile; }
    public Backend backend() { return backend; }
    public boolean eagerDecode() { return eagerDecode; }
    public int decodeMaxFrames() { return decodeMaxFrames; }
    public String framesCol() { return framesCol; }
    public String thumbnailCol() { return thumbnailCol; }
    public String reencodeTo() { return reencodeTo; }
    public int reencodeBitRate() { return reencodeBitRate; }
    public double reencodeFps() { return reencodeFps; }
}
