package org.bytedeco.pytorch.dataframe.io.config;

import org.bytedeco.pytorch.dataframe.io.config.MediaBackend.Backend;

public class SoundFolderOptions {
    private boolean recursive = true;
    private boolean includePath = false;
    private boolean includeSize = false;
    private boolean includeExtension = false;
    private boolean includeModifiedTime = false;
    private boolean includeName = false;

    /**
     * Backend selection.
     * <ul>
     *   <li>{@link Backend#INPUT_STREAM}: legacy pure-Java path; report file
     *       metadata + (optionally) extension/size only.</li>
     *   <li>{@link Backend#FFMPEG_OPENCV} (preferred): use
     *       {@code vision/ffmpeg.AudioFile.meta()} for sampleRate / channels /
     *       numSamples / durationSeconds. Decoded waveform tensors go into a
     *       TENSOR-typed column when {@link #eagerDecode()} is true.</li>
     *   <li>{@link Backend#NATIVE_REQUIRED}: same as FFMPEG_OPENCV but throws
     *       on native-load failure.</li>
     * </ul>
     */
    private Backend backend = Backend.FFMPEG_OPENCV;

    /** Probe-only by default; set true to also load waveforms. */
    private boolean eagerDecode = false;

    /** Column name for the decoded waveform TENSOR cells. {@code null}/blank disables. */
    private String waveformCol = "waveform";

    /** When true, store raw audio bytes alongside the waveform. Default false. */
    private boolean storeBytes = false;
    private String bytesCol = "audio_bytes";

    /** Limit per-file metadata for very large audio (>2GiB). */
    private long maxBytes = 0;

    public static SoundFolderOptions defaults() {
        return new SoundFolderOptions();
    }

    public SoundFolderOptions recursive(boolean v) { this.recursive = v; return this; }
    public SoundFolderOptions includePath(boolean v) { this.includePath = v; return this; }
    public SoundFolderOptions includeSize(boolean v) { this.includeSize = v; return this; }
    public SoundFolderOptions includeExtension(boolean v) { this.includeExtension = v; return this; }
    public SoundFolderOptions includeModifiedTime(boolean v) { this.includeModifiedTime = v; return this; }
    public SoundFolderOptions includeName(boolean v) { this.includeName = v; return this; }
    public SoundFolderOptions backend(Backend v) { this.backend = v == null ? Backend.FFMPEG_OPENCV : v; return this; }
    public SoundFolderOptions eagerDecode(boolean v) { this.eagerDecode = v; return this; }
    public SoundFolderOptions waveformCol(String v) { this.waveformCol = v; return this; }
    public SoundFolderOptions storeBytes(boolean v) { this.storeBytes = v; return this; }
    public SoundFolderOptions bytesCol(String v) { this.bytesCol = v; return this; }
    public SoundFolderOptions maxBytes(long v) { this.maxBytes = v; return this; }

    public boolean recursive() { return recursive; }
    public boolean includePath() { return includePath; }
    public boolean includeSize() { return includeSize; }
    public boolean includeExtension() { return includeExtension; }
    public boolean includeModifiedTime() { return includeModifiedTime; }
    public boolean includeName() { return includeName; }
    public Backend backend() { return backend; }
    public boolean eagerDecode() { return eagerDecode; }
    public String waveformCol() { return waveformCol; }
    public boolean storeBytes() { return storeBytes; }
    public String bytesCol() { return bytesCol; }
    public long maxBytes() { return maxBytes; }
}
