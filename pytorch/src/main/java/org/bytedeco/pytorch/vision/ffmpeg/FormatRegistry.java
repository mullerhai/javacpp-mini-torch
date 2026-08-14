/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option) any later version (collectively, the "License");
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
package org.bytedeco.pytorch.vision.ffmpeg;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Comprehensive format registry — single source of truth for every container
 * extension the {@link VideoFile}, {@link AudioFile}, {@link VideoWriter},
 * {@link AudioWriter} and downstream folder readers support.
 *
 * <p>Each format has a {@link Spec} with:
 * <ul>
 *   <li>{@code videoCodec} — FFmpeg encoder name (e.g. {@code libx264},
 *       {@code libvpx-vp9}, {@code mpeg4}, {@code theora}). {@code null} for
 *       audio-only formats.</li>
 *   <li>{@code audioCodec} — FFmpeg encoder name (e.g. {@code aac}, {@code libmp3lame},
 *       {@code libvorbis}, {@code libopus}). {@code null} for video-only.</li>
 *   <li>{@code mimeType} — RFC 6838 MIME subtype (e.g. {@code mp4},
 *       {@code ogg}, {@code quicktime}).</li>
 *   <li>{@code pixelFormat} — YUV/RGB pixel format for video (e.g. {@code yuv420p}
 *       for browser-safe MP4, {@code yuv444p} for editing).</li>
 * </ul>
 *
 * <p>The registry has no FFmpeg / OpenCV dependency, so it can be used by
 * pure-Java code paths (e.g. sniffers, format-detection, file-extension routing)
 * without ever loading natives.</p>
 */
public final class FormatRegistry {

    private FormatRegistry() {}

    /** Container spec — fields are FFmpeg short names. */
    public static final class Spec {
        public final String container;
        public final String videoCodec;
        public final String audioCodec;
        public final String mimeType;
        public final String pixelFormat;
        public final boolean video;
        public final boolean audio;

        Spec(String container, String videoCodec, String audioCodec,
             String mimeType, String pixelFormat, boolean video, boolean audio) {
            this.container = container;
            this.videoCodec = videoCodec;
            this.audioCodec = audioCodec;
            this.mimeType = mimeType;
            this.pixelFormat = pixelFormat;
            this.video = video;
            this.audio = audio;
        }

        @Override public String toString() {
            return "Spec{" + container + " v=" + videoCodec + " a=" + audioCodec
                    + " mime=" + mimeType + " px=" + pixelFormat + "}";
        }
    }

    // ---- canonical extension → spec map (preserves insertion order) -------

    private static final Map<String, Spec> BY_EXT = new LinkedHashMap<>();

    static {
        // -------- video containers --------
        // MP4 family (H.264 + AAC, browser-friendly yuv420p)
        register(".mp4",  "mp4",          "libx264", "aac",        "mp4",          "yuv420p", true, true);
        register(".m4v",  "mp4",          "libx264", "aac",        "mp4",          "yuv420p", true, true);
        register(".mov",  "mov",          "libx264", "aac",        "quicktime",    "yuv420p", true, true);
        register(".3gp",  "3gp",          "libx264", "aac",        "3gpp",         "yuv420p", true, true);
        register(".3g2",  "3gp2",         "libx264", "aac",        "3gpp2",        "yuv420p", true, true);
        // MKV (Matroska) — flexible, accepts anything
        register(".mkv",  "matroska",     "libx264", "libvorbis",  "x-matroska",   "yuv420p", true, true);
        register(".webm", "webm",         "libvpx-vp9", "libopus", "webm",         "yuv420p", true, true);
        register(".avi",  "avi",          "mpeg4",   "libmp3lame", "x-msvideo",    "yuv420p", true, true);
        // MPEG-1/2 system streams
        register(".mpg",  "mpeg",         "mpeg2video", "mp2",      "mpeg",        "yuv420p", true, true);
        register(".mpeg", "mpeg",         "mpeg2video", "mp2",      "mpeg",        "yuv420p", true, true);
        register(".mpe",  "mpeg",         "mpeg2video", "mp2",      "mpeg",        "yuv420p", true, true);
        register(".m1v",  "mpegvideo",    "mpeg1video", null,       "mpeg",        "yuv420p", true, false);
        register(".m2v",  "mpegvideo",    "mpeg2video", null,       "mpeg",        "yuv420p", true, false);
        register(".mp2",  "mpegvideo",    "mpeg2video", null,       "mpeg",        "yuv420p", true, false);
        register(".mpv",  "mpegvideo",    "mpeg2video", null,       "mpeg",        "yuv420p", true, false);
        // OGG family — Theora + Vorbis (Opus also supported)
        register(".ogv",  "ogg",          "libtheora", "libvorbis","ogg",         "yuv420p", true, true);
        register(".ogg",  "ogg",          "libtheora", "libvorbis","ogg",         "yuv420p", true, true);
        // Transport streams (broadcast)
        register(".ts",   "mpegts",       "libx264", "aac",        "mp2t",         "yuv420p", true, true);
        register(".m2ts", "mpegts",       "libx264", "aac",        "mp2t",         "yuv420p", true, true);
        register(".mts",  "mpegts",       "libx264", "aac",        "mp2t",         "yuv420p", true, true);
        register(".tsv",  "mpegts",       "libx264", "aac",        "mp2t",         "yuv420p", true, true);
        // Flash family
        register(".flv",  "flv",          "libx264", "aac",        "x-flv",        "yuv420p", true, true);
        register(".f4v",  "f4v",          "libx264", "aac",        "mp4",          "yuv420p", true, true);
        register(".f4p",  "f4v",          "libx264", "aac",        "mp4",          "yuv420p", true, true);
        register(".f4a",  "f4v",          null,      "aac",        "mp4",          "yuv420p", false, true);
        register(".f4b",  "f4v",          null,      "aac",        "mp4",          "yuv420p", false, true);
        // Misc video
        register(".wmv",  "asf",          "wmv2",    "wmav2",      "x-ms-wmv",     "yuv420p", true, true);
        // Image-sequence (frame folder) — every frame is a JPG/PNG, container = image2
        register(".jpg",  "image2",       "mjpeg",   null,         "jpeg",         "yuvj420p", true, false);
        register(".jpeg", "image2",       "mjpeg",   null,         "jpeg",         "yuvj420p", true, false);
        register(".png",  "image2",       "png",     null,         "png",          "rgb24",    true, false);
        // -------- audio-only containers --------
        register(".wav",  "wav",          null,      "pcm_s16le", "x-wav",        null,       false, true);
        register(".mp3",  "mp3",          null,      "libmp3lame","mpeg",         null,       false, true);
        register(".flac", "flac",         null,      "flac",      "flac",         null,       false, true);
        register(".m4a",  "ipod",         null,      "aac",       "mp4",          null,       false, true);
        register(".aac",  "adts",         null,      "aac",       "aac",          null,       false, true);
        register(".wma",  "asf",          null,      "wmav2",     "x-ms-wma",     null,       false, true);
        register(".aiff", "aiff",         null,      "pcm_s16be", "aiff",         null,       false, true);
        register(".aif",  "aiff",         null,      "pcm_s16be", "aiff",         null,       false, true);
        register(".opus", "ogg",          null,      "libopus",   "ogg",          null,       false, true);
        register(".mid",  "smf",          null,      "libsoundfont", "sp-midi",   null,       false, true);
        register(".midi", "smf",          null,      "libsoundfont", "sp-midi",   null,       false, true);
        // -------- still images (write paths) --------
        register(".bmp",  null,           null,      null,        "bmp",          null,       false, false);
        register(".webp", null,           null,      null,        "webp",         null,       false, false);
        register(".tiff", null,           null,      null,        "tiff",         null,       false, false);
        register(".tif",  null,           null,      null,        "tiff",         null,       false, false);
        register(".gif",  null,           null,      null,        "gif",          null,       false, false);
        // -------- uppercase duplicates (Windows-friendly) --------
        for (String ext : new String[]{
                ".MP4", ".M4V", ".MOV", ".MKV", ".WEBM", ".AVI", ".FLV", ".WMV",
                ".MPG", ".MPEG", ".OGV", ".OGG", ".3GP", ".3G2", ".TS", ".M2TS", ".MTS",
                ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIFF", ".TIF", ".GIF",
                ".WAV", ".MP3", ".FLAC", ".M4A", ".AAC", ".WMA", ".AIFF", ".AIF",
                ".OPUS", ".MID", ".MIDI"}) {
            Spec lower = BY_EXT.get(ext.toLowerCase(java.util.Locale.ROOT));
            if (lower != null) BY_EXT.put(ext, lower);
        }
    }

    private static void register(String ext, String container, String videoCodec,
                                String audioCodec, String mime, String pix,
                                boolean video, boolean audio) {
        BY_EXT.put(ext, new Spec(container, videoCodec, audioCodec, mime, pix, video, audio));
    }

    // ---- public API -------------------------------------------------------

    /** All supported extensions, lower-case + upper-case. */
    public static Set<String> supportedExtensions() {
        return Collections.unmodifiableSet(new TreeSet<>(BY_EXT.keySet()));
    }

    /** @return spec for extension (with or without leading dot) or {@code null}. */
    public static Spec lookup(String extOrPath) {
        if (extOrPath == null) return null;
        String e = extOrPath;
        int dot = e.lastIndexOf('.');
        if (dot >= 0) e = e.substring(dot);
        return BY_EXT.get(e);
    }

    /** Whether the extension denotes a video container (with or without audio). */
    public static boolean isVideo(String ext) {
        Spec s = lookup(ext);
        return s != null && s.video;
    }

    /** Whether the extension denotes an audio-only container. */
    public static boolean isAudio(String ext) {
        Spec s = lookup(ext);
        return s != null && s.audio && !s.video;
    }

    /** Whether the extension is supported (any kind). */
    public static boolean isSupported(String ext) {
        return lookup(ext) != null;
    }

    /** Get the FFmpeg encoder short name for the video codec of an extension. */
    public static String videoCodecFor(String ext) {
        Spec s = lookup(ext);
        return s == null ? null : s.videoCodec;
    }

    /** Get the FFmpeg encoder short name for the audio codec of an extension. */
    public static String audioCodecFor(String ext) {
        Spec s = lookup(ext);
        return s == null ? null : s.audioCodec;
    }

    /** Container name (e.g. {@code mp4}, {@code matroska}). */
    public static String containerFor(String ext) {
        Spec s = lookup(ext);
        return s == null ? null : s.container;
    }

    /** Recommended pixel format for browser-safe playback (yuv420p) / lossless (rgb24). */
    public static String pixelFormatFor(String ext) {
        Spec s = lookup(ext);
        return s == null ? "yuv420p" : s.pixelFormat;
    }
}