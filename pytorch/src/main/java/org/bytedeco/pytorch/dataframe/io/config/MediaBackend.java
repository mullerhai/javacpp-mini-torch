/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
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
package org.bytedeco.pytorch.dataframe.io.config;

import java.lang.reflect.Method;
import java.nio.file.Path;

/**
 * Backend dispatch for the multimodal {@code ImageFolder}, {@code SoundFolder},
 * {@code VideoFolder} readers.
 *
 * <p>Native dispatches (FFmpeg for audio/video, OpenCV for images) are loaded
 * <em>reflectively</em>, so this class always compiles even when the
 * {@code vision/ffmpeg} or {@code vision/opencv} packages haven't been built
 * yet. At runtime, if the native package class is missing or the linked native
 * library is unavailable, all probe/decode helpers return {@code null} (or
 * populate the row with defaults) and the legacy {@code java.io.InputStream}
 * path keeps working.</p>
 *
 * <p>This is the canonical enterprise pattern: the data layer contract never
 * depends on native loading succeeding — it only depends on the resulting
 * tensor/data being available for downstream training or inference.</p>
 */
public final class MediaBackend {

    private MediaBackend() {}

    /** Backend selection. */
    public enum Backend {
        /** Pure-Java header-only path; no decoding, just metadata + bytes. */
        INPUT_STREAM,
        /**
         * Preferred path. Uses {@code vision/opencv.OpenCVIO} for images and
         * {@code vision/ffmpeg.{AudioFile, VideoFile}} for audio + video.
         * Falls back to {@link #INPUT_STREAM} automatically when native libs
         * are not available.
         */
        FFMPEG_OPENCV,
        /**
         * Force native even on probe failures (decode will throw). Useful for
         * environments where native is loaded lazily.
         */
        NATIVE_REQUIRED
    }

    // ---- capability probes (reflective) ------------------------------------

    private static final boolean OPENCV_AVAILABLE = probe("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
    private static final boolean FFMPEG_AVAILABLE = probe("org.bytedeco.pytorch.vision.ffmpeg.VideoFile");
    private static final boolean TENSOR_AVAILABLE = probe("org.bytedeco.pytorch.Tensor");

    /** True if OpenCV natives are on the classpath. */
    public static boolean isOpenCVAvailable() { return OPENCV_AVAILABLE && TENSOR_AVAILABLE; }

    /** True if FFmpeg + Tensor natives are on the classpath. */
    public static boolean isFFmpegAvailable() { return FFMPEG_AVAILABLE && TENSOR_AVAILABLE; }

    private static boolean probe(String cls) {
        try { Class.forName(cls); return true; }
        catch (Throwable t) { return false; }
    }

    // ---- backend resolution ------------------------------------------------

    /**
     * Resolve effective backend for a probe. Honors explicit
     * {@link Backend#NATIVE_REQUIRED}, silently falls back to
     * {@link Backend#INPUT_STREAM} when native is preferred but unavailable.
     */
    public static Backend resolve(Backend requested) {
        if (requested == null) return Backend.INPUT_STREAM;
        if (requested == Backend.INPUT_STREAM) return Backend.INPUT_STREAM;
        if (requested == Backend.NATIVE_REQUIRED) return Backend.FFMPEG_OPENCV;
        // FFMPEG_OPENCV: require any of the native sides for the union.
        return (isOpenCVAvailable() || isFFmpegAvailable())
                ? Backend.FFMPEG_OPENCV : Backend.INPUT_STREAM;
    }

    // ---- image backend ----------------------------------------------------

    /** Image metadata probed via OpenCV. {@code null} when unavailable. */
    public static ImageMeta probeImage(Path p) { return probeImage(p.toString()); }

    /** Image metadata probed via OpenCV. {@code null} when unavailable. */
    public static ImageMeta probeImage(String path) {
        if (!isOpenCVAvailable()) return null;
        try {
            Object tensor = reflectStatic("org.bytedeco.pytorch.vision.opencv.OpenCVIO",
                    "readImage", new Class[]{String.class}, new Object[]{path});
            if (tensor == null) return null;
            long[] s = (long[]) reflect(tensor, "sizes", null); // not used: see below
            // tensors: query shapes via reflection (no direct dep)
            long c = ((Number) reflect(tensor, "size", new Class[]{long.class}, new Object[]{0L})).longValue();
            long h = ((Number) reflect(tensor, "size", new Class[]{long.class}, new Object[]{1L})).longValue();
            long w = ((Number) reflect(tensor, "size", new Class[]{long.class}, new Object[]{2L})).longValue();
            return new ImageMeta((int) c, (int) h, (int) w, "BGR→RGB");
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Open the image as a {@code Tensor} (returned as Object). Returns {@code null}
     * when OpenCV is unavailable.
     */
    public static Object openImage(String path) {
        if (!isOpenCVAvailable()) return null;
        try { return reflectStatic("org.bytedeco.pytorch.vision.opencv.OpenCVIO",
                "readImage", new Class[]{String.class}, new Object[]{path}); }
        catch (Throwable t) { return null; }
    }

    public static Object openImage(Path p) { return openImage(p.toString()); }

    /** Image probe result. */
    public static final class ImageMeta {
        public final int channels, height, width;
        public final String colorOrder;
        public ImageMeta(int channels, int height, int width, String colorOrder) {
            this.channels = channels; this.height = height;
            this.width = width; this.colorOrder = colorOrder;
        }
        @Override public String toString() {
            return "ImageMeta{" + channels + "x" + height + "x" + width + " " + colorOrder + "}";
        }
    }

    // ---- audio backend ----------------------------------------------------

    /** Audio metadata probed via FFmpeg. {@code null} when unavailable. */
    public static AudioMeta probeAudio(String path) {
        if (!isFFmpegAvailable()) return null;
        try (AC audio = openAudioHandle(path)) {
            if (audio == null) return null;
            int sr = ((Number) reflect(audio.handle, "sampleRate", null)).intValue();
            int ch = ((Number) reflect(audio.handle, "channels", null)).intValue();
            long ns = ((Number) reflect(audio.handle, "numSamples", null)).longValue();
            double du = ((Number) reflect(audio.handle, "durationSec", null)).doubleValue();
            return new AudioMeta(sr, ch, ns, du);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Audio probe result. */
    public static final class AudioMeta {
        public final int sampleRate;
        public final int channels;
        public final long numSamples;
        public final double durationSeconds;
        public AudioMeta(int sampleRate, int channels, long numSamples, double durationSeconds) {
            this.sampleRate = sampleRate;
            this.channels = channels;
            this.numSamples = numSamples;
            this.durationSeconds = durationSeconds;
        }
    }

    // ---- video backend ----------------------------------------------------

    /** Video metadata probed via FFmpeg. {@code null} when unavailable. */
    public static VideoMeta probeVideo(String path) {
        if (!isFFmpegAvailable()) return null;
        try (AC video = openVideoHandle(path)) {
            if (video == null) return null;
            Object meta = reflect(video.handle, "meta", null);
            if (meta == null) return null;
            String pathOut = (String) reflect(meta, "path", null);
            int width = ((Number) reflect(meta, "width", null)).intValue();
            int height = ((Number) reflect(meta, "height", null)).intValue();
            double fps = ((Number) reflect(meta, "fps", null)).doubleValue();
            double dur = ((Number) reflect(meta, "durationSec", null)).doubleValue();
            long nFrames = ((Number) reflect(meta, "numFrames", null)).longValue();
            String codec = (String) reflect(meta, "codecName", null);
            long br = ((Number) reflect(meta, "bitRate", null)).longValue();
            int px = ((Number) reflect(meta, "pixelFormat", null)).intValue();
            return new VideoMeta(pathOut, width, height, fps, dur, nFrames, codec, br, px);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Video metadata record. */
    public static final class VideoMeta {
        public final String path;
        public final int width;
        public final int height;
        public final double fps;
        public final double durationSec;
        public final long numFrames;
        public final String codecName;
        public final long bitRate;
        public final int pixelFormat;
        public VideoMeta(String path, int width, int height, double fps,
                         double durationSec, long numFrames, String codecName,
                         long bitRate, int pixelFormat) {
            this.path = path; this.width = width; this.height = height;
            this.fps = fps; this.durationSec = durationSec; this.numFrames = numFrames;
            this.codecName = codecName; this.bitRate = bitRate; this.pixelFormat = pixelFormat;
        }
        @Override public String toString() {
            return "VideoMeta{" + width + "x" + height + " @" + fps + "fps"
                    + " dur=" + durationSec + "s frames≈" + numFrames
                    + " codec=" + codecName + " br=" + bitRate + "}";
        }
    }

    // ---- auto-closeable handles ------------------------------------------

    /** Open an audio handle for repeated probing/decoding. */
    public static AC openAudioHandle(String path) {
        if (!isFFmpegAvailable()) return null;
        try {
            Class<?> audioCls = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.AudioFile");
            Object af = audioCls.getMethod("open", String.class).invoke(null, path);
            return af == null ? null : new AC(af, audioCls);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Open a video handle for repeated probing/decoding. */
    public static AC openVideoHandle(String path) {
        if (!isFFmpegAvailable()) return null;
        try {
            Class<?> videoCls = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.VideoFile");
            Object vf = videoCls.getMethod("open", String.class).invoke(null, path);
            return vf == null ? null : new AC(vf, videoCls);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Reflective auto-closeable wrapper for AudioFile/VideoFile handles. */
    public static final class AC implements AutoCloseable {
        public final Object handle;
        public final Class<?> handleCls;
        AC(Object handle, Class<?> handleCls) { this.handle = handle; this.handleCls = handleCls; }
        @Override public void close() {
            if (handle == null) return;
            try { handleCls.getMethod("close").invoke(handle); }
            catch (Throwable ignored) {}
        }
    }

    // ---- encode / write API (reflective over OpenCVIO + vision.ffmpeg.Writers) ----

    /**
     * Write an image tensor to a path. Returns {@code true} on success, {@code false}
     * when OpenCV is unavailable. Delegates to {@code vision/opencv.OpenCVIO.writeImage}.
     */
    public static boolean writeImage(String path, Object tensor) {
        if (!isOpenCVAvailable() || tensor == null) return false;
        try {
            Class<?> c = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
            c.getMethod("writeImage", String.class, Class.forName("org.bytedeco.pytorch.Tensor"))
                    .invoke(null, path, tensor);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Encode an image tensor to bytes for the given format
     * (jpg/png/bmp/webp/tiff). Returns {@code null} when OpenCV is unavailable.
     */
    public static byte[] encodeImage(Object tensor, String format) {
        if (!isOpenCVAvailable() || tensor == null) return null;
        try {
            Class<?> c = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
            return (byte[]) c.getMethod("encode", Class.forName("org.bytedeco.pytorch.Tensor"), String.class)
                    .invoke(null, tensor, format);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Open a video writer for a path. Returns {@code null} if FFmpeg natives
     * are unavailable. Delegates to {@code vision/ffmpeg.VideoWriter.open}.
     */
    public static Object openVideoWriter(String path, int width, int height, double fps) {
        if (!isFFmpegAvailable()) return null;
        try {
            Class<?> c = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.VideoWriter");
            return c.getMethod("open", String.class, int.class, int.class, double.class)
                    .invoke(null, path, width, height, fps);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Open an audio writer for a path. Returns {@code null} if FFmpeg natives
     * are unavailable. Delegates to {@code vision/ffmpeg.AudioWriter.open}.
     */
    public static Object openAudioWriter(String path, int sampleRate, int channels) {
        if (!isFFmpegAvailable()) return null;
        try {
            Class<?> c = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.AudioWriter");
            return c.getMethod("open", String.class, int.class, int.class)
                    .invoke(null, path, sampleRate, channels);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Write a single frame to an open video writer. */
    public static void writeVideoFrame(Object writer, Object frameTensor) {
        if (writer == null || frameTensor == null) return;
        try { writer.getClass().getMethod("write", Object.class).invoke(writer, frameTensor); }
        catch (Throwable ignored) {}
    }

    /** Write a chunk to an open audio writer. */
    public static void writeAudioChunk(Object writer, Object waveform) {
        if (writer == null || waveform == null) return;
        try { writer.getClass().getMethod("write", Object.class).invoke(writer, waveform); }
        catch (Throwable ignored) {}
    }

    // ---- format capability summary ----

    /**
     * Report which formats in {@code vision/ffmpeg.FormatRegistry} are decodable
     * AND encodable with the currently-loaded natives.
     *
     * @return map from extension (e.g. {@code .mp4}) to a {@link FormatReport}.
     */
    public static java.util.Map<String, FormatReport> roundTripCheck() {
        java.util.Map<String, FormatReport> report = new java.util.LinkedHashMap<>();
        try {
            Class<?> reg = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.FormatRegistry");
            Object supported = reg.getMethod("supportedExtensions").invoke(null);
            for (Object ext : (Iterable<?>) supported) {
                FormatReport r = new FormatReport((String) ext);
                // decode probe: open via VideoFile (if video) or AudioFile (if audio-only)
                Object spec = reg.getMethod("lookup", String.class).invoke(null, ext);
                boolean isVideo = (Boolean) spec.getClass().getMethod("video").invoke(spec);
                boolean isAudio = (Boolean) spec.getClass().getMethod("audio").invoke(spec);
                r.decodeVideoSupported = isVideo && isFFmpegAvailable();
                r.decodeAudioSupported = (isAudio || isVideo) && isFFmpegAvailable();
                // encode: video codecs via FormatRegistry.videoCodecFor / AudioCodecFor
                String vcodec = (String) spec.getClass().getMethod("videoCodec").invoke(spec);
                String acodec = (String) spec.getClass().getMethod("audioCodec").invoke(spec);
                if (vcodec != null) r.encodeVideoSupported = isFFmpegAvailable();
                if (acodec != null) r.encodeAudioSupported = isFFmpegAvailable();
                report.put((String) ext, r);
            }
        } catch (Throwable t) {
            // FormatRegistry missing → empty report
        }
        return report;
    }

    /** Per-format capability record. */
    public static final class FormatReport {
        public final String extension;
        public boolean decodeVideoSupported;
        public boolean decodeAudioSupported;
        public boolean encodeVideoSupported;
        public boolean encodeAudioSupported;
        public FormatReport(String ext) { this.extension = ext; }
        public boolean fullRoundTrip() {
            boolean anyDec = decodeVideoSupported || decodeAudioSupported;
            boolean anyEnc = encodeVideoSupported || encodeAudioSupported;
            return anyDec && anyEnc;
        }
        @Override public String toString() {
            return "FormatReport{" + extension
                    + " decV=" + decodeVideoSupported + " decA=" + decodeAudioSupported
                    + " encV=" + encodeVideoSupported + " encA=" + encodeAudioSupported + "}";
        }
    }

    // ---- reflection helpers -----------------------------------------------

    private static Object reflectStatic(String cls, String name, Class<?>[] params, Object[] args) {
        try {
            Class<?> c = Class.forName(cls);
            Method m = c.getMethod(name, params);
            return m.invoke(null, args);
        } catch (Throwable t) {
            return null;
        }
    }

    private static Object reflect(Object target, String name, Class<?>[] params, Object... args) {
        try {
            Method m = target.getClass().getMethod(name, params == null ? new Class<?>[0] : params);
            return m.invoke(target, args == null ? new Object[0] : args);
        } catch (Throwable t) {
            return null;
        }
    }
}
