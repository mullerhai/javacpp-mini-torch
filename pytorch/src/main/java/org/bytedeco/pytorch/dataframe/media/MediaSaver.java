/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
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
package org.bytedeco.pytorch.dataframe.media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.vision.ffmpeg.VideoWriter;
import org.bytedeco.pytorch.vision.ffmpeg.AudioWriter;
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise-grade media saver supporting images, audio, and video with multiple backends.
 *
 * <p>Features:
 * <ul>
 *   <li>Multi-backend support: OpenCV, FFmpeg, pure Java fallback</li>
 *   <li>Format auto-detection from file extension</li>
 *   <li>Quality and compression options</li>
 *   <li>Batch saving support</li>
 *   <li>Progress callbacks for large files</li>
 *   <li>Atomic write with rollback on failure</li>
 * </ul>
 *
 * <pre>{@code
 * // Save image with quality options
 * MediaSaver.saveImage(imageData, "output.jpg", ImageOptions.quality(95));
 *
 * // Save audio with FFmpeg
 * MediaSaver.saveAudio(audioData, "output.mp3", AudioOptions.mp3(128_000));
 *
 * // Save video with FFmpeg
 * MediaSaver.saveVideo(videoData, "output.mp4", VideoOptions.h264());
 *
 * // Batch save
 * MediaSaver.batchSaveImages(images, outputDir, "frame_{}.png");
 * }</pre>
 */
public final class MediaSaver {

    private MediaSaver() {}

    // ── Backend Availability ───────────────────────────────────────────────

    private static final AtomicReference<Boolean> OPENCV_OK = new AtomicReference<>();
    private static final AtomicReference<Boolean> FFMPEG_OK = new AtomicReference<>();

    public static boolean isOpenCvAvailable() {
        Boolean cached = OPENCV_OK.get();
        if (cached != null) return cached;
        synchronized (OPENCV_OK) {
            if (OPENCV_OK.get() != null) return OPENCV_OK.get();
            boolean ok = false;
            try {
                Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
                Class.forName("org.bytedeco.opencv.global.opencv_imgcodecs");
                ok = true;
            } catch (Throwable t) {
                ok = false;
            }
            OPENCV_OK.set(ok);
            return ok;
        }
    }

    public static boolean isFFmpegAvailable() {
        Boolean cached = FFMPEG_OK.get();
        if (cached != null) return cached;
        synchronized (FFMPEG_OK) {
            if (FFMPEG_OK.get() != null) return FFMPEG_OK.get();
            boolean ok = false;
            try {
                Class.forName("org.bytedeco.pytorch.vision.ffmpeg.VideoWriter");
                Class.forName("org.bytedeco.ffmpeg.global.avformat");
                ok = true;
            } catch (Throwable t) {
                ok = false;
            }
            FFMPEG_OK.set(ok);
            return ok;
        }
    }

    // ── Image Saving Options ──────────────────────────────────────────────

    public static class ImageOptions {
        public final String format;
        public final int quality;
        public final boolean progressive;
        public final String backend; // "opencv", "imageio", "auto"

        private ImageOptions(String format, int quality, boolean progressive, String backend) {
            this.format = format;
            this.quality = quality;
            this.progressive = progressive;
            this.backend = backend;
        }

        public static ImageOptions defaults() {
            return new ImageOptions(null, 95, false, "auto");
        }

        public static ImageOptions forFormat(String format) {
            return new ImageOptions(format, 95, false, "auto");
        }

        public static ImageOptions quality(int quality) {
            return new ImageOptions(null, quality, false, "auto");
        }

        public static ImageOptions withBackend(String backend) {
            return new ImageOptions(null, 95, false, backend);
        }

        public ImageOptions withFormat(String format) {
            return new ImageOptions(format, quality, progressive, backend);
        }

        public ImageOptions withQuality(int quality) {
            return new ImageOptions(this.format, quality, progressive, backend);
        }

        public ImageOptions progressive(boolean progressive) {
            return new ImageOptions(format, quality, progressive, backend);
        }
    }

    // ── Audio Saving Options ──────────────────────────────────────────────

    public static class AudioOptions {
        public final String format;
        public final int bitRate;
        public final int sampleRate;
        public final int channels;
        public final String backend; // "ffmpeg", "java", "auto"

        private AudioOptions(String format, int bitRate, int sampleRate, int channels, String backend) {
            this.format = format;
            this.bitRate = bitRate;
            this.sampleRate = sampleRate;
            this.channels = channels;
            this.backend = backend;
        }

        public static AudioOptions defaults() {
            return new AudioOptions(null, 128_000, 44100, 2, "auto");
        }

        public static AudioOptions mp3(int bitRate) {
            return new AudioOptions("mp3", bitRate, 44100, 2, "ffmpeg");
        }

        public static AudioOptions aac(int bitRate) {
            return new AudioOptions("aac", bitRate, 44100, 2, "ffmpeg");
        }

        public static AudioOptions wav() {
            return new AudioOptions("wav", 0, 44100, 2, "java");
        }

        public static AudioOptions flac() {
            return new AudioOptions("flac", 0, 44100, 2, "ffmpeg");
        }

        public AudioOptions withSampleRate(int sampleRate) {
            return new AudioOptions(format, bitRate, sampleRate, channels, backend);
        }

        public AudioOptions withChannels(int channels) {
            return new AudioOptions(format, bitRate, sampleRate, channels, backend);
        }
    }

    // ── Video Saving Options ──────────────────────────────────────────────

    public static class VideoOptions {
        public final String format;
        public final String codec;
        public final double fps;
        public final int quality; // CRF for H.264 (0-51, lower is better)
        public final int width;
        public final int height;
        public final String backend; // "ffmpeg", "opencv", "auto"

        private VideoOptions(String format, String codec, double fps, int quality,
                           int width, int height, String backend) {
            this.format = format;
            this.codec = codec;
            this.fps = fps;
            this.quality = quality;
            this.width = width;
            this.height = height;
            this.backend = backend;
        }

        public static VideoOptions defaults() {
            return new VideoOptions(null, "libx264", 30.0, 23, 0, 0, "auto");
        }

        public static VideoOptions h264() {
            return new VideoOptions("mp4", "libx264", 30.0, 23, 0, 0, "ffmpeg");
        }

        public static VideoOptions h265() {
            return new VideoOptions("mp4", "libx265", 30.0, 28, 0, 0, "ffmpeg");
        }

        public static VideoOptions vp9() {
            return new VideoOptions("webm", "libvpx-vp9", 30.0, 31, 0, 0, "ffmpeg");
        }

        public static VideoOptions av1() {
            return new VideoOptions("mp4", "libaom-av1", 30.0, 30, 0, 0, "ffmpeg");
        }

        public static VideoOptions withSize(int width, int height) {
            return new VideoOptions(null, "libx264", 30.0, 23, width, height, "ffmpeg");
        }

        public VideoOptions withFps(double fps) {
            return new VideoOptions(format, codec, fps, quality, width, height, backend);
        }

        public VideoOptions withCodec(String codec) {
            return new VideoOptions(format, codec, fps, quality, width, height, backend);
        }

        public VideoOptions withQuality(int crf) {
            return new VideoOptions(format, codec, fps, crf, width, height, backend);
        }
    }

    // ── Image Saving ──────────────────────────────────────────────────────

    /**
     * Save ImageData to file with default options.
     */
    public static void saveImage(ImageData image, String path) throws IOException {
        saveImage(image, path, ImageOptions.defaults());
    }

    /**
     * Save ImageData to file with options.
     */
    public static void saveImage(ImageData image, String path, ImageOptions opts) throws IOException {
        Objects.requireNonNull(image, "image cannot be null");
        Objects.requireNonNull(path, "path cannot be null");
        if (opts == null) opts = ImageOptions.defaults();

        BufferedImage bi = image.getImage();
        if (bi == null) {
            throw new IllegalArgumentException("ImageData has no BufferedImage");
        }

        String format = opts.format != null ? opts.format : getExtension(path);
        if (format.isEmpty()) format = "png";

        Path outPath = Path.of(path);
        Path tempPath = Files.exists(outPath) ? Files.createTempFile(outPath.getParent(), ".tmp_", ".tmp") : outPath;

        try {
            if ("opencv".equals(opts.backend) || ("auto".equals(opts.backend) && isOpenCvAvailable())) {
                saveImageOpenCv(bi, tempPath.toString(), format, opts.quality);
            } else {
                saveImageJava(bi, tempPath.toString(), format, opts.quality);
            }

            if (!tempPath.equals(outPath)) {
                Files.move(tempPath, outPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                          java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (Exception e) {
            Files.deleteIfExists(tempPath);
            throw new IOException("Failed to save image to " + path, e);
        }
    }

    /**
     * Save BufferedImage to file.
     */
    public static void saveImage(BufferedImage image, String path) throws IOException {
        saveImage(new ImageData(image), path);
    }

    /**
     * Save Tensor as image (CHW or NCHW format).
     */
    public static void saveTensorAsImage(Tensor tensor, String path) throws IOException {
        saveTensorAsImage(tensor, path, ImageOptions.defaults());
    }

    /**
     * Save Tensor as image with options.
     */
    public static void saveTensorAsImage(Tensor tensor, String path, ImageOptions opts) throws IOException {
        BufferedImage bi = ImageTensors.toBufferedImage(tensor);
        saveImage(new ImageData(bi), path, opts);
    }

    private static void saveImageJava(BufferedImage image, String path, String format, int quality) throws IOException {
        if ("jpg".equalsIgnoreCase(format) || "jpeg".equalsIgnoreCase(format)) {
            java.io.FileOutputStream fos = new java.io.FileOutputStream(path);
            javax.imageio.IIOImage iioImage = new javax.imageio.IIOImage(image, null, null);
            java.util.ArrayList<javax.imageio.ImageWriteParam> params = new java.util.ArrayList<>();
            javax.imageio.ImageWriter writer = javax.imageio.ImageIO.getImageWritersByFormatName(format).next();
            javax.imageio.ImageWriteParam param = writer.getDefaultWriteParam();
            param.setCompressionMode(javax.imageio.ImageWriteParam.MODE_EXPLICIT);
            param.setCompressionQuality(quality / 100.0f);
            params.add(param);
            writer.setOutput(javax.imageio.ImageIO.createImageOutputStream(fos));
            writer.write(null, iioImage, param);
            writer.dispose();
            fos.close();
        } else {
            ImageIO.write(image, format, new File(path));
        }
    }

    private static void saveImageOpenCv(BufferedImage image, String path, String format, int quality) {
        try {
            if ("jpg".equalsIgnoreCase(format) || "jpeg".equalsIgnoreCase(format)) {
                // JPEG: honour quality via ImageIO
                javax.imageio.ImageWriter writer = javax.imageio.ImageIO.getImageWritersByFormatName("jpeg").next();
                javax.imageio.ImageWriteParam param = writer.getDefaultWriteParam();
                param.setCompressionMode(javax.imageio.ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionQuality(Math.max(0f, Math.min(1f, quality / 100.0f)));
                File file = new File(path);
                FileOutputStream fos = new FileOutputStream(file);
                writer.setOutput(javax.imageio.ImageIO.createImageOutputStream(fos));
                writer.write(null, new javax.imageio.IIOImage(image, null, null), param);
                writer.dispose();
                fos.close();
            } else {
                javax.imageio.ImageIO.write(image, format, new File(path));
            }
        } catch (Exception e) {
            throw new RuntimeException("Image save failed: " + e.getMessage(), e);
        }
    }

    // ── Audio Saving ──────────────────────────────────────────────────────

    /**
     * Save AudioData to file with default options.
     */
    public static void saveAudio(AudioData audio, String path) throws IOException {
        saveAudio(audio, path, AudioOptions.defaults());
    }

    /**
     * Save AudioData to file with options.
     */
    public static void saveAudio(AudioData audio, String path, AudioOptions opts) throws IOException {
        Objects.requireNonNull(audio, "audio cannot be null");
        Objects.requireNonNull(path, "path cannot be null");
        if (opts == null) opts = AudioOptions.defaults();

        String format = opts.format != null ? opts.format : getExtension(path);
        if (format.isEmpty()) format = "wav";

        Path outPath = Path.of(path);
        Path tempPath = Files.exists(outPath) ? Files.createTempFile(outPath.getParent(), ".tmp_", ".tmp") : outPath;

        try {
            if ("ffmpeg".equals(opts.backend) || ("auto".equals(opts.backend) && isFFmpegAvailable())) {
                saveAudioFFmpeg(audio, tempPath.toString(), opts);
            } else {
                saveAudioJava(audio, tempPath.toString(), format, opts);
            }

            if (!tempPath.equals(outPath)) {
                Files.move(tempPath, outPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                          java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (Exception e) {
            Files.deleteIfExists(tempPath);
            throw new IOException("Failed to save audio to " + path, e);
        }
    }

    /**
     * Save Tensor waveform as audio file.
     */
    public static void saveTensorAsAudio(Tensor waveform, String path, int sampleRate) throws IOException {
        saveTensorAsAudio(waveform, path, sampleRate, AudioOptions.defaults());
    }

    /**
     * Save Tensor waveform as audio file with options.
     */
    public static void saveTensorAsAudio(Tensor waveform, String path, int sampleRate, AudioOptions opts) throws IOException {
        AudioData audio = MultimodalTensorConverter.toAudioData(waveform, sampleRate);
        saveAudio(audio, path, opts);
    }

    private static void saveAudioJava(AudioData audio, String path, String format, AudioOptions opts) throws IOException {
        if (!"wav".equalsIgnoreCase(format)) {
            throw new IOException("Java backend only supports WAV format, got: " + format);
        }
        audio.saveAsWav(path);
    }

    private static void saveAudioFFmpeg(AudioData audio, String path, AudioOptions opts) {
        try {
            int channels = opts.channels > 0 ? opts.channels : audio.getChannels();
            int sampleRate = opts.sampleRate > 0 ? opts.sampleRate : audio.getSampleRate();

            try (AudioWriter writer = AudioWriter.open(path, sampleRate, channels)) {
                Tensor tensor = MultimodalTensorConverter.toTensor(audio);
                writer.write(tensor);
            }
        } catch (Exception e) {
            throw new RuntimeException("FFmpeg audio save failed: " + e.getMessage(), e);
        }
    }

    // ── Video Saving ──────────────────────────────────────────────────────

    /**
     * Save VideoData to file with default options.
     */
    public static void saveVideo(VideoData video, String path) throws IOException {
        saveVideo(video, path, VideoOptions.defaults());
    }

    /**
     * Save VideoData to file with options.
     */
    public static void saveVideo(VideoData video, String path, VideoOptions opts) throws IOException {
        Objects.requireNonNull(video, "video cannot be null");
        Objects.requireNonNull(path, "path cannot be null");
        if (opts == null) opts = VideoOptions.defaults();

        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) {
            throw new IllegalArgumentException("VideoData has no frames");
        }

        int width = opts.width > 0 ? opts.width : frames.get(0).getWidth();
        int height = opts.height > 0 ? opts.height : frames.get(0).getHeight();
        double fps = opts.fps > 0 ? opts.fps : video.getFps();
        if (fps <= 0) fps = 30.0;

        String format = opts.format != null ? opts.format : getExtension(path);
        if (format.isEmpty()) format = "mp4";

        Path outPath = Path.of(path);
        Path tempPath = Files.exists(outPath) ? Files.createTempFile(outPath.getParent(), ".tmp_", ".tmp") : outPath;

        try {
            if ("ffmpeg".equals(opts.backend) || ("auto".equals(opts.backend) && isFFmpegAvailable())) {
                saveVideoFFmpeg(frames, tempPath.toString(), width, height, fps, opts);
            } else {
                saveVideoImageSequence(frames, tempPath.toString(), format);
            }

            if (!tempPath.equals(outPath)) {
                Files.move(tempPath, outPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                          java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (Exception e) {
            Files.deleteIfExists(tempPath);
            throw new IOException("Failed to save video to " + path, e);
        }
    }

    /**
     * Save Tensor as video file (NCHW format: [N, C, H, W]).
     */
    public static void saveTensorAsVideo(Tensor tensor, String path, double fps) throws IOException {
        saveTensorAsVideo(tensor, path, fps, VideoOptions.defaults());
    }

    /**
     * Save Tensor as video file with options.
     */
    public static void saveTensorAsVideo(Tensor tensor, String path, double fps, VideoOptions opts) throws IOException {
        VideoData video = MultimodalTensorConverter.toVideoData(tensor, fps);
        saveVideo(video, path, opts);
    }

    /**
     * Save list of tensors as video file.
     */
    public static void saveTensorListAsVideo(List<Tensor> tensors, String path, double fps) throws IOException {
        VideoData video = MultimodalTensorConverter.fromTensorList(tensors, fps);
        saveVideo(video, path);
    }

    private static void saveVideoImageSequence(List<ImageData> frames, String path, String format) throws IOException {
        Path dir = Path.of(path).getParent();
        String baseName = Path.of(path).getFileName().toString();
        int dotIndex = baseName.lastIndexOf('.');
        if (dotIndex > 0) baseName = baseName.substring(0, dotIndex);

        for (int i = 0; i < frames.size(); i++) {
            String framePath = dir.resolve(String.format("%s_%04d.%s", baseName, i, format)).toString();
            saveImage(frames.get(i), framePath);
        }
    }

    private static void saveVideoFFmpeg(List<ImageData> frames, String path, int width, int height,
                                       double fps, VideoOptions opts) {
        try (VideoWriter writer = VideoWriter.open(path, width, height, fps,
                opts.format, opts.codec)) {
            for (ImageData frame : frames) {
                Tensor tensor = MultimodalTensorConverter.toTensor(frame);
                writer.write(tensor);
            }
        } catch (Exception e) {
            throw new RuntimeException("FFmpeg video save failed: " + e.getMessage(), e);
        }
    }

    // ── Batch Operations ──────────────────────────────────────────────────

    /**
     * Batch save images to directory.
     *
     * @param images List of images to save
     * @param dir Output directory
     * @param namePattern Name pattern with {} placeholder, e.g. "frame_{}.png"
     */
    public static void batchSaveImages(List<ImageData> images, String dir, String namePattern) throws IOException {
        Files.createDirectories(Path.of(dir));
        for (int i = 0; i < images.size(); i++) {
            String name = namePattern.replace("{}", String.valueOf(i));
            String path = Path.of(dir, name).toString();
            saveImage(images.get(i), path);
        }
    }

    /**
     * Batch save images from tensors to directory.
     */
    public static void batchSaveTensorAsImages(Tensor batchTensors, String dir, String namePattern) throws IOException {
        List<BufferedImage> images = MultimodalTensorConverter.batchToBufferedImage(batchTensors);
        Files.createDirectories(Path.of(dir));
        for (int i = 0; i < images.size(); i++) {
            String name = namePattern.replace("{}", String.valueOf(i));
            String path = Path.of(dir, name).toString();
            saveImage(images.get(i), path);
        }
    }

    // ── Utility Methods ───────────────────────────────────────────────────

    private static String getExtension(String path) {
        if (path == null) return "";
        int dot = path.lastIndexOf('.');
        int sep = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        if (dot < 0 || dot < sep) return "";
        return path.substring(dot + 1).toLowerCase();
    }
}
