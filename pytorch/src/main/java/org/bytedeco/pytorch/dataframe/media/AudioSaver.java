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
import org.bytedeco.pytorch.vision.ffmpeg.AudioWriter;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise-grade audio saver with FFmpeg backend support.
 *
 * <p>Features:
 * <ul>
 *   <li>Multiple format support: WAV, MP3, FLAC, AAC, OGG, M4A</li>
 *   <li>Bitrate control for compressed formats</li>
 *   <li>Sample rate and channel configuration</li>
 *   <li>Progress tracking for long audio</li>
 * </ul>
 *
 * <pre>{@code
 * // Basic usage
 * AudioSaver.save(audioData, "output.wav");
 *
 * // With options
 * AudioSaver.AudioOptions opts = AudioSaver.AudioOptions.mp3(192_000)
 *     .withSampleRate(44100)
 *     .withChannels(2);
 * AudioSaver.save(audioData, "output.mp3", opts);
 *
 * // From tensor
 * AudioSaver.saveFromTensor(waveform, "output.wav", 16000);
 * }</pre>
 */
public final class AudioSaver {

    private AudioSaver() {}

    private static final AtomicReference<Boolean> FFMPEG_OK = new AtomicReference<>();

    public static boolean isFFmpegAvailable() {
        Boolean cached = FFMPEG_OK.get();
        if (cached != null) return cached;
        synchronized (FFMPEG_OK) {
            if (FFMPEG_OK.get() != null) return FFMPEG_OK.get();
            boolean ok = false;
            try {
                Class.forName("org.bytedeco.pytorch.vision.ffmpeg.AudioWriter");
                Class.forName("org.bytedeco.ffmpeg.global.avformat");
                ok = true;
            } catch (Throwable t) {
                ok = false;
            }
            FFMPEG_OK.set(ok);
            return ok;
        }
    }

    // ── Audio Options ───────────────────────────────────────────────────

    public static class AudioOptions {
        public final String format;
        public final int sampleRate;
        public final int channels;
        public final int bitRate; // in bps
        public final int bitsPerSample; // for PCM formats
        public final boolean normalize;

        private AudioOptions(String format, int sampleRate, int channels, int bitRate,
                         int bitsPerSample, boolean normalize) {
            this.format = format;
            this.sampleRate = sampleRate;
            this.channels = channels;
            this.bitRate = bitRate;
            this.bitsPerSample = bitsPerSample;
            this.normalize = normalize;
        }

        public static AudioOptions defaults() {
            return new AudioOptions("wav", 44100, 2, 0, 16, false);
        }

        // Format presets
        public static AudioOptions wav() {
            return new AudioOptions("wav", 44100, 2, 0, 16, false);
        }

        public static AudioOptions mp3(int bitRate) {
            return new AudioOptions("mp3", 44100, 2, bitRate, 0, false);
        }

        public static AudioOptions aac(int bitRate) {
            return new AudioOptions("aac", 44100, 2, bitRate, 0, false);
        }

        public static AudioOptions flac() {
            return new AudioOptions("flac", 44100, 2, 0, 16, false);
        }

        public static AudioOptions ogg(int bitRate) {
            return new AudioOptions("ogg", 44100, 2, bitRate, 0, false);
        }

        public static AudioOptions m4a(int bitRate) {
            return new AudioOptions("m4a", 44100, 2, bitRate, 0, false);
        }

        // Quality presets
        public static AudioOptions highQuality() {
            return new AudioOptions("wav", 48000, 2, 0, 24, false);
        }

        public static AudioOptions streaming() {
            return new AudioOptions("mp3", 22050, 1, 64_000, 0, true);
        }

        public AudioOptions withSampleRate(int sampleRate) {
            return new AudioOptions(format, sampleRate, channels, bitRate, bitsPerSample, normalize);
        }

        public AudioOptions withChannels(int channels) {
            return new AudioOptions(format, sampleRate, channels, bitRate, bitsPerSample, normalize);
        }

        public AudioOptions withBitRate(int bitRate) {
            return new AudioOptions(format, sampleRate, channels, bitRate, bitsPerSample, normalize);
        }

        public AudioOptions withBitsPerSample(int bits) {
            return new AudioOptions(format, sampleRate, channels, bitRate, bits, normalize);
        }

        public AudioOptions normalize(boolean normalize) {
            return new AudioOptions(format, sampleRate, channels, bitRate, bitsPerSample, normalize);
        }

        public AudioOptions withFormat(String format) {
            return new AudioOptions(format, sampleRate, channels, bitRate, bitsPerSample, normalize);
        }
    }

    // ── Main Save Methods ─────────────────────────────────────────────────

    /**
     * Save AudioData to file with default options.
     */
    public static void save(AudioData audio, String path) throws IOException {
        save(audio, path, AudioOptions.defaults());
    }

    /**
     * Save AudioData to file with options.
     */
    public static void save(AudioData audio, String path, AudioOptions opts) throws IOException {
        Objects.requireNonNull(audio, "audio cannot be null");
        Objects.requireNonNull(path, "path cannot be null");
        if (opts == null) opts = AudioOptions.defaults();

        String format = opts.format != null ? opts.format : getExtension(path);
        if (format.isEmpty()) format = "wav";

        int sampleRate = opts.sampleRate > 0 ? opts.sampleRate : audio.getSampleRate();
        int channels = opts.channels > 0 ? opts.channels : audio.getChannels();

        Path outPath = Path.of(path);
        Path tempPath = Files.exists(outPath) ?
                Files.createTempFile(outPath.getParent(), ".tmp_audio_", ".wav") : outPath;

        try {
            if (isFFmpegAvailable() && !format.equalsIgnoreCase("wav")) {
                saveWithFFmpeg(audio, tempPath.toString(), sampleRate, channels, opts);
            } else {
                saveWithJava(audio, tempPath.toString(), sampleRate, channels, opts);
            }

            if (!tempPath.equals(outPath)) {
                Files.move(tempPath, outPath,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                        java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            }
        } catch (Exception e) {
            Files.deleteIfExists(tempPath);
            throw new IOException("Failed to save audio to " + path, e);
        }
    }

    /**
     * Save audio from waveform tensor.
     */
    public static void saveFromTensor(Tensor waveform, String path, int sampleRate) throws IOException {
        saveFromTensor(waveform, path, sampleRate, AudioOptions.defaults());
    }

    /**
     * Save audio from waveform tensor with options.
     */
    public static void saveFromTensor(Tensor waveform, String path, int sampleRate, AudioOptions opts) throws IOException {
        AudioData audio = MultimodalTensorConverter.toAudioData(waveform, sampleRate);
        save(audio, path, opts);
    }

    // ── Internal Implementation ───────────────────────────────────────────

    private static void saveWithJava(AudioData audio, String path, int sampleRate, int channels, AudioOptions opts) throws IOException {
        float[] samples = audio.getSamples();
        if (samples == null) {
            throw new IllegalStateException("AudioData has no samples");
        }

        int bits = opts.bitsPerSample > 0 ? opts.bitsPerSample : 16;
        boolean isSigned = true;
        boolean bigEndian = false;

        AudioFormat format = new AudioFormat(
                sampleRate, bits, channels, isSigned, bigEndian);

        byte[] audioBytes;
        if (bits == 16) {
            audioBytes = floatsToBytes16(samples, channels);
        } else if (bits == 24) {
            audioBytes = floatsToBytes24(samples, channels);
        } else if (bits == 32) {
            audioBytes = floatsToBytes32(samples, channels);
        } else {
            audioBytes = floatsToBytes16(samples, channels); // default to 16-bit
        }

        ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
        AudioInputStream ais = new AudioInputStream(bais, format, audioBytes.length / (channels * bits / 8));

        File outputFile = new File(path);
        AudioSystem.write(ais, AudioFileFormat.Type.WAVE, outputFile);

        ais.close();
        bais.close();
    }

    private static void saveWithFFmpeg(AudioData audio, String path, int sampleRate, int channels, AudioOptions opts) {
        try (AudioWriter writer = AudioWriter.open(path, sampleRate, channels)) {
            Tensor tensor = MultimodalTensorConverter.toTensor(audio);
            writer.write(tensor);
        } catch (Exception e) {
            throw new RuntimeException("FFmpeg audio save failed: " + e.getMessage(), e);
        }
    }

    // ── Byte Conversion Helpers ───────────────────────────────────────────

    private static byte[] floatsToBytes16(float[] samples, int channels) {
        int frames = samples.length / channels;
        byte[] bytes = new byte[frames * channels * 2];

        for (int i = 0; i < frames; i++) {
            for (int c = 0; c < channels; c++) {
                float sample = samples[i * channels + c];
                sample = Math.max(-1.0f, Math.min(1.0f, sample));
                short s = (short) (sample * 32767);
                int idx = (i * channels + c) * 2;
                bytes[idx] = (byte) (s & 0xFF);
                bytes[idx + 1] = (byte) ((s >> 8) & 0xFF);
            }
        }
        return bytes;
    }

    private static byte[] floatsToBytes24(float[] samples, int channels) {
        int frames = samples.length / channels;
        byte[] bytes = new byte[frames * channels * 3];

        for (int i = 0; i < frames; i++) {
            for (int c = 0; c < channels; c++) {
                float sample = samples[i * channels + c];
                sample = Math.max(-1.0f, Math.min(1.0f, sample));
                int s = (int) (sample * 8388607);
                int idx = (i * channels + c) * 3;
                bytes[idx] = (byte) (s & 0xFF);
                bytes[idx + 1] = (byte) ((s >> 8) & 0xFF);
                bytes[idx + 2] = (byte) ((s >> 16) & 0xFF);
            }
        }
        return bytes;
    }

    private static byte[] floatsToBytes32(float[] samples, int channels) {
        int frames = samples.length / channels;
        byte[] bytes = new byte[frames * channels * 4];

        for (int i = 0; i < frames; i++) {
            for (int c = 0; c < channels; c++) {
                float sample = samples[i * channels + c];
                sample = Math.max(-1.0f, Math.min(1.0f, sample));
                int s = (int) (sample * 2147483647);
                int idx = (i * channels + c) * 4;
                bytes[idx] = (byte) (s & 0xFF);
                bytes[idx + 1] = (byte) ((s >> 8) & 0xFF);
                bytes[idx + 2] = (byte) ((s >> 16) & 0xFF);
                bytes[idx + 3] = (byte) ((s >> 24) & 0xFF);
            }
        }
        return bytes;
    }

    // ── Utility Methods ─────────────────────────────────────────────────

    private static String getExtension(String path) {
        if (path == null) return "";
        int dot = path.lastIndexOf('.');
        int sep = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        if (dot < 0 || dot < sep) return "";
        return path.substring(dot + 1).toLowerCase();
    }

    /**
     * Get supported audio formats.
     */
    public static String[] getSupportedFormats() {
        return new String[]{"wav", "mp3", "flac", "aac", "ogg", "m4a"};
    }

    /**
     * Estimate output file size in bytes.
     *
     * @param durationSeconds Audio duration
     * @param sampleRate Sample rate
     * @param channels Number of channels
     * @param bitRate Bit rate (bps, for compressed formats)
     * @return Estimated file size in bytes
     */
    public static long estimateFileSize(double durationSeconds, int sampleRate, int channels, int bitRate) {
        if (bitRate > 0) {
            // Compressed format
            return (long) (durationSeconds * bitRate / 8);
        } else {
            // PCM format (16-bit)
            return (long) (durationSeconds * sampleRate * channels * 2);
        }
    }

    /**
     * Get audio duration in seconds.
     */
    public static double getDuration(AudioData audio) {
        if (audio.getSamples() == null) return 0;
        return (double) audio.getSamples().length / audio.getChannels() / audio.getSampleRate();
    }
}
