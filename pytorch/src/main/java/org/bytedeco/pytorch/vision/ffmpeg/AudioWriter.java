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

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.file.Path;

/**
 * High-level FFmpeg audio writer — mirrors torchaudio.save / torchcodec.
 *
 * <p>All FFmpeg types (AVFormatContext, AVCodecContext, AVFrame, AVPacket,
 * SwrContext) are loaded <em>reflectively</em> so this class compiles even
 * when the {@code org.bytedeco.ffmpeg} jars haven't been resolved.</p>
 *
 * <pre>{@code
 * try (AudioWriter w = AudioWriter.open("/tmp/out.wav", 16000, 1)) {
 *     w.write(waveform);   // [C, T] float32 in [-1, 1]
 * }
 * }</pre>
 */
public final class AudioWriter implements AutoCloseable {

    private final String path;
    private final int sampleRate, channels;
    private final String formatName, audioCodec;
    private final long totalSamples; // for fixed-length containers

    private Object fmtCtx;
    private Object codecCtx;
    private Object stream;
    private Object frame;
    private Object packet;
    private Object swrCtx;
    private long pts = 0;
    private boolean headerWritten = false;
    private boolean closed = false;
    private Throwable lastError = null;

    private AudioWriter(String path, int sampleRate, int channels,
                        String formatName, String audioCodec, long totalSamples) {
        this.path = path;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.formatName = formatName;
        this.audioCodec = audioCodec;
        this.totalSamples = totalSamples;
    }

    public static AudioWriter open(String path, int sampleRate, int channels) {
        return open(path, sampleRate, channels, 0L);
    }

    public static AudioWriter open(String path, int sampleRate, int channels, long totalSamples) {
        if (!FormatRegistry.isSupported(path)) {
            throw new UnsupportedOperationException(
                    "Unsupported audio extension: " + path
                    + " — supported: " + FormatRegistry.supportedExtensions());
        }
        String ext = path.substring(path.lastIndexOf('.')).toLowerCase(java.util.Locale.ROOT);
        String container = FormatRegistry.containerFor(ext);
        String audioCodec = FormatRegistry.audioCodecFor(ext);
        if (audioCodec == null) {
            throw new UnsupportedOperationException(
                    "No audio codec for container " + container + " (" + ext + ")");
        }
        AudioWriter w = new AudioWriter(path, sampleRate, channels, container, audioCodec, totalSamples);
        try { w.init(); return w; }
        catch (UnsupportedOperationException uoe) { throw uoe; }
        catch (Throwable t) {
            w.lastError = t;
            throw new UnsupportedOperationException(
                    "FFmpeg natives unavailable; cannot open audio writer for " + path + ": " + t, t);
        }
    }

    public static AudioWriter open(Path path, int sampleRate, int channels) {
        return open(path.toString(), sampleRate, channels);
    }

    private void init() throws Exception {
        loadNativeRuntimes();
        Class<?> fcCls = Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext");
        Class<?> ocCls = Class.forName("org.bytedeco.ffmpeg.avformat.AVOutputFormat");
        fmtCtx = fcCls.getConstructor().newInstance();
        invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_alloc_output_context2",
                new Class<?>[]{fcCls, ocCls, String.class, String.class},
                new Object[]{fmtCtx, null, formatName, path});
        Class<?> ccCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodec");
        Object codec = invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_find_encoder",
                new Class<?>[]{String.class}, new Object[]{audioCodec});
        if (codec == null) throw new IllegalStateException("audio encoder not found: " + audioCodec);
        Class<?> ccCtxCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext");
        codecCtx = ccCtxCls.getConstructor(ccCls).newInstance(codec);
        invokeInstance(codecCtx, "bit_rate", 128_000);
        invokeInstance(codecCtx, "sample_rate", sampleRate);
        // channel layout
        Class<?> chlCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVChannelLayout");
        Object layout = chlCls.getConstructor(int.class).newInstance(channels);
        invokeInstance(codecCtx, "ch_layout", layout);
        invokeInstance(codecCtx, "sample_fmt", 1 /* AV_SAMPLE_FMT_S16 */);
        invokeInstance(codecCtx, "time_base", new VideoWriter.AVRationalBridge(1, sampleRate));
        invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_open2",
                new Class<?>[]{ccCtxCls, ccCls, Class.forName("org.bytedeco.ffmpeg.avutil.AVDictionary")},
                new Object[]{codecCtx, codec, null});
        Class<?> stCls = Class.forName("org.bytedeco.ffmpeg.avformat.AVStream");
        stream = invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_new_stream",
                new Class<?>[]{fcCls, ccCls}, new Object[]{fmtCtx, codec});
        // open output file
        invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avio_open",
                new Class<?>[]{Class.forName("org.bytedeco.javacpp.Pointer"), String.class, int.class},
                new Object[]{invokeInstance(fmtCtx, "pb"), path, 0x0002});
        invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_write_header",
                new Class<?>[]{fcCls, Class.forName("org.bytedeco.ffmpeg.avutil.AVDictionary")},
                new Object[]{fmtCtx, null});
        headerWritten = true;
        // alloc frame
        Class<?> frCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame");
        frame = invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_alloc",
                new Class<?>[0], new Object[0]);
        invokeInstance(frame, "format", 1);
        invokeInstance(frame, "sample_rate", sampleRate);
        invokeInstance(frame, "ch_layout", layout);
        invokeInstance(frame, "nb_samples", 1024);
        invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_get_buffer",
                new Class<?>[]{frCls, int.class}, new Object[]{frame, 0});
        Class<?> pkCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket");
        packet = invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "av_packet_alloc",
                new Class<?>[0], new Object[0]);
    }

    /**
     * Write a chunk of audio samples.
     * Accepts {@code [channels, time]} float32 in {@code [-1, 1]} (multi-channel
     * planar layout). For mono waveforms, use {@code [1, T]}.
     */
    public void write(Object waveform) {
        if (closed) throw new IllegalStateException("writer closed");
        try {
            Class<?> tCls = waveform.getClass();
            long[] shape = (long[]) invokeInstance(waveform, "sizes", new Class<?>[0]);
            int c = (int) shape[0], t = (int) shape[1];
            if (c != channels) throw new IllegalArgumentException(
                    "channel count mismatch: tensor=" + c + ", writer=" + channels);
            // PCM float→S16 resample via SwrContext would go here in a fully-native impl.
            // For now we write the planar tensor directly via frame.data with manual int16 conversion.
            Object contig = invokeInstance(waveform, "contiguous");
            Object cpu = invokeInstance(contig, "cpu");
            Object fp = invokeInstance(cpu, "data_ptr_float");
            Method getF = fp.getClass().getMethod("get", long.class);
            // AVFrame.data[0] is interleaved S16 samples: [L0,R0,L1,R1,…]
            Class<?> frCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame");
            Class<?> bpCls = Class.forName("org.bytedeco.javacpp.BytePointer");
            Object frameData = invokeInstance(frame, "data");
            int frameSamples = Math.min(t, 1024);
            short[] s16 = new short[frameSamples * channels];
            for (int i = 0; i < frameSamples; i++) {
                for (int ch = 0; ch < channels; ch++) {
                    float v = ((Number) getF.invoke(fp, (long) ch * t + i)).floatValue();
                    s16[i * channels + ch] = (short) Math.max(Short.MIN_VALUE,
                            Math.min(Short.MAX_VALUE, (int) (v * 32767f)));
                }
            }
            byte[] s16Bytes = new byte[s16.length * 2];
            for (int i = 0; i < s16.length; i++) {
                s16Bytes[i * 2 + 0] = (byte) (s16[i] & 0xff);
                s16Bytes[i * 2 + 1] = (byte) ((s16[i] >> 8) & 0xff);
            }
            bpCls.getMethod("put", byte[].class).invoke(frameData, s16Bytes);
            invokeInstance(frame, "pts", pts);
            pts += frameSamples;
            // encode + write packet
            Class<?> ccCtxCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext");
            Class<?> pkCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket");
            invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_send_frame",
                    new Class<?>[]{ccCtxCls, frCls}, new Object[]{codecCtx, frame});
            while (true) {
                int ret = ((Number) invokeStatic("org.bytedeco.ffmpeg.global.avcodec",
                        "avcodec_receive_packet",
                        new Class<?>[]{ccCtxCls, pkCls},
                        new Object[]{codecCtx, packet})).intValue();
                if (ret < 0) break;
                invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_packet_rescale_ts",
                        new Class<?>[]{pkCls, VideoWriter.AVRationalBridge.cls(),
                                       VideoWriter.AVRationalBridge.cls()},
                        new Object[]{packet, invokeInstance(codecCtx, "time_base"),
                                     invokeInstance(stream, "time_base")});
                invokeStatic("org.bytedeco.ffmpeg.global.avformat", "av_interleaved_write_frame",
                        new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext"),
                                       Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket")},
                        new Object[]{fmtCtx, packet});
                invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "av_packet_unref",
                        new Class<?>[]{pkCls}, new Object[]{packet});
            }
        } catch (UnsupportedOperationException uoe) {
            throw uoe;
        } catch (Throwable e) {
            lastError = e;
            throw new UnsupportedOperationException("AudioWriter.write failed: " + e, e);
        }
    }

    @Override public void close() {
        if (closed) return;
        closed = true;
        try {
            if (headerWritten) {
                Class<?> ccCtxCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext");
                Class<?> frCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame");
                invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_send_frame",
                        new Class<?>[]{ccCtxCls, frCls}, new Object[]{codecCtx, null});
                Class<?> pkCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket");
                while (((Number) invokeStatic("org.bytedeco.ffmpeg.global.avcodec",
                        "avcodec_receive_packet",
                        new Class<?>[]{ccCtxCls, pkCls},
                        new Object[]{codecCtx, packet})).intValue() >= 0) {
                    invokeStatic("org.bytedeco.ffmpeg.global.avformat", "av_interleaved_write_frame",
                            new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext"), pkCls},
                            new Object[]{fmtCtx, packet});
                }
                invokeStatic("org.bytedeco.ffmpeg.global.avformat", "av_write_trailer",
                        new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext")},
                        new Object[]{fmtCtx});
            }
        } catch (Throwable ignored) {}
        try { if (frame != null)
            invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_free",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame")},
                    new Object[]{frame}); } catch (Throwable ignored) {}
        try { if (packet != null)
            invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "av_packet_free",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket")},
                    new Object[]{packet}); } catch (Throwable ignored) {}
        try { if (codecCtx != null)
            invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_free_context",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext")},
                    new Object[]{codecCtx}); } catch (Throwable ignored) {}
        try { if (fmtCtx != null)
            invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_free_context",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext")},
                    new Object[]{fmtCtx}); } catch (Throwable ignored) {}
    }

    public String path() { return path; }
    public int sampleRate() { return sampleRate; }
    public int channels() { return channels; }
    public String formatName() { return formatName; }
    public String audioCodec() { return audioCodec; }
    public long samplesWritten() { return pts; }
    public Throwable lastError() { return lastError; }

    private static Object invokeStatic(String cls, String name, Class<?>[] params, Object[] args) throws Exception {
        Class<?> c = Class.forName(cls);
        Method m = c.getMethod(name, params);
        return m.invoke(null, args);
    }

    /** Reflective Loader.load(...) — see VideoWriter.loadNativeRuntimes. */
    private static void loadNativeRuntimes() {
        String[] runtimeClasses = {
                "org.bytedeco.ffmpeg.global.avutil",
                "org.bytedeco.ffmpeg.global.avcodec",
                "org.bytedeco.ffmpeg.global.avformat"
        };
        try {
            Class<?> loaderCls = Class.forName("org.bytedeco.javacpp.Loader");
            java.lang.reflect.Method load = loaderCls.getMethod("load", Class.class);
            for (String cls : runtimeClasses) {
                try { load.invoke(null, Class.forName(cls)); }
                catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {}
    }

    private static Object invokeInstance(Object target, String name, Object... args) throws Exception {
        Class<?>[] params = new Class<?>[args.length];
        for (int i = 0; i < args.length; i++) params[i] = args[i] == null ? Object.class : args[i].getClass();
        Method m = target.getClass().getMethod(name, params);
        return m.invoke(target, args);
    }
}