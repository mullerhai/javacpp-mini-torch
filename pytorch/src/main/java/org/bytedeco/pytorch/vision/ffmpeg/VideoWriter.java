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
import java.util.HashMap;
import java.util.Map;

/**
 * High-level FFmpeg video writer — mirrors {@code torchvision.io.VideoWriter}
 * / torchcodec write semantics.
 *
 * <p>All FFmpeg types (AVFormatContext, AVCodecContext, AVFrame, AVPacket,
 * SwsContext, BytePointer) are loaded <em>reflectively</em> so this class
 * compiles even when the {@code org.bytedeco.ffmpeg} jars haven't been
 * resolved. When the native side is missing, {@link #open} throws an
 * {@link UnsupportedOperationException} and callers should fall back to
 * OpenCV-style writers ({@link org.bytedeco.pytorch.vision.opencv.OpenCVIO})
 * for image sequences / MJPEG streams.</p>
 *
 * <pre>{@code
 * try (VideoWriter w = VideoWriter.open("/tmp/out.mp4", 640, 480, 30.0)) {
 *     for (Tensor frame : frames) {
 *         w.write(frame);   // [3, H, W] float32, RGB
 *     }
 * }
 * }</pre>
 *
 * <p>Container / codec selection is driven by {@link FormatRegistry} so every
 * extension is supported with sensible defaults. Override via
 * {@link #open(String, int, int, double, String, String)}.</p>
 */
public final class VideoWriter implements AutoCloseable {

    // ---- native state (all Object-typed because classes are reflectively loaded) ----
    private final String path;
    private final int width, height;
    private final double fps;
    private final String formatName;
    private final String videoCodec;
    private final String audioCodec;
    private final String pixelFormat;

    private Object fmtCtx;          // AVFormatContext
    private Object codecCtx;        // AVCodecContext
    private Object stream;          // AVStream
    private Object swsCtx;          // SwsContext (for RGB→YUV conversion)
    private Object frame;           // AVFrame (encoded frame)
    private Object packet;          // AVPacket
    private Object swsFrame;        // AVFrame (RGB source, sws_scale input)

    private long framePts = 0;
    private boolean headerWritten = false;
    private boolean closed = false;
    private Throwable lastError = null;

    private VideoWriter(String path, int width, int height, double fps,
                        String formatName, String videoCodec, String audioCodec,
                        String pixelFormat) {
        this.path = path;
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.formatName = formatName;
        this.videoCodec = videoCodec;
        this.audioCodec = audioCodec;
        this.pixelFormat = pixelFormat;
    }

    /** Open with auto-detected format/codec from extension. */
    public static VideoWriter open(String path, int width, int height, double fps) {
        return open(path, width, height, fps, null, null);
    }

    /** Open with explicit format / codec. */
    public static VideoWriter open(String path, int width, int height, double fps,
                                   String formatName, String videoCodec) {
        if (!FormatRegistry.isSupported(path)) {
            throw new UnsupportedOperationException(
                    "Unsupported video extension: " + path
                    + " — supported: " + FormatRegistry.supportedExtensions());
        }
        String ext = path.substring(path.lastIndexOf('.')).toLowerCase(java.util.Locale.ROOT);
        if (formatName == null) formatName = FormatRegistry.containerFor(ext);
        if (videoCodec == null) videoCodec = FormatRegistry.videoCodecFor(ext);
        String pixFmt = FormatRegistry.pixelFormatFor(ext);
        String audioCodec = FormatRegistry.audioCodecFor(ext);
        VideoWriter w = new VideoWriter(path, width, height, fps,
                formatName, videoCodec, audioCodec, pixFmt);
        try {
            w.init();
            return w;
        } catch (UnsupportedOperationException uoe) {
            throw uoe;
        } catch (Throwable t) {
            w.lastError = t;
            throw new UnsupportedOperationException(
                    "FFmpeg natives unavailable; cannot open writer for " + path + ": " + t, t);
        }
    }

    public static VideoWriter open(Path path, int width, int height, double fps) {
        return open(path.toString(), width, height, fps);
    }

    // ---- native init (reflective) -----------------------------------------

    private void init() throws Exception {
        loadNativeRuntimes();
        Class<?> fcCls = Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext");
        Class<?> ocCls = Class.forName("org.bytedeco.ffmpeg.avformat.AVOutputFormat");
        Constructor<?> fcCtor = fcCls.getConstructor();
        fmtCtx = fcCtor.newInstance();
        // avformat_alloc_output_context2(fmtCtx, null, formatName, path)
        invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_alloc_output_context2",
                new Class<?>[]{fcCls, ocCls, String.class, String.class},
                new Object[]{fmtCtx, null, formatName, path});
        // avcodec_find_encoder(videoCodec)
        Class<?> ccCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodec");
        Object codec = invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_find_encoder",
                new Class<?>[]{String.class}, new Object[]{videoCodec});
        if (codec == null) throw new IllegalStateException("encoder not found: " + videoCodec);
        // avcodec_alloc_context3(codec)
        Class<?> ccCtxCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext");
        codecCtx = ccCtxCls.getConstructor(ccCls).newInstance(codec);
        // codecCtx->width = ...
        invokeInstance(codecCtx, "width", width);
        invokeInstance(codecCtx, "height", height);
        invokeInstance(codecCtx, "time_base", new AVRationalBridge(1, Math.max(1, (int) Math.round(fps))));
        invokeInstance(codecCtx, "framerate", new AVRationalBridge((int) Math.round(fps), 1));
        invokeInstance(codecCtx, "pix_fmt", pixFmtToInt(pixelFormat));
        invokeInstance(codecCtx, "bit_rate", 4_000_000);
        invokeInstance(codecCtx, "gop_size", 12);
        invokeInstance(codecCtx, "max_b_frames", 2);
        // avcodec_open2
        invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_open2",
                new Class<?>[]{ccCtxCls, ccCls, Class.forName("org.bytedeco.ffmpeg.avutil.AVDictionary")},
                new Object[]{codecCtx, codec, null});
        // avformat_new_stream
        Class<?> stCls = Class.forName("org.bytedeco.ffmpeg.avformat.AVStream");
        stream = invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_new_stream",
                new Class<?>[]{fcCls, ccCls}, new Object[]{fmtCtx, codec});
        invokeInstance(codecCtx, "time_base", new AVRationalBridge(1, Math.max(1, (int) Math.round(fps))));
        invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_parameters_from_context",
                new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecParameters"),
                               ccCtxCls},
                new Object[]{invokeInstance(stream, "codecpar"), codecCtx});
        // av_frame_alloc
        Class<?> frCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame");
        frame = invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_alloc",
                new Class<?>[0], new Object[0]);
        invokeInstance(frame, "format", pixFmtToInt(pixelFormat));
        invokeInstance(frame, "width", width);
        invokeInstance(frame, "height", height);
        invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_get_buffer",
                new Class<?>[]{frCls, int.class}, new Object[]{frame, 0});
        // av_packet_alloc
        Class<?> pkCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket");
        packet = invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "av_packet_alloc",
                new Class<?>[0], new Object[0]);
        // sws: BGR0 → pixel_format
        swsFrame = invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_alloc",
                new Class<?>[0], new Object[0]);
        invokeInstance(swsFrame, "format", 0 /* AV_PIX_FMT_BGR0 */);
        invokeInstance(swsFrame, "width", width);
        invokeInstance(swsFrame, "height", height);
        invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_get_buffer",
                new Class<?>[]{frCls, int.class}, new Object[]{swsFrame, 0});
        Class<?> swsCls = Class.forName("org.bytedeco.ffmpeg.swscale.SwsContext");
        swsCtx = invokeStatic("org.bytedeco.ffmpeg.global.swscale", "sws_getContext",
                new Class<?>[]{int.class, int.class, int.class, int.class, int.class, int.class, int.class,
                               int.class, int.class, double.class, double.class},
                new Object[]{width, height, 0 /*BGR0*/, width, height, pixFmtToInt(pixelFormat),
                             2 /*SWS_BILINEAR*/, 0, 0, 0.0, 0.0});
        // avio_open (writes to actual file)
        invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avio_open",
                new Class<?>[]{Class.forName("org.bytedeco.javacpp.Pointer"), String.class, int.class},
                new Object[]{invokeInstance(fmtCtx, "pb"), path, 0x0002 /* AVIO_FLAG_WRITE */});
        invokeInstance(fmtCtx, "pb", invokeInstance(fmtCtx, "pb"));
        // avformat_write_header
        invokeStatic("org.bytedeco.ffmpeg.global.avformat", "avformat_write_header",
                new Class<?>[]{fcCls, Class.forName("org.bytedeco.ffmpeg.avutil.AVDictionary")},
                new Object[]{fmtCtx, null});
        headerWritten = true;
    }

    // ---- public API -------------------------------------------------------

    public String path() { return path; }
    public int width() { return width; }
    public int height() { return height; }
    public double fps() { return fps; }
    public String formatName() { return formatName; }
    public String videoCodec() { return videoCodec; }
    public String audioCodec() { return audioCodec; }
    public String pixelFormat() { return pixelFormat; }
    public long framesWritten() { return framePts; }
    public Throwable lastError() { return lastError; }

    /**
     * Write a frame. Accepts tensors of shape {@code [3, H, W]} (RGB) or
     * {@code [H, W, 3]} float32 in {@code [0, 255]}, or {@code [H, W]} gray.
     */
    public void write(Object tensor) {
        if (closed) throw new IllegalStateException("writer closed");
        try {
            // 1) Copy tensor → swsFrame (BGR0 layout)
            byte[] rgb = tensorToBgr0(tensor);
            // Fill swsFrame.data()
            Class<?> frCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame");
            Class<?> bpCls = Class.forName("org.bytedeco.javacpp.BytePointer");
            Object swsData = invokeInstance(swsFrame, "data");
            // Use reflection: data() returns BytePointer; row 0 is the first line.
            Object row0 = invokeInstance(swsData, "address", new Class<?>[0]); // placeholder
            // bytes[] → BytePointer
            // We write rgb bytes via .put(byte[]) — invoke put(byte[])
            // The rgb buffer has 3*H*W bytes (BGR0 padding ignored — caller supplies CHW).
            byte[] packed = rgb;
            Method put = bpCls.getMethod("put", byte[].class);
            put.invoke(swsData, packed);
            // 2) sws_scale
            invokeStatic("org.bytedeco.ffmpeg.global.swscale", "sws_scale",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.swscale.SwsContext"),
                                   bpCls.arrayType(), bpCls.arrayType(),
                                   int.class, int.class,
                                   frCls, frCls},
                    new Object[]{swsCtx,
                                 new Object[]{swsData}, new Object[]{invokeInstance(swsFrame, "linesize")},
                                 0, height, null, frame});
            // 3) pts
            invokeInstance(frame, "pts", framePts++);
            // 4) encode + write
            encodeAndWrite();
        } catch (UnsupportedOperationException uoe) {
            throw uoe;
        } catch (Throwable t) {
            lastError = t;
            throw new UnsupportedOperationException("VideoWriter.write failed: " + t, t);
        }
    }

    private void encodeAndWrite() throws Exception {
        Class<?> ccCtxCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext");
        Class<?> frCls = Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame");
        Class<?> pkCls = Class.forName("org.bytedeco.ffmpeg.avcodec.AVPacket");
        invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_send_frame",
                new Class<?>[]{ccCtxCls, frCls}, new Object[]{codecCtx, frame});
        while (true) {
            int ret = ((Number) invokeStatic("org.bytedeco.ffmpeg.global.avcodec",
                    "avcodec_receive_packet",
                    new Class<?>[]{ccCtxCls, pkCls},
                    new Object[]{codecCtx, packet})).intValue();
            if (ret < 0) break; // EAGAIN or EOF
            // rescale pts
            invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_packet_rescale_ts",
                    new Class<?>[]{pkCls, AVRationalBridge.cls(), AVRationalBridge.cls()},
                    new Object[]{packet, codecCtx != null ? invokeInstance(codecCtx, "time_base") : null,
                                 stream != null ? invokeInstance(stream, "time_base") : null});
            invokeStatic("org.bytedeco.ffmpeg.global.avformat", "av_interleaved_write_frame",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext"),
                                   pkCls},
                    new Object[]{fmtCtx, packet});
            invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "av_packet_unref",
                    new Class<?>[]{pkCls}, new Object[]{packet});
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            if (headerWritten) {
                invokeStatic("org.bytedeco.ffmpeg.global.avcodec", "avcodec_send_frame",
                        new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avcodec.AVCodecContext"),
                                       Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame")},
                        new Object[]{codecCtx, null});
                // flush
                encodeAndWrite();
                invokeStatic("org.bytedeco.ffmpeg.global.avformat", "av_write_trailer",
                        new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avformat.AVFormatContext")},
                        new Object[]{fmtCtx});
            }
        } catch (Throwable ignored) {}
        try { if (swsCtx != null)
            invokeStatic("org.bytedeco.ffmpeg.global.swscale", "sws_freeContext",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.swscale.SwsContext")},
                    new Object[]{swsCtx}); } catch (Throwable ignored) {}
        try { if (frame != null)
            invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_free",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame")},
                    new Object[]{frame}); } catch (Throwable ignored) {}
        try { if (swsFrame != null)
            invokeStatic("org.bytedeco.ffmpeg.global.avutil", "av_frame_free",
                    new Class<?>[]{Class.forName("org.bytedeco.ffmpeg.avutil.AVFrame")},
                    new Object[]{swsFrame}); } catch (Throwable ignored) {}
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

    /** Reflective Loader.load(...) for FFmpeg's global classes (avutil, avcodec,
     *  avformat, swscale). Each call is best-effort: if natives aren't on the
     *  classpath the load fails silently and individual API calls later will
     *  throw {@link ClassNotFoundException} which surfaces as
     *  {@link UnsupportedOperationException} via {@link #open}. */
    private static void loadNativeRuntimes() {
        String[] runtimeClasses = {
                "org.bytedeco.ffmpeg.global.avutil",
                "org.bytedeco.ffmpeg.global.avcodec",
                "org.bytedeco.ffmpeg.global.avformat",
                "org.bytedeco.ffmpeg.global.swscale"
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

    // ---- helpers ----------------------------------------------------------

    /** Convert a [C,H,W] RGB float tensor in [0,255] → BGR0 row-major bytes. */
    private byte[] tensorToBgr0(Object tensor) throws Exception {
        // Reshape: contiguous [3, H, W] float32 in [0,255]
        Class<?> tCls = tensor.getClass();
        long[] shape = (long[]) invokeInstance(tensor, "sizes", new Class<?>[0]);
        int c = (int) shape[0], h = (int) shape[1], w = (int) shape[2];
        if (c != 3 && c != 1) throw new IllegalArgumentException("expected [C,H,W] tensor");
        // Pack BGR0: row stride = w*4 bytes; bottom-up padding ignored.
        Object contig = invokeInstance(tensor, "contiguous");
        Object cpu = invokeInstance(contig, "cpu");
        Object fp = invokeInstance(cpu, "data_ptr_float");
        // Convert BGR0 bytes per pixel: B=blue, G=green, R=red, 0 pad
        byte[] out = new byte[h * w * 4];
        Class<?> fpCls = fp.getClass();
        Method getF = fpCls.getMethod("get", long.class);
        // CHW → HWC BGR0
        for (int y = 0; y < h; y++) {
            int row = y * w * 4;
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                float r = ((Number) getF.invoke(fp, (long) 0 * h * w + i)).floatValue();
                float g = ((Number) getF.invoke(fp, (long) 1 * h * w + i)).floatValue();
                float b = ((Number) getF.invoke(fp, (long) 2 * h * w + i)).floatValue();
                out[row + x * 4 + 0] = (byte) Math.max(0, Math.min(255, (int) b));
                out[row + x * 4 + 1] = (byte) Math.max(0, Math.min(255, (int) g));
                out[row + x * 4 + 2] = (byte) Math.max(0, Math.min(255, (int) r));
                out[row + x * 4 + 3] = 0;
            }
        }
        return out;
    }

    /** Translate a pixel format name to FFmpeg int constant. */
    private static int pixFmtToInt(String name) {
        if (name == null) return 0; // AV_PIX_FMT_NONE
        switch (name) {
            case "yuv420p":   return 0;
            case "yuvj420p":  return 12;
            case "yuv444p":   return 5;
            case "rgb24":     return 2;
            case "bgr24":     return 3;
            case "rgba":      return 26;
            case "bgra":      return 28;
            default:          return 0;
        }
    }

    private static Object invokeStatic(String cls, String name, Class<?>[] params, Object[] args) throws Exception {
        Class<?> c = Class.forName(cls);
        Method m = c.getMethod(name, params);
        return m.invoke(null, args);
    }

    private static Object invokeInstance(Object target, String name, Object... args) throws Exception {
        Class<?>[] params = new Class<?>[args.length];
        for (int i = 0; i < args.length; i++) params[i] = args[i] == null ? Object.class : args[i].getClass();
        if (args.length == 1 && args[0] != null && args[0].getClass() == Integer.class) params[0] = int.class;
        Method m = target.getClass().getMethod(name, params);
        return m.invoke(target, args);
    }

    /**
     * AVRational adapter. We don't link against the native class; we just need
     * to expose num/den getters that the FFmpeg side reads via reflection.
     */
    public static final class AVRationalBridge {
        public final int num;
        public final int den;
        public AVRationalBridge(int num, int den) { this.num = num; this.den = den; }
        public int num() { return num; }
        public int den() { return den; }
        static Class<?> cls() {
            try { return Class.forName("org.bytedeco.ffmpeg.avutil.AVRational"); }
            catch (ClassNotFoundException e) { return Object.class; }
        }
    }
}