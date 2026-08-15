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
 */
package org.bytedeco.pytorch.vision.draw;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Top-level enterprise facade for tensor-aware drawing in the
 * {@link org.bytedeco.pytorch.vision.draw} module.
 *
 * <p>{@code VisionDraw} accepts any of the four canonical "image source"
 * representations used in this repo:
 *
 * <ul>
 *   <li>{@link Tensor} — CHW float {@code [0,255]} or {@code [0,1]}, or HWC float, or uint8</li>
 *   <li>{@code float[]} — flat pixel data (CHW or HWC; we pick the most useful layout)</li>
 *   <li>{@code ai.onnxruntime.OnnxTensor} (loaded reflectively so the dep stays optional)</li>
 *   <li>{@link BufferedImage} / {@link org.bytedeco.pytorch.vision.pillow.Image} (passthrough)</li>
 * </ul>
 *
 * <p>and produces either an in-memory image or a {@link Tensor} for downstream
 * pipelines. The class is intentionally stateless — all state is held by the
 * {@link DrawContext} it returns.
 *
 * <pre>{@code
 * Tensor t = ...;                                 // CHW float
 * try (DrawContext d = VisionDraw.onTensor(t)) {
 *     d.setFillColor(DColor.of("lime"));
 *     d.canvas().fillRect(DRect.of(10, 10, 100, 50));
 *     d.canvas().text("hello", 16, 30, DColor.of("black"));
 * }
 * Tensor out = d.toTensor();                      // CHW float [0,255]
 * }</pre>
 *
 * <p>Or for ONNX tensors, the same surface:
 *
 * <pre>{@code
 * OnnxTensor ot = session.getOutput("heatmap");   // float[N,1,H,W]
 * try (DrawContext d = VisionDraw.onOnnxTensor(ot)) {
 *     DHeatmap.apply(d.canvas(), d.floatArray(), 0);
 * }
 * }</pre>
 */
public final class VisionDraw {

    private VisionDraw() {}

    // ---- factories -------------------------------------------------------

    public static DrawContext on(int width, int height) {
        return new DrawContext(DrawingCanvas.create(width, height));
    }

    public static DrawContext on(int width, int height, int imageType) {
        return new DrawContext(DrawingCanvas.create(width, height, imageType));
    }

    public static DrawContext on(BufferedImage img) {
        return new DrawContext(DrawingCanvas.on(img));
    }

    /** Tensor → image. Accepts CHW/HWC/NCHW float {@code [0,1]} or {@code [0,255]}. */
    public static DrawContext onTensor(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        BufferedImage bi = ImageTensors.toBufferedImage(t);
        return new DrawContext(DrawingCanvas.on(bi)).sourceTensor(t);
    }

    /** float[] → image. {@code chw[c * h * w + y * w + x]} (default) or {@code hwc} if requested. */
    public static DrawContext onFloat(float[] data, int c, int h, int w, boolean hwc) {
        Objects.requireNonNull(data, "data");
        if (c <= 0 || h <= 0 || w <= 0) throw new IllegalArgumentException("c,h,w must be > 0");
        if (data.length != c * h * w) throw new IllegalArgumentException("data.length != c*h*w");
        float[] chw = hwc ? toChw(data, c, h, w) : data;
        Tensor t = torch.tensor(chw).reshape(c, h, w);
        return onTensor(t);
    }

    public static DrawContext onFloat(float[] data, int h, int w) {
        return onFloat(data, 1, h, w, false);
    }

    /** RGB bytes (length = h * w * 3, HWC) → image. */
    public static DrawContext onRgbBytes(byte[] rgb, int h, int w) {
        Objects.requireNonNull(rgb, "rgb");
        if (rgb.length != h * w * 3) throw new IllegalArgumentException("rgb.length != h*w*3");
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int[] pixels = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 3;
                int r = rgb[idx] & 0xFF;
                int g = rgb[idx + 1] & 0xFF;
                int b = rgb[idx + 2] & 0xFF;
                pixels[y * w + x] = (r << 16) | (g << 8) | b;
            }
        }
        img.setRGB(0, 0, w, h, pixels, 0, w);
        return new DrawContext(DrawingCanvas.on(img));
    }

    /**
     * ONNX Runtime {@code ai.onnxruntime.OnnxTensor} → image. The {@code OnnxTensor}
     * class is loaded reflectively so this module stays soft-dep friendly.
     */
    public static DrawContext onOnnxTensor(Object onnxTensor) {
        if (onnxTensor == null) throw new IllegalArgumentException("null onnxTensor");
        String cls = onnxTensor.getClass().getName();
        if (!cls.equals("ai.onnxruntime.OnnxTensor")) {
            throw new IllegalArgumentException("expected ai.onnxruntime.OnnxTensor, got " + cls);
        }
        try {
            // Pull info via reflection
            Class<?> otCls = onnxTensor.getClass();
            // The OnnxTensorLike superclass has overloaded getInfo() (TensorInfo / ValueInfo);
            // resolve by walking up to the supertype and picking the TensorInfo variant.
            Class<?> infoOwner = otCls;
            java.lang.reflect.Method getInfo = null;
            while (infoOwner != null) {
                try {
                    getInfo = infoOwner.getDeclaredMethod("getInfo", java.lang.reflect.Modifier.class != null
                            ? new Class<?>[]{} : new Class<?>[]{});
                    break;
                } catch (NoSuchMethodException ex) {
                    infoOwner = infoOwner.getSuperclass();
                }
            }
            if (getInfo == null) throw new NoSuchMethodException("getInfo()");
            getInfo.setAccessible(true);
            Object info = getInfo.invoke(onnxTensor);
            java.lang.reflect.Method getShape = info.getClass().getMethod("getShape");
            long[] shape = (long[]) getShape.invoke(info);
            // Get float buffer
            java.lang.reflect.Method getFloatBuffer = otCls.getMethod("getFloatBuffer");
            java.nio.FloatBuffer fb = (java.nio.FloatBuffer) getFloatBuffer.invoke(onnxTensor);
            float[] data = new float[fb.remaining()];
            fb.get(data);
            // Take last 3 dims as image; treat N=1
            int[] shp = new int[shape.length];
            for (int i = 0; i < shape.length; i++) shp[i] = (int) shape[i];
            if (shp.length < 2) throw new IllegalArgumentException("ONNX tensor must be at least 2D");
            // heuristic: last 3 dims for image; if 2D take grayscale
            int c = 1, h, w;
            if (shp.length >= 3) {
                c = shp[shp.length - 3];
                h = shp[shp.length - 2];
                w = shp[shp.length - 1];
            } else {
                h = shp[shp.length - 2];
                w = shp[shp.length - 1];
            }
            float[] chw = chwFromAny(data, shp, c, h, w);
            // Normalize to [0,255] if looks like [0,1]
            float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
            for (float v : chw) { if (v < min) min = v; if (v > max) max = v; }
            if (max <= 1.5f && min >= -0.001f) {
                for (int i = 0; i < chw.length; i++) chw[i] *= 255f;
            }
            return onFloat(chw, c, h, w, false);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("failed to read OnnxTensor — is onnxruntime on the classpath?", e);
        }
    }

    // ---- output -----------------------------------------------------------

    /** {@link DrawContext} → {@link Tensor} (CHW float {@code [0,255]}). */
    public static Tensor toTensor(DrawContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        return ImageTensors.toTensor(ctx.canvas().image()).mul(new Scalar(255f));
    }

    public static Tensor toTensor(DrawingCanvas canvas) {
        return ImageTensors.toTensor(canvas.image()).mul(new Scalar(255f));
    }

    /** Image → float {@code [0,255]} HWC buffer (used by ONNX output). */
    public static float[] toFloatHwc255(DrawingCanvas canvas) {
        BufferedImage img = canvas.image();
        int w = img.getWidth(), h = img.getHeight();
        int[] pixels = img.getRGB(0, 0, w, h, null, 0, w);
        float[] out = new float[w * h * 3];
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            out[i * 3 + 0] = ((p >> 16) & 0xFF);
            out[i * 3 + 1] = ((p >> 8) & 0xFF);
            out[i * 3 + 2] = (p & 0xFF);
        }
        return out;
    }

    // ---- helpers ----------------------------------------------------------

    private static float[] toChw(float[] hwc, int c, int h, int w) {
        float[] chw = new float[c * h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int src = (y * w + x) * c;
                for (int k = 0; k < c; k++) {
                    chw[k * h * w + y * w + x] = hwc[src + k];
                }
            }
        }
        return chw;
    }

    private static float[] chwFromAny(float[] data, int[] shape, int c, int h, int w) {
        // data is contiguous in some layout (NCHW or NHWC). For now treat as NCHW if last dim
        // matches expected channel placement; otherwise reshape with transpose.
        long total = 1; for (int s : shape) total *= s;
        if (total != data.length) throw new IllegalArgumentException("shape != data size");
        // Simple case: shape.length == 3 → HWC. Reshape to CHW.
        if (shape.length == 3 && shape[2] == c) {
            return toChw(data, c, shape[0], shape[1]);
        }
        if (shape.length == 4 && shape[1] == c) {
            // NCHW — collapse first dim (assume N=1)
            return data;
        }
        if (shape.length == 4 && shape[3] == c) {
            float[] flat = new float[c * h * w];
            for (int n = 0; n < shape[0]; n++) {
                int offset = n * c * h * w;
                float[] slice = new float[c * h * w];
                System.arraycopy(data, offset, slice, 0, c * h * w);
                float[] chw = toChw(slice, c, h, w);
                // first slice wins; ignore N>1
                if (n == 0) System.arraycopy(chw, 0, flat, 0, c * h * w);
            }
            return flat;
        }
        // Fall back: assume already CHW.
        return data;
    }

    // ---- DrawContext ------------------------------------------------------

    /**
     * Lifetime wrapper around a {@link DrawingCanvas}. Use {@link #close()} to
     * release AWT graphics resources; afterwards you must not use the canvas.
     */
    public static final class DrawContext implements AutoCloseable {
        private final DrawingCanvas canvas;
        private Tensor sourceTensor;
        private Object sourceOnnx;

        DrawContext(DrawingCanvas canvas) {
            this.canvas = canvas;
        }

        public DrawingCanvas canvas() { return canvas; }
        public BufferedImage image() { return canvas.image(); }

        DrawContext sourceTensor(Tensor t) { this.sourceTensor = t; return this; }
        public Tensor sourceTensor() { return sourceTensor; }

        /** Best-effort access to the raw float pixel array of the current canvas
         *  (synchronous, RGB packed, 0..255). Useful for benchmarks / onnx I/O. */
        public float[] floatArray() {
            BufferedImage img = canvas.image();
            int w = img.getWidth(), h = img.getHeight();
            int[] pix = img.getRGB(0, 0, w, h, null, 0, w);
            float[] out = new float[w * h * 3];
            for (int i = 0; i < pix.length; i++) {
                int p = pix[i];
                out[i * 3 + 0] = (p >> 16) & 0xFF;
                out[i * 3 + 1] = (p >> 8) & 0xFF;
                out[i * 3 + 2] = p & 0xFF;
            }
            return out;
        }

        public int width() { return canvas.width(); }
        public int height() { return canvas.height(); }

        @Override public void close() { canvas.close(); }
    }
}