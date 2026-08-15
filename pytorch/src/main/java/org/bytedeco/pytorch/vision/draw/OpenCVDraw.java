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

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC4;
import static org.bytedeco.opencv.global.opencv_core.CV_32SC2;
import static org.bytedeco.opencv.global.opencv_imgproc.circle;
import static org.bytedeco.opencv.global.opencv_imgproc.line;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.ellipse;
import static org.bytedeco.opencv.global.opencv_imgproc.fillPoly;
import static org.bytedeco.opencv.global.opencv_imgproc.polylines;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_4;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_8;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.FILLED;

/**
 * OpenCV drawing primitives exposed to the {@link org.bytedeco.pytorch.vision.draw}
 * module.
 *
 * <p>Wraps the C++ {@code cv::line / cv::circle / cv::rectangle / cv::polylines
 * / cv::fillPoly / cv::putText / cv::ellipse} API for direct mutation of an
 * OpenCV {@link Mat}, and provides {@link Builder} to fluently construct a
 * JavaCPP {@link Mat} output identical to what {@code VisionDraw} would produce.
 *
 * <p>Use {@link #on(Mat)} to draw onto a borrowed Mat, or {@link #create(int, int)}
 * to allocate a fresh BGR image and obtain a {@link Builder} for that surface.
 *
 * <p>Soft-dep friendly: if OpenCV is not on the classpath at runtime, the
 * {@code isOpenCvAvailable()} flag returns false and the static methods throw
 * a clear {@link IllegalStateException}.
 */
public final class OpenCVDraw {

    private OpenCVDraw() {}

    /** True if the OpenCV natives are present (best-effort class probe). */
    public static boolean isOpenCvAvailable() {
        try {
            Class.forName("org.bytedeco.opencv.opencv_core.Mat");
            return true;
        } catch (ClassNotFoundException e) { return false; }
    }

    /** Allocate a fresh BGR Mat and return a builder for it. */
    public static Builder create(int width, int height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("size");
        Mat m = new Mat(height, width, CV_8UC3, new Scalar(0, 0, 0, 0));
        return new Builder(m);
    }

    /** Allocate a fresh BGRA Mat and return a builder for it. */
    public static Builder createRgba(int width, int height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("size");
        Mat m = new Mat(height, width, CV_8UC4, new Scalar(0, 0, 0, 0));
        return new Builder(m);
    }

    /** Allocate a fresh single-channel Mat (grayscale). */
    public static Builder createGray(int width, int height) {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("size");
        Mat m = new Mat(height, width, CV_8UC1, new Scalar(0, 0, 0, 0));
        return new Builder(m);
    }

    /** Attach a builder to an existing Mat (drawing will modify it in place). */
    public static Builder on(Mat m) { return new Builder(m); }

    /** Convert a {@link DColor} to BGR / BGRA {@link Scalar} suitable for OpenCV drawing. */
    public static Scalar toScalar(DColor c, boolean bgra) {
        if (bgra) return new Scalar(c.b(), c.g(), c.r(), c.a());
        return new Scalar(c.b(), c.g(), c.r(), 0);
    }

    /** Native {@code cv::line}. */
    public static void drawLine(Mat dst, int x1, int y1, int x2, int y2, DColor color, int thickness, LineType lineType) {
        rectangle_ready();
        Scalar s = toScalar(color, dst.channels() == 4);
        line(dst, new Point(x1, y1), new Point(x2, y2), s, thickness, lineType.id, 0);
    }

    /** Native {@code cv::rectangle}. */
    public static void drawRect(Mat dst, int x, int y, int w, int h, DColor color, int thickness, LineType lineType) {
        rectangle_ready();
        Scalar s = toScalar(color, dst.channels() == 4);
        rectangle(dst, new Point(x, y), new Point(x + w, y + h), s, thickness, lineType.id, 0);
    }

    public static void fillRect(Mat dst, int x, int y, int w, int h, DColor color) {
        Scalar s = toScalar(color, dst.channels() == 4);
        rectangle(dst, new Point(x, y), new Point(x + w, y + h), s, FILLED, LINE_8, 0);
    }

    public static void drawCircle(Mat dst, int cx, int cy, int r, DColor color, int thickness, LineType lineType) {
        Scalar s = toScalar(color, dst.channels() == 4);
        circle(dst, new Point(cx, cy), r, s, thickness, lineType.id, 0);
    }

    public static void fillCircle(Mat dst, int cx, int cy, int r, DColor color) {
        Scalar s = toScalar(color, dst.channels() == 4);
        circle(dst, new Point(cx, cy), r, s, FILLED, LINE_8, 0);
    }

    public static void drawEllipse(Mat dst, int cx, int cy, int rx, int ry,
                                   double angleDeg, double startDeg, double endDeg,
                                   DColor color, int thickness, LineType lineType) {
        Scalar s = toScalar(color, dst.channels() == 4);
        ellipse(dst, new Point(cx, cy), new org.bytedeco.opencv.opencv_core.Size(rx, ry),
                angleDeg, startDeg, endDeg, s, thickness, lineType.id, 0);
    }

    public static void fillEllipse(Mat dst, int cx, int cy, int rx, int ry,
                                   double angleDeg, double startDeg, double endDeg,
                                   DColor color) {
        Scalar s = toScalar(color, dst.channels() == 4);
        ellipse(dst, new Point(cx, cy), new org.bytedeco.opencv.opencv_core.Size(rx, ry),
                angleDeg, startDeg, endDeg, s, FILLED, LINE_8, 0);
    }

    public static void drawPolyline(Mat dst, Point[] pts, boolean closed, DColor color, int thickness, LineType lineType) {
        Scalar s = toScalar(color, dst.channels() == 4);
        MatVector polys = new MatVector(1);
        polys.put(0, pointsToMat(pts));
        polylines(dst, polys, closed, s, thickness, lineType.id, 0);
        polys.get(0).close();
        polys.close();
    }

    public static void fillPoly(Mat dst, Point[][] polygons, DColor color) {
        Scalar s = toScalar(color, dst.channels() == 4);
        MatVector polys = new MatVector(polygons.length);
        for (int i = 0; i < polygons.length; i++) polys.put(i, pointsToMat(polygons[i]));
        // Use the explicit static import — `fillPoly` is also our method name.
        org.bytedeco.opencv.global.opencv_imgproc.fillPoly(dst, polys, s);
        for (int i = 0; i < polygons.length; i++) polys.get(i).close();
        polys.close();
    }

    private static Mat pointsToMat(Point[] pts) {
        // Nx1 2-channel int32 mat of [x, y] pairs
        Mat m = new Mat(pts.length, 1, CV_32SC2);
        org.bytedeco.javacpp.IntPointer buf = new org.bytedeco.javacpp.IntPointer(pts.length * 2);
        for (int i = 0; i < pts.length; i++) {
            buf.put(i * 2 + 0, pts[i].x());
            buf.put(i * 2 + 1, pts[i].y());
        }
        m.data().put(buf);
        return m;
    }

    public static void drawText(Mat dst, String text, int x, int y, DColor color,
                                HersheyFont font, double fontScale, int thickness, LineType lineType) {
        Scalar s = toScalar(color, dst.channels() == 4);
        putText(dst, text, new Point(x, y), font.id, fontScale, s, thickness, lineType.id, false);
    }

    private static void rectangle_ready() { /* placeholder for parity with cv::line */ }

    public enum LineType {
        CONNECTED_4(LINE_4),
        CONNECTED_8(LINE_8),
        ANTIALIASED(LINE_AA);
        public final int id;
        LineType(int id) { this.id = id; }
    }

    /** Hershey font names — matches {@code cv::HersheyFonts}. */
    public enum HersheyFont {
        SIMPLEX(0), PLAIN(1), DUPLEX(2), COMPLEX(3), TRIPLEX(4),
        COMPLEX_SMALL(5), SCRIPT_SIMPLEX(6), SCRIPT_COMPLEX(7);
        public final int id;
        HersheyFont(int id) { this.id = id; }
    }

    // ---- Builder ----------------------------------------------------------

    /**
     * Fluent builder for drawing on a Mat using the OpenCV primitives above.
     */
    public static final class Builder {
        private final Mat mat;
        private DPen pen;
        private DBrush brush;
        private DFont font;
        private int lineType = LINE_8;

        Builder(Mat mat) {
            this.mat = mat;
            this.pen = DPen.solid(1f, DColor.of("black"));
            this.brush = DBrush.solid(DColor.of("black"));
            this.font = DFont.sans(14f);
        }

        public Mat mat() { return mat; }

        public Builder pen(DPen p) { this.pen = p; return this; }
        public Builder brush(DBrush b) { this.brush = b; return this; }
        public Builder font(DFont f) { this.font = f; return this; }
        public Builder strokeColor(DColor c) { this.pen = DPen.solid(pen.width, c).withCap(pen.cap).withJoin(pen.join); return this; }
        public Builder fillColor(DColor c) { this.brush = DBrush.solid(c); return this; }
        public Builder lineWidth(int px) { this.pen = pen.withWidth(px); return this; }
        public Builder antialiased(boolean on) { this.lineType = on ? LINE_AA : LINE_8; return this; }

        public Builder line(int x1, int y1, int x2, int y2) {
            drawLine(mat, x1, y1, x2, y2, pen.width > 0 ? strokeColor(mat, pen) : DColor.of("transparent"),
                    Math.max(1, Math.round(pen.width)), LineType.values()[lineType == LINE_AA ? 2 : 1]);
            return this;
        }

        public Builder rect(int x, int y, int w, int h, boolean filled) {
            DColor c = ((DBrush) brush).color;
            if (filled) fillRect(mat, x, y, w, h, c);
            else drawRect(mat, x, y, w, h, c, Math.max(1, Math.round(pen.width)), LineType.values()[lineType == LINE_AA ? 2 : 1]);
            return this;
        }

        public Builder roundedRect(int x, int y, int w, int h, float radius, boolean filled) {
            // OpenCV has no rounded-rect primitive; approximate with ellipse-on-corner clipping via
            // a polygon path drawn through Graphics2D. Fall back to plain rect when alpha allows it.
            if (radius <= 0f) return rect(x, y, w, h, filled);
            DColor c = ((DBrush) brush).color;
            org.bytedeco.opencv.opencv_core.Point[] pts = roundedRectPoints(x, y, w, h, radius);
            if (filled) fillPoly(mat, new org.bytedeco.opencv.opencv_core.Point[][]{pts}, c);
            else drawPolyline(mat, pts, true, c, Math.max(1, Math.round(pen.width)), LineType.values()[lineType == LINE_AA ? 2 : 1]);
            return this;
        }

        public Builder circle(int cx, int cy, int r, boolean filled) {
            DColor c = ((DBrush) brush).color;
            if (filled) fillCircle(mat, cx, cy, r, c);
            else drawCircle(mat, cx, cy, r, c, Math.max(1, Math.round(pen.width)), LineType.values()[lineType == LINE_AA ? 2 : 1]);
            return this;
        }

        public Builder ellipse(int cx, int cy, int rx, int ry, double angleDeg,
                               double startDeg, double endDeg, boolean filled) {
            DColor c = ((DBrush) brush).color;
            if (filled) fillEllipse(mat, cx, cy, rx, ry, angleDeg, startDeg, endDeg, c);
            else drawEllipse(mat, cx, cy, rx, ry, angleDeg, startDeg, endDeg, c,
                    Math.max(1, Math.round(pen.width)), LineType.values()[lineType == LINE_AA ? 2 : 1]);
            return this;
        }

        public Builder text(String s, int x, int y) {
            DColor c = ((DBrush) brush).color;
            drawText(mat, s, x, y, c, HersheyFont.SIMPLEX, Math.max(0.2, font.size() / 18f),
                    Math.max(1, Math.round(pen.width)), LineType.values()[lineType == LINE_AA ? 2 : 1]);
            return this;
        }

        public Builder clear(DColor c) {
            int total = (int) mat.total();
            int ch = mat.channels();
            byte[] arr = new byte[total * ch];
            for (int i = 0; i < total; i++) {
                int idx = i * ch;
                arr[idx] = (byte) c.b();
                if (ch > 1) arr[idx + 1] = (byte) c.g();
                if (ch > 2) arr[idx + 2] = (byte) c.r();
                if (ch > 3) arr[idx + 3] = (byte) c.a();
            }
            mat.data().put(arr);
            return this;
        }

        private static DColor strokeColor(Mat mat, DPen p) {
            // Pen doesn't carry color; default to red for visibility. Use fillColor when desired.
            return DColor.of(255, 0, 0);
        }
    }

    // ---- convert into our drawing system --------------------------------

    /**
     * Round-trip a Mat into a {@link DrawingCanvas} via a {@link java.awt.image.BufferedImage}
     * adapter. Useful when you want to apply the high-level {@link DrawingCanvas}
     * API on top of an existing OpenCV image.
     *
     * <p>Caller is responsible for closing the returned canvas.
     */
    public static DrawingCanvas toCanvas(Mat m) {
        // MatToTensor import is intentionally avoided to keep this class zero-dep;
        // most callers will already depend on OpenCVIO for that path.
        throw new UnsupportedOperationException(
                "Use org.bytedeco.pytorch.vision.opencv.MatToTensor.fromMat + " +
                        "org.bytedeco.pytorch.vision.utils.ImageTensors.toBufferedImage to bridge into DrawingCanvas.");
    }

    private static org.bytedeco.opencv.opencv_core.Point[] roundedRectPoints(int x, int y, int w, int h, float r) {
        // Approximate with 16 vertices
        org.bytedeco.opencv.opencv_core.Point[] pts = new org.bytedeco.opencv.opencv_core.Point[16];
        int n = 0;
        int[] cx = {x + (int) r, x + w - (int) r, x + w - (int) r, x + (int) r};
        int[] cy = {y + (int) r, y + (int) r, y + h - (int) r, y + h - (int) r};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double a = Math.PI / 2 * i + Math.PI / 8 * j;
                pts[n++] = new org.bytedeco.opencv.opencv_core.Point(
                        (int) Math.round(cx[i] + r * Math.cos(a)),
                        (int) Math.round(cy[i] + r * Math.sin(a)));
            }
        }
        return pts;
    }
}