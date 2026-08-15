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
 *     http://www.gnu.org/software/classpath/license.html.
 */

/**
 * Enterprise 2D / 3D drawing module for the javacpp-pytorch vision stack.
 *
 * <h2>Overview</h2>
 *
 * <p>This package complements the existing vision modules
 * ({@code org.bytedeco.pytorch.vision.opencv}, {@code …vision.pillow}) with a
 * complete drawing surface that is independent of any specific image
 * representation. The drawing API is built on top of the JDK
 * {@link java.awt.image.BufferedImage} / {@link java.awt.Graphics2D}
 * primitives, so it requires no native code and works in every Java environment.
 *
 * <h2>Classes</h2>
 *
 * <ul>
 *   <li>{@link DrawingCanvas} — the central render surface. Holds state (pen,
 *       brush, font, transform, clip, opacity) and exposes all 2D primitives.</li>
 *   <li>{@link DColor} — immutable RGBA color (parsed from hex, named, HSV,
 *       HSL, Lab, YUV).</li>
 *   <li>{@link DPen}, {@link DBrush}, {@link DFont}, {@link DTransform} — state
 *       building blocks.</li>
 *   <li>{@link DPoint}, {@link DRect}, {@link DEllipse}, {@link DPath} —
 *       geometry value types.</li>
 *   <li>{@link DText} — high-level typography (shadow, outline, glow, block
 *       layout, vertical CJK, bitmap raster).</li>
 *   <li>{@link D3D}, {@link D3DRenderer} — wireframe cube / sphere, axes,
 *       grid, arrows.</li>
 *   <li>{@link DHeatmap}, {@link DGrid}, {@link DMosaic}, {@link DOverlay},
 *       {@link DCharts}, {@link DContour} — high-level visualization helpers.</li>
 *   <li>{@link DAnnotations} — boxes, keypoints, polygons, label badges.</li>
 *   <li>{@link ImageDraw} — Pillow-compatible facade for users who already
 *       know PIL's {@code ImageDraw}.</li>
 *   <li>{@link OpenCVDraw} — exposes native {@code cv::line / circle / rect /
 *       putText / polylines} for soft-dep OpenCV users.</li>
 *   <li>{@link VisionDraw} — Tensor / {@code float[]} / {@code OnnxTensor} /
 *       {@link java.awt.image.BufferedImage} aware facade, the main entry
 *       point of the module.</li>
 * </ul>
 *
 * <h2>Typical usage</h2>
 *
 * <pre>{@code
 * Tensor t = ...; // CHW float [0,255]
 * try (VisionDraw.DrawContext d = VisionDraw.onTensor(t)) {
 *     d.canvas().setFillColor(DColor.of("#1e90ff"));
 *     d.canvas().fillRect(DRect.of(20, 20, 200, 60));
 *     d.canvas().text("Result", 30, 55, DColor.of("white"));
 *
 *     DAnnotations.drawBox(d.canvas(),
 *         DAnnotations.Box.of(50, 100, 250, 200, "car",
 *             DAnnotations.colorForClass(7), 0.94f), 2, true);
 *
 *     DHeatmap.overlay(d.canvas(), heatmap, h, w,
 *         DHeatmap.Colormap.VIRIDIS, 0.4f);
 * }
 * Tensor out = VisionDraw.toTensor(d);
 * }</pre>
 */
package org.bytedeco.pytorch.vision.draw;