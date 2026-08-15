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

import org.bytedeco.pytorch.vision.draw.D3D.Camera;
import org.bytedeco.pytorch.vision.draw.D3D.Mat4;
import org.bytedeco.pytorch.vision.draw.D3D.V3;

import java.util.Objects;

/**
 * 3D drawing facade for the {@link org.bytedeco.pytorch.vision.draw} module.
 *
 * <p>Builds a perspective view × projection matrix from a {@link Camera}, then
 * draws wireframe / solid primitives directly on a {@link DrawingCanvas}.
 *
 * <p>Supported primitives: cube wireframe, cube filled (with hidden-surface
 * back-face culling), sphere wireframe, axes, grid, arrows, point clouds,
 * generic line set.
 *
 * <pre>{@code
 * try (DrawingCanvas c = DrawingCanvas.create(800, 600)) {
 *     c.clear(DColor.of("#111"));
 *     D3DRenderer r3d = new D3DRenderer(c, Camera.defaultCamera(800f / 600f));
 *     r3d.drawAxes(1.5f);
 *     r3d.drawGrid(2f, 10);
 *     r3d.drawCubeWireframe(1f, DColor.of("lime"));
 *     r3d.drawSphereWireframe(0.6f, 16, 16, DColor.of("deepskyblue"));
 * }
 * }</pre>
 */
public final class D3DRenderer {

    private final DrawingCanvas canvas;
    private Camera camera;
    private Mat4 viewProj;
    /** Axis-aligned bounding box that maps to the canvas viewport, in NDC. */
    private float viewportX = -1f, viewportY = -1f, viewportW = 2f, viewportH = 2f;

    public D3DRenderer(DrawingCanvas canvas, Camera camera) {
        this.canvas = Objects.requireNonNull(canvas, "canvas");
        setCamera(camera);
    }

    public Camera camera() { return camera; }

    public D3DRenderer setCamera(Camera camera) {
        this.camera = Objects.requireNonNull(camera, "camera");
        camera.aspect = (canvas.width() * viewportW) / (canvas.height() * viewportH);
        this.viewProj = camera.projectionMatrix().mul(camera.viewMatrix());
        return this;
    }

    /** Custom NDC viewport mapping (default = full NDC). */
    public D3DRenderer setViewport(float ndcX, float ndcY, float ndcW, float ndcH) {
        this.viewportX = ndcX;
        this.viewportY = ndcY;
        this.viewportW = ndcW;
        this.viewportH = ndcH;
        return setCamera(camera);
    }

    private DPoint toScreen(V3 v) {
        V3 q = viewProj.transformPoint(v);
        if (Math.abs(q.z) > 1f) {
            // outside the unit cube — clamp to canvas so degenerate draws do not throw
            return DPoint.of(-10000, -10000);
        }
        float sx = canvas.width() * (0.5f * (q.x * viewportW + viewportW + 2f * viewportX) / viewportW + 0.5f);
        float sy = canvas.height() * (0.5f * (-q.y * viewportH + viewportH + 2f * viewportY) / viewportH + 0.5f);
        return DPoint.of(sx, sy);
    }

    /** Depth-buffer-free perspective-sort key (smaller = farther). */
    private float depth(V3 v) {
        V3 q = viewProj.transformPoint(v);
        return q.z;
    }

    // ---- primitives -------------------------------------------------------

    public D3DRenderer drawAxes(float len) {
        V3[][] ax = D3D.axes(len);
        drawLine(ax[0][0], ax[0][1], DColor.of(255, 80, 80), 2f);
        drawLine(ax[1][0], ax[1][1], DColor.of(80, 255, 80), 2f);
        drawLine(ax[2][0], ax[2][1], DColor.of(80, 80, 255), 2f);
        return this;
    }

    public D3DRenderer drawGrid(float size, int cells) {
        V3[][] lines = D3D.grid2D(size, cells);
        canvas.save();
        canvas.setStrokeColor(DColor.of(80, 80, 80));
        canvas.setPen(DPen.solid(1f, DColor.of(80, 80, 80)));
        for (V3[] line : lines) {
            canvas.drawLine(toScreen(line[0]), toScreen(line[1]));
        }
        canvas.restore();
        return this;
    }

    public D3DRenderer drawCubeWireframe(float size, DColor color) {
        V3[] verts = D3D.cubeVertices(size);
        int[][] edges = D3D.cubeEdges();
        canvas.save();
        canvas.setPen(DPen.solid(1.5f, color).colored(color));
        canvas.setStrokeColor(color);
        for (int[] edge : edges) {
            canvas.drawLine(toScreen(verts[edge[0]]), toScreen(verts[edge[1]]));
        }
        canvas.restore();
        return this;
    }

    /** Filled cube with back-face culling + simple painter sort (depth = avg of 4 verts). */
    public D3DRenderer drawCubeFilled(float size, DBrush brush, DColor edge) {
        V3[] v = D3D.cubeVertices(size);
        int[][] faces = D3D.cubeFaces();
        // compute depths
        float[] d = new float[faces.length];
        Integer[] order = new Integer[faces.length];
        for (int i = 0; i < faces.length; i++) {
            float avg = 0;
            for (int k = 0; k < 4; k++) avg += depth(v[faces[i][k]]);
            d[i] = avg / 4f;
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Float.compare(d[b], d[a])); // farthest first
        canvas.save();
        if (brush != null) canvas.setBrush(brush);
        if (edge != null) {
            canvas.setPen(DPen.solid(1f, edge).colored(edge));
            canvas.setStrokeColor(edge);
        }
        for (int idx : order) {
            DPoint[] pts = new DPoint[4];
            for (int k = 0; k < 4; k++) pts[k] = toScreen(v[faces[idx][k]]);
            canvas.fillPolygon(pts);
            if (edge != null) canvas.drawPolygon(pts);
        }
        canvas.restore();
        return this;
    }

    public D3DRenderer drawSphereWireframe(float r, int latBands, int lonBands, DColor color) {
        V3[] verts = D3D.sphereVertices(r, latBands, lonBands);
        canvas.save();
        canvas.setStrokeColor(color);
        canvas.setPen(DPen.solid(1f, color).colored(color));
        for (int lat = 0; lat < latBands; lat++) {
            for (int lon = 0; lon < lonBands; lon++) {
                int first = lat * (lonBands + 1) + lon;
                int second = first + lonBands + 1;
                canvas.drawLine(toScreen(verts[first]), toScreen(verts[first + 1]));
                canvas.drawLine(toScreen(verts[first]), toScreen(verts[second]));
            }
        }
        canvas.restore();
        return this;
    }

    /** Point cloud: each {@code points[i]} → radius {@code px}. */
    public D3DRenderer drawPointCloud(V3[] points, float px, DColor color) {
        canvas.save();
        canvas.setFillColor(color);
        for (V3 p : points) {
            DPoint s = toScreen(p);
            float r = px / 2f;
            canvas.fillEllipse(DEllipse.of(s.x, s.y, r, r));
        }
        canvas.restore();
        return this;
    }

    /** Draw a 3D line segment with width and color. */
    public D3DRenderer drawLine(V3 a, V3 b, DColor color, float width) {
        canvas.save();
        canvas.setPen(DPen.solid(width, color).colored(color));
        canvas.setStrokeColor(color);
        canvas.drawLine(toScreen(a), toScreen(b));
        canvas.restore();
        return this;
    }

    /** Draw a polyline through a 3D point list. */
    public D3DRenderer drawPolyline(V3[] points, DColor color, float width) {
        canvas.save();
        canvas.setPen(DPen.solid(width, color).colored(color));
        canvas.setStrokeColor(color);
        for (int i = 1; i < points.length; i++) {
            canvas.drawLine(toScreen(points[i - 1]), toScreen(points[i]));
        }
        canvas.restore();
        return this;
    }

    /** Draw an axis arrow (a simple thick line + small filled head). */
    public D3DRenderer drawArrow(V3 from, V3 to, DColor color, float width) {
        canvas.save();
        canvas.setPen(DPen.solid(width, color).colored(color));
        canvas.setStrokeColor(color);
        canvas.drawLine(toScreen(from), toScreen(to));
        DPoint s = toScreen(to);
        float r = width * 1.6f;
        canvas.setFillColor(color);
        canvas.fillEllipse(DEllipse.of(s.x, s.y, r, r));
        canvas.restore();
        return this;
    }

    /** Draw the 3D coordinate origin (small sphere) — useful in scientific plots. */
    public D3DRenderer drawOrigin(float r, DColor color) {
        V3[] v = D3D.sphereVertices(r, 8, 8);
        int[] idx = D3D.sphereIndices(8, 8);
        DPoint[] screen = new DPoint[v.length];
        for (int i = 0; i < v.length; i++) screen[i] = toScreen(v[i]);
        canvas.save();
        canvas.setFillColor(color);
        canvas.setStrokeColor(color);
        canvas.setPen(DPen.solid(0.5f, color).colored(color));
        for (int i = 0; i < idx.length; i += 3) {
            DPoint p1 = screen[idx[i]], p2 = screen[idx[i + 1]], p3 = screen[idx[i + 2]];
            canvas.fillPolygon(p1, p2, p3);
        }
        canvas.restore();
        return this;
    }

    /** Label a 3D point with 2D text. */
    public D3DRenderer label(V3 at, String text, DColor color) {
        DPoint s = toScreen(at);
        canvas.save();
        canvas.setFont(DFont.sans(11f));
        canvas.text(text, s.x + 4, s.y - 4, color);
        canvas.restore();
        return this;
    }
}