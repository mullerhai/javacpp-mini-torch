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

/**
 * 3D vector and geometry primitives used by {@link D3DRenderer}.
 *
 * <p>Designed for didactic / illustrative 3D drawing — wireframe cubes,
 * spheres, axes, point clouds. Not a full general-purpose 3D engine.
 */
public final class D3D {

    private D3D() {}

    // ---- 3D vector -------------------------------------------------------

    public static final class V3 {
        public final float x, y, z;
        public V3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public V3 add(V3 o) { return new V3(x + o.x, y + o.y, z + o.z); }
        public V3 sub(V3 o) { return new V3(x - o.x, y - o.y, z - o.z); }
        public V3 mul(float s) { return new V3(x * s, y * s, z * s); }
        public float dot(V3 o) { return x * o.x + y * o.y + z * o.z; }
        public V3 cross(V3 o) {
            return new V3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
        }
        public float len() { return (float) Math.sqrt(x * x + y * y + z * z); }
        public V3 normalize() { float l = len(); return l < 1e-9f ? this : new V3(x / l, y / l, z / l); }

        public V3 rotateX(double a) {
            float c = (float) Math.cos(a), s = (float) Math.sin(a);
            return new V3(x, c * y - s * z, s * y + c * z);
        }
        public V3 rotateY(double a) {
            float c = (float) Math.cos(a), s = (float) Math.sin(a);
            return new V3(c * x + s * z, y, -s * x + c * z);
        }
        public V3 rotateZ(double a) {
            float c = (float) Math.cos(a), s = (float) Math.sin(a);
            return new V3(c * x - s * y, s * x + c * y, z);
        }
        public V3 rotateAxis(V3 axis, double angle) {
            V3 k = axis.normalize();
            float c = (float) Math.cos(angle), s = (float) Math.sin(angle);
            float dot = k.dot(this);
            V3 cr = k.cross(this);
            return k.mul(dot * (1f - c))
                    .add(cr.mul(s))
                    .add(this.mul(c));
        }
        @Override public String toString() { return String.format("V3[%.3f,%.3f,%.3f]", x, y, z); }
    }

    public static final class Mat4 {
        // Row-major 4x4
        public final float[] m;
        public Mat4(float[] m) {
            if (m.length != 16) throw new IllegalArgumentException("need 16 floats");
            this.m = m;
        }
        public static Mat4 identity() { return new Mat4(new float[]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}); }

        public static Mat4 translate(float tx, float ty, float tz) {
            return new Mat4(new float[]{1,0,0,0, 0,1,0,0, 0,0,1,0, tx,ty,tz,1});
        }
        public static Mat4 scale(float sx, float sy, float sz) {
            return new Mat4(new float[]{sx,0,0,0, 0,sy,0,0, 0,0,sz,0, 0,0,0,1});
        }
        public static Mat4 rotateX(double a) {
            float c = (float) Math.cos(a), s = (float) Math.sin(a);
            return new Mat4(new float[]{1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1});
        }
        public static Mat4 rotateY(double a) {
            float c = (float) Math.cos(a), s = (float) Math.sin(a);
            return new Mat4(new float[]{c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1});
        }
        public static Mat4 rotateZ(double a) {
            float c = (float) Math.cos(a), s = (float) Math.sin(a);
            return new Mat4(new float[]{c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1});
        }
        public Mat4 mul(Mat4 rhs) {
            float[] a = this.m, b = rhs.m, o = new float[16];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    float s = 0;
                    for (int k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j];
                    o[i * 4 + j] = s;
                }
            }
            return new Mat4(o);
        }
        public V3 transformPoint(V3 p) {
            float x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
            float y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
            float z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];
            float w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15];
            if (Math.abs(w) < 1e-9f) return new V3(x, y, z);
            return new V3(x / w, y / w, z / w);
        }
    }

    // ---- camera / projection ---------------------------------------------

    public static final class Camera {
        public V3 eye;
        public V3 target;
        public V3 up;
        public float fovYDeg;
        public float aspect;
        public float zNear;
        public float zFar;

        public Camera(V3 eye, V3 target, V3 up, float fovYDeg, float aspect, float zNear, float zFar) {
            this.eye = eye; this.target = target; this.up = up;
            this.fovYDeg = fovYDeg; this.aspect = aspect;
            this.zNear = zNear; this.zFar = zFar;
        }
        public static Camera defaultCamera(float aspect) {
            return new Camera(new V3(0, 0, 5), new V3(0, 0, 0), new V3(0, 1, 0), 45f, aspect, 0.1f, 100f);
        }
        public Mat4 viewMatrix() {
            V3 f = target.sub(eye).normalize();
            V3 s = f.cross(up).normalize();
            V3 u = s.cross(f);
            Mat4 m = new Mat4(new float[]{
                    s.x,  s.y,  s.z, -s.dot(eye),
                    u.x,  u.y,  u.z, -u.dot(eye),
                    -f.x, -f.y, -f.z,  f.dot(eye),
                    0, 0, 0, 1
            });
            return m;
        }
        public Mat4 projectionMatrix() {
            float f = (float) (1.0 / Math.tan(Math.toRadians(fovYDeg) / 2.0));
            float fn = 1f / (zNear - zFar);
            return new Mat4(new float[]{
                    f / aspect, 0, 0, 0,
                    0, f, 0, 0,
                    0, 0, (zFar + zNear) * fn, 2f * zFar * zNear * fn,
                    0, 0, -1, 0
            });
        }
    }

    // ---- primitive geometry ---------------------------------------------

    public static V3[] cubeVertices(float size) {
        float s = size * 0.5f;
        return new V3[]{
                new V3(-s, -s, -s), new V3(s, -s, -s), new V3(s, s, -s), new V3(-s, s, -s),
                new V3(-s, -s, s), new V3(s, -s, s), new V3(s, s, s), new V3(-s, s, s),
        };
    }

    public static int[][] cubeEdges() {
        return new int[][]{
                {0,1},{1,2},{2,3},{3,0},
                {4,5},{5,6},{6,7},{7,4},
                {0,4},{1,5},{2,6},{3,7}
        };
    }

    public static int[][] cubeFaces() {
        return new int[][]{
                {0,1,2,3}, {4,7,6,5}, {0,3,7,4}, {1,5,6,2}, {3,2,6,7}, {0,4,5,1}
        };
    }

    /**
     * Generate a unit-sphere mesh (V3 array) at radius {@code r} with {@code latBands}
     * latitude bands and {@code lonBands} longitude bands.
     */
    public static V3[] sphereVertices(float r, int latBands, int lonBands) {
        if (latBands < 2 || lonBands < 3) throw new IllegalArgumentException("need lat>=2, lon>=3");
        java.util.List<V3> v = new java.util.ArrayList<>((latBands + 1) * (lonBands + 1));
        for (int lat = 0; lat <= latBands; lat++) {
            double theta = lat * Math.PI / latBands;
            double sinT = Math.sin(theta), cosT = Math.cos(theta);
            for (int lon = 0; lon <= lonBands; lon++) {
                double phi = lon * 2 * Math.PI / lonBands;
                double sinP = Math.sin(phi), cosP = Math.cos(phi);
                float x = (float) (cosP * sinT);
                float y = (float) cosT;
                float z = (float) (sinP * sinT);
                v.add(new V3(x * r, y * r, z * r));
            }
        }
        return v.toArray(new V3[0]);
    }

    public static int[] sphereIndices(int latBands, int lonBands) {
        java.util.List<Integer> idx = new java.util.ArrayList<>();
        for (int lat = 0; lat < latBands; lat++) {
            for (int lon = 0; lon < lonBands; lon++) {
                int first = lat * (lonBands + 1) + lon;
                int second = first + lonBands + 1;
                idx.add(first);
                idx.add(second);
                idx.add(first + 1);
                idx.add(second);
                idx.add(second + 1);
                idx.add(first + 1);
            }
        }
        int[] out = new int[idx.size()];
        for (int i = 0; i < out.length; i++) out[i] = idx.get(i);
        return out;
    }

    /** Generate a 3D axes triad (origin → +X,+Y,+Z). Returns 3 (from, to) pairs. */
    public static V3[][] axes(float len) {
        return new V3[][]{
                {new V3(0, 0, 0), new V3(len, 0, 0)},
                {new V3(0, 0, 0), new V3(0, len, 0)},
                {new V3(0, 0, 0), new V3(0, 0, len)},
        };
    }

    /** Generate a 3D grid in the X-Y plane at Z = 0. */
    public static V3[][] grid2D(float size, int cells) {
        float step = size / cells;
        java.util.List<V3[]> lines = new java.util.ArrayList<>();
        float h = size * 0.5f;
        for (int i = 0; i <= cells; i++) {
            float t = -h + i * step;
            lines.add(new V3[]{new V3(-h, t, 0), new V3(h, t, 0)});
            lines.add(new V3[]{new V3(t, -h, 0), new V3(t, h, 0)});
        }
        return lines.toArray(new V3[0][]);
    }

    /** Generate an arrow (cylinder body approximated by a thick line + cone). */
    public static V3[] arrow(float length, float headRatio) {
        float head = length * Math.max(0.05f, headRatio);
        float body = length - head;
        return new V3[]{
                new V3(0, 0, 0),
                new V3(body, 0, 0),
                new V3(body, 0, 0),
                new V3(length, 0, 0),
        };
    }

    // ---- project to screen ------------------------------------------------

    /**
     * Project a 3D point through the view × projection matrix and return its
     * normalized device coordinates {@code (x, y)} in {@code [-1, 1]} plus depth
     * {@code z} in {@code [-1, 1]}. Caller maps NDC to canvas pixels.
     */
    public static float[] project(V3 p, Mat4 viewProj) {
        V3 q = viewProj.transformPoint(p);
        return new float[]{q.x, q.y, q.z};
    }
}