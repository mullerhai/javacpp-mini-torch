package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Hot-path distance kernels — used by HNSW / Flat fallback.
 *
 * <p>Optimizations: hoisted base pointers (JIT range-checks once), dim%8 fast-path
 * (no tail compare), FMA-friendly statement merging on the L2 accumulators.
 */
public final class DistanceKernel {
    private DistanceKernel() {}

    /** Squared L2 between two dense vectors. */
    public static float l2(float[] a, float[] b, int dim) {
        return l2Row(a, 0, b, 0, dim);
    }

    /** Inner product between two dense vectors. */
    public static float ip(float[] a, float[] b, int dim) {
        return ipRow(a, 0, b, 0, dim);
    }

    /** Squared L2: query vs row-major matrix row. */
    public static float l2Row(float[] q, float[] matrix, int rowBase, int dim) {
        return l2Row(q, 0, matrix, rowBase, dim);
    }

    /** Inner product: query vs row-major matrix row. */
    public static float ipRow(float[] q, float[] matrix, int rowBase, int dim) {
        return ipRow(q, 0, matrix, rowBase, dim);
    }

    /** Squared L2 with query offset into a packed query matrix. */
    public static float l2Row(float[] queries, int qOff, float[] matrix, int rowBase, int dim) {
        // Hot path. Local copies let the JIT elide one range-check + CSE the base.
        final float[] qb = queries;
        final float[] mb = matrix;
        final int qBase = qOff;
        final int mBase = rowBase;
        if ((dim & 7) == 0) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = 0; i < dim; i += 8) {
                float d0 = qb[qBase + i    ] - mb[mBase + i    ];
                float d1 = qb[qBase + i + 1] - mb[mBase + i + 1];
                float d2 = qb[qBase + i + 2] - mb[mBase + i + 2];
                float d3 = qb[qBase + i + 3] - mb[mBase + i + 3];
                float d4 = qb[qBase + i + 4] - mb[mBase + i + 4];
                float d5 = qb[qBase + i + 5] - mb[mBase + i + 5];
                float d6 = qb[qBase + i + 6] - mb[mBase + i + 6];
                float d7 = qb[qBase + i + 7] - mb[mBase + i + 7];
                s0 += d0 * d0 + d1 * d1;
                s1 += d2 * d2 + d3 * d3;
                s2 += d4 * d4 + d5 * d5;
                s3 += d6 * d6 + d7 * d7;
            }
            return s0 + s1 + s2 + s3;
        }
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            float d0 = qb[qBase + i    ] - mb[mBase + i    ];
            float d1 = qb[qBase + i + 1] - mb[mBase + i + 1];
            float d2 = qb[qBase + i + 2] - mb[mBase + i + 2];
            float d3 = qb[qBase + i + 3] - mb[mBase + i + 3];
            float d4 = qb[qBase + i + 4] - mb[mBase + i + 4];
            float d5 = qb[qBase + i + 5] - mb[mBase + i + 5];
            float d6 = qb[qBase + i + 6] - mb[mBase + i + 6];
            float d7 = qb[qBase + i + 7] - mb[mBase + i + 7];
            s0 += d0 * d0 + d1 * d1;
            s1 += d2 * d2 + d3 * d3;
            s2 += d4 * d4 + d5 * d5;
            s3 += d6 * d6 + d7 * d7;
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) {
            float d = qb[qBase + i] - mb[mBase + i];
            s += d * d;
        }
        return s;
    }

    /** Inner product with query offset into a packed query matrix. */
    public static float ipRow(float[] queries, int qOff, float[] matrix, int rowBase, int dim) {
        final float[] qb = queries;
        final float[] mb = matrix;
        final int qBase = qOff;
        final int mBase = rowBase;
        if ((dim & 7) == 0) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = 0; i < dim; i += 8) {
                s0 += qb[qBase + i    ] * mb[mBase + i    ]
                    + qb[qBase + i + 1] * mb[mBase + i + 1];
                s1 += qb[qBase + i + 2] * mb[mBase + i + 2]
                    + qb[qBase + i + 3] * mb[mBase + i + 3];
                s2 += qb[qBase + i + 4] * mb[mBase + i + 4]
                    + qb[qBase + i + 5] * mb[mBase + i + 5];
                s3 += qb[qBase + i + 6] * mb[mBase + i + 6]
                    + qb[qBase + i + 7] * mb[mBase + i + 7];
            }
            return s0 + s1 + s2 + s3;
        }
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            s0 += qb[qBase + i    ] * mb[mBase + i    ]
                + qb[qBase + i + 1] * mb[mBase + i + 1];
            s1 += qb[qBase + i + 2] * mb[mBase + i + 2]
                + qb[qBase + i + 3] * mb[mBase + i + 3];
            s2 += qb[qBase + i + 4] * mb[mBase + i + 4]
                + qb[qBase + i + 5] * mb[mBase + i + 5];
            s3 += qb[qBase + i + 6] * mb[mBase + i + 6]
                + qb[qBase + i + 7] * mb[mBase + i + 7];
        }
        float s = s0 + s1 + s2 + s3;
        for (; i < dim; i++) s += qb[qBase + i] * mb[mBase + i];
        return s;
    }

    /** Squared L2 norm of one vector. */
    public static float sqNorm(float[] v, int off, int dim) {
        final float[] vb = v;
        final int base = off;
        if ((dim & 7) == 0) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int i = 0; i < dim; i += 8) {
                float a0 = vb[base + i    ], a1 = vb[base + i + 1];
                float a2 = vb[base + i + 2], a3 = vb[base + i + 3];
                float a4 = vb[base + i + 4], a5 = vb[base + i + 5];
                float a6 = vb[base + i + 6], a7 = vb[base + i + 7];
                s0 += a0 * a0 + a1 * a1;
                s1 += a2 * a2 + a3 * a3;
                s2 += a4 * a4 + a5 * a5;
                s3 += a6 * a6 + a7 * a7;
            }
            return s0 + s1 + s2 + s3;
        }
        float s = 0;
        int end = base + dim;
        for (int i = base; i < end; i++) {
            float a = vb[i];
            s += a * a;
        }
        return s;
    }

    /** In-place L2 normalize rows of row-major matrix. */
    public static void normalizeL2(float[] x, int n, int d) {
        for (int i = 0; i < n; i++) {
            int base = i * d;
            float sum = 0f;
            int j = 0;
            for (; j + 7 < d; j += 8) {
                float a0 = x[base + j    ], a1 = x[base + j + 1];
                float a2 = x[base + j + 2], a3 = x[base + j + 3];
                float a4 = x[base + j + 4], a5 = x[base + j + 5];
                float a6 = x[base + j + 6], a7 = x[base + j + 7];
                sum += a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5 + a6*a6 + a7*a7;
            }
            float s = sum;
            for (; j < d; j++) { float a = x[base + j]; s += a * a; }
            if (s > 0f) {
                float inv = (float) (1.0 / Math.sqrt(s));
                for (j = 0; j < d; j++) x[base + j] *= inv;
            }
        }
    }

    /** Copy+normalize into dst. */
    public static void normalizeL2Copy(float[] src, float[] dst, int n, int d) {
        for (int i = 0; i < n; i++) {
            int base = i * d;
            float sum = 0f;
            int j = 0;
            for (; j + 7 < d; j += 8) {
                float a0 = src[base + j    ], a1 = src[base + j + 1];
                float a2 = src[base + j + 2], a3 = src[base + j + 3];
                float a4 = src[base + j + 4], a5 = src[base + j + 5];
                float a6 = src[base + j + 6], a7 = src[base + j + 7];
                sum += a0*a0 + a1*a1 + a2*a2 + a3*a3 + a4*a4 + a5*a5 + a6*a6 + a7*a7;
            }
            float s = sum;
            for (; j < d; j++) { float a = src[base + j]; s += a * a; }
            float inv = s > 0f ? (float) (1.0 / Math.sqrt(s)) : 1f;
            for (j = 0; j < d; j++) dst[base + j] = src[base + j] * inv;
        }
    }
}
