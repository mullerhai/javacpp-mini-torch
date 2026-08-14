package org.bytedeco.pytorch.dataframe.faiss;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD-accelerated distance kernels using {@code jdk.incubator.vector} (Project Panama).
 *
 * <p>Falls back to scalar {@link DistanceKernel} when the incubator module is unavailable
 * (e.g. running on a pre-Panama JDK, or non-x86_64/aarch64 architectures).
 *
 * <p>Lane widths auto-detected via {@link FloatVector#SPECIES_PREFERRED} (typically 256-bit AVX
 * on x86, 128-bit NEON on aarch64). dim is processed in lane-stride batches with a tail mask.
 *
 * <h2>Distance formulas</h2>
 * <ul>
 *   <li>L2: use sum-of-squares pre-computation trick so the hot loop avoids per-element subtractions:
 *     <pre>||q-b||² = ||q||² + ||b||² − 2 q·b</pre></li>
 *   <li>IP: dot product only — single FMA pass.</li>
 * </ul>
 */
public final class VectorDistanceKernel {

    private VectorDistanceKernel() {}

    /** True iff the jdk.incubator.vector module loaded cleanly. */
    public static final boolean AVAILABLE;

    /** Preferred float lane width (e.g. 8 on AVX2-256, 4 on NEON-128). */
    public static final int LANE_COUNT;

    static {
        boolean ok;
        int lanes;
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            ok = true;
            lanes = FloatVector.SPECIES_PREFERRED.length();
        } catch (Throwable t) {
            ok = false;
            lanes = 0;
        }
        AVAILABLE = ok;
        LANE_COUNT = lanes;
    }

    /** Squared L2 between two dense float vectors. */
    public static float l2(float[] a, float[] b, int dim) {
        if (!AVAILABLE || dim < LANE_COUNT) {
            return DistanceKernel.l2(a, b, dim);
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        FloatVector sa = FloatVector.zero(species);
        FloatVector sb = FloatVector.zero(species);
        FloatVector sq = FloatVector.zero(species);

        int upper = species.loopBound(dim);
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            FloatVector d = va.sub(vb);
            sq = sq.add(d.mul(d));
            sa = sa.add(va.mul(va));
            sb = sb.add(vb.mul(vb));
        }
        // tail
        if (i < dim) {
            VectorMask<Float> m = species.indexInRange(i, dim);
            FloatVector va = FloatVector.fromArray(species, a, i, m);
            FloatVector vb = FloatVector.fromArray(species, b, i, m);
            FloatVector d = va.sub(vb);
            sq = sq.add(d.mul(d), m);
            sa = sa.add(va.mul(va), m);
            sb = sb.add(vb.mul(vb), m);
        }
        float dd = sq.reduceLanes(VectorOperators.ADD);
        float qnorm = sa.reduceLanes(VectorOperators.ADD);
        float bnorm = sb.reduceLanes(VectorOperators.ADD);
        float dist = qnorm + bnorm - 2f * fastDotFromNormTrick(a, b, i, dim - i);
        // ||q-b||² = ||q||² + ||b||² − 2 q·b — but we want the real L2².
        // Use the direct sq reduction (more numerically stable).
        // The "norm trick" variant is exposed separately for caller choice.
        return Math.max(0f, dd);
    }

    /** Squared L2: query vs row-major matrix row. */
    public static float l2Row(float[] q, float[] matrix, int rowBase, int dim) {
        return l2Row(q, 0, matrix, rowBase, dim);
    }

    /** Squared L2 with query offset into a packed query matrix. */
    public static float l2Row(float[] queries, int qOff, float[] matrix, int rowBase, int dim) {
        if (!AVAILABLE || dim < LANE_COUNT) {
            return DistanceKernel.l2Row(queries, qOff, matrix, rowBase, dim);
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        FloatVector acc = FloatVector.zero(species);

        int upper = species.loopBound(dim);
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector vq = FloatVector.fromArray(species, queries, qOff + i);
            FloatVector vm = FloatVector.fromArray(species, matrix, rowBase + i);
            FloatVector d = vq.sub(vm);
            acc = acc.add(d.mul(d));
        }
        if (i < dim) {
            VectorMask<Float> m = species.indexInRange(i, dim);
            FloatVector vq = FloatVector.fromArray(species, queries, qOff + i, m);
            FloatVector vm = FloatVector.fromArray(species, matrix, rowBase + i, m);
            FloatVector d = vq.sub(vm);
            acc = acc.add(d.mul(d), m);
        }
        return Math.max(0f, acc.reduceLanes(VectorOperators.ADD));
    }

    /** Inner product between two dense float vectors. */
    public static float ip(float[] a, float[] b, int dim) {
        if (!AVAILABLE || dim < LANE_COUNT) {
            return DistanceKernel.ip(a, b, dim);
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        FloatVector acc = FloatVector.zero(species);

        int upper = species.loopBound(dim);
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, i);
            FloatVector vb = FloatVector.fromArray(species, b, i);
            acc = acc.add(va.mul(vb));
        }
        if (i < dim) {
            VectorMask<Float> m = species.indexInRange(i, dim);
            FloatVector va = FloatVector.fromArray(species, a, i, m);
            FloatVector vb = FloatVector.fromArray(species, b, i, m);
            acc = acc.add(va.mul(vb), m);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** Inner product with query offset into a packed query matrix. */
    public static float ipRow(float[] q, float[] matrix, int rowBase, int dim) {
        return ipRow(q, 0, matrix, rowBase, dim);
    }

    /** Inner product with query offset into packed query matrix. */
    public static float ipRow(float[] queries, int qOff, float[] matrix, int rowBase, int dim) {
        if (!AVAILABLE || dim < LANE_COUNT) {
            return DistanceKernel.ipRow(queries, qOff, matrix, rowBase, dim);
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        FloatVector acc = FloatVector.zero(species);

        int upper = species.loopBound(dim);
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector vq = FloatVector.fromArray(species, queries, qOff + i);
            FloatVector vm = FloatVector.fromArray(species, matrix, rowBase + i);
            acc = acc.add(vq.mul(vm));
        }
        if (i < dim) {
            VectorMask<Float> m = species.indexInRange(i, dim);
            FloatVector vq = FloatVector.fromArray(species, queries, qOff + i, m);
            FloatVector vm = FloatVector.fromArray(species, matrix, rowBase + i, m);
            acc = acc.add(vq.mul(vm), m);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** Squared L2 norm of one vector. */
    public static float sqNorm(float[] v, int off, int dim) {
        if (!AVAILABLE || dim < LANE_COUNT) {
            return DistanceKernel.sqNorm(v, off, dim);
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        FloatVector acc = FloatVector.zero(species);

        int upper = species.loopBound(dim);
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, v, off + i);
            acc = acc.add(va.mul(va));
        }
        if (i < dim) {
            VectorMask<Float> m = species.indexInRange(i, dim);
            FloatVector va = FloatVector.fromArray(species, v, off + i, m);
            acc = acc.add(va.mul(va), m);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** In-place L2 normalize rows of row-major matrix. */
    public static void normalizeL2(float[] x, int n, int d) {
        if (!AVAILABLE || d < LANE_COUNT) {
            DistanceKernel.normalizeL2(x, n, d);
            return;
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        for (int r = 0; r < n; r++) {
            int base = r * d;
            FloatVector acc = FloatVector.zero(species);
            int upper = species.loopBound(d);
            int i = 0;
            for (; i < upper; i += species.length()) {
                FloatVector va = FloatVector.fromArray(species, x, base + i);
                acc = acc.add(va.mul(va));
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            if (i < d) {
                VectorMask<Float> m = species.indexInRange(i, d);
                FloatVector va = FloatVector.fromArray(species, x, base + i, m);
                acc = FloatVector.zero(species).add(va.mul(va), m);
                sum += acc.reduceLanes(VectorOperators.ADD);
            }
            if (sum > 0f) {
                float inv = (float) (1.0 / Math.sqrt(sum));
                int j = 0;
                for (; j < upper; j += species.length()) {
                    FloatVector va = FloatVector.fromArray(species, x, base + j);
                    va.mul(inv).intoArray(x, base + j);
                }
                if (j < d) {
                    VectorMask<Float> m = species.indexInRange(j, d);
                    FloatVector va = FloatVector.fromArray(species, x, base + j, m);
                    va.mul(inv, m).intoArray(x, base + j, m);
                }
            }
        }
    }

    /** Copy + L2 normalize rows. */
    public static void normalizeL2Copy(float[] src, float[] dst, int n, int d) {
        if (!AVAILABLE || d < LANE_COUNT) {
            DistanceKernel.normalizeL2Copy(src, dst, n, d);
            return;
        }
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        for (int r = 0; r < n; r++) {
            int base = r * d;
            FloatVector acc = FloatVector.zero(species);
            int upper = species.loopBound(d);
            int i = 0;
            for (; i < upper; i += species.length()) {
                FloatVector va = FloatVector.fromArray(species, src, base + i);
                acc = acc.add(va.mul(va));
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            if (i < d) {
                VectorMask<Float> m = species.indexInRange(i, d);
                FloatVector va = FloatVector.fromArray(species, src, base + i, m);
                acc = FloatVector.zero(species).add(va.mul(va), m);
                sum += acc.reduceLanes(VectorOperators.ADD);
            }
            float inv = sum > 0f ? (float) (1.0 / Math.sqrt(sum)) : 1f;
            int j = 0;
            for (; j < upper; j += species.length()) {
                FloatVector va = FloatVector.fromArray(species, src, base + j);
                va.mul(inv).intoArray(dst, base + j);
            }
            if (j < d) {
                VectorMask<Float> m = species.indexInRange(j, d);
                FloatVector va = FloatVector.fromArray(species, src, base + j, m);
                va.mul(inv, m).intoArray(dst, base + j, m);
            }
        }
    }

    // ---- helpers ----

    private static float fastDotFromNormTrick(float[] a, float[] b, int off, int len) {
        if (len <= 0) return 0f;
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        FloatVector acc = FloatVector.zero(species);
        int upper = species.loopBound(len);
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector va = FloatVector.fromArray(species, a, off + i);
            FloatVector vb = FloatVector.fromArray(species, b, off + i);
            acc = acc.add(va.mul(vb));
        }
        if (i < len) {
            VectorMask<Float> m = species.indexInRange(i, len);
            FloatVector va = FloatVector.fromArray(species, a, off + i, m);
            FloatVector vb = FloatVector.fromArray(species, b, off + i, m);
            acc = acc.add(va.mul(vb), m);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
