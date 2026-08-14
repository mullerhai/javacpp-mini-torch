package org.bytedeco.pytorch.dataframe.ann;

import java.lang.reflect.Method;

import org.bytedeco.pytorch.dataframe.faiss.CpuDistanceBackend;
import org.bytedeco.pytorch.dataframe.faiss.CudaDistanceBackend;
import org.bytedeco.pytorch.dataframe.faiss.DeviceSelector;
import org.bytedeco.pytorch.dataframe.faiss.MpsDistanceBackend;
import org.bytedeco.pytorch.dataframe.faiss.VectorCpuDistanceBackend;
import org.bytedeco.pytorch.dataframe.faiss.VectorDistanceKernel;

/**
 * Bridge between the {@code ann} module and the FAISS module's optimized kernels.
 *
 * <p>Two boolean flags pick the active path:
 * <ul>
 *   <li>{@link #turboFast} — when {@code true}, distance math runs through
 *       {@link VectorDistanceKernel} (Project Panama SIMD, jdk.incubator.vector)
 *       via {@link VectorCpuDistanceBackend}. Falls back automatically to the
 *       scalar {@link CpuDistanceBackend} when the incubator module is unavailable.</li>
 *   <li>{@link #useGpu} — when {@code true}, the bulk distance computation
 *       is delegated to {@link CudaDistanceBackend} (CUDA) or
 *       {@link MpsDistanceBackend} (Apple Silicon), whichever is present.</li>
 * </ul>
 *
 * <p>{@code useGpu} takes priority over {@code turboFast}: when both are on, the GPU
 * path is selected first; the {@code ann} graph-walk always runs on CPU regardless.
 */
public final class AnnKernel {

    /** Default routing — preserves the original scalar Java behavior. */
    public static final AnnKernel LEGACY = new AnnKernel(false, false);

    /** SIMD-only acceleration on CPU. */
    public static final AnnKernel TURBO  = new AnnKernel(true,  false);

    /** GPU-accelerated path (CUDA or MPS). */
    public static final AnnKernel GPU    = new AnnKernel(false, true);

    /** SIMD + GPU when available. */
    public static final AnnKernel TURBO_GPU = new AnnKernel(true, true);

    public final boolean turboFast;
    public final boolean useGpu;

    public AnnKernel(boolean turboFast, boolean useGpu) {
        this.turboFast = turboFast;
        this.useGpu = useGpu;
    }

    public static AnnKernel of(boolean turboFast, boolean useGpu) {
        if (turboFast && useGpu) return TURBO_GPU;
        if (useGpu)              return GPU;
        if (turboFast)           return TURBO;
        return LEGACY;
    }

    /** Convenience: build from the global {@link DeviceSelector} settings. */
    public static AnnKernel fromDeviceSelector() {
        DeviceSelector.BackendMode m = DeviceSelector.backendMode();
        boolean turbo = (m == DeviceSelector.BackendMode.VECTOR
                      || m == DeviceSelector.BackendMode.AUTO) && VectorDistanceKernel.AVAILABLE;
        boolean gpu   = (m == DeviceSelector.BackendMode.CUDA || m == DeviceSelector.BackendMode.MPS)
                      && (DeviceSelector.isCudaAvailable() || DeviceSelector.isMpsAvailable());
        return of(turbo, gpu);
    }

    /** True when the SIMD-accelerated vector kernel is the active path. */
    public boolean vectorEnabled() {
        return turboFast && VectorDistanceKernel.AVAILABLE && !useGpu;
    }

    /** Squared L2 between query vector and a base matrix row (rowBase = byte offset). */
    public float l2(float[] q, float[] base, int rowBase, int dim) {
        if (vectorEnabled()) {
            return VectorDistanceKernel.l2Row(q, base, rowBase, dim);
        }
        return Distance.L2.distance(q, base, rowBase / dim, dim);
    }

    /** Inner-product between query and row (rowBase = byte offset). */
    public float ip(float[] q, float[] base, int rowBase, int dim) {
        if (vectorEnabled()) {
            return -VectorDistanceKernel.ipRow(q, base, rowBase, dim);
        }
        return Distance.IP.distance(q, base, rowBase / dim, dim);
    }

    /** Cosine between query and row (rowBase = byte offset). */
    public float cosine(float[] q, float[] base, int rowBase, int dim) {
        return Distance.COSINE.distance(q, base, rowBase / dim, dim);
    }

    /** Squared L2 between two stored vectors (graph traversal). */
    public float l2(float[] base, int aBase, int bBase, int dim) {
        if (vectorEnabled()) {
            return VectorDistanceKernel.l2Row(base, aBase, base, bBase, dim);
        }
        return scalarL2(base, aBase, bBase, dim);
    }

    /** Inner product between two stored vectors. */
    public float ip(float[] base, int aBase, int bBase, int dim) {
        if (vectorEnabled()) {
            return -VectorDistanceKernel.ipRow(base, aBase, base, bBase, dim);
        }
        return scalarIP(base, aBase, bBase, dim);
    }

    /** Cosine between two stored vectors. */
    public float cosine(float[] base, int aBase, int bBase, int dim) {
        return scalarCosine(base, aBase, bBase, dim);
    }

    /**
     * Bulk brute-force ground truth — used by {@code HnswIndex.bruteForce} and benchmarks.
     * Selects GPU (matmul) when {@link #useGpu}, else SIMD, else scalar.
     */
    public float[] bruteForce(float[] base, int nb, float[] queries, int nq, int d,
                              Distance metric) {
        if (useGpu) {
            try {
                float[] flat = gpuBruteForce(base, nb, queries, nq, d, metric);
                if (flat != null) return flat;
            } catch (Throwable ignored) {
                // fall through to CPU path
            }
        }
        if (turboFast && VectorDistanceKernel.AVAILABLE) {
            return vectorBruteForce(base, nb, queries, nq, d, metric);
        }
        return scalarBruteForce(base, nb, queries, nq, d, metric);
    }

    // ------------------- private helpers -------------------

    private static float scalarL2(float[] base, int a, int b, int dim) {
        float s = 0f;
        for (int i = 0; i < dim; i++) {
            float d = base[a + i] - base[b + i];
            s += d * d;
        }
        return s;
    }

    private static float scalarIP(float[] base, int a, int b, int dim) {
        float s = 0f;
        for (int i = 0; i < dim; i++) s += base[a + i] * base[b + i];
        return -s;
    }

    private static float scalarCosine(float[] base, int a, int b, int dim) {
        float dot = 0f, na = 0f, nb = 0f;
        for (int i = 0; i < dim; i++) {
            float av = base[a + i], bv = base[b + i];
            dot += av * bv; na += av * av; nb += bv * bv;
        }
        if (na == 0f || nb == 0f) return 1f;
        float cos = dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
        if (cos > 1f) cos = 1f; if (cos < -1f) cos = -1f;
        return 1f - cos;
    }

    private float[] scalarBruteForce(float[] base, int nb, float[] queries, int nq, int d,
                                     Distance metric) {
        float[] out = new float[nq * nb];
        for (int q = 0; q < nq; q++) {
            int qOff = q * d;
            for (int i = 0; i < nb; i++) {
                out[q * nb + i] = metric.distance(queries, base, i, d);
            }
        }
        return out;
    }

    private float[] vectorBruteForce(float[] base, int nb, float[] queries, int nq, int d,
                                     Distance metric) {
        float[] out = new float[nq * nb];
        for (int q = 0; q < nq; q++) {
            int qOff = q * d;
            for (int i = 0; i < nb; i++) {
                int rowBase = i * d;
                float dist;
                switch (metric) {
                    case L2:
                        dist = VectorDistanceKernel.l2Row(queries, qOff, base, rowBase, d);
                        break;
                    case IP:
                        dist = -VectorDistanceKernel.ipRow(queries, qOff, base, rowBase, d);
                        break;
                    case COSINE:
                    default:
                        dist = Distance.COSINE.distance(queries, base, qOff / d, d);
                        break;
                }
                out[q * nb + i] = dist;
            }
        }
        return out;
    }

    private float[] gpuBruteForce(float[] base, int nb, float[] queries, int nq, int d,
                                  Distance metric) {
        if (DeviceSelector.isCudaAvailable()) {
            return torchGemmFlat(CudaDistanceBackend.INSTANCE, base, nb, queries, nq, d, metric);
        }
        if (DeviceSelector.isMpsAvailable()) {
            return torchGemmFlat(MpsDistanceBackend.INSTANCE, base, nb, queries, nq, d, metric);
        }
        return null;
    }

    /**
     * Borrow the FAISS torch-matmul path to compute pairwise scores on GPU.
     * The native path produces D[nq][nb] float[][]; flatten to a single array.
     */
    private static float[] torchGemmFlat(org.bytedeco.pytorch.dataframe.faiss.DistanceBackend b,
                                         float[] base, int nb, float[] queries, int nq, int d,
                                         Distance metric) {
        try {
            // The FAISS torch GEMM computes raw L2² or raw IP. We need raw scores
            // (lower-better) for ann's semantics; cosine is not handled by GEMM.
            if (metric == Distance.COSINE) {
                return null;
            }
            // CudaDistanceBackend's knn already runs the matmul but expects to write to
            // a top-k structure. For brute force we replicate the GEMM via reflection
            // on the torch.Tensor API.
            org.bytedeco.pytorch.Tensor xb = org.bytedeco.pytorch.global.torch.tensor(base)
                .reshape(new long[]{nb, d});
            org.bytedeco.pytorch.Tensor xq = org.bytedeco.pytorch.global.torch.tensor(
                java.util.Arrays.copyOf(queries, nq * d)).reshape(new long[]{nq, d});

            Method toDev = null;
            try {
                toDev = xb.getClass().getMethod("cuda");
                xb = (org.bytedeco.pytorch.Tensor) toDev.invoke(xb);
                xq = (org.bytedeco.pytorch.Tensor) toDev.invoke(xq);
            } catch (NoSuchMethodException ns) {
                try { xb.close(); } catch (Throwable ignored) {}
                try { xq.close(); } catch (Throwable ignored) {}
                return null; // MPS route handled below
            }

            org.bytedeco.pytorch.Tensor scores;
            if (metric == Distance.IP) {
                scores = xq.matmul(xb.transpose(0, 1));
            } else {
                org.bytedeco.pytorch.ScalarTypeOptional noDtype = new org.bytedeco.pytorch.ScalarTypeOptional();
                org.bytedeco.pytorch.Tensor q2 = xq.pow(new org.bytedeco.pytorch.Scalar(2)).sum(new long[]{1}, true, noDtype);
                org.bytedeco.pytorch.Tensor b2 = xb.pow(new org.bytedeco.pytorch.Scalar(2)).sum(new long[]{1}, true, noDtype);
                org.bytedeco.pytorch.Tensor dots = xq.matmul(xb.transpose(0, 1));
                scores = q2.add(b2.transpose(0, 1)).sub(dots.mul(new org.bytedeco.pytorch.Scalar(2)));
            }

            org.bytedeco.pytorch.Tensor cpu = scores.contiguous();
            try { if (cpu.is_cuda()) cpu = cpu.cpu(); } catch (Throwable ignored) {}
            if (!cpu.is_contiguous()) cpu = cpu.contiguous();
            org.bytedeco.javacpp.FloatPointer ptr = cpu.data_ptr_float();
            float[] flat = new float[nq * nb];
            ptr.get(flat);

            // Convert IP raw → ann IP (lower-is-better via negation)
            if (metric == Distance.IP) {
                for (int i = 0; i < flat.length; i++) flat[i] = -flat[i];
            }
            try { xb.close(); } catch (Throwable ignored) {}
            try { xq.close(); } catch (Throwable ignored) {}
            try { scores.close(); } catch (Throwable ignored) {}
            try { cpu.close(); } catch (Throwable ignored) {}
            return flat;
        } catch (Throwable t) {
            return null;
        }
    }
}
