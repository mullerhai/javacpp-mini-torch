package org.bytedeco.pytorch.dataframe.faiss;

import java.lang.reflect.Method;

/**
 * MPS (Apple Silicon) distance backend via javacpp-pytorch.
 *
 * <p>Mirrors {@link CudaDistanceBackend} but pushes the pairwise scoring onto the
 * Metal Performance Shaders device through {@code tensor.mps()}. Falls back to
 * {@link VectorCpuDistanceBackend} (or {@link CpuDistanceBackend} if vector
 * unavailable) when MPS is not present.
 */
public final class MpsDistanceBackend implements DistanceBackend {
    public static final MpsDistanceBackend INSTANCE = new MpsDistanceBackend();

    private MpsDistanceBackend() {}

    @Override
    public String name() {
        return DeviceSelector.isMpsAvailable()
            ? "mps-torch"
            : (VectorDistanceKernel.AVAILABLE
                ? "mps-fallback-vector"
                : "mps-fallback-cpu");
    }

    @Override
    public SearchResult knn(float[] base, int nb, float[] queries, int nq,
                            int d, int k, MetricType metric, long[] ids) {
        if (!DeviceSelector.isMpsAvailable()) {
            return VectorDistanceKernel.AVAILABLE
                ? VectorCpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids)
                : CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
        }
        // Reuse the torch matmul path: only matters whether the tensors went to MPS instead of CUDA.
        try {
            org.bytedeco.pytorch.Tensor xb = org.bytedeco.pytorch.global.torch.tensor(base)
                .reshape(new long[]{nb, d});
            org.bytedeco.pytorch.Tensor xq = org.bytedeco.pytorch.global.torch.tensor(
                java.util.Arrays.copyOf(queries, nq * d)).reshape(new long[]{nq, d});

            try {
                Method mpsM = xb.getClass().getMethod("mps");
                xb = (org.bytedeco.pytorch.Tensor) mpsM.invoke(xb);
                xq = (org.bytedeco.pytorch.Tensor) mpsM.invoke(xq);
            } catch (Throwable e) {
                // stay on CPU — fall through to fallback
                try { xb.close(); } catch (Throwable ignored) {}
                try { xq.close(); } catch (Throwable ignored) {}
                return VectorDistanceKernel.AVAILABLE
                    ? VectorCpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids)
                    : CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
            }

            org.bytedeco.pytorch.Tensor scores;
            if (metric == MetricType.METRIC_INNER_PRODUCT) {
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

            boolean lower = metric.lowerIsBetter();
            float[][] D = new float[nq][k];
            long[][] I = new long[nq][k];
            for (int q = 0; q < nq; q++) {
                TopK heap = new TopK(k, lower);
                int row = q * nb;
                for (int i = 0; i < nb; i++) {
                    heap.offer(ids != null ? ids[i] : i, flat[row + i]);
                }
                heap.export(D[q], I[q]);
            }
            try { xb.close(); } catch (Throwable ignored) {}
            try { xq.close(); } catch (Throwable ignored) {}
            try { scores.close(); } catch (Throwable ignored) {}
            try { cpu.close(); } catch (Throwable ignored) {}
            return new SearchResult(D, I);
        } catch (Throwable e) {
            return VectorDistanceKernel.AVAILABLE
                ? VectorCpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids)
                : CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
        }
    }

    @Override
    public RangeSearchResult range(float[] base, int nb, float[] queries, int nq,
                                   int d, float radius, MetricType metric, long[] ids) {
        return VectorDistanceKernel.AVAILABLE
            ? VectorCpuDistanceBackend.INSTANCE.range(base, nb, queries, nq, d, radius, metric, ids)
            : CpuDistanceBackend.INSTANCE.range(base, nb, queries, nq, d, radius, metric, ids);
    }
}
