package org.bytedeco.pytorch.dataframe.faiss;

/**
 * CUDA distance backend via javacpp-pytorch Tensor matmul.
 *
 * <p>For the bulk k-NN loop the GPU kernel is {@code (xq · xbᵀ)} (with the L2 norm-trick
 * for squared L2 distances). Pairwise scoring is computed as a single GEMM and the
 * top-k selection is done on the CPU side (returns nq × nb float matrix).
 *
 * <p>Falls back to {@link CpuDistanceBackend} when CUDA is unavailable.
 * Range search stays on CPU to avoid variable-length GPU→CPU transfers.
 */
public final class CudaDistanceBackend implements DistanceBackend {
    public static final CudaDistanceBackend INSTANCE = new CudaDistanceBackend();

    private CudaDistanceBackend() {}

    @Override
    public String name() {
        return DeviceSelector.isCudaAvailable() ? "cuda-torch" : "cuda-fallback-cpu";
    }

    @Override
    public SearchResult knn(float[] base, int nb, float[] queries, int nq,
                            int d, int k, MetricType metric, long[] ids) {
        if (!DeviceSelector.isCudaAvailable()) {
            return CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
        }
        SearchResult r = CpuDistanceBackend.knnTorch(base, nb, queries, nq, d, k, metric, ids, true);
        if (r != null) return r;
        return CpuDistanceBackend.INSTANCE.knn(base, nb, queries, nq, d, k, metric, ids);
    }

    @Override
    public RangeSearchResult range(float[] base, int nb, float[] queries, int nq,
                                   int d, float radius, MetricType metric, long[] ids) {
        return CpuDistanceBackend.INSTANCE.range(base, nb, queries, nq, d, radius, metric, ids);
    }
}
