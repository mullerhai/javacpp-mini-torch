package org.bytedeco.pytorch.dataframe.ann;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import org.bytedeco.pytorch.dataframe.faiss.VectorDistanceKernel;

/**
 * Static factory and helpers for ANN indexes — produces pre-configured
 * {@link HnswIndex} instances and exposes a flat brute-force search routine
 * that delegates to {@link AnnKernel} for SIMD / GPU acceleration.
 *
 * <pre>
 *   HnswIndex idx = AnnFactory.hnsw(256, Distance.COSINE)
 *       .M(32).efConstruction(200).efSearch(64)
 *       .turboFast(true)        // enable jdk.incubator.vector SIMD
 *       .build();
 *
 *   AnnSearchResult r = AnnFactory.flatBruteForce(base, query, dim, 10, Distance.L2,
 *       AnnKernel.TURBO);
 * </pre>
 */
public final class AnnFactory {

    private AnnFactory() {}

    /** Begin building an HNSW index in {@code dim} dimensions. */
    public static HnswBuilder hnsw(int dim) {
        return new HnswBuilder(dim);
    }

    /**
     * Brute-force k-NN — supports {@code k} up to {@code Integer.MAX_VALUE}.
     * Returns {@code null} if base is empty.
     */
    public static AnnSearchResult flatBruteForce(float[] base, int nb, float[] query,
                                                int dim, int k, Distance space, AnnKernel kernel) {
        if (nb <= 0) return new AnnSearchResult(new int[0], new float[0], new long[0]);
        if (query == null || query.length != dim)
            throw new IllegalArgumentException("query dim mismatch");

        if (kernel.useGpu && (space == Distance.L2 || space == Distance.IP)) {
            AnnSearchResult r = tryGpuBruteForce(base, nb, query, dim, k, space, kernel);
            if (r != null) return r;
        }
        return cpuFlatBruteForce(base, nb, query, dim, k, space, kernel);
    }

    private static AnnSearchResult cpuFlatBruteForce(float[] base, int nb, float[] query,
                                                    int dim, int k, Distance space, AnnKernel kernel) {
        boolean parallel = nb >= 8_000 && ForkJoinPool.commonPool().getParallelism() > 1;

        // top-k via simple insertion sort (k is small)
        int[] bestIdx = new int[k];
        float[] bestDist = new float[k];
        for (int i = 0; i < k; i++) {
            bestIdx[i] = -1;
            bestDist[i] = Float.POSITIVE_INFINITY;
        }

        if (parallel) {
            int chunk = Math.max(1, nb / ForkJoinPool.commonPool().getParallelism());
            final int dLocal = dim;
            final int K = k;
            final int nbLocal = nb;
            float[][] partialBestD = new float[ForkJoinPool.commonPool().getParallelism()][];
            int[][] partialBestI = new int[ForkJoinPool.commonPool().getParallelism()][];
            List<RecursiveAction> tasks = new ArrayList<>();
            for (int start = 0; start < nb; start += chunk) {
                final int s = start;
                final int e = Math.min(nb, start + chunk);
                tasks.add(new RecursiveAction() {
                    @Override protected void compute() {
                        int tid = (int) (Thread.currentThread().getId() % partialBestD.length + partialBestD.length) % partialBestD.length;
                        float[] localD = new float[K];
                        int[] localI = new int[K];
                        for (int i = 0; i < K; i++) localD[i] = Float.POSITIVE_INFINITY;
                        for (int i = s; i < e; i++) {
                            float dist = kernelDistance(kernel, space, query, base, i * dLocal, dLocal);
                            if (dist < localD[K - 1]) {
                                int j = K - 1;
                                while (j > 0 && localD[j - 1] > dist) { localD[j] = localD[j - 1]; localI[j] = localI[j - 1]; j--; }
                                localD[j] = dist;
                                localI[j] = i;
                            }
                        }
                        partialBestD[tid] = localD;
                        partialBestI[tid] = localI;
                    }
                });
            }
            RecursiveAction.invokeAll(tasks);
            // merge partial results
            for (int t = 0; t < partialBestD.length; t++) {
                if (partialBestD[t] == null) continue;
                for (int j = 0; j < k; j++) {
                    float pd = partialBestD[t][j];
                    int pidx = partialBestI[t][j];
                    if (pd < bestDist[k - 1]) {
                        int m = k - 1;
                        while (m > 0 && bestDist[m - 1] > pd) { bestDist[m] = bestDist[m - 1]; bestIdx[m] = bestIdx[m - 1]; m--; }
                        bestDist[m] = pd;
                        bestIdx[m] = pidx;
                    }
                }
            }
        } else {
            for (int i = 0; i < nb; i++) {
                float dist = kernelDistance(kernel, space, query, base, i * dim, dim);
                if (dist < bestDist[k - 1]) {
                    int j = k - 1;
                    while (j > 0 && bestDist[j - 1] > dist) {
                        bestDist[j] = bestDist[j - 1];
                        bestIdx[j] = bestIdx[j - 1];
                        j--;
                    }
                    bestDist[j] = dist;
                    bestIdx[j] = i;
                }
            }
        }

        int valid = k;
        while (valid > 0 && bestIdx[valid - 1] == -1) valid--;
        int[] outI = new int[valid];
        float[] outD = new float[valid];
        long[] outIds = new long[valid];
        System.arraycopy(bestIdx, 0, outI, 0, valid);
        System.arraycopy(bestDist, 0, outD, 0, valid);
        return new AnnSearchResult(outI, outD, outIds);
    }

    private static float kernelDistance(AnnKernel k, Distance space, float[] q, float[] base,
                                        int rowBase, int dim) {
        switch (space) {
            case L2:     return k.l2(q, base, rowBase, dim);
            case IP:     return k.ip(q, base, rowBase, dim);
            case COSINE: return k.cosine(q, base, rowBase, dim);
            default:     return Float.MAX_VALUE;
        }
    }

    private static AnnSearchResult tryGpuBruteForce(float[] base, int nb, float[] query, int dim,
                                                    int k, Distance space, AnnKernel kernel) {
        float[] flat = kernel.bruteForce(base, nb, query, 1, dim, space);
        if (flat == null) return null;
        int[] bestI = new int[k];
        float[] bestD = new float[k];
        for (int i = 0; i < k; i++) { bestI[i] = -1; bestD[i] = Float.POSITIVE_INFINITY; }
        for (int i = 0; i < nb; i++) {
            float dist = flat[i];
            if (dist < bestD[k - 1]) {
                int j = k - 1;
                while (j > 0 && bestD[j - 1] > dist) { bestD[j] = bestD[j - 1]; bestI[j] = bestI[j - 1]; j--; }
                bestD[j] = dist;
                bestI[j] = i;
            }
        }
        int valid = k;
        while (valid > 0 && bestI[valid - 1] == -1) valid--;
        int[] outI = new int[valid];
        float[] outD = new float[valid];
        long[] outIds = new long[valid];
        System.arraycopy(bestI, 0, outI, 0, valid);
        System.arraycopy(bestD, 0, outD, 0, valid);
        return new AnnSearchResult(outI, outD, outIds);
    }

    /** Compute approximate recall vs brute-force ground truth. */
    public static double recall(AnnSearchResult approx, AnnSearchResult truth) {
        if (truth == null || truth.indices() == null || truth.indices().length == 0) return 1.0;
        if (approx == null || approx.indices() == null || approx.indices().length == 0) return 0.0;
        int[] t = truth.indices();
        java.util.Set<Integer> truthSet = new java.util.HashSet<>();
        for (int id : t) truthSet.add(id);
        int hits = 0;
        for (int id : approx.indices()) if (truthSet.contains(id)) hits++;
        return hits / (double) t.length;
    }

    // ------------------- HNSW builder -------------------

    public static final class HnswBuilder {
        private final int dim;
        private int M = 16;
        private int efConstruction = 200;
        private Distance space = Distance.L2;
        private boolean normalize = false;
        private int initialCap = 1024;
        private float[] matrix;
        private int n;
        private long[] ids;
        private int efSearch = 0;
        private boolean turboFast = false;
        private boolean useGpu = false;
        private String name = "hnsw";

        HnswBuilder(int dim) { this.dim = dim; }

        public HnswBuilder M(int v) { this.M = v; return this; }
        public HnswBuilder efConstruction(int v) { this.efConstruction = v; return this; }
        public HnswBuilder efSearch(int v) { this.efSearch = v; return this; }
        public HnswBuilder space(Distance v) { this.space = v; return this; }
        public HnswBuilder normalize(boolean v) { this.normalize = v; return this; }
        public HnswBuilder initialCapacity(int v) { this.initialCap = v; return this; }
        public HnswBuilder name(String v) { this.name = v; return this; }
        public HnswBuilder turboFast(boolean v) {
            this.turboFast = v && VectorDistanceKernel.AVAILABLE;
            return this;
        }
        public HnswBuilder useGpu(boolean v) { this.useGpu = v; return this; }
        public HnswBuilder kernel(AnnKernel k) {
            this.turboFast = k != null && k.turboFast;
            this.useGpu = k != null && k.useGpu;
            return this;
        }

        public HnswBuilder vectors(float[] matrix, int n) { this.matrix = matrix; this.n = n; return this; }
        public HnswBuilder vectors(float[][] rows) {
            if (rows == null || rows.length == 0) { this.matrix = new float[0]; this.n = 0; return this; }
            int d = rows[0].length;
            float[] m = new float[rows.length * d];
            for (int i = 0; i < rows.length; i++) {
                if (rows[i] == null || rows[i].length != d)
                    throw new IllegalArgumentException("ragged vectors at " + i);
                System.arraycopy(rows[i], 0, m, i * d, d);
            }
            this.matrix = m; this.n = rows.length; return this;
        }
        public HnswBuilder ids(long[] ids) { this.ids = ids; return this; }

        public HnswIndex build() {
            HnswIndex idx = new HnswIndex(dim, M, efConstruction, space, normalize,
                matrix != null ? Math.max(initialCap, n) : initialCap, efSearch);
            if (name != null) idx.setName(name);
            idx.setKernel(AnnKernel.of(turboFast, useGpu));
            if (matrix != null && n > 0) {
                idx.add(matrix, n, ids);
            }
            return idx;
        }
    }
}