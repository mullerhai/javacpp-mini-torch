package org.bytedeco.pytorch.dataframe.faiss;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * CPU brute-force k-NN / range search using {@link VectorDistanceKernel} (Project Panama SIMD).
 *
 * <p>If the incubator vector module is unavailable at runtime, this backend is a near-perfect
 * alias for {@link CpuDistanceBackend} (the kernel auto-falls back to scalar math).
 */
public final class VectorCpuDistanceBackend implements DistanceBackend {
    public static final VectorCpuDistanceBackend INSTANCE = new VectorCpuDistanceBackend();

    private VectorCpuDistanceBackend() {}

    @Override
    public String name() {
        return VectorDistanceKernel.AVAILABLE
            ? "cpu-vector-" + VectorDistanceKernel.LANE_COUNT + "w"
            : "cpu-vector-fallback-scalar";
    }

    @Override
    public SearchResult knn(float[] base, int nb, float[] queries, int nq,
                            int d, int k, MetricType metric, long[] ids) {
        if (nq <= 0 || k <= 0 || nb <= 0) {
            float[][] D = new float[Math.max(0, nq)][Math.max(0, k)];
            long[][] I = new long[Math.max(0, nq)][Math.max(0, k)];
            for (int q = 0; q < nq; q++) {
                for (int j = 0; j < k; j++) {
                    D[q][j] = metric.lowerIsBetter() ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                    I[q][j] = -1;
                }
            }
            return new SearchResult(D, I);
        }
        k = Math.min(k, nb);

        boolean lower = metric.lowerIsBetter();
        boolean l2 = metric == MetricType.METRIC_L2;
        TopK[] heaps = new TopK[nq];
        for (int q = 0; q < nq; q++) heaps[q] = TopK.borrow(k, lower);

        if (nq == 1) {
            scanQueryVector(base, nb, queries, 0, d, heaps[0], l2, ids);
        } else {
            int parallelism = Math.max(1, ForkJoinPool.commonPool().getParallelism());
            int chunk = Math.max(1, (nq + parallelism - 1) / parallelism);
            RecursiveAction root = new RecursiveAction() {
                @Override protected void compute() {
                    List<RecursiveAction> tasks = new ArrayList<>();
                    for (int start = 0; start < nq; start += chunk) {
                        final int s = start;
                        final int e = Math.min(nq, start + chunk);
                        tasks.add(new RecursiveAction() {
                            @Override protected void compute() {
                                for (int q = s; q < e; q++) {
                                    scanQueryVector(base, nb, queries, q, d, heaps[q], l2, ids);
                                }
                            }
                        });
                    }
                    invokeAll(tasks);
                }
            };
            ForkJoinPool.commonPool().invoke(root);
        }
        return TopK.toSearchResult(heaps, k);
    }

    @Override
    public RangeSearchResult range(float[] base, int nb, float[] queries, int nq,
                                   int d, float radius, MetricType metric, long[] ids) {
        if (nq <= 0 || nb <= 0) {
            return new RangeSearchResult(new long[]{0}, new float[0], new long[0]);
        }
        boolean l2 = metric == MetricType.METRIC_L2;
        int[] counts = new int[nq];
        long[] lims = new long[nq + 1];

        // Pass 1: count
        for (int q = 0; q < nq; q++) {
            int qOff = q * d;
            int c = 0;
            for (int i = 0; i < nb; i++) {
                float dist = l2
                    ? VectorDistanceKernel.l2Row(queries, qOff, base, i * d, d)
                    : VectorDistanceKernel.ipRow(queries, qOff, base, i * d, d);
                if (l2 ? dist <= radius : dist >= radius) c++;
            }
            counts[q] = c;
            lims[q + 1] = lims[q] + c;
        }
        long total = lims[nq];
        float[] D = new float[(int) total];
        long[] I = new long[(int) total];
        for (int q = 0; q < nq; q++) {
            int qOff = q * d;
            int c = counts[q];
            if (c == 0) continue;
            float[] tmpD = new float[c];
            long[] tmpI = new long[c];
            int m = 0;
            for (int i = 0; i < nb; i++) {
                float dist = l2
                    ? VectorDistanceKernel.l2Row(queries, qOff, base, i * d, d)
                    : VectorDistanceKernel.ipRow(queries, qOff, base, i * d, d);
                if (l2 ? dist <= radius : dist >= radius) {
                    tmpD[m] = dist;
                    tmpI[m] = ids != null ? ids[i] : i;
                    m++;
                }
            }
            sortPairs(tmpD, tmpI, m, l2);
            int baseIdx = (int) lims[q];
            System.arraycopy(tmpD, 0, D, baseIdx, m);
            System.arraycopy(tmpI, 0, I, baseIdx, m);
        }
        return new RangeSearchResult(lims, D, I);
    }

    private static final ThreadLocal<float[]> QUERY_SCRATCH = new ThreadLocal<>();

    private static void scanQueryVector(float[] base, int nb, float[] queries, int q,
                                       int d, TopK heap, boolean l2, long[] ids) {
        int qOff = q * d;
        float[] qv = QUERY_SCRATCH.get();
        if (qv == null || qv.length < d) {
            qv = new float[Math.max(d, 256)];
            QUERY_SCRATCH.set(qv);
        }
        System.arraycopy(queries, qOff, qv, 0, d);
        for (int i = 0; i < nb; i++) {
            float dist = l2
                ? VectorDistanceKernel.l2Row(qv, base, i * d, d)
                : VectorDistanceKernel.ipRow(qv, base, i * d, d);
            long id = ids != null ? ids[i] : i;
            heap.offer(id, dist);
        }
    }

    private static void sortPairs(float[] d, long[] ids, int n, boolean lowerIsBetter) {
        if (n > 64) {
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) order[i] = i;
            if (lowerIsBetter) {
                java.util.Arrays.sort(order, (a, b) -> Float.compare(d[a], d[b]));
            } else {
                java.util.Arrays.sort(order, (a, b) -> Float.compare(d[b], d[a]));
            }
            float[] nd = new float[n];
            long[] ni = new long[n];
            for (int i = 0; i < n; i++) {
                nd[i] = d[order[i]];
                ni[i] = ids[order[i]];
            }
            System.arraycopy(nd, 0, d, 0, n);
            System.arraycopy(ni, 0, ids, 0, n);
            return;
        }
        for (int i = 1; i < n; i++) {
            float key = d[i];
            long kid = ids[i];
            int j = i - 1;
            if (lowerIsBetter) {
                while (j >= 0 && d[j] > key) { d[j + 1] = d[j]; ids[j + 1] = ids[j]; j--; }
            } else {
                while (j >= 0 && d[j] < key) { d[j + 1] = d[j]; ids[j + 1] = ids[j]; j--; }
            }
            d[j + 1] = key;
            ids[j + 1] = kid;
        }
    }
}
