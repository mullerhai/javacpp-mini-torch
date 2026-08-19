package org.bytedeco.pytorch.dataframe.faiss;

/**
 * Bounded top-k collector. Maintains the worst element at {@code size-1}
 * (max-at-end invariant for lower-is-better; min-at-end otherwise) so
 * {@link #offer(long, float)} runs in amortized O(1) for the rejection check
 * and at most one compare-and-swap to restore the invariant.
 *
 * <p>For L2 (lower better) keeps smallest k; for IP (higher better) keeps largest k.
 */
public final class TopK {
    private final int k;
    private final boolean lowerIsBetter;
    private final float[] dist;
    private final long[] id;
    private int size;

    public TopK(int k, boolean lowerIsBetter) {
        if (k <= 0) throw new IllegalArgumentException("k must be > 0");
        this.k = k;
        this.lowerIsBetter = lowerIsBetter;
        this.dist = new float[k];
        this.id = new long[k];
        this.size = 0;
        float init = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            dist[i] = init;
            id[i] = -1;
        }
    }

    /** Reset for reuse (avoid re-alloc). */
    public void reset() {
        size = 0;
        float init = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            dist[i] = init;
            id[i] = -1;
        }
    }

    public void offer(long idx, float d) {
        if (size < k) {
            dist[size] = d;
            id[size] = idx;
            size++;
            // After insert, restore worst-at-end invariant with a single sift.
            siftIntoPosition(size - 1);
            return;
        }
        // Full: cheap rejection against the worst (always at size-1).
        if (lowerIsBetter) {
            if (d >= dist[size - 1]) return;
        } else {
            if (d <= dist[size - 1]) return;
        }
        // Replace the worst slot, then restore invariant with at most one compare+sift.
        dist[size - 1] = d;
        id[size - 1] = idx;
        siftIntoPosition(size - 1);
    }

    /** Single-pass sift to maintain worst-at-end. */
    private void siftIntoPosition(int i) {
        if (lowerIsBetter) {
            // worst at end (largest dist at end) → bubble the new element up toward smaller-index slots.
            while (i > 0 && dist[i] < dist[i - 1]) {
                swap(i, i - 1);
                i--;
            }
        } else {
            // worst at end (smallest dist at end for higher-is-better) → bubble toward smaller-index slots.
            while (i > 0 && dist[i] > dist[i - 1]) {
                swap(i, i - 1);
                i--;
            }
        }
    }

    private void swap(int a, int b) {
        float td = dist[a]; dist[a] = dist[b]; dist[b] = td;
        long ti = id[a]; id[a] = id[b]; id[b] = ti;
    }

    /** Worst (boundary) distance currently held; O(1). */
    public float worst() {
        if (size == 0) {
            return lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        return dist[size - 1];
    }

    public boolean isFull() { return size >= k; }

    public int size() { return size; }

    /** Sorted best→worst into out arrays (length k, pad with init / -1). */
    public void export(float[] outD, long[] outI) {
        int n = Math.min(k, outD.length);
        // The internal array is already sorted best→worst, so just copy.
        if (n > size) {
            float init = lowerIsBetter ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            for (int i = size; i < n; i++) {
                outD[i] = init;
                outI[i] = -1;
            }
            n = size;
        }
        System.arraycopy(dist, 0, outD, 0, n);
        System.arraycopy(id, 0, outI, 0, n);
    }

    public static SearchResult toSearchResult(TopK[] perQuery, int k) {
        int nq = perQuery.length;
        float[][] D = new float[nq][k];
        long[][] I = new long[nq][k];
        for (int q = 0; q < nq; q++) {
            perQuery[q].export(D[q], I[q]);
        }
        return new SearchResult(D, I);
    }

    /**
     * Allocate a fresh collector. Must NOT be keyed only by {@code k}:
     * batch knn does {@code heaps[q] = borrow(k)} for every query, so a
     * per-k singleton would alias all queries onto one heap and collapse
     * recall to ~0 (Flat GT, IVFPQ, VectorCpu backend).
     */
    public static TopK borrow(int k, boolean lowerIsBetter) {
        return new TopK(k, lowerIsBetter);
    }
}
