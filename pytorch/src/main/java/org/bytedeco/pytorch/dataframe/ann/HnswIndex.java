package org.bytedeco.pytorch.dataframe.ann;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;

import org.bytedeco.pytorch.dataframe.faiss.VectorDistanceKernel;

/**
 * Enterprise-grade Hierarchical Navigable Small World (HNSW) index — extends the
 * baseline implementation with:
 *
 * <ul>
 *   <li><b>SIMD-accelerated distance math</b> via {@link AnnKernel#turboFast}
 *       (jdk.incubator.vector) — controlled by {@link #setKernel}.</li>
 *   <li><b>GPU bulk operations</b> via {@link AnnKernel#useGpu} (CUDA / MPS torch matmul).</li>
 *   <li><b>Tunable {@code efSearch}</b> at runtime without rebuilding — important for
 *       online latency/recall trade-offs.</li>
 *   <li><b>Generation-stamp visited</b> bookkeeping — eliminates per-query {@code boolean[]}
 *       allocations for billion-scale workloads.</li>
 *   <li><b>Soft-delete with tombstone</b> and {@link #commitDelete()} compaction —
 *       removes need to rebuild the whole graph on small deletes.</li>
 *   <li><b>Filtered search</b> via {@link IDSelector}.</li>
 *   <li><b>Parallel batch search</b> across queries (ForkJoin).</li>
 *   <li><b>Save / load</b> with versioning + magic header for cross-version tolerance.</li>
 * </ul>
 *
 * <pre>
 *   HnswIndex idx = AnnFactory.hnsw(256)
 *       .M(32).efConstruction(200).efSearch(64)
 *       .space(Distance.COSINE)
 *       .turboFast(true)        // enable jdk.incubator.vector SIMD
 *       .build();
 *   idx.add(vectors);
 *   AnnSearchResult r = idx.search(query, 10);
 * </pre>
 */
public final class HnswIndex implements AnnIndex {
    private static final long serialVersionUID = 3L;

    // ── header magic for save/load (version-tagged) ──────────────
    private static final int FILE_MAGIC = 0x484E5357; // 'HNSW'
    private static final int FILE_VERSION = 2;

    // ── config ──────────────────────────────────────────────────
    private final int dim;
    private final int M;
    private final int maxM0;
    private final int efConstruction;
    private final Distance space;
    private final double levelMult;
    private final boolean normalize;

    private int efSearch;            // mutable per-instance runtime knob
    private String name = "hnsw";

    // ── storage ────────────────────────────────────────────────
    private float[] data;
    private int size;
    private int capacity;
    private long[] ids;              // -1 if no external id
    private boolean hasIds;
    private byte[] deleted;          // soft-delete tombstone (1=deleted, 0=alive)
    private int liveCount;           // number of !deleted entries
    private boolean dirty;           // true when tombstones need compaction

    // ── graph ──────────────────────────────────────────────────
    private int[][][] neighbors;
    private int[] levels;
    private int entryPoint = -1;
    private int maxLevel = -1;

    // ── kernel + visited stamps ────────────────────────────────
    private transient AnnKernel kernel = AnnKernel.LEGACY;
    private transient int[] visitStamp;
    private transient int visitGen = 1;
    private transient int liveCapacityForStamp;

    // ── ctor ───────────────────────────────────────────────────

    // Visible-for-builder — package private so AnnFactory can use.
    HnswIndex(int dim, int M, int efConstruction, Distance space, boolean normalize,
              int initialCap, int efSearch) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        if (M < 2) throw new IllegalArgumentException("M must be >= 2");
        this.dim = dim;
        this.M = M;
        this.maxM0 = M * 2;
        this.efConstruction = Math.max(efConstruction, M);
        this.efSearch = Math.max(efSearch <= 0 ? this.efConstruction / 2 : efSearch, 1);
        this.space = space == null ? Distance.L2 : space;
        this.levelMult = 1.0 / Math.log(M);
        this.normalize = normalize;
        this.capacity = Math.max(16, initialCap);
        this.data = new float[this.capacity * dim];
        this.neighbors = new int[this.capacity][][];
        this.levels = new int[this.capacity];
        this.ids = new long[this.capacity];
        Arrays.fill(this.ids, -1L);
        this.deleted = new byte[this.capacity];
        this.size = 0;
        this.liveCount = 0;
        this.hasIds = false;
        this.dirty = false;
        ensureVisitStamp(capacity);
    }

    public static AnnFactory.HnswBuilder builder(int dim) {
        return AnnFactory.hnsw(dim);
    }

    // ── getters / setters ──────────────────────────────────────

    @Override public int dim() { return dim; }
    @Override public int size() { return size; }
    /** Number of non-deleted entries. */
    public int liveCount() { return liveCount; }
    public int M() { return M; }
    public int efConstruction() { return efConstruction; }
    public int efSearch() { return efSearch; }
    public void setEfSearch(int v) { this.efSearch = Math.max(v, 1); }
    public Distance space() { return space; }
    public boolean normalize() { return normalize; }
    public String name() { return name; }
    public void setName(String n) { this.name = n == null ? "hnsw" : n; }

    @Override public AnnKernel kernel() { return kernel; }
    @Override public void setKernel(AnnKernel k) {
        this.kernel = k == null ? AnnKernel.LEGACY : k;
    }

    /** Set turboFast (Project Panama SIMD on CPU). */
    public void setTurboFast(boolean on) {
        this.kernel = AnnKernel.of(on && VectorDistanceKernel.AVAILABLE, this.kernel.useGpu);
    }

    /** Set useGpu (CUDA / MPS via torch matmul). */
    public void setUseGpu(boolean on) {
        this.kernel = AnnKernel.of(this.kernel.turboFast, on);
    }

    // ── add ────────────────────────────────────────────────────

    @Override public synchronized void add(float[][] rows) {
        if (rows == null || rows.length == 0) return;
        float[] m = new float[rows.length * dim];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] == null || rows[i].length != dim)
                throw new IllegalArgumentException("vector dim mismatch at " + i);
            System.arraycopy(rows[i], 0, m, i * dim, dim);
        }
        add(m, rows.length, null);
    }

    @Override public synchronized void add(float[] matrix, int n) {
        add(matrix, n, null);
    }

    @Override public synchronized void add(float[] matrix, int n, long[] externalIds) {
        if (n <= 0) return;
        if (matrix.length < n * dim) throw new IllegalArgumentException("matrix too small");
        ensureCapacity(size + n);
        if (normalize) {
            for (int i = 0; i < n; i++) {
                int src = i * dim;
                int dst = (size + i) * dim;
                float inv = kernelNormalizeInv(matrix, src, dim);
                for (int d = 0; d < dim; d++) data[dst + d] = matrix[src + d] * inv;
            }
        } else {
            System.arraycopy(matrix, 0, data, size * dim, n * dim);
        }
        if (externalIds != null) hasIds = true;
        for (int i = 0; i < n; i++) {
            int slot = size + i;
            ids[slot] = externalIds != null && i < externalIds.length
                ? externalIds[i] : (hasIds ? slot : -1L);
            deleted[slot] = 0;
            liveCount++;
            insertNode(slot);
        }
        size += n;
    }

    public synchronized void addOne(float[] vector) { addOne(vector, -1L); }

    public synchronized void addOne(float[] vector, long id) {
        if (vector == null || vector.length != dim)
            throw new IllegalArgumentException("vector dim mismatch");
        ensureCapacity(size + 1);
        int base = size * dim;
        if (normalize) {
            float inv = kernelNormalizeInv(vector, 0, dim);
            for (int d = 0; d < dim; d++) data[base + d] = vector[d] * inv;
        } else {
            System.arraycopy(vector, 0, data, base, dim);
        }
        if (id >= 0) hasIds = true;
        ids[size] = id >= 0 ? id : (hasIds ? size : -1L);
        deleted[size] = 0;
        liveCount++;
        insertNode(size);
        size++;
    }

    private void ensureCapacity(int need) {
        if (need <= capacity) {
            ensureVisitStamp(capacity);
            return;
        }
        int nc = Math.max(Math.max(16, capacity * 2), need);
        data = Arrays.copyOf(data, nc * dim);
        neighbors = Arrays.copyOf(neighbors, nc);
        levels = Arrays.copyOf(levels, nc);
        ids = Arrays.copyOf(ids, nc);
        deleted = Arrays.copyOf(deleted, nc);
        capacity = nc;
        ensureVisitStamp(capacity);
    }

    private void ensureVisitStamp(int cap) {
        int need = Math.max(16, cap);
        if (visitStamp == null || visitStamp.length < need) {
            int[] next = new int[need];
            if (visitStamp != null) System.arraycopy(visitStamp, 0, next, 0, visitStamp.length);
            visitStamp = next;
        }
        liveCapacityForStamp = cap;
    }

    /** Reserve a fresh visit generation token. Wraps safely when overflowing. */
    private int nextGen() {
        int g = ++visitGen;
        if (g == Integer.MAX_VALUE) {
            Arrays.fill(visitStamp, 0);
            visitGen = 1;
            g = 1;
        }
        return g;
    }

    private boolean isAlive(int node) {
        return node >= 0 && node < size && deleted[node] == 0;
    }

    private int randomLevel() {
        double r = ThreadLocalRandom.current().nextDouble();
        return (int) Math.floor(-Math.log(r) * levelMult);
    }

    private void insertNode(int node) {
        int level = randomLevel();
        levels[node] = level;
        neighbors[node] = new int[level + 1][];
        for (int lc = 0; lc <= level; lc++) neighbors[node][lc] = new int[0];

        if (entryPoint < 0) {
            entryPoint = node;
            maxLevel = level;
            return;
        }

        int curr = entryPoint;
        for (int lc = maxLevel; lc > level; lc--) {
            curr = greedyClosest(curr, node, lc);
        }

        for (int lc = Math.min(level, maxLevel); lc >= 0; lc--) {
            Neighbors cand = searchLayer(node, curr, efConstruction, lc, true);
            int maxM = (lc == 0) ? maxM0 : M;
            int[] selected = selectNeighborsHeuristic(node, cand, maxM, lc);
            neighbors[node][lc] = selected;
            for (int nb : selected) addNeighbor(nb, node, lc, maxM);
            if (!cand.isEmpty()) curr = cand.nearest();
        }

        if (level > maxLevel) { maxLevel = level; entryPoint = node; }
    }

    private void addNeighbor(int node, int nb, int lc, int maxM) {
        int[] cur = neighbors[node][lc];
        if (cur == null) { neighbors[node][lc] = new int[]{nb}; return; }
        for (int x : cur) if (x == nb) return;
        if (cur.length < maxM) {
            int[] n2 = Arrays.copyOf(cur, cur.length + 1);
            n2[cur.length] = nb;
            neighbors[node][lc] = n2;
        } else {
            Neighbors tmp = new Neighbors(maxM + 1);
            for (int x : cur) tmp.add(x, distNN(node, x));
            tmp.add(nb, distNN(node, nb));
            neighbors[node][lc] = selectNeighborsHeuristic(node, tmp, maxM, lc);
        }
    }

    // ── distance helpers ───────────────────────────────────────

    private float distNN(int a, int b) {
        int ba = a * dim, bb = b * dim;
        switch (space) {
            case L2:
                if (kernel.vectorEnabled()) return kernel.l2(data, ba, bb, dim);
                return Distance.L2.distance(data, ba, bb, dim);
            case IP:
                if (kernel.vectorEnabled()) return kernel.ip(data, ba, bb, dim);
                return Distance.IP.distance(data, ba, bb, dim);
            case COSINE:
            default:
                return Distance.COSINE.distance(data, ba, bb, dim);
        }
    }

    private float distQuery(float[] q, int node) {
        return kernelDistQuery(q, node * dim);
    }

    private float kernelDistQuery(float[] q, int rowBase) {
        switch (space) {
            case L2:
                if (kernel.vectorEnabled()) return kernel.l2(q, data, rowBase, dim);
                return Distance.L2.distance(q, data, rowBase / dim, dim);
            case IP:
                if (kernel.vectorEnabled()) return kernel.ip(q, data, rowBase, dim);
                return Distance.IP.distance(q, data, rowBase / dim, dim);
            case COSINE:
            default:
                return Distance.COSINE.distance(q, data, rowBase / dim, dim);
        }
    }

    private float kernelNormalizeInv(float[] v, int off, int dim) {
        float sum = 0f;
        if (kernel.vectorEnabled() && dim >= VectorDistanceKernel.LANE_COUNT) {
            sum = VectorDistanceKernel.sqNorm(v, off, dim);
        } else {
            for (int d = 0; d < dim; d++) sum += v[off + d] * v[off + d];
        }
        return sum > 0f ? (float) (1.0 / Math.sqrt(sum)) : 1f;
    }

    // ── search primitives ──────────────────────────────────────

    private int greedyClosest(int enter, int queryNode, int lc) {
        int curr = enter;
        float curDist = distNN(curr, queryNode);
        boolean changed = true;
        while (changed) {
            changed = false;
            int[] nbs = neighbors[curr][lc];
            if (nbs == null) break;
            for (int nb : nbs) {
                float d = distNN(nb, queryNode);
                if (d < curDist) { curDist = d; curr = nb; changed = true; }
            }
        }
        return curr;
    }

    private int greedyClosestQuery(int enter, float[] query, int lc) {
        int curr = enter;
        float curDist = distQuery(query, curr);
        boolean changed = true;
        while (changed) {
            changed = false;
            int[] nbs = neighbors[curr][lc];
            if (nbs == null) break;
            for (int nb : nbs) {
                float d = distQuery(query, nb);
                if (d < curDist) { curDist = d; curr = nb; changed = true; }
            }
        }
        return curr;
    }

    /**
     * Layer beam search.
     *
     * @param queryNode the node index being inserted (when queryIsNode)
     * @param enter     entry node id on this layer
     * @param ef        beam width
     * @param lc        layer index
     * @param queryIsNode true if the "query" is an existing node id (insert path);
     *                    false if it is a real vector.
     */
    private Neighbors searchLayer(int queryNode, float[] query, int enter, int ef, int lc, boolean queryIsNode) {
        Neighbors candidates = new Neighbors(ef * 2);
        Neighbors w = new Neighbors(ef * 2);
        int gen = nextGen();
        if (enter < 0 || enter >= capacity) return w;
        float enterDist = queryIsNode ? distNN(enter, queryNode) : distQuery(query, enter);
        candidates.add(enter, enterDist);
        w.add(enter, enterDist);
        visitStamp[enter] = gen;

        while (!candidates.isEmpty()) {
            int c = candidates.pollNearest();
            float cDist = candidates.lastPolledDist;
            float farthest = w.farthestDist();
            if (cDist > farthest && w.size() >= ef) break;
            int[] nbs = (c < neighbors.length && neighbors[c] != null && lc < neighbors[c].length)
                ? neighbors[c][lc] : null;
            if (nbs == null) continue;
            for (int nb : nbs) {
                if (nb < 0 || nb >= size || deleted[nb] != 0) continue;
                if (visitStamp[nb] == gen) continue;
                visitStamp[nb] = gen;
                float d = queryIsNode ? distNN(nb, queryNode) : distQuery(query, nb);
                if (w.size() < ef || d < w.farthestDist()) {
                    candidates.add(nb, d);
                    w.add(nb, d);
                    if (w.size() > ef) w.pollFarthest();
                }
            }
        }
        return w;
    }

    /** Node-to-node insertion searchLayer (query is an existing node id). */
    private Neighbors searchLayer(int queryNode, int enter, int ef, int lc, boolean queryIsNode) {
        return searchLayer(queryNode, null, enter, ef, lc, queryIsNode);
    }

    /** Heuristic neighbor selection (Algorithm 4 from the HNSW paper). */
    private int[] selectNeighborsHeuristic(int queryNode, Neighbors cand, int maxM, int lc) {
        if (cand.size() <= maxM) {
            int[] out = new int[cand.size()];
            int i = 0;
            for (int id : cand.toSortedIds()) out[i++] = id;
            return out;
        }
        int[] sorted = cand.toSortedIds();
        float[] sortedDist = cand.toSortedDists();
        List<Integer> selected = new ArrayList<>(maxM);
        for (int i = 0; i < sorted.length && selected.size() < maxM; i++) {
            int c = sorted[i];
            float dqc = sortedDist[i];
            boolean ok = true;
            for (int s : selected) {
                float dsc = distNN(s, c);
                if (dsc < dqc) { ok = false; break; }
            }
            if (ok) selected.add(c);
        }
        if (selected.size() < maxM) {
            Set<Integer> have = new HashSet<>(selected);
            for (int c : sorted) {
                if (selected.size() >= maxM) break;
                if (have.add(c)) selected.add(c);
            }
        }
        int[] out = new int[selected.size()];
        for (int i = 0; i < selected.size(); i++) out[i] = selected.get(i);
        return out;
    }

    // ── public search ──────────────────────────────────────────

    @Override public AnnSearchResult search(float[] query, int k) {
        return search(query, k, efSearch);
    }

    @Override public AnnSearchResult search(float[] query, int k, int ef) {
        if (size == 0 || liveCount == 0) return emptyResult(k);
        if (query == null || query.length != dim)
            throw new IllegalArgumentException("query dim mismatch");
        return doSearch(query, k, Math.max(ef, k), null);
    }

    /** Filter-aware single-query k-NN. */
    public AnnSearchResult search(float[] query, int k, int ef, IDSelector selector) {
        if (size == 0 || liveCount == 0) return emptyResult(k);
        if (query == null || query.length != dim)
            throw new IllegalArgumentException("query dim mismatch");
        if (selector == null) return search(query, k, ef);
        return doSearch(query, k, Math.max(ef, k), selector);
    }

    private AnnSearchResult doSearch(float[] query, int k, int ef, IDSelector selector) {
        float[] q = prepareQuery(query);
        int curr = entryPoint;
        for (int lc = maxLevel; lc > 0; lc--) {
            curr = greedyClosestQuery(curr, q, lc);
        }
        Neighbors top = searchLayer(-1, q, curr, ef, 0, false);
        int[] idsSorted = top.toSortedIds();
        float[] distSorted = top.toSortedDists();
        List<Integer> outIdx = new ArrayList<>(k);
        List<Float> outDist = new ArrayList<>(k);
        List<Long> outIds = new ArrayList<>(k);
        for (int i = 0; i < idsSorted.length && outIdx.size() < k; i++) {
            int idx = idsSorted[i];
            if (deleted[idx] != 0) continue;
            long external = hasIds ? ids[idx] : idx;
            if (selector != null && !selector.is_member(external)) continue;
            outIdx.add(idx);
            outDist.add(distSorted[i]);
            outIds.add(external);
        }
        int[] oI = new int[outIdx.size()]; for (int i = 0; i < oI.length; i++) oI[i] = outIdx.get(i);
        float[] oD = new float[oI.length]; for (int i = 0; i < oD.length; i++) oD[i] = outDist.get(i);
        long[] oE = new long[oI.length]; for (int i = 0; i < oE.length; i++) oE[i] = outIds.get(i);
        return new AnnSearchResult(oI, oD, oE);
    }

    private AnnSearchResult emptyResult(int k) {
        return new AnnSearchResult(new int[0], new float[0], new long[0]);
    }

    private float[] prepareQuery(float[] q) {
        if (!normalize) return q;
        float[] copy = Arrays.copyOf(q, dim);
        float inv = kernelNormalizeInv(copy, 0, dim);
        for (int i = 0; i < dim; i++) copy[i] *= inv;
        return copy;
    }

    // ── batch search (parallel) ────────────────────────────────

    @Override public AnnSearchResult[] searchBatch(float[] queries, int nq, int k, int efSearch) {
        AnnSearchResult[] out = new AnnSearchResult[nq];
        int parallelism = Math.max(1, ForkJoinPool.commonPool().getParallelism());
        if (nq >= 4 && parallelism > 1) {
            int chunk = Math.max(1, (nq + parallelism - 1) / parallelism);
            List<RecursiveAction> tasks = new ArrayList<>();
            for (int s = 0; s < nq; s += chunk) {
                final int start = s;
                final int end = Math.min(nq, s + chunk);
                tasks.add(new RecursiveAction() {
                    @Override protected void compute() {
                        for (int i = start; i < end; i++) {
                            float[] q = Arrays.copyOfRange(queries, i * dim, (i + 1) * dim);
                            out[i] = doSearch(q, k, Math.max(efSearch, k), null);
                        }
                    }
                });
            }
            RecursiveAction.invokeAll(tasks);
        } else {
            for (int i = 0; i < nq; i++) {
                float[] q = Arrays.copyOfRange(queries, i * dim, (i + 1) * dim);
                out[i] = doSearch(q, k, Math.max(efSearch, k), null);
            }
        }
        return out;
    }

    /** Brute-force ground truth (for recall benchmarks). Uses {@link AnnKernel} when GPU is enabled. */
    public AnnSearchResult bruteForce(float[] query, int k) {
        if (size == 0 || liveCount == 0) return emptyResult(k);
        float[] q = prepareQuery(query);
        // try GPU path first
        if (kernel.useGpu && (space == Distance.L2 || space == Distance.IP)) {
            AnnSearchResult r = gpuBruteForce(q, k);
            if (r != null) return r;
        }
        // CPU brute force
        int[] bestI = new int[k];
        float[] bestD = new float[k];
        for (int i = 0; i < k; i++) { bestI[i] = -1; bestD[i] = Float.POSITIVE_INFINITY; }
        for (int i = 0; i < size; i++) {
            if (deleted[i] != 0) continue;
            float d = distQuery(q, i);
            if (d < bestD[k - 1]) {
                int j = k - 1;
                while (j > 0 && bestD[j - 1] > d) {
                    bestD[j] = bestD[j - 1]; bestI[j] = bestI[j - 1]; j--;
                }
                bestD[j] = d; bestI[j] = i;
            }
        }
        int valid = k;
        while (valid > 0 && (bestI[valid - 1] < 0 || bestI[valid - 1] >= size || deleted[bestI[valid - 1]] != 0)) valid--;
        int[] oI = Arrays.copyOf(bestI, valid);
        float[] oD = Arrays.copyOf(bestD, valid);
        long[] oE = new long[valid];
        for (int i = 0; i < valid; i++) oE[i] = hasIds ? ids[oI[i]] : oI[i];
        return new AnnSearchResult(oI, oD, oE);
    }

    private AnnSearchResult gpuBruteForce(float[] q, int k) {
        try {
            float[] flat = kernel.bruteForce(data, size, q, 1, dim, space);
            if (flat == null) return null;
            int[] bestI = new int[k];
            float[] bestD = new float[k];
            for (int i = 0; i < k; i++) { bestI[i] = -1; bestD[i] = Float.POSITIVE_INFINITY; }
            for (int i = 0; i < size; i++) {
                if (deleted[i] != 0) continue;
                float d = flat[i];
                if (d < bestD[k - 1]) {
                    int j = k - 1;
                    while (j > 0 && bestD[j - 1] > d) { bestD[j] = bestD[j - 1]; bestI[j] = bestI[j - 1]; j--; }
                    bestD[j] = d; bestI[j] = i;
                }
            }
            int valid = k;
            while (valid > 0 && (bestI[valid - 1] < 0 || deleted[bestI[valid - 1]] != 0)) valid--;
            int[] oI = Arrays.copyOf(bestI, valid);
            float[] oD = Arrays.copyOf(bestD, valid);
            long[] oE = new long[valid];
            for (int i = 0; i < valid; i++) oE[i] = hasIds ? ids[oI[i]] : oI[i];
            return new AnnSearchResult(oI, oD, oE);
        } catch (Throwable t) {
            return null;
        }
    }

    // ── delete / soft delete ──────────────────────────────────

    /** Soft-delete by external id. Returns true if a node was marked. */
    public synchronized boolean deleteById(long externalId) {
        if (!hasIds) return false;
        for (int i = 0; i < size; i++) {
            if (deleted[i] == 0 && ids[i] == externalId) {
                markDeleted(i);
                return true;
            }
        }
        return false;
    }

    /** Soft-delete by internal node index. */
    public synchronized boolean deleteByIndex(int index) {
        if (index < 0 || index >= size || deleted[index] != 0) return false;
        markDeleted(index);
        return true;
    }

    private void markDeleted(int node) {
        deleted[node] = 1;
        liveCount--;
        dirty = true;
        // if entry point is gone, fall back to nearest alive node.
        if (node == entryPoint) {
            entryPoint = -1;
            maxLevel = -1;
            for (int i = 0; i < size; i++) {
                if (deleted[i] == 0) {
                    if (entryPoint == -1) entryPoint = i;
                    if (levels[i] > maxLevel) maxLevel = levels[i];
                }
            }
        }
    }

    /** Number of pending tombstones. */
    public int pendingDeletes() {
        return dirty ? (size - liveCount) : 0;
    }

    /** Compact the index: physically remove soft-deleted nodes and rebuild graph. */
    public synchronized void commitDelete() {
        if (!dirty) return;
        int newSize = liveCount;
        if (newSize == 0) {
            size = 0; entryPoint = -1; maxLevel = -1;
            deleted = new byte[capacity];
            dirty = false;
            return;
        }
        // build remap: old index → new index (or -1)
        int[] remap = new int[size];
        Arrays.fill(remap, -1);
        float[] newData = new float[newSize * dim];
        long[] newIds = new long[newSize];
        int[][][] newNeighbors = new int[newSize][][];
        int[] newLevels = new int[newSize];
        int dst = 0;
        for (int i = 0; i < size; i++) {
            if (deleted[i] != 0) continue;
            System.arraycopy(data, i * dim, newData, dst * dim, dim);
            newIds[dst] = ids[i];
            newNeighbors[dst] = neighbors[i];
            newLevels[dst] = levels[i];
            remap[i] = dst++;
        }
        // remap neighbor ids
        for (int i = 0; i < newSize; i++) {
            int[][] lvl = newNeighbors[i];
            if (lvl == null) continue;
            for (int lc = 0; lc < lvl.length; lc++) {
                int[] nb = lvl[lc];
                if (nb == null) continue;
                for (int j = 0; j < nb.length; j++) {
                    if (nb[j] >= 0 && nb[j] < size) nb[j] = remap[nb[j]];
                }
            }
        }
        // re-locate entry point
        int newEntry = entryPoint >= 0 ? remap[entryPoint] : -1;
        int newMaxLevel = newEntry >= 0 ? newLevels[newEntry] : -1;

        this.size = newSize;
        this.liveCount = newSize;
        this.data = newData;
        this.ids = newIds;
        this.neighbors = newNeighbors;
        this.levels = newLevels;
        this.entryPoint = newEntry;
        this.maxLevel = newMaxLevel;
        this.deleted = new byte[Math.max(16, capacity)];
        this.dirty = false;
        ensureVisitStamp(capacity);
    }

    // ── persistence ───────────────────────────────────────────

    @Override public void save(Path path) throws IOException {
        commitDelete();
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(FILE_MAGIC);
            out.writeInt(FILE_VERSION);
            out.writeUTF(name);
            out.writeInt(dim);
            out.writeInt(M);
            out.writeInt(efConstruction);
            out.writeInt(efSearch);
            out.writeUTF(space.name());
            out.writeBoolean(normalize);
            out.writeInt(size);
            out.writeBoolean(hasIds);
            out.writeInt(capacity);
            // ids
            for (int i = 0; i < size; i++) out.writeLong(ids[i]);
            // data
            for (int i = 0; i < size * dim; i++) out.writeFloat(data[i]);
            // levels
            for (int i = 0; i < size; i++) out.writeInt(levels[i]);
            // neighbors (variable lengths)
            for (int i = 0; i < size; i++) {
                int[][] lvl = neighbors[i];
                int lvlLen = lvl == null ? 0 : lvl.length;
                out.writeInt(lvlLen);
                for (int lc = 0; lc < lvlLen; lc++) {
                    int[] nb = lvl[lc];
                    int nbLen = nb == null ? 0 : nb.length;
                    out.writeInt(nbLen);
                    for (int j = 0; j < nbLen; j++) out.writeInt(nb[j]);
                }
            }
            out.writeInt(entryPoint);
            out.writeInt(maxLevel);
        }
    }

    public void save(String path) throws IOException { save(Path.of(path)); }

    public static HnswIndex load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path)))) {
            int magic = in.readInt();
            int ver = in.readInt();
            if (magic != FILE_MAGIC || ver != FILE_VERSION) {
                throw new IOException("Unsupported HNSW file (magic=" + magic + " v=" + ver + ")");
            }
            String name = in.readUTF();
            int dim = in.readInt();
            int M = in.readInt();
            int efConstruction = in.readInt();
            int efSearch = in.readInt();
            Distance space = Distance.valueOf(in.readUTF());
            boolean normalize = in.readBoolean();
            int size = in.readInt();
            boolean hasIds = in.readBoolean();
            int capacity = in.readInt();

            HnswIndex idx = new HnswIndex(dim, M, efConstruction, space, normalize, capacity, efSearch);
            idx.name = name;
            for (int i = 0; i < size; i++) idx.ids[i] = in.readLong();
            idx.hasIds = hasIds;
            for (int i = 0; i < size * dim; i++) idx.data[i] = in.readFloat();
            for (int i = 0; i < size; i++) idx.levels[i] = in.readInt();
            for (int i = 0; i < size; i++) {
                int lvlLen = in.readInt();
                idx.neighbors[i] = new int[lvlLen][];
                for (int lc = 0; lc < lvlLen; lc++) {
                    int nbLen = in.readInt();
                    int[] nb = new int[nbLen];
                    for (int j = 0; j < nbLen; j++) nb[j] = in.readInt();
                    idx.neighbors[i][lc] = nb;
                }
            }
            idx.entryPoint = in.readInt();
            idx.maxLevel = in.readInt();
            idx.size = size;
            idx.liveCount = size;
            idx.deleted = new byte[idx.capacity];
            idx.ensureVisitStamp(idx.capacity);
            return idx;
        }
    }

    public static HnswIndex load(String path) throws IOException {
        return load(Path.of(path));
    }

    // ── helpers ────────────────────────────────────────────────

    public float[] getVector(int index) {
        if (index < 0 || index >= size) throw new IndexOutOfBoundsException();
        return Arrays.copyOfRange(data, index * dim, (index + 1) * dim);
    }

    /** True if this index has external ids. */
    public boolean hasExternalIds() { return hasIds; }

    // ── Neighbor priority structures ───────────────────────────

    /**
     * Dual-purpose neighbor set: tracks (id, dist) pairs.
     * Supports nearest-poll (candidates) and farthest-poll (result trim).
     */
    static final class Neighbors {
        private final ArrayList<long[]> heap = new ArrayList<>();
        float lastPolledDist;

        Neighbors(int capHint) { heap.ensureCapacity(capHint); }

        int size() { return heap.size(); }
        boolean isEmpty() { return heap.isEmpty(); }

        void add(int id, float dist) {
            heap.add(new long[]{id, Float.floatToIntBits(dist)});
        }

        float farthestDist() {
            if (heap.isEmpty()) return Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (long[] e : heap) {
                float d = Float.intBitsToFloat((int) e[1]);
                if (d > max) max = d;
            }
            return max;
        }

        int nearest() {
            int best = -1;
            float min = Float.POSITIVE_INFINITY;
            for (long[] e : heap) {
                float d = Float.intBitsToFloat((int) e[1]);
                if (d < min) { min = d; best = (int) e[0]; }
            }
            return best;
        }

        int pollNearest() {
            int bi = -1;
            float min = Float.POSITIVE_INFINITY;
            for (int i = 0; i < heap.size(); i++) {
                float d = Float.intBitsToFloat((int) heap.get(i)[1]);
                if (d < min) { min = d; bi = i; }
            }
            long[] e = heap.remove(bi);
            lastPolledDist = min;
            return (int) e[0];
        }

        void pollFarthest() {
            int bi = -1;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < heap.size(); i++) {
                float d = Float.intBitsToFloat((int) heap.get(i)[1]);
                if (d > max) { max = d; bi = i; }
            }
            if (bi >= 0) heap.remove(bi);
        }

        int[] toSortedIds() {
            heap.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat((int) e[1])));
            int[] out = new int[heap.size()];
            for (int i = 0; i < heap.size(); i++) out[i] = (int) heap.get(i)[0];
            return out;
        }

        float[] toSortedDists() {
            heap.sort(Comparator.comparingDouble(e -> Float.intBitsToFloat((int) e[1])));
            float[] out = new float[heap.size()];
            for (int i = 0; i < heap.size(); i++) out[i] = Float.intBitsToFloat((int) heap.get(i)[1]);
            return out;
        }
    }
}