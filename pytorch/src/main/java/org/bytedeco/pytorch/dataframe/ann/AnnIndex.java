package org.bytedeco.pytorch.dataframe.ann;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Path;

/**
 * Common ANN index contract used across the {@code ann} module.
 *
 * <p>Implementations: {@link HnswIndex} (graph-based) and the pure-Java flat
 * search used by {@link AnnFactory#flatBruteForce}.
 *
 * <p>Each index owns an {@link AnnKernel} that selects among:
 * <ul>
 *   <li>{@code turboFast=true}: jdk.incubator.vector SIMD on CPU</li>
 *   <li>{@code useGpu=true}: torch matmul on CUDA / MPS</li>
 *   <li>default: pure-Java scalar fallback</li>
 * </ul>
 */
public interface AnnIndex extends Serializable {

    /** Vector dimension. */
    int dim();

    /** Number of currently stored vectors. */
    int size();

    /** Add vectors (row-major matrix or row array). */
    void add(float[][] rows);

    void add(float[] matrix, int n);

    void add(float[] matrix, int n, long[] ids);

    /** Single-query k-NN. */
    AnnSearchResult search(float[] query, int k);

    /** Single-query k-NN with efSearch override. */
    AnnSearchResult search(float[] query, int k, int efSearch);

    /** Batch k-NN. */
    AnnSearchResult[] searchBatch(float[] queries, int nq, int k, int efSearch);

    /** Active kernel selection. */
    AnnKernel kernel();

    /** Switch kernel. Resets visited-state scratch buffers lazily. */
    void setKernel(AnnKernel k);

    /** Persist index to disk (best-effort, format depends on implementation). */
    void save(Path path) throws IOException;

    static AnnIndex load(Path path) throws IOException, ClassNotFoundException {
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(
                new java.io.BufferedInputStream(java.nio.file.Files.newInputStream(path)))) {
            return (AnnIndex) ois.readObject();
        }
    }
}