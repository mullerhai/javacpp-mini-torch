/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.util.ArrayList;
import java.util.List;

/**
 * SentenceTransformer-style embeddings. Pure Java TF-IDF approximation that produces
 * 384-dimensional L2-normalized vectors. Used by Ex23 (CAG evaluation cosine similarity).
 */
public final class SentenceTransformer {

    private final int dim;
    private final String modelName;

    public SentenceTransformer(String modelName) {
        this(modelName, 384);
    }

    public SentenceTransformer(String modelName, int dim) {
        this.modelName = modelName;
        this.dim = dim;
    }

    public int dimension() { return dim; }
    public String modelName() { return modelName; }

    public float[] encode(String text) {
        if (text == null || text.isEmpty()) return new float[dim];
        int[] hashes = new int[dim];
        for (String tok : text.toLowerCase().split("\\W+")) {
            if (tok.isEmpty()) continue;
            int h = tok.hashCode();
            int pos = Math.floorMod(h, dim);
            hashes[pos]++;
        }
        float[] v = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) { v[i] = (float) hashes[i]; norm += hashes[i] * hashes[i]; }
        if (norm > 0) {
            norm = Math.sqrt(norm);
            for (int i = 0; i < dim; i++) v[i] /= norm;
        }
        return v;
    }

    public List<float[]> encodeBatch(List<String> texts) {
        List<float[]> out = new ArrayList<>();
        for (String t : texts) out.add(encode(t));
        return out;
    }

    public static double cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null) return 0.0;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}