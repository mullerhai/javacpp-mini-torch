/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * In-memory vector store compatible with LangChain's {@code VectorStore} interface
 * (search-by-similarity, add_documents, mmr, etc.). Used by the CAG example.
 */
public final class MemoryVectorStore {

    public static final class Match {
        public final Document doc;
        public final double score;
        public Match(Document doc, double score) { this.doc = doc; this.score = score; }
    }

    public interface Embeddings {
        float[] embed(String text);
    }

    private final Embeddings embeddings;
    private final List<Document> documents = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();

    public MemoryVectorStore(Embeddings embeddings) {
        this.embeddings = embeddings;
    }

    public MemoryVectorStore addDocuments(List<Document> docs) {
        for (Document d : docs) {
            float[] v = embeddings.embed(d.pageContent);
            documents.add(d);
            vectors.add(v);
        }
        return this;
    }

    public List<Match> similaritySearch(String query, int k) {
        float[] q = embeddings.embed(query);
        double[] qv = new double[q.length];
        for (int i = 0; i < q.length; i++) qv[i] = q[i];
        // cosine
        List<Match> out = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) {
            out.add(new Match(documents.get(i), SentenceTransformer.cosineSimilarity(q, vectors.get(i))));
        }
        out.sort((a, b) -> Double.compare(b.score, a.score));
        return out.subList(0, Math.min(k, out.size()));
    }

    public List<Match> mmr(String query, int k, double lambda) {
        List<Match> base = similaritySearch(query, Math.max(k, 10));
        List<Match> selected = new ArrayList<>();
        List<Match> cands = new ArrayList<>(base);
        for (int i = 0; i < k && !cands.isEmpty(); i++) {
            Match best = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (Match m : cands) {
                double div = 0;
                for (Match s : selected) {
                    div = Math.max(div, SentenceTransformer.cosineSimilarity(
                            embeddings.embed(m.doc.pageContent),
                            embeddings.embed(s.doc.pageContent)));
                }
                double score = lambda * m.score - (1 - lambda) * div;
                if (score > bestScore) { bestScore = score; best = m; }
            }
            if (best != null) {
                selected.add(best);
                cands.remove(best);
            }
        }
        return selected;
    }

    public List<Document> documents() { return documents; }
    public List<float[]> vectors() { return vectors; }
}