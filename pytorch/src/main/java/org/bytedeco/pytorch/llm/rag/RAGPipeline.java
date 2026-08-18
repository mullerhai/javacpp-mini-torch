/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * RAG pipeline driver: takes a query and a vector store, optionally re-ranks via MMR,
 * and pipes the joined context into an LLM bundle generator.
 */
public final class RAGPipeline {

    public interface LLMGenerator {
        String generate(String prompt);
    }

    private final MemoryVectorStore store;
    private final LLMGenerator generator;
    private final int topK;
    private final double mmrLambda;

    public RAGPipeline(MemoryVectorStore store, LLMGenerator generator, int topK, double mmrLambda) {
        this.store = store;
        this.generator = generator;
        this.topK = topK;
        this.mmrLambda = mmrLambda;
    }

    public String query(String userQuery) {
        List<MemoryVectorStore.Match> matches = store.mmr(userQuery, topK, mmrLambda);
        StringBuilder prompt = new StringBuilder("Use the following context to answer.\n\n");
        for (MemoryVectorStore.Match m : matches) {
            prompt.append("Context: ").append(m.doc.pageContent).append("\n");
        }
        prompt.append("\nQuestion: ").append(userQuery).append("\nAnswer:");
        return generator.generate(prompt.toString());
    }

    public MemoryVectorStore store() { return store; }
}