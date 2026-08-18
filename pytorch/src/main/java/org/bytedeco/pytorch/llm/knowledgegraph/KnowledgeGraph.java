/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.knowledgegraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Store for triples with a query interface. Used by graph-storing RAG pipelines.
 */
public final class KnowledgeGraph {

    private final List<Triple> triples = new ArrayList<>();

    public KnowledgeGraph addTriples(List<Triple> t) {
        triples.addAll(t);
        return this;
    }

    public List<Triple> triples() { return triples; }

    public List<Triple> query(String cypher) {
        // Toy implementation: matches when triple contains the cypher text as substring on any field.
        List<Triple> out = new ArrayList<>();
        for (Triple t : triples) {
            if (t.subject.contains(cypher) || t.object.contains(cypher) || t.predicate.contains(cypher)) {
                out.add(t);
            }
        }
        return out;
    }

    public String executeCypher(String cypher) {
        // Returns a synthetic tabular response.
        StringBuilder sb = new StringBuilder();
        sb.append("Triples matched:\n");
        for (Triple t : query(cypher)) sb.append("  ").append(t.subject).append(" -> ").append(t.predicate).append(" -> ").append(t.object).append('\n');
        return sb.toString();
    }
}
