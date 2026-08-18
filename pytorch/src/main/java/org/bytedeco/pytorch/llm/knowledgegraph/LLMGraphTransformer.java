/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.knowledgegraph;

import org.bytedeco.pytorch.llm.rag.Document;

import java.util.ArrayList;
import java.util.List;

/**
 * Mirror of LangChain's {@code LLMGraphTransformer.convert_to_graph_documents(docs)}. Calls a
 * triple-extractor LLM chunk-by-chunk and accumulates the triples into a knowledge graph.
 */
public final class LLMGraphTransformer {

    public interface TripleExtractor {
        List<Triple> extract(Document doc);
    }

    public static final class SimplePromptExtractor implements TripleExtractor {
        @Override
        public List<Triple> extract(Document doc) {
            // Use a deterministic politeness heuristic: split sentences, find " X is Y " patterns.
            List<Triple> out = new ArrayList<>();
            for (String sentence : doc.pageContent.split("\\.")) {
                String s = sentence.trim();
                if (s.isEmpty()) continue;
                int ix = s.indexOf(" is ");
                if (ix > 0 && ix + 5 < s.length()) {
                    String sub = s.substring(0, ix).trim();
                    String obj = s.substring(ix + 4).trim();
                    if (!sub.isEmpty() && !obj.isEmpty()) out.add(new Triple(sub, "is", obj));
                }
            }
            return out;
        }
    }

    private final TripleExtractor extractor;

    public LLMGraphTransformer() { this(new SimplePromptExtractor()); }
    public LLMGraphTransformer(TripleExtractor extractor) { this.extractor = extractor; }

    public List<Triple> convertToGraphDocuments(List<Document> docs) {
        List<Triple> all = new ArrayList<>();
        for (Document d : docs) all.addAll(extractor.extract(d));
        return all;
    }
}