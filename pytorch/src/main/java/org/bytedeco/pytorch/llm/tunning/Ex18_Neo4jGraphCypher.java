/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.knowledgegraph.GraphCypherQAChain;
import org.bytedeco.pytorch.llm.knowledgegraph.Neo4jClient;
import org.bytedeco.pytorch.llm.knowledgegraph.LLMGraphTransformer;
import org.bytedeco.pytorch.llm.knowledgegraph.Triple;
import org.bytedeco.pytorch.llm.rag.Document;
import org.bytedeco.pytorch.llm.rag.RAGPipeline;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex18 — Neo4j GraphCypherQAChain demo. */
public final class Ex18_Neo4jGraphCypher {

    public static final String NAME = "Ex18_Neo4jGraphCypher";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(18, "Neo4j GraphCypherQAChain");

        Neo4jClient neo = new Neo4jClient("bolt://localhost:7687", "neo4j", "pw");
        Document d = new Document("France is a country. Paris is in France. Eiffel Tower is in Paris.", new java.util.LinkedHashMap<>());
        List<Triple> triples = new LLMGraphTransformer().convertToGraphDocuments(List.of(d));
        neo.addTriples(triples);
        System.out.println("Seeded triples: " + triples.size());

        RAGPipeline.LLMGenerator gen = prompt -> prompt.contains("Paris") ? "MATCH (n)-[r]->(m) WHERE n.country='Paris' RETURN m" : "MATCH (n) RETURN n";
        GraphCypherQAChain chain = GraphCypherQAChain.fromLlm(gen, neo, true);
        String answer = chain.run("What's the capital of France?");
        System.out.println("Q: What's the capital of France?");
        System.out.println("A: " + answer);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("Neo4jGraphCypher")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
