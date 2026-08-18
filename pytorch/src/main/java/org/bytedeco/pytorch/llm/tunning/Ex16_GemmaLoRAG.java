/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;

import org.bytedeco.pytorch.llm.knowledgegraph.GraphCypherQAChain;
import org.bytedeco.pytorch.llm.knowledgegraph.LLMGraphTransformer;
import org.bytedeco.pytorch.llm.knowledgegraph.Neo4jClient;
import org.bytedeco.pytorch.llm.knowledgegraph.Triple;
import org.bytedeco.pytorch.llm.rag.Document;
import org.bytedeco.pytorch.llm.rag.RAGPipeline;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex16 — Gemma-2B LoRA + Neo4j knowledge graph. */
public final class Ex16_GemmaLoRAG {

    public static final String NAME = "Ex16_GemmaLoRAG";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(16, "Gemma LoRA + Neo4j Knowledge Graph");

        Neo4jClient neo = new Neo4jClient("bolt://localhost:7687", "neo4j", "pw");
        Document d = new Document("ChatGPT is a chatbot. GPT-4 is a multimodal Model.", new LinkedHashMap<>());
        List<Triple> triples = new LLMGraphTransformer().convertToGraphDocuments(List.of(d));
        neo.addTriples(triples);
        System.out.println("Triples: " + triples.size());

        RAGPipeline.LLMGenerator gen = prompt -> "MATCH (n) RETURN n";
        GraphCypherQAChain chain = GraphCypherQAChain.fromLlm(gen, neo, true);
        String answer = chain.run("What is ChatGPT?");
        System.out.println("Answer: " + answer);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("gemma-2b")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
