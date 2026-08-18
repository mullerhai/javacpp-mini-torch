/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.knowledgegraph;

import org.bytedeco.pytorch.llm.rag.RAGPipeline;

/**
 * {@code GraphCypherQAChain} analog. Turns a natural-language question into a Cypher query
 * (via the LLM) and runs it against a {@link Neo4jClient} (or {@link KnowledgeGraph}).
 */
public final class GraphCypherQAChain {

    private final RAGPipeline.LLMGenerator generator;
    private final Neo4jClient graph;
    private final boolean verbose;

    public GraphCypherQAChain(RAGPipeline.LLMGenerator generator, Neo4jClient graph, boolean verbose) {
        this.generator = generator;
        this.graph = graph;
        this.verbose = verbose;
    }

    public static GraphCypherQAChain fromLlm(RAGPipeline.LLMGenerator llm, Neo4jClient graph) {
        return new GraphCypherQAChain(llm, graph, false);
    }

    public static GraphCypherQAChain fromLlm(RAGPipeline.LLMGenerator llm, Neo4jClient graph, boolean verbose) {
        return new GraphCypherQAChain(llm, graph, verbose);
    }

    public String run(String question) {
        String cypher = generator.generate(
                "Translate the question to a Cypher query.\nQuestion: " + question + "\nCypher:");
        String result = graph.graph().executeCypher(cypher);
        String answer = generator.generate(
                "Use the graph query result to answer the user's question.\n\nQuestion: " + question +
                "\n\nQuery result:\n" + result + "\nAnswer:");
        if (verbose) {
            System.out.println("[GraphCypherQAChain] generated cypher: " + cypher);
        }
        return answer;
    }
}