///*
// * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
// *
// * Apache License 2.0.
// */
//package org.bytedeco.pytorch.llm.tunning;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Map;
//
//import org.bytedeco.pytorch.llm.knowledgegraph.GraphCypherQAChain;
//import org.bytedeco.pytorch.llm.knowledgegraph.LLMGraphTransformer;
//import org.bytedeco.pytorch.llm.knowledgegraph.Neo4jClient;
//import org.bytedeco.pytorch.llm.knowledgegraph.Triple;
//import org.bytedeco.pytorch.llm.knowledgegraph.WikipediaEntityRecognizer;
//import org.bytedeco.pytorch.llm.rag.Document;
//import org.bytedeco.pytorch.llm.rag.RAGPipeline;
//import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
//
///** Ex17 — Wikipedia entity linking + Neo4j. */
//public final class Ex17_WikipediaEntityLink {
//
//    public static final String NAME = "Ex17_WikipediaEntityLink";
//
//    public static void run(FastTokenizer tokenizer) {
//        TunningSupport.banner(17, "Wikipedia Entity Linking + Neo4j");
//
//        Document d = new Document("Albert Einstein is a physicist. Isaac Newton is a mathematician. Albert Einstein is German.", new java.util.LinkedHashMap<>());
//        List<String> entities = new WikipediaEntityRecognizer().extract(d.text);
//        System.out.println("Detected entities: " + entities);
//
//        Neo4jClient neo = new Neo4jClient("bolt://localhost:7687", "neo4j", "pw");
//        neo.addTriples(new ArrayList<>());
//        List<Triple> triples = new LLMGraphTransformer().convertToGraphDocuments(List.of(d));
//        neo.addTriples(triples);
//
//        RAGPipeline.LLMGenerator gen = prompt -> "MATCH (n)-[r]->(m) RETURN n, r, m";
//        GraphCypherQAChain chain = GraphCypherQAChain.fromLlm(gen, neo, true);
//        String answer = chain.run("Who is a physicist?");
//        System.out.println("Answer: " + answer);
//    }
//
//    public static void main(String[] args) {
//        try (FastTokenizer t = TunningSupport.tokenizerFor("WikipediaEntityLink")) {
//            run(t);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//    }
//}
