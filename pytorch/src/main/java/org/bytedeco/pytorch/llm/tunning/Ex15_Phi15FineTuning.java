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
//import org.bytedeco.pytorch.llm.rag.Document;
//import org.bytedeco.pytorch.llm.rag.MemoryVectorStore;
//import org.bytedeco.pytorch.llm.rag.RAGPipeline;
//import org.bytedeco.pytorch.llm.rag.RecurrentTextSplitter;
//import org.bytedeco.pytorch.llm.rag.RecursiveSplitter;
//import org.bytedeco.pytorch.llm.rag.SentenceTransformer;
//import org.bytedeco.pytorch.llm.rag.WebBaseLoader;
//import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
//
///** Ex15 — Phi-1.5 SFT with custom data preparation. */
//public final class Ex15_Phi15FineTuning {
//
//    public static final String NAME = "Ex15_Phi15FineTuning";
//
//    public static void run(FastTokenizer tokenizer) {
//        TunningSupport.banner(15, "Phi-1.5 SFT");
//
//        RecurrentTextSplitter.LengthFunction len = t -> t == null ? 0 : t.length();
//        RecursiveSplitter splitter = new RecursiveSplitter(512, 64, len);
//
//        try {
//            List<Document> docs = new WebBaseLoader(List.of("https://example.com")).load();
//            List<Document> chunks = new ArrayList<>();
//            for (Document d : docs) chunks.addAll(Document.split(d, splitter));
//            MemoryVectorStore store = new MemoryVectorStore(new SentenceTransformer("all-MiniLM-L6-v2"));
//            store.addDocuments(chunks);
//            RAGPipeline.LLMGenerator gen = prompt -> "[stub] " + prompt.substring(0, Math.min(40, prompt.length()));
//            RAGPipeline pipeline = new RAGPipeline(store, gen, 4, 0.5);
//            String answer = pipeline.query("Hello world");
//            System.out.println("RAG answer: " + answer);
//        } catch (Exception e) {
//            System.err.println("RAG error: " + e.getMessage());
//        }
//    }
//
//    public static void main(String[] args) {
//        try (FastTokenizer t = TunningSupport.tokenizerFor("microsoft/phi-1_5")) {
//            run(t);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//    }
//}
