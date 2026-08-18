///*
// * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
// *
// * Apache License 2.0.
// */
//package org.bytedeco.pytorch.llm.tunning;
//
//import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
//import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
//import org.bytedeco.pytorch.llm.unsloth.FastLanguageModel;
//
///** Ex33 — Unsloth LLaMA-3 4-bit fine-tune. */
//public final class Ex33_UnslothLLaMA3FineTune {
//
//    public static final String NAME = "Ex33_UnslothLLaMA3FineTune";
//
//    public static void run(FastTokenizer tokenizer) {
//        TunningSupport.banner(33, "Unsloth LLaMA-3 QLoRA");
//
//        FastLanguageModel.FastConfig cfg = FastLanguageModel.FastConfig.builder()
//                .maxSeqLength(2048)
//                .loadIn4bit(true)
//                .dtype("bf16")
//                .fullFinetuning(false)
//                .useGradientCheckpointing("unsloth")
//                .trustRemoteCode(true)
//                .build();
//
//        System.out.println("Model: llama-3-8b-instruct-bnb-4bit");
//        System.out.println("Quantization: nf4");
//        System.out.println("Gradient checkpointing: unsloth");
//        System.out.println("FastConfig: " + cfg);
//    }
//
//    public static void main(String[] args) {
//        try (FastTokenizer t = TunningSupport.tokenizerFor("llama-3-8b")) {
//            run(t);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//    }
//}
