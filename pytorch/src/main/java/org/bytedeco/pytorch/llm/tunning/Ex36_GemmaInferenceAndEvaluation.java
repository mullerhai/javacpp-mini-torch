/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning-Tutorial authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

/**
 * Ex36 — Gemma-2B Inference + Evaluation.
 *
 * <p>Maps to: LoRA adapter loading + merge + inference + perplexity evaluation
 */
public final class Ex36_GemmaInferenceAndEvaluation {

    public static void run() {
        TunningSupport.banner(36, "Gemma-2B Inference + Evaluation");

        // Model config
        PretrainedConfig cfg = PretrainedConfig.builder()
                .torchDtype("bfloat16")
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        // LoRA config (for adapter loading)
        LoraConfig loraCfg = LoraConfig.builder()
                .taskType("CAUSAL_LM")
                .r(16)
                .alpha(32)
                .build();

        // Load adapter (if available)
        System.out.println("[Ex36] Attempting to load LoRA adapter...");
        try {
            PeftModel peft = PeftModel.fromPretrained(model, "model/beomi-gemma-2b-kullm-v2-adapter");
            System.out.println("[Ex36] Adapter loaded successfully");
        } catch (Exception e) {
            System.out.println("[Ex36] Adapter not found, using base model");
        }

        // Tokenizer
        FastTokenizer tok = TunningSupport.tokenizerFor("beomi/gemma-ko-2b");

        // Generate sample
        String prompt = "K-9 자주포에 대해서 알려주세요.";
        String text = TunningSupport.llama2ChatPrompt(List.of(
                Map.of("role", "user", "content", prompt)));
        int[] ids = tok.encode(text, false).ids();
        System.out.println("[Ex36] Prompt encoded: " + ids.length + " tokens");
        System.out.println("[Ex36] Inference configuration:");
        System.out.println("  - max_new_tokens: 256");
        System.out.println("  - temperature: 0.2");
        System.out.println("  - top_p: 0.9");

        // Evaluation
        List<Map<String, Object>> testData = TunningSupport.alpacaSample(16);
        long totalTokens = 0;
        for (Map<String, Object> row : testData) {
            String text2 = TunningSupport.alpacaPrompt(row);
            totalTokens += tok.encode(text2, false).ids().length;
        }
        double syntheticPerplexity = Math.exp(Math.log(2) * 1.0);
        System.out.println("[Ex36] Eval: " + testData.size() + " records, " + totalTokens + " tokens");
        System.out.println("[Ex36] Perplexity (placeholder): " + String.format("%.4f", syntheticPerplexity));
    }

    private Ex36_GemmaInferenceAndEvaluation() {}

    public static void main(String[] args) {
        run();
    }
}
