/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;
import java.util.Map;

import org.bytedeco.pytorch.llm.llamafactory.LlamaFactoryConfig;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex21 — LlamaFactory CLI orchestration. */
public final class Ex21_LlamaFactoryConfig {

    public static final String NAME = "Ex21_LlamaFactoryConfig";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(21, "LlamaFactory Config");

        Map<String, Object> cfg = LlamaFactoryConfig.builder()
                .llamaPath("meta-llama/Llama-2-7b-hf")
                .stage("sft")
                .dataset("alpaca_zh")
                .template("llama2")
                .outputDir("build/llama-factory")
                .loraRank(8)
                .loraAlpha(16.0)
                .loraDropout(0.05)
                .batchSize(2)
                .gradAccum(8)
                .epochs(3)
                .lr(2e-4)
                .scheduler("cosine")
                .warmupSteps(50)
                .weightDecay(0.0)
                .optimizer("adamw")
                .fp16(false)
                .bf16(true)
                .bnbConfig(BitsAndBytesConfig.builder().loadIn4Bit(true).build())
                .gradientCheckpointing(true)
                .flashAttn("fa2")
                .seed(42)
                .build();
        System.out.println("Config:\n" + LlamaFactoryConfig.serializeYaml(cfg));
        boolean ok = LlamaFactoryConfig.run(cfg);
        System.out.println("Run dispatch: " + ok);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("Llama-2-7b")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
