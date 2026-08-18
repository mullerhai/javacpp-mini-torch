/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex29 — QLoRA pipeline for beginners. */
public final class Ex29_QLoRAPipelineBeginner {

    public static final String NAME = "Ex29_QLoRAPipelineBeginner";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(29, "QLoRA Pipeline Beginner");

        List<Map<String, Object>> raw = TunningSupport.alpacaSample(64);
        List<Map<String, Object>> formatted = raw.stream().map(row -> {
            Map<String, Object> r = new LinkedHashMap<>(row);
            r.put("text", TunningSupport.alpacaPrompt(row));
            return r;
        }).toList();

        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .build();
        System.out.println("Model: mistral-7b");
        System.out.println("Quantization: " + bnb);
//        System.out.println("Precision: " + HardwareSupport.pickPrecision(true));

        double[] losses = TunningSupport.simulateTrainingLoop(1, 0.93);
        System.out.println("Initial loss: " + losses[0]);
        System.out.println("Final loss: " + losses[losses.length - 1]);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("mistral-7b")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
