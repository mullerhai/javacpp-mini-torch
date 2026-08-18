/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;

import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.quantization.HQQQuantizer;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex20 — Llama-2 7B HQQ 1-bit quantization. */
public final class Ex20_LlamaHQQ1bit {

    public static final String NAME = "Ex20_LlamaHQQ1bit";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(20, "HQQ 1-bit Quantization");

        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder().loadIn8Bit(true).build();
        // Simulate a flatten of weights.
        float[] weights = new float[256];
        for (int i = 0; i < weights.length; i++) weights[i] = ((float) Math.sin(i * 0.07)) * 0.5f;

        HQQQuantizer.Quantum q = HQQQuantizer.quantize1Bit(weights, 64);
        float[] dequant = HQQQuantizer.dequantize(q);
        long sizeBytes = q.zeros.length * 4 + q.scales.length * 4 + q.codes.length * 8;
        System.out.println("HQQ quant: bytes=" + sizeBytes + " original=" + (weights.length * 4) + " ratio=" +
                (sizeBytes * 1.0 / (weights.length * 4)));
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("Llama-2-7b")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
