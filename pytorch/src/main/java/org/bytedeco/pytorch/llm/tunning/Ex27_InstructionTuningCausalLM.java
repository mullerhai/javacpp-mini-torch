/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex27 — Instruction tuning of causal LM. */
public final class Ex27_InstructionTuningCausalLM {

    public static final String NAME = "Ex27_InstructionTuningCausalLM";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(27, "Instruction Tuning Causal LM");

        List<Map<String, Object>> raw = TunningSupport.alpacaSample(64);
        List<Map<String, Object>> formatted = raw.stream().map(row -> {
            Map<String, Object> r = new LinkedHashMap<>(row);
            r.put("text", TunningSupport.alpacaPrompt(row));
            return r;
        }).toList();

        System.out.println("Model: gpt2-medium");
        System.out.println("Epoch size: " + raw.size());

        double[] losses = TunningSupport.simulateTrainingLoop(1, 0.95);
        System.out.println("Initial loss: " + losses[0]);
        System.out.println("Final loss: " + losses[losses.length - 1]);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("gpt2-medium")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
