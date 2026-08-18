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

/** Ex31 — ChatGLM LoRA tuning. */
public final class Ex31_ChatGMLoRATuning {

    public static final String NAME = "Ex31_ChatGMLoRATuning";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(31, "ChatGLM LoRA Tuning");

        List<Map<String, Object>> raw = TunningSupport.chatmlSample(64);
        List<Map<String, Object>> formatted = raw.stream().map(row -> {
            Map<String, Object> r = new LinkedHashMap<>(row);
            String text = TunningSupport.chatglmPrompt((List<Map<String, String>>) row.get("messages"));
            r.put("text", text);
            return r;
        }).toList();

        System.out.println("Model: chatglm-6b");
        System.out.println("Template: ChatGLM");

        double[] losses = TunningSupport.simulateTrainingLoop(1, 0.95);
        System.out.println("Initial loss: " + losses[0]);
        System.out.println("Final loss: " + losses[losses.length - 1]);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("chatglm-6b")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
