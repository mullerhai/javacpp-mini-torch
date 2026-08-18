/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex30 — Instruction data tokenization. */
public final class Ex30_InstructionDataTokenization {

    public static final String NAME = "Ex30_InstructionDataTokenization";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(30, "Instruction Data Tokenization");

        List<Map<String, Object>> raw = TunningSupport.alpacaSample(64);
        Map<String, Object> sample = raw.get(0);
        String text = TunningSupport.alpacaPrompt(sample);
        int[] ids = tokenizer.encode(text, false).ids();
        System.out.println("Tokenized sample: input_ids.length=" + ids.length);
        System.out.println("Dataset rows: " + raw.size());
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("instruct-ds")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
