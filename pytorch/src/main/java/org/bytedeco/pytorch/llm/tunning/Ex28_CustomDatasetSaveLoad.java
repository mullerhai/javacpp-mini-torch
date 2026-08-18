/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex28 — Custom dataset save/load. */
public final class Ex28_CustomDatasetSaveLoad {

    public static final String NAME = "Ex28_CustomDatasetSaveLoad";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(28, "Custom Dataset Save/Load");

        List<Map<String, Object>> raw = TunningSupport.alpacaSample(48);
        String jsonl = TunningSupport.writeTmpJsonl(raw);
        System.out.println("Wrote " + raw.size() + " rows to " + jsonl);
        System.out.println("Dataset operations completed.");
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("custom-ds")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
