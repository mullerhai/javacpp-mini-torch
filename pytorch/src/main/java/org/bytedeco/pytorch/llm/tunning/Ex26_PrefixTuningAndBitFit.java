/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;

import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.nn.Module;

/** Ex26 — Prefix Tuning + BitFit. */
public final class Ex26_PrefixTuningAndBitFit {

    public static final String NAME = "Ex26_PrefixTuningAndBitFit";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(26, "Prefix Tuning + BitFit");

        List<Map<String, Object>> raw = TunningSupport.alpacaSample(64);
        List<Map<String, Object>> formatted = raw.stream().map(row -> {
            Map<String, Object> r = new LinkedHashMap<>(row);
            r.put("text", TunningSupport.alpacaPrompt(row));
            return r;
        }).toList();

        Module model = new Module("bitfit-base");
        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .build();

        double[] losses = TunningSupport.simulateTrainingLoop(1, 0.93);
        System.out.println("Method: BitFit");
        System.out.println("Trained params: bias + LayerNorm");
        System.out.println("Initial loss: " + losses[0]);
        System.out.println("Final loss: " + losses[losses.length - 1]);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("bitfit-base")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
