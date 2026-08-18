/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.List;

//import org.bytedeco.pytorch.llm.hardware.HardwareSupport;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.nn.Module;

/** Ex25 — LoRA PEFT tuning. */
public final class Ex25_LoRAPeftTuning {

    public static final String NAME = "Ex25_LoRAPeftTuning";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(25, "LoRA PEFT Tuning");

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
        Module model = new Module("opt-1.3b");

        double[] losses = TunningSupport.simulateTrainingLoop(1, 0.93);
        System.out.println("Model: facebook/opt-1.3b");
        System.out.println("PEFT: LoRA");
//        System.out.println("Precision: " + HardwareSupport.pickPrecision(true));
        System.out.println("Initial loss: " + losses[0]);
        System.out.println("Final loss: " + losses[losses.length - 1]);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("opt-1.3b")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
