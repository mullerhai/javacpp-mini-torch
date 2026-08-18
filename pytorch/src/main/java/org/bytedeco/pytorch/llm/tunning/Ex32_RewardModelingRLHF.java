/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex32 — Reward modeling for RLHF. */
public final class Ex32_RewardModelingRLHF {

    public static final String NAME = "Ex32_RewardModelingRLHF";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(32, "Reward Modeling RLHF");

        List<Map<String, Object>> raw = TunningSupport.preferenceSample(16);
        System.out.println("Preference samples: " + raw.size());

        // Naive pairwise reward loss probe
        double chosenReward = 1.2;
        double rejectedReward = 0.5;
        double loss = Math.log1p(Math.exp(chosenReward - rejectedReward));
        System.out.println("Pairwise reward loss: " + loss);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("reward-base")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
