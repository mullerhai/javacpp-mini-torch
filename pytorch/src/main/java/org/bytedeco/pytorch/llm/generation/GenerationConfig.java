/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.generation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.IntConsumer;
import java.util.function.DoubleConsumer;

/**
 * Hugging Face {@code GenerationConfig} analog.
 */
public final class GenerationConfig {

    public int maxNewTokens = 256;
    public int minNewTokens = 0;
    public double temperature = 1.0;
    public double topP = 1.0;
    public int topK = 50;
    public double repetitionPenalty = 1.0;
    public double lengthPenalty = 1.0;
    public boolean doSample = false;
    public int numBeams = 1;
    public int numReturnSequences = 1;
    public int earlyStopping = 0;
    public boolean padTokenIdSet = false;
    public int padTokenId = 0;
    public int eosTokenId = 2;
    public int bosTokenId = 1;
    public double noRepeatNgramSize = 0;
    public boolean returnDict = true;

    public static GenerationConfig fromDict(java.util.Map<String, Object> d) {
        GenerationConfig c = new GenerationConfig();
        if (d == null) return c;
        oInt(d, "max_new_tokens", v -> c.maxNewTokens = v);
        oInt(d, "min_new_tokens", v -> c.minNewTokens = v);
        oDbl(d, "temperature", v -> c.temperature = v);
        oDbl(d, "top_p", v -> c.topP = v);
        oInt(d, "top_k", v -> c.topK = v);
        oDbl(d, "repetition_penalty", v -> c.repetitionPenalty = v);
        oDbl(d, "length_penalty", v -> c.lengthPenalty = v);
        oBool(d, "do_sample", v -> c.doSample = (v != 0));
        oInt(d, "num_beams", v -> c.numBeams = v);
        oInt(d, "num_return_sequences", v -> c.numReturnSequences = v);
        oInt(d, "early_stopping", v -> c.earlyStopping = v);
        oInt(d, "pad_token_id", v -> { c.padTokenId = v; c.padTokenIdSet = true; });
        oInt(d, "eos_token_id", v -> c.eosTokenId = v);
        oInt(d, "bos_token_id", v -> c.bosTokenId = v);
        oDbl(d, "no_repeat_ngram_size", v -> c.noRepeatNgramSize = v);
        return c;
    }

    private static void oInt(java.util.Map<String, Object> d, String k, IntConsumer c) {
        Object v = d.get(k);
        if (v instanceof Number) c.accept(((Number) v).intValue());
    }

    private static void oDbl(java.util.Map<String, Object> d, String k, DoubleConsumer c) {
        Object v = d.get(k);
        if (v instanceof Number) c.accept(((Number) v).doubleValue());
    }

    private static void oBool(java.util.Map<String, Object> d, String k, java.util.function.IntConsumer c) {
        Object v = d.get(k);
        if (v instanceof Boolean) c.accept(((Boolean) v) ? 1 : 0);
        else if (v instanceof Number) c.accept(((Number) v).intValue());
    }

    /** Simple temperature + top-p sampling dispatcher. */
    public static int sample(List<Integer> logitsAsList, double temperature, double topP, int topK, long seed) {
        if (temperature <= 0) return argmax(logitsAsList);
        Random r = new Random(seed);
        double max = Double.NEGATIVE_INFINITY;
        for (int v : logitsAsList) if (v > max) max = v;
        double sum = 0;
        double[] probs = new double[logitsAsList.size()];
        for (int i = 0; i < logitsAsList.size(); i++) {
            double p = Math.exp((logitsAsList.get(i) - max) / temperature);
            probs[i] = p;
            sum += p;
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;

        // Sort indices by probability (descending)
        Integer[] order = new Integer[probs.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer idx) -> probs[idx]).reversed());

        if (topK > 0 && topK < order.length) {
            double total = 0;
            for (int i = 0; i < topK; i++) total += probs[order[i]];
            for (int i = 0; i < topK; i++) probs[order[i]] /= total;
            for (int i = topK; i < order.length; i++) probs[order[i]] = 0;
        }
        if (topP < 1.0) {
            double cum = 0;
            for (int i = 0; i < order.length; i++) {
                cum += probs[order[i]];
                if (cum > topP) {
                    for (int j = i + 1; j < order.length; j++) probs[order[j]] = 0;
                    break;
                }
            }
        }
        double pick = r.nextDouble();
        double cursor = 0;
        for (int i = 0; i < probs.length; i++) {
            cursor += probs[i];
            if (pick <= cursor) return i;
        }
        return argmax(logitsAsList);
    }

    public static int argmax(List<Integer> logits) {
        int best = 0; int bestVal = Integer.MIN_VALUE;
        for (int i = 0; i < logits.size(); i++) {
            int v = logits.get(i);
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    public List<String> toStringList() {
        List<String> out = new ArrayList<>();
        out.add("max_new_tokens=" + maxNewTokens);
        out.add("temperature=" + temperature);
        out.add("top_p=" + topP);
        out.add("top_k=" + topK);
        out.add("repetition_penalty=" + repetitionPenalty);
        out.add("do_sample=" + doSample);
        return out;
    }
}
