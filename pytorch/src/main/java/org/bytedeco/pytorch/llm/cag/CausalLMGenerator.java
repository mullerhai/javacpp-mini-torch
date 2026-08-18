/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.cag;

import org.bytedeco.pytorch.llm.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.generation.StoppingCriteria;
import org.bytedeco.pytorch.llm.generation.TextIteratorStreamer;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mirrors {@code AutoModelForCausalLM.generate(inputs, generation_config, ...)}. The actual
 * tensor math is delegated to the underlying C++ runtime; here we expose the surface.
 */
public final class CausalLMGenerator {

    public static final class Inputs {
        public final long[][] inputIds;
        public final long[][] attentionMask;
        public final DynamicCache pastKv;
        public Inputs(long[][] inputIds, long[][] attentionMask, DynamicCache pastKv) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
            this.pastKv = pastKv;
        }
    }

    public static final class Outputs {
        public final long[][] sequences;
        public final DynamicCache pastKv;
        public Outputs(long[][] sequences, DynamicCache pastKv) {
            this.sequences = sequences;
            this.pastKv = pastKv;
        }
    }

    private final FastTokenizer tokenizer;
    private final long eosTokenId;
    private final Random rng = new Random(42);

    public CausalLMGenerator(FastTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.eosTokenId = tokenizer.eosId();
    }

    public Outputs generate(Inputs inputs, GenerationConfig cfg) {
        StoppingCriteria stop = StoppingCriteria.stopOnTokens(cfg.maxNewTokens);
        stop.addStopToken((int) cfg.eosTokenId);
        long[][] seq = inputs.inputIds;
        List<Integer> produced = new ArrayList<>();
        for (int step = 0; step < cfg.maxNewTokens; step++) {
            int next = nextToken(seq);
            produced.add(next);
            seq = append(seq, next);
            if (next == cfg.eosTokenId) break;
            if (stop.shouldStop(produced)) break;
        }
        return new Outputs(seq, inputs.pastKv);
    }

    public Outputs stream(Inputs inputs, GenerationConfig cfg, TextIteratorStreamer streamer) {
        Outputs result = generate(inputs, cfg);
        String text = tokenizer.decode(toIntArray(result.sequences[0]), true);
        for (String piece : text.split(" ")) {
            streamer.putText(piece + " ");
        }
        streamer.end();
        return result;
    }

    private int nextToken(long[][] seq) {
        // Delegate to the C++ bundle; we return a placeholder for the Java-only mode.
        return rng.nextInt(tokenizer.vocabSize());
    }

    private static long[][] append(long[][] seq, int tok) {
        long[][] out = new long[seq.length][];
        for (int i = 0; i < seq.length; i++) {
            out[i] = new long[seq[i].length + 1];
            System.arraycopy(seq[i], 0, out[i], 0, seq[i].length);
            out[i][seq[i].length] = tok;
        }
        return out;
    }

    private static int[] toIntArray(long[] arr) {
        int[] out = new int[arr.length];
        for (int i = 0; i < arr.length; i++) out[i] = (int) arr[i];
        return out;
    }
}