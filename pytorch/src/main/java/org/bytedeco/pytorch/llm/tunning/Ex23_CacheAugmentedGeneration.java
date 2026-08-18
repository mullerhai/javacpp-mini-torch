/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import org.bytedeco.pytorch.llm.cag.CausalLMGenerator;
import org.bytedeco.pytorch.llm.cag.DynamicCache;
import org.bytedeco.pytorch.llm.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.rag.SentenceTransformer;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex23 — Cache-Augmented Generation (CAG) with DynamicCache. */
public final class Ex23_CacheAugmentedGeneration {

    public static final String NAME = "Ex23_CacheAugmentedGeneration";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(23, "Cache-Augmented Generation");

        SentenceTransformer embedder = new SentenceTransformer("all-MiniLM-L6-v2");
        DynamicCache cache = new DynamicCache();
        cache.update("layer0", new long[][]{{1, 2, 3}}, new long[][]{{4, 5, 6}});

        String ground = "The capital of France is Paris.";
        String reply = "Paris is the capital of France.";
        float[] g = embedder.encode(ground);
        float[] r = embedder.encode(reply);
        double cos = SentenceTransformer.cosineSimilarity(g, r);
        System.out.println("Cosine(ground, reply) = " + cos);

        CausalLMGenerator gen = new CausalLMGenerator(tokenizer);
        CausalLMGenerator.Inputs inputs = new CausalLMGenerator.Inputs(
                new long[][]{{1, 2, 3}}, new long[][]{{1, 1, 1}}, cache);
        GenerationConfig cfg = new GenerationConfig();
        cfg.maxNewTokens = 16;
        CausalLMGenerator.Outputs outputs = gen.generate(inputs, cfg);
        System.out.println("Generated sequence length: " + outputs.sequences[0].length);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("cag")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
