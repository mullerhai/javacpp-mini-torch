/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.prompt;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;

import static org.bytedeco.pytorch.global.torch.randn;

/**
 * Java analog of HuggingFace {@code peft.tuners.prompt.PromptEmbedding}.
 *
 * <p>Used by Prompt Tuning / CPT / Multitask Prompt Tuning: a learnable
 * {@code [num_virtual_tokens, token_dim]} embedding. Forward: repeat for each
 * batch element.
 */
public class PromptEmbedding extends Module {

    private final EmbeddingImpl embedding;
    private final long numVirtualTokens;

    public PromptEmbedding(long numVirtualTokens, long tokenDim) {
        super("PromptEmbedding");
        this.numVirtualTokens = numVirtualTokens;
        this.embedding = new EmbeddingImpl(numVirtualTokens, tokenDim);
        // Default init: randn (HF uses same)
        Tensor w = randn(new long[]{numVirtualTokens, tokenDim});
        embedding.weight(w);
        register_module("embedding", embedding);
    }

    public Tensor forward(long batchSize) {
        Tensor e = embedding.weight();        // [num_virtual_tokens, token_dim]
        return e.unsqueeze(0).expand(new long[]{batchSize, numVirtualTokens, e.size(1)});
    }

    public EmbeddingImpl embedding() { return embedding; }
}
