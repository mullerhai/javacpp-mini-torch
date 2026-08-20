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

import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.tuners.prompt.PromptEncoder}.
 *
 * <p>Used by P-Tuning: encodes a learnable soft prompt (continuous embedding) into
 * the per-layer prompts via an MLP. Forward produces {@code [batch_size, num_layers,
 * num_virtual_tokens * token_dim]}.
 */
public class PromptEncoder extends Module {

    private final long numVirtualTokens;
    private final long tokenDim;
    private final long numLayers;
    private final long numAttentionHeads;

    public PromptEncoder(long numVirtualTokens, long tokenDim, long numLayers, long numAttentionHeads) {
        super("PromptEncoder");
        this.numVirtualTokens = numVirtualTokens;
        this.tokenDim = tokenDim;
        this.numLayers = numLayers;
        this.numAttentionHeads = numAttentionHeads;
    }

    /** Embedding parameter (lazy-initialised in subclass). */
    public Tensor embedding() { return new Tensor(); }

    /** Forward: returns [batch, num_layers * num_virtual_tokens * 2 * token_dim] for P-Tuning v2. */
    public Tensor forward(long batchSize) {
        // Placeholder: production builds the multi-layer prompt with per-position reparameterisation.
        return embedding().unsqueeze(0).expand(batchSize, -1, -1);
    }

    public List<Tensor> namedParameters() { return java.util.Collections.emptyList(); }
}
