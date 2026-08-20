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

import static org.bytedeco.pytorch.global.torch.randn;

/**
 * Java analog of HuggingFace {@code peft.tuners.prefix_tuning.PrefixEncoder}.
 *
 * <p>Prefix Tuning reparameterises the past_key_values with a learnable prefix
 * per layer. Forward produces {@code [batch, num_layers, 2, num_attention_heads,
 * num_virtual_tokens, head_dim]}.
 */
public class PrefixEncoder extends Module {

    private final long numVirtualTokens;
    private final long tokenDim;
    private final long numLayers;
    private final long numAttentionHeads;
    private final long prefixProjectionHidden;

    /** The learnable prefix tensor (num_layers * 2 * num_virtual_tokens * token_dim). */
    private Tensor prefixProjected;

    public PrefixEncoder(long numVirtualTokens, long numLayers, long numAttentionHeads,
                         long tokenDim, long prefixProjectionHidden) {
        super("PrefixEncoder");
        this.numVirtualTokens = numVirtualTokens;
        this.tokenDim = tokenDim;
        this.numLayers = numLayers;
        this.numAttentionHeads = numAttentionHeads;
        this.prefixProjectionHidden = prefixProjectionHidden;
        // Allocate the learnable parameters.
        Tensor w = randn(new long[]{numVirtualTokens, prefixProjectionHidden > 0 ? prefixProjectionHidden : tokenDim});
        Tensor mlpOut = randn(new long[]{numVirtualTokens, numLayers, 2, numAttentionHeads, tokenDim / numAttentionHeads});
        register_module("prefix_proj_w", new Module("W") {
            @Override public Tensor forward(Tensor x) { return w; }
            @Override public org.bytedeco.pytorch.TensorVector parameters() {
                org.bytedeco.pytorch.TensorVector v = new org.bytedeco.pytorch.TensorVector();
                v.push_back(w); return v;
            }
        });
        prefixProjected = mlpOut;
    }

    public Tensor forward(long batchSize) {
        // [num_layers, 2, num_attention_heads, num_virtual_tokens, head_dim]
        Tensor t = prefixProjected.permute(new long[]{1, 2, 3, 0, 4});
        // [batch, num_layers, 2, num_attention_heads, num_virtual_tokens, head_dim]
        return t.unsqueeze(0).expand(batchSize, -1, -1, -1, -1, -1);
    }

    public long getNumVirtualTokens() { return numVirtualTokens; }
    public long getTokenDim() { return tokenDim; }
    public long getNumLayers() { return numLayers; }
    public long getNumAttentionHeads() { return numAttentionHeads; }
}
