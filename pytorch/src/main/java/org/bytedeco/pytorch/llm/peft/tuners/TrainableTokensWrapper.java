/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;

/**
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.TrainableTokensWrapper}.
 */
public class TrainableTokensWrapper extends Module {

    private final EmbeddingImpl base;

    public TrainableTokensWrapper(EmbeddingImpl base, int[] trainableIndices) {
        super("TrainableTokensWrapper");
        this.base = base;
        register_module("base", base);
    }

    @Override public Tensor forward(Tensor input) {
        return base.forward(input);
    }
}
