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

/**
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.BaseTunerLayer} plus the
 * AuxiliaryTrainingWrapper concept: a Module that holds additional trainable buffers
 * (e.g. activation tokens for XLora) alongside the standard adapter logic.
 */
public class AuxiliaryTrainingWrapper extends Module {

    private final Module base;
    private final Module auxiliary;

    public AuxiliaryTrainingWrapper(Module base, Module auxiliary) {
        super("AuxiliaryTrainingWrapper");
        this.base = base;
        this.auxiliary = auxiliary;
        register_module("base", base);
        register_module("auxiliary", auxiliary);
    }

    @Override public Tensor forward(Tensor x) {
        Tensor main = base.forward(x);
        // Auxiliary tensors are trained via their own forward; the main forward stays unchanged.
        return main;
    }

    public Module auxiliary() { return auxiliary; }
}
