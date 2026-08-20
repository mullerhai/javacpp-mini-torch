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
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.ModulesToSaveWrapper}.
 *
 * <p>Wraps a base module that should be fully trained (not just adapter-fine-tuned)
 * alongside PEFT. Common use: classifier head.
 */
public class ModulesToSaveWrapper extends Module {

    private final Module original;
    private final Module modulesToSave;

    public ModulesToSaveWrapper(Module original, Module modulesToSave) {
        super("ModulesToSaveWrapper");
        this.original = original;
        this.modulesToSave = modulesToSave;
        register_module("original_module", original);
        register_module("modules_to_save", modulesToSave);
    }

    public Module originalModule() { return original; }
    public Module modulesToSave() { return modulesToSave; }

    @Override public Tensor forward(Tensor x) {
        return modulesToSave.forward(x);
    }

    @Override public org.bytedeco.pytorch.TensorVector parameters() {
        return modulesToSave.parameters();
    }
}
