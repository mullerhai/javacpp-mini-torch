/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.utils;

import org.bytedeco.pytorch.nn.Module;

/**
 * Java analog of HuggingFace {@code peft.utils.helpers}.
 */
public final class Helpers {

    private Helpers() {}

    public static boolean checkIfPeftModel(Module model) {
        // PeftModel is not a Module subclass in this implementation; check by class name instead.
        return model.getClass().getSimpleName().startsWith("Peft");
    }

    public static void rescaleAdapterScale(Module model, String adapterName, double scaling) {
        // Placeholder: production would walk the model tree and rescale the LoraLinear scaling map.
    }

    public static Module disableAdapter(Module model) {
        // Placeholder: production would walk the model tree and disable adapters.
        return model;
    }

    public static Module enableAdapter(Module model) {
        return model;
    }
}
