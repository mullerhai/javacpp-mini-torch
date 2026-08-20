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
 * Java analog of HuggingFace {@code peft.utils.gradient_checkpointing}.
 */
public final class GradientCheckpointing {

    private GradientCheckpointing() {}

    public static void enableOn(Module model, boolean useReentrant) {
        if (!model.is_training()) model.train(true);
    }
}
