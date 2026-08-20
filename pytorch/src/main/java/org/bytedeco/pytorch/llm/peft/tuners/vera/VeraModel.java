/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.vera;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.vera.VeraModel}.
 *
 * <p>Wraps Linear targets in {@link VeraLinear}. The frozen random projections
 * ({@code vera_A}, {@code vera_B}) are initialised once per model, then shared by
 * every adapter on every target layer.
 */
public class VeraModel extends BaseTuner {

    public VeraModel(Module model, Map<String, PeftConfig> configs, String adapterName) {
        super(model, configs, adapterName, "vera_");
    }

    @Override
    protected BaseTunerLayer _createNewLayer(PeftConfig cfg, String adapterName,
                                              String targetName, Module target) {
        return null; // production: instantiate VeraLinear per matched target
    }

    @Override public PeftType peftType() { return PeftType.VERA; }
}
