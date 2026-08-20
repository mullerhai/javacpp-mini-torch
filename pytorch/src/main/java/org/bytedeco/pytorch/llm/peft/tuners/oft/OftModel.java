/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.oft;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.oft.OftModel} / {@code BoftModel}.
 */
public class OftModel extends BaseTuner {

    public OftModel(Module model, Map<String, PeftConfig> configs, String adapterName) {
        super(model, configs, adapterName, "oft_");
    }

    @Override
    protected BaseTunerLayer _createNewLayer(PeftConfig cfg, String adapterName,
                                              String targetName, Module target) {
        return null;
    }

    @Override public PeftType peftType() { return PeftType.OFT; }
}
