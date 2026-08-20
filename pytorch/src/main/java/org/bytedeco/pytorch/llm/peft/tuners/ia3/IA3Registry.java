/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.ia3;

import org.bytedeco.pytorch.llm.peft.IA3Config;
import org.bytedeco.pytorch.llm.peft.PeftMethodRegistry;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.TunerLayerRegistry;

/** Static initialiser that wires IA3 into the global registry. */
public final class IA3Registry {
    private IA3Registry() {}

    static {
        PeftMethodRegistry.instance().register(
                PeftType.IA3,
                IA3Config.class,
                IA3Model.class,
                "ia3_",
                false);
        TunerLayerRegistry.registerLayer(PeftType.IA3, IA3Linear.class);
    }
}