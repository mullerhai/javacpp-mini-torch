/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftMethodRegistry;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.TunerLayerRegistry;

/**
 * Static initialiser that registers the LoRA-family PEFT types with the global
 * {@link PeftMethodRegistry} and {@link TunerLayerRegistry}.
 */
public final class LoraRegistry {
    private LoraRegistry() {}

    static {
        // Canonical LORA tuner / config mapping
        PeftMethodRegistry.instance().register(
                PeftType.LORA,
                LoraConfig.class,
                LoraModel.class,
                /*prefix=*/"lora_",
                /*mixed=*/true);
        // LoRA family aliases share the same LoraConfig / LoraModel implementation.
        PeftType[] loraAliases = {
                PeftType.ADALORA, PeftType.ADAMSS, PeftType.DELORA, PeftType.C3A,
                PeftType.DEFT, PeftType.GLORA, PeftType.GRALORA, PeftType.LILY,
                PeftType.MISS, PeftType.OSF, PeftType.PEANUT, PeftType.PSOFT,
                PeftType.RANDLORA, PeftType.SHIRA, PeftType.TINYLORA, PeftType.UNILORA,
                PeftType.VBLORA
        };
        for (PeftType t : loraAliases) {
            if (t == null) continue;
            PeftMethodRegistry.instance().register(t, LoraConfig.class, LoraModel.class, "lora_", t == PeftType.ADALORA || t == PeftType.SHIRA);
        }
        // SHIRA is mixed-compatible (HF COMPATIBLE_TUNER_TYPES).
        // QLORA: BNB-specific config; keep the regular LoraConfig for the runtime layer.
        PeftMethodRegistry.instance().register(PeftType.QLORA, LoraConfig.class, LoraModel.class, "lora_", false);

        // Layer registration
        TunerLayerRegistry.registerLayer(PeftType.LORA, LoraLinear.class);
        for (PeftType t : loraAliases) {
            if (t != null) TunerLayerRegistry.registerLayer(t, LoraLinear.class);
        }
        TunerLayerRegistry.registerLayer(PeftType.QLORA, LoraLinear.class);
    }
}