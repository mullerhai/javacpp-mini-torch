/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners;

import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftMethodRegistry;

import java.util.EnumMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.TunerLayerRegistry}.
 *
 * <p>Allows a single {@link BaseTunerLayer} implementation to be shared by multiple
 * tuners (e.g. LoraLinear is used by {@code LORA}, {@code ADALORA}, {@code LOHA}, etc.).
 */
public final class TunerLayerRegistry {

    private static final Map<PeftType, Class<? extends BaseTuner>> TUNER_CLASSES = new EnumMap<>(PeftType.class);
    private static final Map<PeftType, Class<? extends BaseTunerLayer>> LAYER_CLASSES = new EnumMap<>(PeftType.class);

    private TunerLayerRegistry() {}

    public static void registerTuner(PeftType type, Class<? extends BaseTuner> tuner) {
        TUNER_CLASSES.put(type, tuner);
    }
    public static void registerLayer(PeftType type, Class<? extends BaseTunerLayer> layer) {
        LAYER_CLASSES.put(type, layer);
    }
    public static Class<? extends BaseTuner> tunerFor(PeftType type) {
        Class<? extends BaseTuner> c = TUNER_CLASSES.get(type);
        if (c == null) {
            // tunerClassFor method not available in PeftMethodRegistry
            // Return null to indicate not found
            return null;
        }
        return c;
    }
    public static Class<? extends BaseTunerLayer> layerFor(PeftType type) {
        return LAYER_CLASSES.get(type);
    }
    public static Map<PeftType, Class<? extends BaseTuner>> tuners() {
        return java.util.Collections.unmodifiableMap(TUNER_CLASSES);
    }
}