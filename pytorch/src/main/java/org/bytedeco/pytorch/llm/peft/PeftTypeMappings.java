/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft;

import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;

import java.util.EnumMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.peft_types.PEFT_TYPE_TO_*_MAPPING}.
 *
 * <p>Holds the EnumMap-based dispatch from {@link PeftType} to its associated
 * {@link PeftConfig} / {@link BaseTuner} implementation. Populated lazily via
 * the per-tuner Registry classes (e.g. {@code LoraRegistry}, {@code IA3Registry}).
 */
public final class PeftTypeMappings {

    private PeftTypeMappings() {}

    private static final Map<PeftType, Class<? extends PeftConfig>> CONFIG = new EnumMap<>(PeftType.class);
    private static final Map<PeftType, Class<? extends BaseTuner>>   TUNER  = new EnumMap<>(PeftType.class);
    private static final Map<PeftType, String>                       PREFIX = new EnumMap<>(PeftType.class);
    private static final Map<PeftType, Boolean>                      MIXED  = new EnumMap<>(PeftType.class);

    public static synchronized void put(PeftType t, Class<? extends PeftConfig> cfgKlass,
                                         Class<? extends BaseTuner> tunerKlass, String prefix, boolean mixed) {
        CONFIG.put(t, cfgKlass);
        TUNER.put(t, tunerKlass);
        PREFIX.put(t, prefix);
        MIXED.put(t, mixed);
    }

    public static Class<? extends PeftConfig> config(PeftType t) { return CONFIG.get(t); }
    public static Class<? extends BaseTuner>   tuner(PeftType t)  { return TUNER.get(t); }
    public static String                       prefix(PeftType t) { return PREFIX.getOrDefault(t, t.prefix()); }
    public static boolean                      mixedCompatible(PeftType t) { return MIXED.getOrDefault(t, false); }
}
