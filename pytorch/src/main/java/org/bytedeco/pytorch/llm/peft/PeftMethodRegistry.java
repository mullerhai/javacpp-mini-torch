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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Static registry mapping {@link PeftType} → its {@link PeftConfig} / {@link BaseTuner} classes.
 *
 * <p>This is the Java analog of the module-level registries used by HuggingFace PEFT
 * ({@code LoraRegistry}, {@code IA3Registry}, etc.). Each tuner package's static
 * initializer calls {@link #instance()#register(PeftType, Class, Class, String, boolean)}
 * to register itself.
 */
public class PeftMethodRegistry {

    private static final PeftMethodRegistry INSTANCE = new PeftMethodRegistry();
    public static PeftMethodRegistry instance() { return INSTANCE; }

    private final Map<PeftType, Entry> registry = new LinkedHashMap<>();

    private static class Entry {
        Class<? extends PeftConfig> configClass;
        Class<? extends BaseTuner>   tunerClass;
        String prefix;
        boolean mixed;
        Entry(Class<? extends PeftConfig> c, Class<? extends BaseTuner> t, String p, boolean m) {
            this.configClass = c; this.tunerClass = t; this.prefix = p; this.mixed = m;
        }
    }

    public synchronized void register(PeftType type,
                                       Class<? extends PeftConfig> configClass,
                                       Class<? extends BaseTuner> tunerClass,
                                       String prefix, boolean mixed) {
        registry.put(type, new Entry(configClass, tunerClass, prefix, mixed));
        PeftTypeMappings.put(type, configClass, tunerClass, prefix, mixed);
    }

    public Class<? extends PeftConfig> configFor(PeftType type) {
        Entry e = registry.get(type);
        return e == null ? null : e.configClass;
    }

    public Class<? extends BaseTuner> tunerFor(PeftType type) {
        Entry e = registry.get(type);
        return e == null ? null : e.tunerClass;
    }

    public String prefixFor(PeftType type) {
        Entry e = registry.get(type);
        return e == null ? type.prefix() : e.prefix;
    }

    public boolean mixedFor(PeftType type) {
        Entry e = registry.get(type);
        return e != null && e.mixed;
    }

    /** Construct the {@link PeftConfig} object from a parsed adapter_config.json map. */
    public PeftConfig fromDict(Map<String, Object> dict) {
        Object t = dict.get("peft_type");
        PeftType type = t == null ? PeftType.LORA : PeftType.valueOf(t.toString().toUpperCase());
        Class<? extends PeftConfig> klass = configFor(type);
        if (klass == null) {
            // Fallback: emit a warning and return base PeftConfig.
            return new PeftConfig(PeftConfig.builder());
        }
        try {
            java.lang.reflect.Method m = klass.getMethod("fromDict", Map.class);
            return (PeftConfig) m.invoke(null, dict);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("fromDict failed for " + type + ": " + e.getMessage(), e);
        }
    }
}
