/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.peft_model.PeftMixedModel}.
 *
 * <p>Wraps a base model with a single {@link BaseTuner} that supports multiple
 * adapter types (e.g. LoRA + IA3 on the same backbone). Only PeftTypes returned by
 * {@link PeftType#isMixedCompatible()} may be used together.
 */
public class PeftMixedModel extends Module {

    private final Module baseModel;
    private final BaseTuner tuner;
    private final Map<String, PeftConfig> peftConfigs;

    public PeftMixedModel(Module baseModel, Map<String, PeftConfig> configs, String adapterName) {
        super("PeftMixedModel");
        this.baseModel = baseModel;
        this.peftConfigs = configs;
        PeftConfig cfg = configs.get(adapterName);
        if (cfg == null) cfg = configs.values().iterator().next();
        Class<? extends BaseTuner> klass = PeftMethodRegistry.instance().tunerFor(cfg.peftType());
        try {
            java.lang.reflect.Constructor<? extends BaseTuner> ctor = klass.getConstructor(
                    Module.class, Map.class, String.class);
            this.tuner = ctor.newInstance(baseModel, configs, adapterName);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Mixed tuner construction failed: " + e.getMessage(), e);
        }
        register_module("base_model", baseModel);
        register_module("tuner", tuner);
    }

    @Override public org.bytedeco.pytorch.Tensor forward(org.bytedeco.pytorch.Tensor x) {
        return tuner.forward(x);
    }

    public Module baseModel() { return baseModel; }
    public BaseTuner tuner() { return tuner; }
    public Map<String, PeftConfig> peftConfigs() { return peftConfigs; }
}
