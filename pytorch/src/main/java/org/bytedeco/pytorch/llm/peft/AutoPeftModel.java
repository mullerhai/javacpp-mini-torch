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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.auto.AutoPeftModel}.
 *
 * <p>Auto-loader for an adapter on top of a base model. Parses the {@code adapter_config.json}
 * from a directory, identifies the {@link PeftType} and target architecture, and instantiates
 * the correct {@link PeftModel} subclass.
 */
public class AutoPeftModel {

    public static PeftModel forCausalLM(Module baseModel, String adapterDir) {
        Map<String, Object> cfg = AdapterConfigIO.read(adapterDir);
        PeftConfig config = Peft.getPeftConfig(cfg);
        return Peft.getPeftModel(baseModel, config);
    }

    public static PeftModel forSeq2SeqLM(Module baseModel, String adapterDir) {
        return forCausalLM(baseModel, adapterDir);
    }

    public static PeftModel forSeqCls(Module baseModel, String adapterDir) {
        return forCausalLM(baseModel, adapterDir);
    }

    public static PeftModel forTokenCls(Module baseModel, String adapterDir) {
        return forCausalLM(baseModel, adapterDir);
    }

    public static PeftModel forQuestionAnswering(Module baseModel, String adapterDir) {
        return forCausalLM(baseModel, adapterDir);
    }

    public static PeftModel forFeatureExtraction(Module baseModel, String adapterDir) {
        return forCausalLM(baseModel, adapterDir);
    }

    public static PeftModel fromPretrained(Module baseModel, String adapterDir) {
        return forCausalLM(baseModel, adapterDir);
    }

    /** Static utility class — do not instantiate. */
    private AutoPeftModel() {}
}
