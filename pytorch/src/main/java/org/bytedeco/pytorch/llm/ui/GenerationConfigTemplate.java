/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Named {@code GenerationConfig} presets shown in the Web UI. Equivalent to HF's
 * {@code generation_config.json} snapshot. Three built-ins ship with the engine:
 * {@link #deterministic()}, {@link #balanced()}, {@link #creative()}.
 *
 * <p>Per-turn overrides from the request body always take precedence over the template
 * baseline (see {@code ChatTurn.RequestParams.applyTo(...)}).
 */
public final class GenerationConfigTemplate {

    public final String name;
    public final boolean doSample;
    public final double temperature;
    public final int topK;
    public final double topP;
    public final double repetitionPenalty;
    public final int maxNewTokens;
    public final List<String> stopStrings = new ArrayList<>();
    public final int repetitionLimit;

    public GenerationConfigTemplate(String name, boolean doSample, double temperature,
                                    int topK, double topP, double repetitionPenalty,
                                    int maxNewTokens, int repetitionLimit) {
        this.name = name;
        this.doSample = doSample;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repetitionPenalty = repetitionPenalty;
        this.maxNewTokens = maxNewTokens;
        this.repetitionLimit = repetitionLimit;
    }

    public static GenerationConfigTemplate deterministic() {
        return new GenerationConfigTemplate("deterministic", false, 0.0, 1, 1.0, 1.0, 512, 0);
    }

    public static GenerationConfigTemplate balanced() {
        return new GenerationConfigTemplate("balanced", true, 0.7, 40, 0.9, 1.0, 512, 0);
    }

    public static GenerationConfigTemplate creative() {
        return new GenerationConfigTemplate("creative", true, 1.0, 80, 0.95, 1.05, 512, 8);
    }

    public static List<GenerationConfigTemplate> defaults() {
        List<GenerationConfigTemplate> list = new ArrayList<>();
        list.add(deterministic());
        list.add(balanced());
        list.add(creative());
        return list;
    }

    /** Materialize a {@link GenerationConfig} from this template's baseline. */
    public GenerationConfig toGenerationConfig() {
        GenerationConfig.Builder b = GenerationConfig.builder()
                .doSample(doSample)
                .temperature(temperature)
                .topK(topK)
                .topP(topP)
                .repetitionPenalty(repetitionPenalty)
                .maxNewTokens(maxNewTokens);
        return b.build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", name);
        m.put("doSample", doSample);
        m.put("temperature", temperature);
        m.put("topK", topK);
        m.put("topP", topP);
        m.put("repetitionPenalty", repetitionPenalty);
        m.put("maxNewTokens", maxNewTokens);
        m.put("repetitionLimit", repetitionLimit);
        if (!stopStrings.isEmpty()) m.put("stopStrings", stopStrings);
        return m;
    }

    public GenerationConfigTemplate withName(String newName) {
        return new GenerationConfigTemplate(newName, doSample, temperature, topK, topP,
                repetitionPenalty, maxNewTokens, repetitionLimit);
    }
}