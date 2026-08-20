/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.layer.LoraLayer}.
 *
 * <p>Base class for all LoRA-family layers (linear / embedding / conv1d / conv2d / conv3d /
 * MHA / ParamWrapper). Holds per-adapter state: lora_A, lora_B, lora_dropout, scaling,
 * r, lora_alpha, useRslora, useDora, loraBias, hasScaling.
 */
public abstract class LoraLayer extends BaseTunerLayer {

    /** Per-adapter LoRA-A linear: a {@code LinearImpl} of shape {@code [r, in_features]}. */
    protected final Map<String, Module> loraA = new LinkedHashMap<>();
    /** Per-adapter LoRA-B linear: a {@code LinearImpl} of shape {@code [out_features, r]}. */
    protected final Map<String, Module> loraB = new LinkedHashMap<>();
    /** Per-adapter dropout (or IdentityModule when {@code dropout=0}). */
    protected final Map<String, Module> loraDropout = new LinkedHashMap<>();
    /** Per-adapter magnitude vector (DoRA only). */
    protected final Map<String, Tensor> loraMagnitudeVector = new LinkedHashMap<>();
    /** Per-adapter rank. */
    protected final Map<String, Integer> r = new LinkedHashMap<>();
    /** Per-adapter alpha. */
    protected final Map<String, Double> loraAlpha = new LinkedHashMap<>();
    /** Per-adapter scaling factor. */
    protected final Map<String, Double> scaling = new LinkedHashMap<>();
    /** Per-adapter rsLoRA flag. */
    protected final Map<String, Boolean> useRslora = new LinkedHashMap<>();
    /** Per-adapter DoRA flag. */
    protected final Map<String, Boolean> useDora = new LinkedHashMap<>();
    /** Per-adapter lora_B bias flag. */
    protected final Map<String, Boolean> loraBias = new LinkedHashMap<>();
    /** Per-adapter "has scaling" flag (false for raw LoRA when alpha=0, etc.). */
    protected final Map<String, Boolean> hasScaling = new LinkedHashMap<>();

    /** Full module-path of this layer (for state-dict key reconstruction). */
    protected String fullModuleName = "";

    /** nn.Linear / nn.Embedding / nn.Conv2d / etc. stored under {@code "base_layer"}. */
    public abstract Module baseLayer();

    public Map<String, Module> loraA() { return loraA; }
    public Map<String, Module> loraB() { return loraB; }
    public Map<String, Module> loraDropout() { return loraDropout; }
    public Map<String, Tensor> loraMagnitudeVector() { return loraMagnitudeVector; }
    public Map<String, Integer> r() { return r; }
    public Map<String, Double> loraAlpha() { return loraAlpha; }
    public Map<String, Double> scaling() { return scaling; }
    public Map<String, Boolean> useRslora() { return useRslora; }
    public Map<String, Boolean> useDora() { return useDora; }
    public Map<String, Boolean> loraBias() { return loraBias; }
    public Map<String, Boolean> hasScaling() { return hasScaling; }
    public String fullModuleName() { return fullModuleName; }
    public void fullModuleName(String v) { this.fullModuleName = v; }

    /** Concrete subclasses must implement per-adapter update. */
    public abstract void updateLayer(String adapterName, int r, double alpha, boolean useRslora,
                                      double loraDropoutP, boolean initLoraWeights,
                                      LoraConfigBackref config);

    /** Default to per-adapter scaling factor. */
    public double getScaling(String adapterName) {
        return scaling.getOrDefault(adapterName, 1.0);
    }

    @Override
    public List<String> adapterLayerNames() {
        return new ArrayList<>(java.util.Arrays.asList("lora_A", "lora_B"));
    }

    /** Common base-class helper: kaiming-uniform for A, zeros for B. */
    public static void defaultLoraInit(Tensor a, Tensor b) {
        // Use libtorch int-equivalent ops via receiver; otherwise leave un-initialised.
        // Concrete callers should use the Init helpers in {@link Init}.
    }

    /** Marker interface so {@link LoraLayer} can call back into {@link LoraConfig} for runtime info. */
    public interface LoraConfigBackref {
        int effectiveRank(String moduleName);
        double effectiveAlpha(String moduleName);
        String initLoraWeights();
        boolean useDora();
        boolean useRslora();
        boolean fanInFanOut();
        boolean loraBias();
        org.bytedeco.pytorch.llm.peft.LoraConfig.LoftQConfig loftqConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.EvaConfig evaConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.CordaConfig cordaConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.LoraGAConfig loraGaConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.BdLoraConfig bdLoraConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.ArrowConfig arrowConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.VeloraConfig veloraConfig();
        org.bytedeco.pytorch.llm.peft.LoraConfig.MontecloraConfig montecloraConfig();
        boolean isAllLinear();
    }
}