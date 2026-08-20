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
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

import java.util.ArrayList;
import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.ParamWrapper}.
 *
 * <p>Wraps a single {@link org.bytedeco.pytorch.Tensor} parameter (LoRA on parameter,
 * not on a Linear / Conv / Embedding). Used when {@code target_parameters} is set on
 * a {@link org.bytedeco.pytorch.llm.peft.LoraConfig}.
 */
public class LoraParamWrapper extends LoraLayer {

    private final Tensor baseParameter;

    public LoraParamWrapper(String name, Tensor baseParam) {
        super();
        this.baseParameter = baseParam;
        this.fullModuleName = name;
    }

    @Override public Module baseLayer() { return null; }
    public Tensor baseParameter() { return baseParameter; }

    @Override
    public void updateLayer(String adapterName, int r, double alpha, boolean useRslora,
                             double loraDropoutP, boolean initLoraWeights, LoraConfigBackref config) {
        int effR = r;
        double effAlpha = alpha;
        if (config != null) {
            effR = config.effectiveRank(fullModuleName);
            effAlpha = config.effectiveAlpha(fullModuleName);
        }
        this.r.put(adapterName, effR);
        this.loraAlpha.put(adapterName, effAlpha);
        this.scaling.put(adapterName, useRslora ? effAlpha / Math.sqrt(effR) : effAlpha / effR);
        this.hasScaling.put(adapterName, true);
        this.useRslora.put(adapterName, useRslora);
        this.useDora.put(adapterName, false);
        this.loraBias.put(adapterName, false);

        long[] shape = baseParameter.shape();
        long inF = shape[shape.length - 1];
        long outF = shape[0];
        org.bytedeco.pytorch.nn.modules.LinearImpl a = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(effR, inF).bias(false));
        org.bytedeco.pytorch.nn.modules.LinearImpl b = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(outF, effR).bias(false));
        register_module("lora_A." + adapterName, a);
        register_module("lora_B." + adapterName, b);
        this.loraA.put(adapterName, a);
        this.loraB.put(adapterName, b);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor forward(Tensor ignored) {
        // Merged path: return baseParameter directly.
        return baseParameter;
    }

    @Override public Tensor getDeltaWeight(String adapterName) {
        Tensor aWeight = ((org.bytedeco.pytorch.nn.modules.LinearImpl) loraA.get(adapterName)).weight();
        Tensor bWeight = ((org.bytedeco.pytorch.nn.modules.LinearImpl) loraB.get(adapterName)).weight();
        return bWeight.mm(aWeight).mul(new org.bytedeco.pytorch.Scalar(scaling.get(adapterName)));
    }

    @Override public void merge(String adapterName, boolean safeMerge) {
        if (merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            boolean rg = false;
            try { rg = baseParameter.requires_grad(); } catch (Exception ignored) {}
            if (rg) baseParameter.requires_grad_(false);
            baseParameter.add_(delta);
            if (rg) baseParameter.requires_grad_(true);
        }
        merged.put(adapterName, true);
    }

    @Override public void unmerge(String adapterName) {
        if (!merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            boolean rg = false;
            try { rg = baseParameter.requires_grad(); } catch (Exception ignored) {}
            if (rg) baseParameter.requires_grad_(false);
            baseParameter.sub_(delta);
            if (rg) baseParameter.requires_grad_(true);
        }
        merged.put(adapterName, false);
    }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (loraA.containsKey(a)) out.add(loraA.get(a));
            if (loraB.containsKey(a)) out.add(loraB.get(a));
        }
        return out;
    }
}