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
import org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

import java.util.ArrayList;
import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.MultiheadAttention}.
 *
 * <p>For multi-head attention, LoRA cannot easily hook the {@code in_proj_weight} path.
 * The recommended approach is to merge the adapter on enter and unmerge on exit so that
 * a single contiguous forward happens through the base MHA.
 */
public class LoraMHA extends LoraLayer {

    private final MultiheadAttentionImpl baseLayer;

    public LoraMHA(MultiheadAttentionImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public MultiheadAttentionImpl base() { return baseLayer; }

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

        long inFeatures = baseLayer.in_proj_weight().size(1);
        long outFeatures = baseLayer.in_proj_weight().size(0);
        org.bytedeco.pytorch.nn.modules.LinearImpl a = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(effR, inFeatures).bias(false));
        org.bytedeco.pytorch.nn.modules.LinearImpl b = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(outFeatures, effR).bias(false));
        register_module("lora_A." + adapterName, a);
        register_module("lora_B." + adapterName, b);
        this.loraA.put(adapterName, a);
        this.loraB.put(adapterName, b);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override
    public Tensor forward(Tensor input) {
        // MHA requires q, k, v -> forward(q, k, v, ...). We collapse to the first positional arg.
        return forward(input, input, input);
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v) {
        // Merge before forward; unmerge after.
        for (String a : activeAdapters) merge(a, true);
        try {
            org.bytedeco.pytorch.T_TensorTensor_T out = baseLayer.forwardT_TensorTensor_T(q, k, v);
            return out.get0();
        } finally {
            for (String a : activeAdapters) unmerge(a);
        }
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
            Tensor w = baseLayer.in_proj_weight();
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            w.add_(delta);
            if (rg) w.requires_grad_(true);
        }
        merged.put(adapterName, true);
    }

    @Override public void unmerge(String adapterName) {
        if (!merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            Tensor w = baseLayer.in_proj_weight();
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            w.sub_(delta);
            if (rg) w.requires_grad_(true);
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