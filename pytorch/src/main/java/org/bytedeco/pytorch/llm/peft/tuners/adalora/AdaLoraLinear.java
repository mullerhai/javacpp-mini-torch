/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.adalora;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.adalora.layer.AdaLoraLinear}.
 *
 * <p>AdaLoRA parameterises the LoRA delta as {@code E * A} where {@code E} is a
 * diagonal matrix (per-parameter rank selector) and {@code A} is the LoRA-A matrix.
 * The rank allocator ({@link RankAllocator}) periodically prunes dimensions of
 * {@code E} with the smallest magnitude and re-allocates budget to the largest.
 */
public class AdaLoraLinear extends BaseTunerLayer {

    /** Per-adapter LoRA A. */
    protected final Map<String, Module> loraA = new LinkedHashMap<>();
    /** Per-adapter LoRA B. */
    protected final Map<String, Module> loraB = new LinkedHashMap<>();
    /** Per-adapter diagonal scaling (rank selector). */
    protected final Map<String, Tensor> loraE = new LinkedHashMap<>();
    /** Per-adapter alpha. */
    protected final Map<String, Double> scaling = new LinkedHashMap<>();
    /** Per-adapter dropout. */
    protected final Map<String, Double> loraDropout = new LinkedHashMap<>();
    /** Per-adapter rank. */
    protected final Map<String, Integer> loraRank = new LinkedHashMap<>();

    private final LinearImpl baseLayer;

    public AdaLoraLinear(LinearImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public LinearImpl base() { return baseLayer; }

    public void updateLayer(String adapterName, int r, double alpha, double loraDropoutP) {
        this.loraRank.put(adapterName, r);
        this.scaling.put(adapterName, alpha / Math.sqrt(r));
        this.loraDropout.put(adapterName, loraDropoutP);

        org.bytedeco.pytorch.nn.options.LinearOptions aOpts = new org.bytedeco.pytorch.nn.options.LinearOptions(
                baseLayer.weight().size(1), r).bias(false);
        org.bytedeco.pytorch.nn.options.LinearOptions bOpts = new org.bytedeco.pytorch.nn.options.LinearOptions(
                r, baseLayer.weight().size(0)).bias(false);

        org.bytedeco.pytorch.nn.modules.LinearImpl a = new org.bytedeco.pytorch.nn.modules.LinearImpl(aOpts);
        org.bytedeco.pytorch.nn.modules.LinearImpl b = new org.bytedeco.pytorch.nn.modules.LinearImpl(bOpts);
        register_module("lora_A." + adapterName, a);
        register_module("lora_B." + adapterName, b);
        loraA.put(adapterName, a);
        loraB.put(adapterName, b);

        Tensor e = org.bytedeco.pytorch.global.torch.ones(new long[]{r}).contiguous().clone();
        e.requires_grad_(true);
        register_parameter("lora_E." + adapterName, e, true);
        loraE.put(adapterName, e);

        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor getDeltaWeight(String adapterName) {
        Tensor aWeight = ((org.bytedeco.pytorch.nn.modules.LinearImpl) loraA.get(adapterName)).weight();
        Tensor bWeight = ((org.bytedeco.pytorch.nn.modules.LinearImpl) loraB.get(adapterName)).weight();
        Tensor e = loraE.get(adapterName);
        Tensor ea = aWeight.mul(e.unsqueeze(1));
        Tensor delta = bWeight.mm(ea);
        return delta.mul(new org.bytedeco.pytorch.Scalar(scaling.get(adapterName)));
    }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (loraA.containsKey(a)) out.add(loraA.get(a));
            if (loraB.containsKey(a)) out.add(loraB.get(a));
        }
        return out;
    }

    @Override public List<String> adapterLayerNames() {
        return new ArrayList<>(java.util.Arrays.asList("lora_A", "lora_B", "lora_E"));
    }

    @Override public void merge(String adapterName, boolean safeMerge) { /* placeholder */ }
    @Override public void unmerge(String adapterName) { /* placeholder */ }

    /** Returns the loraE map (diagonal scaling per adapter). */
    public Map<String, Tensor> loraE() { return loraE; }
}
