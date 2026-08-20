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
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

import java.util.ArrayList;
import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.Conv1d}.
 */
public class LoraConv1d extends LoraLayer {

    private final Conv1dImpl baseLayer;

    public LoraConv1d(Conv1dImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public Conv1dImpl base() { return baseLayer; }

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

        long inFeatures = baseLayer.options().in_channels().get(0);
        long outFeatures = baseLayer.options().out_channels().get(0);
        long kernel = baseLayer.options().kernel_size().get(0);
        long stride = baseLayer.options().stride().get(0);
        Conv1dOptions aOpts = new Conv1dOptions(inFeatures, effR, new LongPointer(kernel));
        aOpts.stride(new LongPointer(stride)).bias(false);
        Conv1dOptions bOpts = new Conv1dOptions(outFeatures, effR, new LongPointer(1));
        bOpts.stride(new LongPointer(1)).bias(false);
        Conv1dImpl a = new Conv1dImpl(aOpts);
        Conv1dImpl b = new Conv1dImpl(bOpts);
        register_module("lora_A." + adapterName, a);
        register_module("lora_B." + adapterName, b);
        this.loraA.put(adapterName, a);
        this.loraB.put(adapterName, b);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor forward(Tensor input) {
        Tensor result = baseLayer.forward(input);
        if (this.disableAdapters) return result;
        for (String a : activeAdapters) {
            if (merged.getOrDefault(a, false)) continue;
            Conv1dImpl aLayer = (Conv1dImpl) loraA.get(a);
            Conv1dImpl bLayer = (Conv1dImpl) loraB.get(a);
            Tensor mid = aLayer.forward(input);
            Tensor delta = bLayer.forward(mid).mul(new org.bytedeco.pytorch.Scalar(scaling.get(a)));
            result = result.add(delta);
        }
        return result;
    }

    @Override public Tensor getDeltaWeight(String adapterName) {
        return null;
    }
    @Override public void merge(String adapterName, boolean safeMerge) { /* no-op */ }
    @Override public void unmerge(String adapterName) { /* no-op */ }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (loraA.containsKey(a)) out.add(loraA.get(a));
            if (loraB.containsKey(a)) out.add(loraB.get(a));
        }
        return out;
    }
}
