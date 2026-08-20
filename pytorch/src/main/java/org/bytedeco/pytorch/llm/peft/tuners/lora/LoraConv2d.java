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
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

import java.util.ArrayList;
import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.Conv2d}.
 */
public class LoraConv2d extends LoraLayer {

    private final Conv2dImpl baseLayer;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final int groups;

    public LoraConv2d(Conv2dImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
        this.kernelSize = (int) baseLayer.options().kernel_size().get(0);
        this.stride = (int) baseLayer.options().stride().get(0);
        org.bytedeco.pytorch.enumtype.Conv2dPadding padVar = baseLayer.options().padding();
        this.padding = (int) padVar.get0().get(0);
        this.groups = (int) baseLayer.options().groups().get(0);
    }

    @Override
    public Module baseLayer() { return baseLayer; }
    public Conv2dImpl base() { return baseLayer; }

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
        org.bytedeco.pytorch.enumtype.Conv2dPadding paddingVal =
                new org.bytedeco.pytorch.enumtype.Conv2dPadding(new org.bytedeco.javacpp.LongPointer(padding, padding));
        org.bytedeco.pytorch.enumtype.Conv2dPadding zeroPadding =
                new org.bytedeco.pytorch.enumtype.Conv2dPadding(new org.bytedeco.javacpp.LongPointer(0, 0));
        Conv2dOptions aOpts = new Conv2dOptions(inFeatures, effR, new LongPointer(kernelSize, kernelSize));
        aOpts.stride(new LongPointer(stride, stride)).padding(paddingVal).bias(false).groups(groups);
        Conv2dOptions bOpts = new Conv2dOptions(effR, outFeatures, new LongPointer(1, 1));
        bOpts.stride(new LongPointer(1, 1)).padding(zeroPadding).bias(false).groups(groups);
        Conv2dImpl a = new Conv2dImpl(aOpts);
        Conv2dImpl b = new Conv2dImpl(bOpts);
        register_module("lora_A." + adapterName, a);
        register_module("lora_B." + adapterName, b);
        this.loraA.put(adapterName, a);
        this.loraB.put(adapterName, b);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor result = baseLayer.forward(input);
        if (this.disableAdapters) return result;
        for (String a : activeAdapters) {
            if (merged.getOrDefault(a, false)) continue;
            Conv2dImpl aLayer = (Conv2dImpl) loraA.get(a);
            Conv2dImpl bLayer = (Conv2dImpl) loraB.get(a);
            Tensor mid = aLayer.forward(input);
            Tensor delta = bLayer.forward(mid).mul(new org.bytedeco.pytorch.Scalar(scaling.get(a)));
            result = result.add(delta);
        }
        return result;
    }

    @Override
    public Tensor getDeltaWeight(String adapterName) {
        Conv2dImpl aLayer = (Conv2dImpl) loraA.get(adapterName);
        Conv2dImpl bLayer = (Conv2dImpl) loraB.get(adapterName);
        Tensor aWeight = aLayer.weight();
        Tensor bWeight = bLayer.weight();
        long kH = aWeight.size(2), kW = aWeight.size(3);
        if (kH == 1 && kW == 1) {
            Tensor aFlat = aWeight.reshape(r.get(adapterName), aWeight.size(1));
            Tensor bFlat = bWeight.reshape(bWeight.size(0), r.get(adapterName));
            return bFlat.mm(aFlat).mul(new org.bytedeco.pytorch.Scalar(scaling.get(adapterName)));
        }
        return null;
    }

    @Override
    public void merge(String adapterName, boolean safeMerge) {
        if (merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        if (delta == null) return;
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            Tensor w = baseLayer.weight();
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            w.add_(delta);
            if (rg) w.requires_grad_(true);
        }
        merged.put(adapterName, true);
    }

    @Override
    public void unmerge(String adapterName) {
        if (!merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        if (delta == null) return;
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            Tensor w = baseLayer.weight();
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            w.sub_(delta);
            if (rg) w.requires_grad_(true);
        }
        merged.put(adapterName, false);
    }

    @Override
    public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (loraA.containsKey(a)) out.add(loraA.get(a));
            if (loraB.containsKey(a)) out.add(loraB.get(a));
        }
        return out;
    }
}
