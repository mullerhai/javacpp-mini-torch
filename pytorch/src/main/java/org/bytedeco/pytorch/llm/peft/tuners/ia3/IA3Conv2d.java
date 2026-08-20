/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.ia3;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.llm.peft.tuners.ia3.IA3Layer.IA3ConfigBackref;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.ones;

/**
 * Java analog of HuggingFace {@code peft.tuners.ia3.Conv2d}.
 */
public class IA3Conv2d extends IA3Layer {

    private final Conv2dImpl baseLayer;
    private final long outChannels;

    public IA3Conv2d(Conv2dImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
        this.outChannels = baseLayer.options().out_channels().get(0);
    }

    @Override public Module baseLayer() { return baseLayer; }
    public Conv2dImpl base() { return baseLayer; }

    @Override
    public void updateLayer(String adapterName, boolean initWeights, IA3ConfigBackref config) {
        long size = isFeedforward.getOrDefault(adapterName, false) ? baseLayer.options().in_channels().get(0) : outChannels;
        Tensor l = ones(new long[]{size, 1, 1}).contiguous().clone();
        l.requires_grad_(true);
        register_parameter("ia3_l." + adapterName, l, true);
        ia3L.put(adapterName, l);
        INITIAL.put(adapterName, true);
        sValue.put(adapterName, 1.0);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor forward(Tensor input) {
        Tensor result = baseLayer.forward(input);
        if (this.disableAdapters) return result;
        for (String a : activeAdapters) {
            if (merged.getOrDefault(a, false)) continue;
            Tensor l = ia3L.get(a);
            result = result.mul(l);
        }
        return result;
    }

    @Override public Tensor getDeltaWeight(String adapterName) { return null; }
    @Override public void merge(String adapterName, boolean safeMerge) { /* no-op for conv2d */ }
    @Override public void unmerge(String adapterName) { /* no-op for conv2d */ }

    public void setIsFeedforward(boolean value) {
        for (String a : activeAdapters) isFeedforward.put(a, value);
    }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            Tensor l = ia3L.get(a);
            out.add(new IA3Linear.TensorWrapper(l, "ia3_l." + a));
        }
        return out;
    }
}