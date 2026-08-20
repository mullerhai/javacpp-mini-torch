/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lycoris;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code lycoris.modules.loha.LohaLayer}.
 *
 * <p>LoHa represents the delta as {@code W1 @ W2} (elementwise Hadamard product)
 * where {@code W1 = A1 @ B1} and {@code W2 = A2 @ B2}. Equivalent to a rank-r
 * decomposition composed with the Hadamard product.
 */
public class LoHaLinear extends LycorisLayer {

    /** Per-adapter W1 = A1 @ B1. */
    protected final Map<String, Module> hadaA1 = new LinkedHashMap<>();
    protected final Map<String, Module> hadaB1 = new LinkedHashMap<>();
    /** Per-adapter W2 = A2 @ B2. */
    protected final Map<String, Module> hadaA2 = new LinkedHashMap<>();
    protected final Map<String, Module> hadaB2 = new LinkedHashMap<>();

    private final LinearImpl baseLayer;

    public LoHaLinear(LinearImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public LinearImpl base() { return baseLayer; }

    public void updateLayer(String adapterName, int r, double alpha, boolean useRslora,
                             double loraDropoutP, boolean initWeights) {
        this.r.put(adapterName, r);
        this.loraAlpha.put(adapterName, alpha);
        this.scaling.put(adapterName, useRslora ? alpha / Math.sqrt(r) : alpha / r);
        org.bytedeco.pytorch.nn.modules.LinearImpl a1 = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(r, baseLayer.weight().size(1)).bias(false));
        org.bytedeco.pytorch.nn.modules.LinearImpl b1 = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(baseLayer.weight().size(0), r).bias(false));
        org.bytedeco.pytorch.nn.modules.LinearImpl a2 = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(r, baseLayer.weight().size(1)).bias(false));
        org.bytedeco.pytorch.nn.modules.LinearImpl b2 = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(baseLayer.weight().size(0), r).bias(false));
        register_module("hada_A1." + adapterName, a1);
        register_module("hada_B1." + adapterName, b1);
        register_module("hada_A2." + adapterName, a2);
        register_module("hada_B2." + adapterName, b2);
        hadaA1.put(adapterName, a1); hadaB1.put(adapterName, b1);
        hadaA2.put(adapterName, a2); hadaB2.put(adapterName, b2);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor computeDelta(String adapterName) {
        Tensor a1w = ((org.bytedeco.pytorch.nn.modules.LinearImpl) hadaA1.get(adapterName)).weight();
        Tensor b1w = ((org.bytedeco.pytorch.nn.modules.LinearImpl) hadaB1.get(adapterName)).weight();
        Tensor a2w = ((org.bytedeco.pytorch.nn.modules.LinearImpl) hadaA2.get(adapterName)).weight();
        Tensor b2w = ((org.bytedeco.pytorch.nn.modules.LinearImpl) hadaB2.get(adapterName)).weight();
        Tensor w1 = b1w.mm(a1w);
        Tensor w2 = b2w.mm(a2w);
        return w1.mul(w2).mul(new org.bytedeco.pytorch.Scalar(scaling.get(adapterName)));
    }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (hadaA1.containsKey(a)) out.add(hadaA1.get(a));
            if (hadaB1.containsKey(a)) out.add(hadaB1.get(a));
            if (hadaA2.containsKey(a)) out.add(hadaA2.get(a));
            if (hadaB2.containsKey(a)) out.add(hadaB2.get(a));
        }
        return out;
    }

    @Override public List<String> adapterLayerNames() {
        return new ArrayList<>(java.util.Arrays.asList("hada_A1", "hada_B1", "hada_A2", "hada_B2"));
    }
}
