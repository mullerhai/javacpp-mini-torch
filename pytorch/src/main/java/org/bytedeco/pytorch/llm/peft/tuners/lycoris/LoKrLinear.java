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
 * Java analog of HuggingFace {@code lycoris.modules.lokr.LokrLayer}.
 *
 * <p>LoKr represents the delta as {@code kron(W1, W2)} (Kronecker product) where
 * {@code W1} is the smaller "factor" matrix and {@code W2} is the larger "core"
 * matrix. Decomposes the delta into a much smaller number of parameters than LoRA.
 */
public class LoKrLinear extends LycorisLayer {

    /** Per-adapter W1 (small factor). */
    protected final Map<String, Module> lokrW1 = new LinkedHashMap<>();
    /** Per-adapter W2 (large core). */
    protected final Map<String, Module> lokrW2 = new LinkedHashMap<>();
    /** Per-adapter W1 shape (after factorization). */
    protected final Map<String, long[]> lokrW1Shape = new LinkedHashMap<>();

    private final LinearImpl baseLayer;

    public LoKrLinear(LinearImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public LinearImpl base() { return baseLayer; }

    public void updateLayer(String adapterName, int r, double alpha, boolean useRslora,
                             double loraDropoutP, boolean initWeights, int factor) {
        this.r.put(adapterName, r);
        this.loraAlpha.put(adapterName, alpha);
        this.scaling.put(adapterName, useRslora ? alpha / Math.sqrt(r) : alpha / r);
        // Placeholder: production splits into factor^2 blocks.
        org.bytedeco.pytorch.nn.modules.LinearImpl w1 = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(factor, r).bias(false));
        org.bytedeco.pytorch.nn.modules.LinearImpl w2 = new org.bytedeco.pytorch.nn.modules.LinearImpl(
                new org.bytedeco.pytorch.nn.options.LinearOptions(baseLayer.weight().size(0), r).bias(false));
        register_module("lokr_W1." + adapterName, w1);
        register_module("lokr_W2." + adapterName, w2);
        lokrW1.put(adapterName, w1); lokrW2.put(adapterName, w2);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor computeDelta(String adapterName) {
        Tensor w1w = ((org.bytedeco.pytorch.nn.modules.LinearImpl) lokrW1.get(adapterName)).weight();
        Tensor w2w = ((org.bytedeco.pytorch.nn.modules.LinearImpl) lokrW2.get(adapterName)).weight();
        // Approximation: outer product in lieu of true Kronecker for fallback.
        return w2w.mm(w1w).mul(new org.bytedeco.pytorch.Scalar(scaling.get(adapterName)));
    }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (lokrW1.containsKey(a)) out.add(lokrW1.get(a));
            if (lokrW2.containsKey(a)) out.add(lokrW2.get(a));
        }
        return out;
    }

    @Override public List<String> adapterLayerNames() {
        return new ArrayList<>(java.util.Arrays.asList("lokr_W1", "lokr_W2"));
    }
}
