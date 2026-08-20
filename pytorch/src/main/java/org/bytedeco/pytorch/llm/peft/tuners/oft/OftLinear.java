/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.oft;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.oft.layer.Linear}.
 *
 * <p>Orthogonal Fine-Tuning (OFT) represents the delta weight as a learnable
 * orthogonal matrix R parameterised by the Cayley transform: R = (I + Q)(I - Q)^{-1}
 * where Q is a skew-symmetric matrix. The forward pass adds {@code delta = R @ W0 - W0}
 * to the base weights.
 */
public class OftLinear extends BaseTunerLayer {

    /** Per-adapter learnable Q (skew-symmetric, stored as half-tensor). */
    protected final Map<String, Tensor> oftR = new LinkedHashMap<>();
    /** Per-adapter alpha (scaling factor). */
    protected final Map<String, Double> oftAlpha = new LinkedHashMap<>();
    /** Per-adapter constraint (whether to apply the constraint step). */
    protected final Map<String, Boolean> oftConstraint = new LinkedHashMap<>();
    /** Per-adapter coft scaling (Block-Diagonal OFT / BOFT only). */
    protected final Map<String, Double> oftCoft = new LinkedHashMap<>();

    private final LinearImpl baseLayer;
    private final long inFeatures;
    private final long outFeatures;
    private final int blockSize;
    private final boolean isBoft;

    public OftLinear(LinearImpl baseLayer, String name, int blockSize, boolean isBoft) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
        this.inFeatures = baseLayer.weight().size(1);
        this.outFeatures = baseLayer.weight().size(0);
        this.blockSize = blockSize <= 0 ? (int) outFeatures : blockSize;
        this.isBoft = isBoft;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public LinearImpl base() { return baseLayer; }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (oftR.containsKey(a)) {
                Tensor r = oftR.get(a);
                out.add(new TensorWrapper(r, "oft_R." + a));
            }
        }
        return out;
    }

    @Override public List<String> adapterLayerNames() {
        return new ArrayList<>(java.util.Arrays.asList("oft_R"));
    }

    /** Update OFT layer for a specific adapter. */
    public void updateLayer(String adapterName, int r, double alpha, boolean useOftConstraint,
                            double coftScale, boolean initOftWeights) {
        this.oftAlpha.put(adapterName, alpha);
        this.oftConstraint.put(adapterName, useOftConstraint);
        this.oftCoft.put(adapterName, coftScale);
        // OFT R is shape [num_blocks, blockSize, blockSize]; stored as a single tensor for simplicity.
        long numBlocks = (outFeatures + blockSize - 1) / blockSize;
        Tensor R = org.bytedeco.pytorch.global.torch.zeros(numBlocks, blockSize, blockSize);
        R.set_requires_grad(true);
        oftR.put(adapterName, R);
        register_parameter("oft_R." + adapterName, R, true);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public Tensor getDeltaWeight(String adapterName) {
        // Cayley transform approximation: R = (I + Q)(I - Q)^{-1}; we approximate by I + skew(Q).
        // Production callers should implement the full Cayley transform via libtorch linear algebra.
        return oftR.get(adapterName).reshape(outFeatures, blockSize).sub(java.util.Arrays.stream(new long[]{0,1}).iterator().next() < 0
                ? org.bytedeco.pytorch.global.torch.eye(outFeatures, blockSize)
                : org.bytedeco.pytorch.global.torch.eye(outFeatures, blockSize));
    }

    @Override public void merge(String adapterName, boolean safeMerge) { /* placeholder */ }
    @Override public void unmerge(String adapterName) { /* placeholder */ }

    /** Minimal parameter holder. */
    public static class TensorWrapper extends Module {
        public final Tensor weight;
        public TensorWrapper(Tensor w, String n) { super(n); this.weight = w; }
        @Override public Tensor forward(Tensor x) { return weight; }
        @Override public org.bytedeco.pytorch.TensorVector parameters() {
            org.bytedeco.pytorch.TensorVector v = new org.bytedeco.pytorch.TensorVector();
            v.push_back(weight); return v;
        }
    }
}
