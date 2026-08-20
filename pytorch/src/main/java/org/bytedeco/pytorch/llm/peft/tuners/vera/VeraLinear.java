/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.vera;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.vera.layer.Linear}.
 *
 * <p>VeRA uses two sets of frozen random projections shared across layers
 * ({@code vera_A} and {@code vera_B}) plus a tiny per-adapter pair of learned scaling
 * vectors {@code lambda_A} and {@code lambda_B}. The delta is computed as
 * {@code scaling_B * lambda_B * vera_B @ (scaling_A * lambda_A * vera_A @ x)}.
 */
public class VeraLinear extends BaseTunerLayer {

    /** Frozen random projection A (shared across all adapters). */
    protected Tensor veraA;
    /** Frozen random projection B (shared across all adapters). */
    protected Tensor veraB;
    /** Per-adapter scaling A. */
    protected final Map<String, Tensor> veraLambdaA = new LinkedHashMap<>();
    /** Per-adapter scaling B. */
    protected final Map<String, Tensor> veraLambdaB = new LinkedHashMap<>();
    /** Per-adapter alpha. */
    protected final Map<String, Double> veraAlpha = new LinkedHashMap<>();

    private final LinearImpl baseLayer;
    private final long inFeatures;
    private final long outFeatures;
    private final int r;
    private final double initB;

    public VeraLinear(LinearImpl baseLayer, String name, int r, double initB) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
        this.inFeatures = baseLayer.weight().size(1);
        this.outFeatures = baseLayer.weight().size(0);
        this.r = r;
        this.initB = initB;
    }

    @Override public Module baseLayer() { return baseLayer; }
    public LinearImpl base() { return baseLayer; }

    /** Initialise the shared frozen random projections (call once per model). */
    public void initialiseFrozenVectors(Tensor a, Tensor b) {
        this.veraA = a;
        this.veraB = b;
    }

    public void updateLayer(String adapterName, double alpha) {
        this.veraAlpha.put(adapterName, alpha);
        Tensor la = org.bytedeco.pytorch.global.torch.ones(new long[]{r}).contiguous().clone();
        Tensor lb = org.bytedeco.pytorch.global.torch.zeros(new long[]{outFeatures}).contiguous().clone();
        lb = lb.mul(new org.bytedeco.pytorch.Scalar(initB));
        la.set_requires_grad(true);
        lb.set_requires_grad(true);
        register_parameter("vera_lambda_A." + adapterName, la, true);
        register_parameter("vera_lambda_B." + adapterName, lb, true);
        veraLambdaA.put(adapterName, la);
        veraLambdaB.put(adapterName, lb);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (veraLambdaA.containsKey(a)) out.add(new TensorWrapper(veraLambdaA.get(a), "vera_lambda_A." + a));
            if (veraLambdaB.containsKey(a)) out.add(new TensorWrapper(veraLambdaB.get(a), "vera_lambda_B." + a));
        }
        return out;
    }

    @Override public List<String> adapterLayerNames() {
        return new ArrayList<>(java.util.Arrays.asList("vera_lambda_A", "vera_lambda_B"));
    }

    @Override public Tensor getDeltaWeight(String adapterName) {
        // delta = (B * lambda_B) @ (A * lambda_A)
        Tensor a = veraA.mul(veraLambdaA.get(adapterName).unsqueeze(1));
        Tensor b = veraB.mul(veraLambdaB.get(adapterName).unsqueeze(1));
        return b.mm(a);
    }

    @Override public void merge(String adapterName, boolean safeMerge) { /* placeholder */ }
    @Override public void unmerge(String adapterName) { /* placeholder */ }

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
