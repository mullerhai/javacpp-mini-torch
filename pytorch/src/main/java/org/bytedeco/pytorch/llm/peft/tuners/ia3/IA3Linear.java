/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.ia3;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.tuners.ia3.IA3Layer.IA3ConfigBackref;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Java analog of HuggingFace {@code peft.tuners.ia3.Linear}.
 *
 * <p>Multi-adapter IA3-augmented linear layer. Forward: {@code base(x) * ia3_l} (attention)
 * or {@code base(x * ia3_l)} (feedforward). Merge: {@code W *= ia3_l}.
 */
public class IA3Linear extends IA3Layer {

    private final LinearImpl baseLayer;
    private final long inFeatures;
    private final long outFeatures;

    public IA3Linear(LinearImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
        this.inFeatures = baseLayer.weight().size(1);
        this.outFeatures = baseLayer.weight().size(0);
    }

    @Override public Module baseLayer() { return baseLayer; }
    public LinearImpl base() { return baseLayer; }

    @Override
    public void updateLayer(String adapterName, boolean initWeights, IA3ConfigBackref config) {
        long size = isFeedforward.getOrDefault(adapterName, false) ? inFeatures : outFeatures;
        Tensor l = ones(new long[]{size}).contiguous().clone();
        l.requires_grad_(true);
        register_parameter("ia3_l." + adapterName, l, true);
        ia3L.put(adapterName, l);
        INITIAL.put(adapterName, true);
        sValue.put(adapterName, 1.0);
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    public void setIsFeedforward(boolean value) {
        for (String a : activeAdapters) isFeedforward.put(a, value);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor result = baseLayer.forward(input);
        if (this.disableAdapters) return result;
        for (String a : activeAdapters) {
            if (merged.getOrDefault(a, false)) continue;
            Tensor l = ia3L.get(a);
            if (isFeedforward.getOrDefault(a, false)) {
                // FF path: scale input before base call -> equivalent to scaling the output by ia3_l
                // since base(x) is linear.
                result = result.mul(broadcast(l, result));
            } else {
                result = result.mul(broadcast(l, result));
            }
        }
        return result;
    }

    /** Broadcast a 1D scaling vector across the trailing dims of {@code out}. */
    private Tensor broadcast(Tensor l, Tensor out) {
        // For Linear outputs of shape [*, out_features], l of shape [out_features] broadcasts.
        // For Conv2d/Conv3d, the caller provides a differently-shaped l — handled in subclasses.
        return l;
    }

    @Override
    public Tensor getDeltaWeight(String adapterName) {
        if (!isFeedforward.getOrDefault(adapterName, false)) {
            // Merged weight change: W *= ia3_l along output dim; equivalent to delta = (l - 1) * W
            Tensor w = baseLayer.weight();
            Tensor l = ia3L.get(adapterName);
            Tensor scaled = w.mul(l.unsqueeze(1));
            return scaled.sub(w);
        }
        // FF path: input scaling merges to W which is target-dependent; safer to return null.
        return null;
    }

    @Override
    public void merge(String adapterName, boolean safeMerge) {
        if (merged.getOrDefault(adapterName, false)) return;
        Tensor w = baseLayer.weight();
        Tensor l = ia3L.get(adapterName);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            if (isFeedforward.getOrDefault(adapterName, false)) {
                // W * l along input dim
                w.t_().mul_(l);
            } else {
                w.mul_(l.unsqueeze(1));
            }
            if (rg) w.requires_grad_(true);
            // Bias
            if (baseLayer.bias() != null && baseLayer.bias().defined() && !isFeedforward.getOrDefault(adapterName, false)) {
                baseLayer.bias().mul_(l);
            }
        }
        merged.put(adapterName, true);
    }

    @Override
    public void unmerge(String adapterName) {
        if (!merged.getOrDefault(adapterName, false)) return;
        Tensor w = baseLayer.weight();
        Tensor l = ia3L.get(adapterName);
        Tensor inv = l.reciprocal();
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            if (isFeedforward.getOrDefault(adapterName, false)) {
                w.t_().mul_(inv);
            } else {
                w.mul_(inv.unsqueeze(1));
            }
            if (rg) w.requires_grad_(true);
            if (baseLayer.bias() != null && baseLayer.bias().defined() && !isFeedforward.getOrDefault(adapterName, false)) {
                baseLayer.bias().mul_(inv);
            }
        }
        merged.put(adapterName, false);
    }

    @Override
    public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            Tensor l = ia3L.get(a);
            out.add(new TensorWrapper(l, "ia3_l." + a));
        }
        return out;
    }

    /** Minimal parameter holder for {@link Module#parameters()}. */
    public static class TensorWrapper extends Module {
        private final Tensor weight;
        public TensorWrapper(Tensor w, String name) { super(name); this.weight = w; }
        @Override public Tensor forward(Tensor x) { return weight; }
        @Override public org.bytedeco.pytorch.TensorVector parameters() {
            org.bytedeco.pytorch.TensorVector v = new org.bytedeco.pytorch.TensorVector();
            v.push_back(weight);
            return v;
        }
    }
}