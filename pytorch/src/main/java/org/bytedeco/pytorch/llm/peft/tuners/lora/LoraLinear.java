/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.dropout;
import static org.bytedeco.pytorch.global.torch.mm;
import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.Linear}.
 *
 * <p>Multi-adapter LoRA-augmented linear layer. Stores per-adapter A / B / dropout /
 * scaling / r / alpha / useRslora / useDora / loraBias. Forward computes
 * {@code base(x) + scaling * (dropout(x) @ A^T @ B^T)} for each active adapter, in the
 * order tracked by {@link #activeAdapters}.
 */
public class LoraLinear extends LoraLayer {

    private final LinearImpl baseLayer;

    public LoraLinear(LinearImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override
    public Module baseLayer() { return baseLayer; }

    public LinearImpl base() { return baseLayer; }

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
        this.useRslora.put(adapterName, useRslora);
        this.scaling.put(adapterName, useRslora ? effAlpha / Math.sqrt(effR) : effAlpha / effR);
        this.hasScaling.put(adapterName, true);
        boolean useDora = config != null && config.useDora();
        this.useDora.put(adapterName, useDora);

        // Dropout module
        if (loraDropoutP > 0.0) {
            this.loraDropout.put(adapterName, new org.bytedeco.pytorch.nn.modules.DropoutImpl(loraDropoutP));
        } else {
            this.loraDropout.put(adapterName, identityModule());
        }

        long inFeatures = baseLayer.weight().size(1);
        long outFeatures = baseLayer.weight().size(0);
        boolean bias = config != null && config.loraBias();

        // Per-adapter A and B — registered as sub-modules so optim discovery picks them up.
        LinearImpl a = new LinearImpl(new org.bytedeco.pytorch.nn.options.LinearOptions(effR, inFeatures).bias(false));
        LinearImpl b = new LinearImpl(new org.bytedeco.pytorch.nn.options.LinearOptions(outFeatures, effR).bias(bias));
        this.loraA.put(adapterName, a);
        this.loraB.put(adapterName, b);
        this.loraBias.put(adapterName, bias);

        // Register as sub-modules for named_modules() / parameters() discovery.
        register_module("lora_A." + adapterName, a);
        register_module("lora_B." + adapterName, b);

        // Initialise weight via Init utility
        Init.resetLoraParameters(this, adapterName, initLoraWeights);

        // DoRA magnitude vector
        if (useDora) {
            Tensor w = baseLayer.weight();
            Tensor norms = w.norm(
                    new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(2)),
                    new long[]{1}, true,
                    org.bytedeco.pytorch.global.torch.kFloat())
                    .reshape(new long[]{outFeatures}).clone().detach();
            register_parameter("lora_magnitude_vector." + adapterName, norms, true);
            this.loraMagnitudeVector.put(adapterName, norms);
        }

        // Move adapter to base device
        moveAdapterToDevice(adapterName);

        // Pass through to config for any runtime bookkeeping (variant init etc.)
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor result = baseLayer.forward(input);
        if (this.disableAdapters) return result;
        for (String a : activeAdapters) {
            if (merged.getOrDefault(a, false)) continue;
            Tensor x = input;
            double p = dropoutP(a);
            if (p > 0 && is_training()) {
                x = dropout(x, p, true);
            }
            // A: [r, in]   B: [out, r]
            Tensor aWeight = ((LinearImpl) loraA.get(a)).weight();
            Tensor bWeight = ((LinearImpl) loraB.get(a)).weight();
            Tensor aT = aWeight.t();
            Tensor bT = bWeight.t();
            Tensor mid = matmulLast(x, aT);
            Tensor delta = matmulLast(mid, bT).mul(new Scalar(scaling.get(a)));
            result = result.add(delta);
        }
        return result;
    }

    @Override
    public Tensor getDeltaWeight(String adapterName) {
        Tensor aWeight = ((LinearImpl) loraA.get(adapterName)).weight();
        Tensor bWeight = ((LinearImpl) loraB.get(adapterName)).weight();
        Tensor delta = mm(bWeight, aWeight);
        double scale = scaling.get(adapterName);
        if (scale != 1.0) delta = delta.mul(new Scalar(scale));
        return delta;
    }

    @Override
    public void merge(String adapterName, boolean safeMerge) {
        if (merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            Tensor w = baseLayer.weight();
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            if (safeMerge) {
                Tensor cloned = w.clone();
                cloned.add_(delta);
                if (cloned.isfinite().all().item_bool()) w.copy_(cloned);
            } else {
                w.add_(delta);
            }
            if (rg) w.requires_grad_(true);
            if (loraBias.getOrDefault(adapterName, false)) {
                Tensor bBias = ((LinearImpl) loraB.get(adapterName)).bias();
                baseLayer.bias().add_(bBias.mul(new Scalar(scaling.get(adapterName))));
            }
        }
        merged.put(adapterName, true);
    }

    @Override
    public void unmerge(String adapterName) {
        if (!merged.getOrDefault(adapterName, false)) return;
        Tensor delta = getDeltaWeight(adapterName);
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            Tensor w = baseLayer.weight();
            boolean rg = false;
            try { rg = w.requires_grad(); } catch (Exception ignored) {}
            if (rg) w.requires_grad_(false);
            w.sub_(delta);
            if (rg) w.requires_grad_(true);
            if (loraBias.getOrDefault(adapterName, false)) {
                Tensor bBias = ((LinearImpl) loraB.get(adapterName)).bias();
                baseLayer.bias().sub_(bBias.mul(new Scalar(scaling.get(adapterName))));
            }
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

    private double dropoutP(String adapterName) {
        // DropoutImpl stores probability – we read via .options.p()
        Module d = loraDropout.get(adapterName);
        if (d instanceof org.bytedeco.pytorch.nn.modules.DropoutImpl) {
            return ((org.bytedeco.pytorch.nn.modules.DropoutImpl) d).options().p().get(0);
        }
        return 0.0;
    }

    private Module identityModule() {
        return new Module("identity") {
            @Override public Tensor forward(Tensor x) { return x; }
        };
    }

    private void moveAdapterToDevice(String adapterName) {
        // Best-effort: in JavaCPP we have a single device; nothing to move.
    }

    private static Tensor matmulLast(Tensor a, Tensor b) {
        if (a.dim() == 2 && b.dim() == 2) return mm(a, b);
        long[] aSizes = a.shape();
        long in = aSizes[aSizes.length - 1];
        long rest = 1;
        for (int i = 0; i < aSizes.length - 1; i++) rest *= aSizes[i];
        Tensor flat = a.reshape(rest, in);
        Tensor out2d = mm(flat, b);
        long out = b.size(1);
        long[] outShape = new long[aSizes.length];
        System.arraycopy(aSizes, 0, outShape, 0, aSizes.length - 1);
        outShape[outShape.length - 1] = out;
        return out2d.reshape(outShape);
    }
}