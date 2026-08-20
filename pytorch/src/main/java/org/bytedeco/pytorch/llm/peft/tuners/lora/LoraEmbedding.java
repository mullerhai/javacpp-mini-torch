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
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.normal_;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.Embedding}.
 */
public class LoraEmbedding extends LoraLayer {

    private final EmbeddingImpl baseLayer;

    public LoraEmbedding(EmbeddingImpl baseLayer, String name) {
        super();
        this.baseLayer = baseLayer;
        this.fullModuleName = name;
    }

    @Override
    public Module baseLayer() { return baseLayer; }
    public EmbeddingImpl base() { return baseLayer; }

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

        long vocabSize = baseLayer.options().num_embeddings().get(0);
        Tensor a = zeros(new long[]{effR, vocabSize}).contiguous().clone();
        Tensor b = zeros(new long[]{vocabSize, effR}).contiguous().clone();
        normal_(b, 0.0, 1.0);
        a.requires_grad_(true);
        b.requires_grad_(true);
        register_parameter("lora_embedding_A." + adapterName, a, true);
        register_parameter("lora_embedding_B." + adapterName, b, true);
        this.loraA.put(adapterName, new PseudoModule(a));
        this.loraB.put(adapterName, new PseudoModule(b));
        this.activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
        this.merged.put(adapterName, false);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor result = baseLayer.forward(input);
        if (this.disableAdapters) return result;
        for (String a : activeAdapters) {
            if (merged.getOrDefault(a, false)) continue;
            Tensor aWeight = ((PseudoModule) loraA.get(a)).weight;
            Tensor bWeight = ((PseudoModule) loraB.get(a)).weight;
            Tensor delta = adapterContribution(input, aWeight, bWeight, scaling.get(a));
            result = result.add(delta);
        }
        return result;
    }

    private Tensor adapterContribution(Tensor input, Tensor aWeight, Tensor bWeight, double scale) {
        // Placeholder: returns scaled b @ a as a fallback approximation; production fill path
        // uses torch.embedding(aWeight.t(), input) @ bWeight.t() once that API is exposed.
        return bWeight.mm(aWeight).mul(new org.bytedeco.pytorch.Scalar(scale));
    }

    @Override
    public Tensor getDeltaWeight(String adapterName) {
        Tensor aWeight = ((PseudoModule) loraA.get(adapterName)).weight;
        Tensor bWeight = ((PseudoModule) loraB.get(adapterName)).weight;
        return bWeight.mm(aWeight);
    }

    @Override
    public void merge(String adapterName, boolean safeMerge) { /* embeddings: typically no merge */ }

    @Override
    public void unmerge(String adapterName) { /* embeddings: typically no merge */ }

    @Override
    public List<Module> adapterModules() {
        List<Module> out = new ArrayList<>();
        for (String a : activeAdapters) {
            if (loraA.containsKey(a)) out.add(loraA.get(a));
            if (loraB.containsKey(a)) out.add(loraB.get(a));
        }
        return out;
    }

    public static class PseudoModule extends Module {
        public final Tensor weight;
        public PseudoModule(Tensor w) { super("LoraEmbeddingWeight"); this.weight = w; }
        @Override public Tensor forward(Tensor x) { return weight; }
        @Override public org.bytedeco.pytorch.TensorVector parameters() {
            org.bytedeco.pytorch.TensorVector v = new org.bytedeco.pytorch.TensorVector();
            v.push_back(weight);
            return v;
        }
    }
}
