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
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code lycoris.modules.lycoris.LycorisLayer}.
 *
 * <p>Base class for the Lycoris family (LoHa / LoKr / OFT3 / GLoRA) of LoRA-like
 * adapters. Stores the multi-adapter scaling factor and per-adapter rank.
 */
public abstract class LycorisLayer extends BaseTunerLayer {

    /** Per-adapter learned rank. */
    protected final Map<String, Integer> r = new LinkedHashMap<>();
    /** Per-adapter alpha. */
    protected final Map<String, Double> loraAlpha = new LinkedHashMap<>();
    /** Per-adapter scaling factor. */
    protected final Map<String, Double> scaling = new LinkedHashMap<>();

    public Map<String, Integer> r() { return r; }
    public Map<String, Double> loraAlpha() { return loraAlpha; }
    public Map<String, Double> scaling() { return scaling; }

    @Override public Tensor getDeltaWeight(String adapterName) {
        return computeDelta(adapterName);
    }

    /** Subclasses implement the actual delta formula (Hadamard product for LoHa,
     *  Kronecker product for LoKr, etc.). */
    public abstract Tensor computeDelta(String adapterName);

    @Override public void merge(String adapterName, boolean safeMerge) { /* placeholder */ }
    @Override public void unmerge(String adapterName) { /* placeholder */ }
}
