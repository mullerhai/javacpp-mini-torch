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
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.ia3.layer.IA3Layer}.
 *
 * <p>For each adapter, holds the learned {@code ia3_l} scale vector and a flag indicating
 * whether it scales input (feedforward) or output (attention).
 */
public abstract class IA3Layer extends BaseTunerLayer {

    /** Per-adapter learned scale vector (shape matches the dimension being scaled). */
    protected final Map<String, Tensor> ia3L = new LinkedHashMap<>();
    /** Per-adapter flag: true = scaling input (FF), false = scaling output (attention). */
    protected final Map<String, Boolean> isFeedforward = new LinkedHashMap<>();
    /** Per-adapter flag: was initialised as ones (peft default). */
    protected final Map<String, Boolean> INITIAL = new LinkedHashMap<>();
    /** Per-adapter scaling factor (default 1.0; used by tests). */
    protected final Map<String, Double> sValue = new LinkedHashMap<>();

    public Map<String, Tensor> ia3L() { return ia3L; }
    public Map<String, Boolean> isFeedforward() { return isFeedforward; }
    public Map<String, Boolean> initial() { return INITIAL; }
    public Map<String, Double> sValue() { return sValue; }
    public double sValue(String adapterName) { return sValue.getOrDefault(adapterName, 1.0); }

    public abstract void updateLayer(String adapterName, boolean initWeights, IA3ConfigBackref config);

    public abstract org.bytedeco.pytorch.Tensor getDeltaWeight(String adapterName);

    @Override public List<String> adapterLayerNames() {
        return new java.util.ArrayList<>(java.util.Arrays.asList("ia3_l"));
    }

    /** Backref interface for runtime config fields. */
    public interface IA3ConfigBackref {
        int fanInFanOut();
        boolean initWeights();
        String[] targetModules();
        String[] feedforwardModules();
    }
}