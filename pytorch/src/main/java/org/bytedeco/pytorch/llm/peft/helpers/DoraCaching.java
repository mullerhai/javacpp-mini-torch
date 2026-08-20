/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.helpers;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLinear;

import java.util.HashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.dora_utils.MaybeQuantizeAndDequantizeWeights}.
 */
public class DoraCaching {

    private final Map<String, Tensor> cache = new HashMap<>();

    public Tensor unitV(LoraLinear layer, String adapterName, Tensor weight) {
        String key = layer.fullModuleName() + "/" + adapterName;
        Tensor cached = cache.get(key);
        if (cached != null) return cached;
        // norm(dim=1, keepdim=True) in PyTorch:
        // norm(ScalarOptional, long[], boolean, ScalarType)
        Tensor norm = weight.norm(
                new org.bytedeco.pytorch.ScalarOptional(new org.bytedeco.pytorch.Scalar(2)),
                new long[]{1},
                true,
                org.bytedeco.pytorch.global.torch.kFloat());
        Tensor u = weight.div(norm.add(new org.bytedeco.pytorch.Scalar(1e-8)));
        cache.put(key, u);
        return u;
    }

    public void invalidate(String layerFullName, String adapterName) {
        cache.remove(layerFullName + "/" + adapterName);
    }

    public void clear() { cache.clear(); }
}
