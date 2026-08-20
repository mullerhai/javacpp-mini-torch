/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.optimizers;

import org.bytedeco.pytorch.Tensor;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Java analog of HuggingFace {@code peft.optimizers.create_loraplus_optimizer}.
 *
 * <p>LoRA+ uses a different learning rate for the LoRA-B matrix than for the LoRA-A
 * matrix (typically B is given a larger learning rate by a factor of {@code lr_ratio}).
 * This class wraps an existing optimiser-like state and dispatches the appropriate
 * effective LR per parameter.
 */
public final class LoRAPlusOptimizer {

    private LoRAPlusOptimizer() {}

    /** Compute the effective learning rate for a parameter name. */
    public static double effectiveLearningRate(String paramName, double baseLr, double lrRatio) {
        if (paramName.contains("lora_B")) return baseLr * lrRatio;
        return baseLr;
    }

    /** Build a LoRA+ param-group split (analog to peft.optimizers.create_loraplus_optimizer_grouped_parameters). */
    public static java.util.List<Map<String, Object>> groupedParameters(
            java.util.Map<String, Tensor> namedParams, double lr, double lrRatio, double weightDecay) {
        java.util.List<Map<String, Object>> groups = new java.util.ArrayList<>();
        Map<String, Object> groupA = new LinkedHashMap<>();
        Map<String, Object> groupB = new LinkedHashMap<>();
        groupA.put("lr", lr);
        groupA.put("weight_decay", weightDecay);
        groupB.put("lr", lr * lrRatio);
        groupB.put("weight_decay", 0.0);
        java.util.List<Tensor> psA = new java.util.ArrayList<>();
        java.util.List<Tensor> psB = new java.util.ArrayList<>();
        for (Map.Entry<String, Tensor> e : namedParams.entrySet()) {
            if (e.getKey().contains("lora_B")) psB.add(e.getValue()); else psA.add(e.getValue());
        }
        groupA.put("params", psA);
        groupB.put("params", psB);
        groups.add(groupA);
        groups.add(groupB);
        return groups;
    }
}
