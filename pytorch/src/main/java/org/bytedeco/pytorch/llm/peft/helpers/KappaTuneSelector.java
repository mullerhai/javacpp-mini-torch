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
import org.bytedeco.pytorch.llm.peft.tuners.adalora.AdaLoraLinear;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.kappa_tuner.KappaTuneSelector}.
 */
public class KappaTuneSelector {

    private final Map<String, double[]> latestScores = new LinkedHashMap<>();

    public double computeKappa(AdaLoraLinear layer, String adapterName) {
        String key = adapterName;
        double[] mag = latestScores.computeIfAbsent(key, k -> {
            Tensor t = layer.loraE().get(k);
            if (t == null || !t.defined()) return new double[0];
            double[] arr = new double[(int) t.size(0)];
            // Use item_double() to get scalar value directly
            for (int i = 0; i < arr.length; i++) {
                arr[i] = Math.abs(t.select(0, i).item_double());
            }
            return arr;
        });
        if (mag.length == 0) return 0.0;
        double sum = 0;
        for (double v : mag) sum += v;
        return sum / mag.length;
    }

    public double meanKappa(java.util.Map<String, AdaLoraLinear> layers) {
        if (layers.isEmpty()) return 0.0;
        double total = 0;
        for (Map.Entry<String, AdaLoraLinear> e : layers.entrySet()) {
            total += computeKappa(e.getValue(), e.getKey());
        }
        return total / layers.size();
    }

    public void reset() { latestScores.clear(); }
}
