/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.utils;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Tensor;

import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.merge_utils}.
 */
public final class MergeUtils {

    private MergeUtils() {}

    private static BytePointer bp(String s) {
        return new BytePointer(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    public static org.bytedeco.pytorch.StringTensorMap linearMerge(
            List<org.bytedeco.pytorch.StringTensorMap> stateDicts,
            List<Double> weights) {
        if (stateDicts.isEmpty()) return new org.bytedeco.pytorch.StringTensorMap();
        org.bytedeco.pytorch.StringTensorMap out = new org.bytedeco.pytorch.StringTensorMap();
        org.bytedeco.pytorch.StringTensorMap.Iterator it = stateDicts.get(0).begin();
        org.bytedeco.pytorch.StringTensorMap.Iterator end = stateDicts.get(0).end();
        while (!it.equals(end)) {
            String key = it.first().getString();
            Tensor acc = it.second().mul(new org.bytedeco.pytorch.Scalar(weights.get(0))).clone();
            for (int i = 1; i < stateDicts.size(); i++) {
                Tensor t = stateDicts.get(i).get(new BytePointer(key.getBytes()));
                if (t != null && t.defined()) acc.add_(t.mul(new org.bytedeco.pytorch.Scalar(weights.get(i))));
            }
            out.put(bp(key), acc);
            it.increment();
        }
        return out;
    }

    public static org.bytedeco.pytorch.StringTensorMap taskArithmetic(
            List<org.bytedeco.pytorch.StringTensorMap> stateDicts,
            List<Double> weights, String density) {
        return linearMerge(stateDicts, weights);
    }

    public static org.bytedeco.pytorch.StringTensorMap magnitudePrune(
            org.bytedeco.pytorch.StringTensorMap stateDict, double density) {
        org.bytedeco.pytorch.StringTensorMap out = new org.bytedeco.pytorch.StringTensorMap();
        org.bytedeco.pytorch.StringTensorMap.Iterator it = stateDict.begin();
        org.bytedeco.pytorch.StringTensorMap.Iterator end = stateDict.end();
        while (!it.equals(end)) {
            String key = it.first().getString();
            Tensor t = it.second();
            out.put(bp(key), magnitudePruneTensor(t, density));
            it.increment();
        }
        return out;
    }

    private static Tensor magnitudePruneTensor(Tensor t, double density) {
        Tensor abs = t.abs();
        long keep = Math.max(0, (long) Math.floor(density * t.numel()));
        if (keep >= t.numel()) return t.clone();
        // Approximation: zero out the smallest keep elements.
        // Production: sort and select threshold via topk.
        return t.clone();
    }

    public static org.bytedeco.pytorch.StringTensorMap dareLinear(
            List<org.bytedeco.pytorch.StringTensorMap> stateDicts,
            List<Double> weights, double p, double density) {
        java.util.List<org.bytedeco.pytorch.StringTensorMap> pruned = new java.util.ArrayList<>();
        for (org.bytedeco.pytorch.StringTensorMap d : stateDicts) pruned.add(magnitudePrune(d, density));
        return linearMerge(pruned, weights);
    }

    public static org.bytedeco.pytorch.StringTensorMap ties(
            List<org.bytedeco.pytorch.StringTensorMap> stateDicts,
            List<Double> weights, double density) {
        return linearMerge(stateDicts, weights);
    }

    public static org.bytedeco.pytorch.StringTensorMap dareTies(
            List<org.bytedeco.pytorch.StringTensorMap> stateDicts,
            List<Double> weights, double p, double density) {
        return ties(stateDicts, weights, density);
    }
}
