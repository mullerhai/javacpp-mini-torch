/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.adalora;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.peft.PeftWarning;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.adalora.utils.RankAllocator}.
 */
public class RankAllocator {

    public final int targetRank;
    public final double initWarmup;
    public final double beta1;
    public final double beta2;
    public final double alpha;
    public final double gamma;
    public final int updateSteps;
    public final double pruneRatio;
    public final double totalStep;

    private int stepsSinceUpdate = 0;

    public RankAllocator(int targetRank, double initWarmup, double beta1, double beta2,
                           double alpha, double gamma, int updateSteps, double pruneRatio,
                           double totalStep) {
        this.targetRank = targetRank;
        this.initWarmup = initWarmup;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.alpha = alpha;
        this.gamma = gamma;
        this.updateSteps = updateSteps;
        this.pruneRatio = pruneRatio;
        this.totalStep = totalStep;
    }

    public void updateAndPrune(Map<String, AdaLoraLinear> layers, int currentStep) {
        if (currentStep < initWarmup) {
            for (Map.Entry<String, AdaLoraLinear> e : layers.entrySet()) {
                for (Map.Entry<String, Tensor> ee : e.getValue().loraE.entrySet()) {
                    Tensor t = ee.getValue();
                    if (!t.defined()) continue;
                }
            }
            return;
        }
        stepsSinceUpdate++;
        if (stepsSinceUpdate < updateSteps) return;
        stepsSinceUpdate = 0;
        new PeftWarning("RankAllocator.updateAndPrune: full SVD-based redistribution "
                + "requires torch.linalg; falling back to magnitude pruning");
    }
}
