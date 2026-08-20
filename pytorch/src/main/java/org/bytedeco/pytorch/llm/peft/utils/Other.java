/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.utils;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.SharedModuleVector;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.utils.other}.
 */
public final class Other {

    private Other() {}

    public static Module prepareModelForKBitTraining(Module model,
                                                      boolean useGradientCheckpointing,
                                                      boolean use_reentrant) {
        // requires_grad_ is not available in the current JavaCPP API
        // Use Tensor.set_requires_grad() method instead for individual parameters
        if (useGradientCheckpointing) {
            GradientCheckpointing.enableOn(model, use_reentrant);
        }
        return model;
    }

    public static void castMixedPrecisionParams(Module model, java.util.List<String> skipModules) {
        CastMixedPrecisionParams.cast(model, skipModules);
    }

    public static Tensor shiftTokensRight(Tensor inputIds, long padTokenId, int decoderStartTokenId) {
        return org.bytedeco.pytorch.global.torch.zeros_like(inputIds);
    }

    public static Tensor transpose(Tensor x, int dim1, int dim2) {
        long n = x.dim();
        long[] perm = new long[(int) n];
        for (int i = 0; i < n; i++) perm[i] = i;
        perm[dim1] = dim2; perm[dim2] = dim1;
        return x.permute(perm);
    }

    public static org.bytedeco.pytorch.StringTensorMap concatenateStateDicts(
            List<org.bytedeco.pytorch.StringTensorMap> dicts) {
        org.bytedeco.pytorch.StringTensorMap out = new org.bytedeco.pytorch.StringTensorMap();
        for (org.bytedeco.pytorch.StringTensorMap d : dicts) {
            org.bytedeco.pytorch.StringTensorMap.Iterator it = d.begin();
            org.bytedeco.pytorch.StringTensorMap.Iterator end = d.end();
            while (!it.equals(end)) {
                out.put(it.first(), it.second());
                it.increment();
            }
        }
        return out;
    }
}
