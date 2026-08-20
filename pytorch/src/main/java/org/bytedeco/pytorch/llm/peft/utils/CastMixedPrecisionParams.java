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
import java.util.HashSet;
import java.util.Set;

/**
 * Java analog of HuggingFace {@code peft.utils.cast_mixed_precision_params}.
 */
public final class CastMixedPrecisionParams {

    private CastMixedPrecisionParams() {}

    public static void cast(Module model, java.util.List<String> skipModules) {
        Set<String> skip = new HashSet<>(skipModules);
        walk(model, skip);
    }

    private static void walk(Module m, Set<String> skip) {
        String name = m.name().getString(StandardCharsets.UTF_8);
        boolean shouldCast = true;
        for (String s : skip) {
            if (name.toLowerCase().contains(s.toLowerCase())) { shouldCast = false; break; }
        }
        if (shouldCast) {
            org.bytedeco.pytorch.TensorVector ps = m.parameters();
            for (int j = 0; j < (int) ps.size(); j++) {
                Tensor p = ps.get(j);
                if (p.defined()) p.toType(org.bytedeco.pytorch.global.torch.kFloat());
            }
        }
        SharedModuleVector children = m.children();
        for (int i = 0; i < (int) children.size(); i++) walk(children.get(i), skip);
    }
}
