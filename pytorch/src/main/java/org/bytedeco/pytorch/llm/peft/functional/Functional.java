/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.functional;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.SharedModuleVector;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftMethodRegistry;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;

import java.util.Map;
import java.util.function.Consumer;

/**
 * Lower-level functional helpers for PEFT injection, casting, adapter switching.
 */
public final class Functional {

    private Functional() {}

    public static Module injectAdapterInModel(Module model, Map<String, PeftConfig> configs, String adapterName) {
        PeftConfig cfg = configs.get(adapterName);
        if (cfg == null) cfg = configs.values().iterator().next();
        PeftType t = cfg.peftType();
        Class<? extends BaseTuner> klass = PeftMethodRegistry.instance().tunerFor(t);
        try {
            java.lang.reflect.Constructor<? extends BaseTuner> ctor = klass.getConstructor(
                    Module.class, Map.class, String.class);
            return ctor.newInstance(model, configs, adapterName);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Failed to inject adapter " + adapterName + ": " + e.getMessage(), e);
        }
    }

    public static void castAdapterDtype(Module model, int dtype) {
        // dtype is a ScalarType ordinal; use torch.ScalarType.fromScalarType.
        // Placeholder: production would convert dtype to ScalarType and call toType.
    }

    public static void setAdapterOnTree(Module model, String adapterName) {
        walk(model, m -> {
            if (m instanceof org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer l) {
                l.setAdapter(adapterName, false);
            }
        });
    }

    public static void deleteAdapterFromTree(Module model, String adapterName) {
        walk(model, m -> {
            if (m instanceof org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer l) {
                l.merged().remove(adapterName);
                if (l.activeAdapters().contains(adapterName)) {
                    java.util.List<String> rest = new java.util.ArrayList<>(l.activeAdapters());
                    rest.remove(adapterName);
                    l.setAdapters(rest, false);
                }
            }
        });
    }

    public static void setRequiresGrad(Module model, boolean requiresGrad) {
        walk(model, m -> {
            org.bytedeco.pytorch.TensorVector ps = m.parameters();
            for (int j = 0; j < (int) ps.size(); j++) {
                if (ps.get(j).defined()) ps.get(j).requires_grad_(requiresGrad);
            }
        });
    }

    private static void walk(Module m, Consumer<Module> fn) {
        fn.accept(m);
        SharedModuleVector children = m.children();
        for (int i = 0; i < (int) children.size(); i++) walk(children.get(i), fn);
    }
}
