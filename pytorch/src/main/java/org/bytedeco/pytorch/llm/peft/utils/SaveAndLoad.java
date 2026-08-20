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
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.SharedModuleVector;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.nio.charset.StandardCharsets;

/**
 * Java analog of HuggingFace {@code peft.utils.save_and_load}.
 */
public final class SaveAndLoad {

    private SaveAndLoad() {}

    private static BytePointer bp(String s) {
        return new BytePointer(s.getBytes(StandardCharsets.UTF_8));
    }

    public static org.bytedeco.pytorch.StringTensorMap getPeftModelStateDict(PeftModel model, String adapterName) {
        org.bytedeco.pytorch.StringTensorMap out = new org.bytedeco.pytorch.StringTensorMap();
        Module m = model.baseModel();
        if (m != null) collectModule(m, adapterName, out);
        return out;
    }

    private static void collectModule(Module m, String adapterName,
                                       org.bytedeco.pytorch.StringTensorMap out) {
        String prefix = m.name().getString(StandardCharsets.UTF_8) + ".";
        if (m instanceof BaseTunerLayer) {
            BaseTunerLayer l = (BaseTunerLayer) m;
            for (Module am : l.adapterModules()) {
                org.bytedeco.pytorch.TensorVector ps = am.parameters();
                for (int j = 0; j < (int) ps.size(); j++) {
                    String key = prefix + am.name().getString(StandardCharsets.UTF_8) + "." + adapterName;
                    out.put(bp(key), ps.get(j));
                }
            }
        }
        SharedModuleVector children = m.children();
        for (int i = 0; i < (int) children.size(); i++) {
            collectModule(children.get(i), adapterName, out);
        }
    }

    public static org.bytedeco.pytorch.StringTensorMap getBaseModelStateDict(PeftModel model) {
        Module m = model.baseModel();
        if (m == null) return new org.bytedeco.pytorch.StringTensorMap();
        // state_dict() is not available in the current JavaCPP API - stubbed placeholder
        return new org.bytedeco.pytorch.StringTensorMap();
    }

    public static void setPeftModelStateDict(PeftModel model,
                                               org.bytedeco.pytorch.StringTensorMap stateDict,
                                               String adapterName) {
    }

    public static void setBaseModelStateDict(PeftModel model,
                                               org.bytedeco.pytorch.StringTensorMap stateDict) {
    }
}
