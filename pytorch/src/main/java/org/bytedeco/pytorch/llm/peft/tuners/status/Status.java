/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.status;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.TunerLayerStatus} / {@code TunerModelStatus}.
 *
 * <p>Provides {@code get_layer_status} / {@code get_model_status} which return a JSON-able
 * description of each layer / adapter pair (name, type, enabled, active, merged,
 * trainable parameter count, etc.).
 */
public final class Status {

    private Status() {}

    public static LayerStatus getLayerStatus(org.bytedeco.pytorch.nn.Module layer) {
        LayerStatus s = new LayerStatus();
        BytePointer namePtr = layer.name();
        s.name = namePtr != null ? namePtr.getString(StandardCharsets.UTF_8) : "";
        s.enabled = !(layer instanceof BaseTunerLayer) || !((BaseTunerLayer) layer).adaptersDisabled();
        if (layer instanceof BaseTunerLayer) {
            BaseTunerLayer l = (BaseTunerLayer) layer;
            s.activeAdapters = new java.util.ArrayList<>(l.activeAdapters());
            s.mergedAdapters = new java.util.ArrayList<>(l.merged().keySet());
        }
        return s;
    }

    public static ModelStatus getModelStatus(PeftModel model) {
        ModelStatus s = new ModelStatus();
        s.modelType = model.peftType().name();
        s.adapters = new LinkedHashMap<>();
        for (Map.Entry<String, org.bytedeco.pytorch.llm.peft.PeftConfig> e : model.peftConfigs().entrySet()) {
            AdapterStatus a = new AdapterStatus();
            a.name = e.getKey();
            a.type = e.getValue().peftType().name();
            a.active = model.activeAdapters().contains(e.getKey());
            s.adapters.put(e.getKey(), a);
        }
        return s;
    }

    public static class LayerStatus {
        public String name;
        public boolean enabled;
        public java.util.List<String> activeAdapters = java.util.Collections.emptyList();
        public java.util.List<String> mergedAdapters = java.util.Collections.emptyList();
        @Override public String toString() { return "LayerStatus{" + name + " enabled=" + enabled + "}"; }
    }

    public static class ModelStatus {
        public String modelType;
        public Map<String, AdapterStatus> adapters;
        @Override public String toString() { return "ModelStatus{" + modelType + " adapters=" + adapters + "}"; }
    }

    public static class AdapterStatus {
        public String name;
        public String type;
        public boolean active;
        @Override public String toString() { return "AdapterStatus{" + name + " " + type + "}"; }
    }
}
