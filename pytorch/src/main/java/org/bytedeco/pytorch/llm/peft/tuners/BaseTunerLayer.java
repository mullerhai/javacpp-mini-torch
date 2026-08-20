/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 * You may not use this file except in compliance with the License.
 */
package org.bytedeco.pytorch.llm.peft.tuners;

import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.BaseTunerLayer}.
 *
 * <p>Tracks the per-adapter state held by every tuner layer: which adapters are active,
 * which are merged, and which are enabled. The concrete subclasses (e.g.
 * {@code LoraLayer}, {@code IA3Layer}) attach their own per-adapter maps on top of this.
 */
public abstract class BaseTunerLayer extends Module {

    /** Map from adapter name to merged-status flag. */
    protected final Map<String, Boolean> merged = new LinkedHashMap<>();

    /** Adapter names currently selected for the forward pass. */
    protected List<String> activeAdapters = new ArrayList<>(java.util.Arrays.asList("default"));

    /** Whether the adapter delta is added at all (forward-time disable). */
    protected boolean disableAdapters = false;

    /** Auto-disabling input-dtype casting flag (used by {@code autocast_adapter_dtype=False}). */
    protected boolean castInputDtypeEnabled = true;

    /** Full module-path of this layer (for state-dict key reconstruction). */
    protected String fullModuleName = "";

    /** Default per-adapter name list (if no setter has been called yet). */
    public void setAdapter(String adapterName, boolean inferenceMode) {
        if (!merged.containsKey(adapterName)) merged.put(adapterName, false);
        activeAdapters = new ArrayList<>(java.util.Arrays.asList(adapterName));
    }

    /** Underlying base module (LoRA-A linear, Conv2d, Embedding, IA3 linear, OFT linear, etc.). */
    public abstract Module baseLayer();

    public String fullModuleName() { return fullModuleName; }
    public void fullModuleName(String v) { this.fullModuleName = v; }

    public void setAdapters(List<String> adapterNames, boolean inferenceMode) {
        if (adapterNames == null) adapterNames = java.util.Arrays.asList("default");
        for (String a : adapterNames) {
            if (!merged.containsKey(a)) merged.put(a, false);
        }
        activeAdapters = new ArrayList<>(adapterNames);
    }

    /** Enable/disable this tuner's adapter path at forward time. */
    public void enableAdapters(boolean enabled) {
        this.disableAdapters = !enabled;
    }

    /** Toggle merging the adapter into the base layer (safe-merge aware). */
    public abstract void merge(String adapterName, boolean safeMerge);

    /** Reverse the merge. */
    public abstract void unmerge(String adapterName);

    /** Returns the active adapter names. */
    public List<String> activeAdapters() {
        return Collections.unmodifiableList(activeAdapters);
    }

    public boolean adaptersDisabled() { return disableAdapters; }
    public boolean isMerged(String adapterName) {
        return merged.getOrDefault(adapterName, false);
    }

    public void setMerge(String adapterName, boolean mergedNow) {
        merged.put(adapterName, mergedNow);
    }

    public void deleteAdapter(String adapterName) {
        merged.remove(adapterName);
        activeAdapters.remove(adapterName);
    }

    /** Returns the unmodifiable merged map. */
    public Map<String, Boolean> merged() {
        return Collections.unmodifiableMap(merged);
    }

    public boolean castInputDtypeEnabled() { return castInputDtypeEnabled; }
    public void castInputDtypeEnabled(boolean v) { this.castInputDtypeEnabled = v; }

    /**
     * Subclasses must return the per-adapter trainable {@link Module} list
     * (LoRA A/B, IA3 l, OFT R, etc.) so that {@link BaseTuner#trainableParameters()}
     * can collect them.
     */
    public abstract List<Module> adapterModules();

    /** Per-adapter parameter names (used by save/load state-dict). */
    public abstract List<String> adapterLayerNames();

    /** Per-adapter parameter names (used by save/load state-dict). Override if non-default. */
    public List<String> otherParamNames() { return new java.util.ArrayList<>(); }

    /** Subclasses override to return the additive delta weight for an adapter. */
    public abstract org.bytedeco.pytorch.Tensor getDeltaWeight(String adapterName);
}