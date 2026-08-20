/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.ia3;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv3dImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.IA3Config;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;
import org.bytedeco.pytorch.llm.peft.tuners.ia3.IA3Layer.IA3ConfigBackref;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.ia3.IA3Model}.
 */
public class IA3Model extends BaseTuner {

    public IA3Model(Module model, Map<String, IA3Config> configs, String adapterName) {
        super(model, (Map<String, PeftConfig>) (Map) configs, adapterName, "ia3_");
    }

    public IA3Model(Module model, IA3Config config, String adapterName) {
        super(model, singleConfigMap(config), adapterName, "ia3_");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, PeftConfig> singleConfigMap(IA3Config config) {
        Map<String, PeftConfig> m = new LinkedHashMap<>();
        m.put("default", config);
        return m;
    }

    @Override
    protected BaseTunerLayer _createNewLayer(PeftConfig cfg, String adapterName,
                                              String targetName, Module target) {
        IA3Config config = (IA3Config) cfg;
        boolean isFF = isFeedforwardTarget(targetName, config.feedforwardModules());
        IA3ConfigBackref bridge = new IA3ConfigBackref() {
            @Override public int fanInFanOut() { return config.fanInFanOut(); }
            @Override public boolean initWeights() { return config.initIa3Weights(); }
            @Override public String[] targetModules() { return config.targetModules(); }
            @Override public String[] feedforwardModules() { return config.feedforwardModules(); }
        };
        if (target instanceof LinearImpl) {
            IA3Linear layer = new IA3Linear((LinearImpl) target, targetName);
            layer.updateLayer(adapterName, config.initIa3Weights(), bridge);
            if (isFF) layer.setIsFeedforward(true);
            return layer;
        }
        if (target instanceof Conv2dImpl) {
            IA3Conv2d layer = new IA3Conv2d((Conv2dImpl) target, targetName);
            layer.updateLayer(adapterName, config.initIa3Weights(), bridge);
            if (isFF) layer.setIsFeedforward(true);
            return layer;
        }
        if (target instanceof Conv3dImpl) {
            IA3Conv3d layer = new IA3Conv3d((Conv3dImpl) target, targetName);
            layer.updateLayer(adapterName, config.initIa3Weights(), bridge);
            if (isFF) layer.setIsFeedforward(true);
            return layer;
        }
        return null;
    }

    /** Match a target name against the feedforward modules list (full, suffix, or regex). */
    public static boolean isFeedforwardTarget(String targetName, String[] feedforwardModules) {
        if (feedforwardModules == null) return false;
        for (String pat : feedforwardModules) {
            if (targetName.equals(pat) || targetName.endsWith(pat) || targetName.matches(pat)) return true;
        }
        return false;
    }

    @Override public PeftType peftType() { return PeftType.IA3; }
}