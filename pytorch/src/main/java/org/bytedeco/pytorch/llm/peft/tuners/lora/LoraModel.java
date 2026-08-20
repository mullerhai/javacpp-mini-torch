/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv3dImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTuner;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.LoraModel}.
 *
 * <p>Walks the host model, identifies LoRA targets, and replaces each with a
 * {@link LoraLinear} / {@link LoraConv2d} / {@link LoraEmbedding} (or Conv1d / Conv3d /
 * MHA / ParamWrapper) wrapper.
 */
public class LoraModel extends BaseTuner {

    public LoraModel(Module model, Map<String, LoraConfig> configs, String adapterName) {
        super(model, (Map<String, PeftConfig>) (Map) configs, adapterName, "lora_");
    }

    public LoraModel(Module model, LoraConfig config, String adapterName) {
        super(model, singleConfigMap(config), adapterName, "lora_");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, PeftConfig> singleConfigMap(LoraConfig config) {
        Map<String, PeftConfig> m = new LinkedHashMap<>();
        m.put("default", config);
        return m;
    }

    @Override
    protected BaseTunerLayer _createNewLayer(PeftConfig cfg, String adapterName,
                                              String targetName, Module target) {
        LoraConfig config = (LoraConfig) cfg;
        LoraConfigBackref bridge = new LoraModel_LoraConfigBridge(config);
        boolean init = config.initLoraWeights() != null;
        if (target instanceof LinearImpl) {
            LoraLinear layer = new LoraLinear((LinearImpl) target, targetName);
            layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                              config.dropout(), init, bridge);
            return layer;
        }
        if (target instanceof EmbeddingImpl) {
            LoraEmbedding layer = new LoraEmbedding((EmbeddingImpl) target, targetName);
            layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                              config.dropout(), init, bridge);
            return layer;
        }
        if (target instanceof Conv2dImpl) {
            LoraConv2d layer = new LoraConv2d((Conv2dImpl) target, targetName);
            layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                              config.dropout(), init, bridge);
            return layer;
        }
        if (target instanceof Conv1dImpl) {
            LoraConv1d layer = new LoraConv1d((Conv1dImpl) target, targetName);
            layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                              config.dropout(), init, bridge);
            return layer;
        }
        if (target instanceof Conv3dImpl) {
            LoraConv3d layer = new LoraConv3d((Conv3dImpl) target, targetName);
            layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                              config.dropout(), init, bridge);
            return layer;
        }
        if (target instanceof org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl) {
            LoraMHA layer = new LoraMHA((org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl) target, targetName);
            layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                              config.dropout(), init, bridge);
            return layer;
        }
        // ParamWrapper fallback
        Tensor firstParam = (target.parameters() != null && target.parameters().size() > 0)
                            ? target.parameters().get(0)
                            : null;
        LoraParamWrapper layer = new LoraParamWrapper(targetName, firstParam);
        layer.updateLayer(adapterName, config.r(), config.alpha(), config.useRslora(),
                          config.dropout(), init, bridge);
        return layer;
    }

    @Override
    public PeftType peftType() { return PeftType.LORA; }
}