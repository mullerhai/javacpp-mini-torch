/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.tuners.lora.LoraLayer.LoraConfigBackref;

/**
 * Adapter that makes {@link LoraConfig} viewable as {@link LoraConfigBackref}.
 *
 * <p>The LoraLayer subclasses only need a small set of {@code effectiveRank/AlphA} /
 * {@code initLoraWeights} / nested-config fields; this bridges
 * {@link LoraConfig} to that interface without polluting the public surface.
 */
public class LoraModel_LoraConfigBridge implements LoraConfigBackref {

    private final LoraConfig cfg;

    public LoraModel_LoraConfigBridge(LoraConfig cfg) { this.cfg = cfg; }

    @Override public int effectiveRank(String n) { return cfg.effectiveRank(n); }
    @Override public double effectiveAlpha(String n) { return cfg.effectiveAlpha(n); }
    @Override public String initLoraWeights() { return cfg.initLoraWeights(); }
    @Override public boolean useDora() { return cfg.useDora(); }
    @Override public boolean useRslora() { return cfg.useRslora(); }
    @Override public boolean fanInFanOut() { return cfg.fanInFanOut(); }
    @Override public boolean loraBias() { return cfg.loraBias() == 1; }
    @Override public LoraConfig.LoftQConfig loftqConfig() { return cfg.loftqConfig(); }
    @Override public LoraConfig.EvaConfig evaConfig() { return cfg.evaConfig(); }
    @Override public LoraConfig.CordaConfig cordaConfig() { return cfg.cordaConfig(); }
    @Override public LoraConfig.LoraGAConfig loraGaConfig() { return cfg.loraGaConfig(); }
    @Override public LoraConfig.BdLoraConfig bdLoraConfig() { return cfg.bdLoraConfig(); }
    @Override public LoraConfig.ArrowConfig arrowConfig() { return cfg.arrowConfig(); }
    @Override public LoraConfig.VeloraConfig veloraConfig() { return cfg.veloraConfig(); }
    @Override public LoraConfig.MontecloraConfig montecloraConfig() { return cfg.montecloraConfig(); }
    @Override public boolean isAllLinear() { return cfg.isAllLinear(); }
}