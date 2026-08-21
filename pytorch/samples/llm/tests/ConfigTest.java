/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tests;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.PromptTuningConfig;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for {@link LoraConfig} and {@link PromptTuningConfig}.
 */
public class ConfigTest {

    @Test
    public void loraConfigRoundtrip() {
        LoraConfig cfg = new LoraConfig.Builder()
                .r(8).loraAlpha(16).targetModules("q_proj", "v_proj")
                .loraDropout(0.05).taskType("CAUSAL_LM").build();
        Map<String, Object> d = cfg.toDict();
        assertEquals(8, d.get("r"));
        assertEquals(16, d.get("lora_alpha"));
        LoraConfig restored = (LoraConfig) org.bytedeco.pytorch.llm.peft.PeftMethodRegistry.instance().fromDict(d);
        assertEquals(cfg.r(), restored.r());
        assertEquals(cfg.loraAlpha(), restored.loraAlpha());
    }

    @Test
    public void pissaVariantDetection() {
        assertTrue(LoraConfig.isPissaVariant("pissa"));
        assertTrue(LoraConfig.isPissaVariant("pissa_niter_4"));
        assertEquals(4, LoraConfig.pissaNiter("pissa_niter_4"));
        assertEquals(0, LoraConfig.pissaNiter("pissa"));
        assertFalse(LoraConfig.isPissaVariant("gaussian"));
    }

    @Test
    public void peftTypePrefix() {
        assertEquals("lora_", PeftType.LORA.prefix());
        assertEquals("ia3_", PeftType.IA3.prefix());
        assertEquals("oft_", PeftType.OFT.prefix());
        assertEquals("hada_", PeftType.LOHA.prefix());
        assertEquals("lokr_", PeftType.LOKR.prefix());
    }

    @Test
    public void promptTuningDefaults() {
        PromptTuningConfig cfg = new PromptTuningConfig.Builder().build();
        assertEquals(0, cfg.numVirtualTokens());
    }

    @Test
    public void peftMethodRegistryRoundtrip() {
        Map<String, Object> d = Map.of("peft_type", "LORA", "r", 8, "lora_alpha", 16);
        Object cfg = org.bytedeco.pytorch.llm.peft.PeftMethodRegistry.instance().fromDict(d);
        assertNotNull(cfg);
    }
}
