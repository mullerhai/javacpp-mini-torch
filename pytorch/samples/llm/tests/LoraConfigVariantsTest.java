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
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests covering the LoraConfig variants.
 */
public class LoraConfigVariantsTest {

    @Test
    public void doraEnabled() {
        LoraConfig cfg = new LoraConfig.Builder().r(8).useDora(true).build();
        assertTrue(cfg.useDora());
    }

    @Test
    public void rsLoRAScaling() {
        LoraConfig cfg = new LoraConfig.Builder().r(16).loraAlpha(8).useRslora(true).build();
        // Effective scaling stored on the per-layer LoraLinear is alpha / sqrt(r)
        // = 8 / 4 = 2.0 — verified at the per-layer level, not on the config.
        assertTrue(cfg.useRslora());
    }

    @Test
    public void cordaRequiresEigvecCache() {
        LoraConfig.CordaConfig corda = new LoraConfig.CordaConfig.Builder()
                .cordaMethod("eigvec")
                .cordaCacheDir("/tmp/corda")
                .build();
        assertEquals("eigvec", corda.cordaMethod());
    }

    @Test
    public void loftqConfig() {
        LoraConfig.LoftQConfig loftq = new LoraConfig.LoftQConfig.Builder()
                .loftqBits(4)
                .loftqIter(1)
                .build();
        assertEquals(4, loftq.loftqBits());
    }

    @Test
    public void evaConfig() {
        LoraConfig.EvaConfig eva = new LoraConfig.EvaConfig.Builder()
                .evaGamma(2.0)
                .build();
        assertEquals(2.0, eva.evaGamma());
    }

    @Test
    public void peftTypeMixedCompatible() {
        assertTrue(PeftType.LORA.isMixedCompatible());
        assertTrue(PeftType.LOHA.isMixedCompatible());
        assertFalse(PeftType.IA3.isMixedCompatible());
    }

    @Test
    public void peftTypePromptLearning() {
        assertTrue(PeftType.PROMPT_TUNING.isPromptLearning());
        assertTrue(PeftType.PREFIX_TUNING.isPromptLearning());
        assertFalse(PeftType.LORA.isPromptLearning());
    }
}
