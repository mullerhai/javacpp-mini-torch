/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tests;

import org.bytedeco.pytorch.llm.peft.utils.MergeUtils;
import org.bytedeco.pytorch.llm.peft.utils.Other;
import org.bytedeco.pytorch.llm.peft.utils.IncrementalPCA;
import org.bytedeco.pytorch.llm.peft.utils.TargetModules;
import org.bytedeco.pytorch.llm.peft.utils.TransformersUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for the utilities package.
 */
public class UtilsTest {

    @Test
    public void targetModulesHasLlama() {
        assertTrue(TargetModules.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.containsKey("llama"));
    }

    @Test
    public void otherUtilsExposed() {
        assertNotNull(Other.class);
        assertNotNull(MergeUtils.class);
        assertNotNull(IncrementalPCA.class);
        assertNotNull(TransformersUtils.class);
    }

    @Test
    public void transformersUtilsRename() {
        // bloom conversion should not crash and should still expose a non-null mapping.
        assertTrue(TargetModules.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.containsKey("bloom"));
    }
}
