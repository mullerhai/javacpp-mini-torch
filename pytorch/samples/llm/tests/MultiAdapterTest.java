/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tests;

import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.IA3Config;
import org.bytedeco.pytorch.llm.peft.helpers.Helpers;
import org.bytedeco.pytorch.llm.peft.utils.SaveAndLoad;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for the multi-adapter lifecycle.
 */
public class MultiAdapterTest {

    @Test
    public void addAndRemoveAdapter() {
        PeftModel model = new PeftModel(null,
                new LoraConfig.Builder().r(8).loraAlpha(16).targetModules("q_proj").build(),
                "default");
        assertEquals(1, model.peftConfigs().size());
        model.addAdapter("adapter_b",
                new LoraConfig.Builder().r(16).loraAlpha(32).targetModules("v_proj").build());
        assertEquals(2, model.peftConfigs().size());
        model.deleteAdapter("adapter_b");
        assertEquals(1, model.peftConfigs().size());
    }

    @Test
    public void mergeUnmergeAdapter() {
        PeftModel model = new PeftModel(null,
                new LoraConfig.Builder().r(8).loraAlpha(16).targetModules("q_proj").build(),
                "default");
        // mergeAdapter / unmergeAdapter should be callable even with a null base in the smoke test.
        model.mergeAdapter();
        model.unmergeAdapter();
    }

    @Test
    public void disableAdapterContext() {
        try (Helpers.DisableAdapter ctx = Helpers.disable(null)) {
            assertNotNull(ctx);
        }
    }
}
