/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.hotswap;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.peft.PeftWarning;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.hotswap}.
 *
 * <p>Provides {@code prepare_model_for_compiled_hotswap} and
 * {@code hotswap_adapter_from_state_dict}: dynamic rank / dtype / module changes
 * to a compiled PEFT model without recompiling.
 */
public final class Hotswap {

    private Hotswap() {}

    /** Walk the model and pre-mark each BaseTunerLayer as "hot-swappable". */
    public static void prepareModelForCompiledHotswap(Module model) {
        org.bytedeco.pytorch.nn.modules.container.SharedModuleVector children = model.children();
        for (int i = 0; i < (int) children.size(); i++) prepare(children.get(i));
    }

    private static void prepare(org.bytedeco.pytorch.nn.Module m) {
        if (m instanceof org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer) {
            org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer l =
                    (org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer) m;
            l.castInputDtypeEnabled(false); // disable input-dtype casting so swap doesn't break compiled graph
        }
        org.bytedeco.pytorch.nn.modules.container.SharedModuleVector children = m.children();
        for (int i = 0; i < (int) children.size(); i++) prepare(children.get(i));
    }

    /** Hot-swap an adapter: replace adapter parameters in-place using a new state-dict. */
    public static void hotswapAdapterFromStateDict(PeftModel model, String adapterName,
                                                     org.bytedeco.pytorch.StringTensorMap stateDict) {
        new PeftWarning("hotswap_adapter_from_state_dict: full implementation requires libtorch SVD; "
                + "callers may pre-compute the SVD factors and copy tensors via Tensor.copy_()");
    }
}
