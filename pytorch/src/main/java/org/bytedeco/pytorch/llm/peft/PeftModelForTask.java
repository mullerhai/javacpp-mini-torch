/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.peft_model.PeftModelForCausalLM} / {@code SeqCls} / etc.
 *
 * <p>The base {@link PeftModel} already covers the multi-adapter lifecycle; this class adds
 * task-specific forward helpers (generation, classifier head, QA head). For most tasks
 * a simple {@code model.forward(input)} is sufficient; we expose convenience signatures
 * matching HuggingFace to keep drop-in parity.
 */
public class PeftModelForTask {

    public static final String CAUSAL_LM = "CAUSAL_LM";
    public static final String SEQ_CLS = "SEQ_CLS";
    public static final String TOKEN_CLS = "TOKEN_CLS";
    public static final String SEQ_2_SEQ = "SEQ_2_SEQ";
    public static final String QUESTION_ANS = "QUESTION_ANS";
    public static final String FEATURE_EXTRACTION = "FEATURE_EXTRACTION";

    /** Forward dispatch used by the {@code transformers.Trainer} integration. */
    public static org.bytedeco.pytorch.Tensor forward(Module model, String taskType, Object... inputs) {
        return model.forward(inputs.length == 1 ? (org.bytedeco.pytorch.Tensor) inputs[0]
                                                 : (org.bytedeco.pytorch.Tensor) inputs[0]);
    }
}
