/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.utils;

import org.bytedeco.pytorch.nn.Module;

import java.util.HashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.transformers_constants}.
 */
public final class TransformersUtils {

    private TransformersUtils() {}

    private static org.bytedeco.javacpp.BytePointer bp(String s) {
        return new org.bytedeco.javacpp.BytePointer(s.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    public static final Map<String, String> TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING;
    static {
        Map<String, String> m = new HashMap<>();
        m.put("bloom", "bloom_model_postprocess_past_key_value");
        m.put("gpt2", "transformers.models.gpt2.modeling_gpt2.gpt2_model_postprocess_past_key_value");
        TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = m;
    }

    public static final Map<String, String> TRANSFORMERS_WEIGHT_CONVERSION_PEFT;
    static {
        Map<String, String> m = new HashMap<>();
        m.put("layer.0.", "transformer.h.0.");
        m.put("attn.c_attn.", "attn.qkv_proj.");
        TRANSFORMERS_WEIGHT_CONVERSION_PEFT = m;
    }

    public static org.bytedeco.pytorch.StringTensorMap convert(org.bytedeco.pytorch.StringTensorMap sd) {
        org.bytedeco.pytorch.StringTensorMap out = new org.bytedeco.pytorch.StringTensorMap();
        org.bytedeco.pytorch.StringTensorMap.Iterator it = sd.begin();
        org.bytedeco.pytorch.StringTensorMap.Iterator end = sd.end();
        while (!it.equals(end)) {
            String oldKey = it.first().getString();
            String newKey = rename(oldKey);
            out.put(bp(newKey), it.second());
            it.increment();
        }
        return out;
    }

    private static String rename(String key) {
        for (Map.Entry<String, String> e : TRANSFORMERS_WEIGHT_CONVERSION_PEFT.entrySet()) {
            if (key.contains(e.getKey())) return key.replace(e.getKey(), e.getValue());
        }
        return key;
    }

    public static org.bytedeco.pytorch.Tensor bloomModelPostprocessPastKeyValue(org.bytedeco.pytorch.Tensor past) {
        return past;
    }

    public static org.bytedeco.pytorch.Tensor fuseMoEExperts(java.util.List<org.bytedeco.pytorch.Tensor> expertWeights) {
        if (expertWeights.isEmpty()) return new org.bytedeco.pytorch.Tensor();
        org.bytedeco.pytorch.TensorVector v = new org.bytedeco.pytorch.TensorVector();
        for (org.bytedeco.pytorch.Tensor t : expertWeights) v.push_back(t);
        return org.bytedeco.pytorch.global.torch.stack(v, 0);
    }
}
