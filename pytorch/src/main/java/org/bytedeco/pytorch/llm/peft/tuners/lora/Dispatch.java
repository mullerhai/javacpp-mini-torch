/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv3dImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.MultiheadAttentionImpl;

/**
 * Maps a host model module to the appropriate {@link LoraLayer} subclass.
 *
 * <p>Mirrors HuggingFace {@code peft.tuners.lora.layer.dispatch_default}.
 */
public final class Dispatch {

    private Dispatch() {}

    public static LoraLayer create(Class<?> targetClass, Module target, String name) {
        if (LinearImpl.class.isAssignableFrom(targetClass)) {
            return new LoraLinear((LinearImpl) target, name);
        }
        if (EmbeddingImpl.class.isAssignableFrom(targetClass)) {
            return new LoraEmbedding((EmbeddingImpl) target, name);
        }
        if (Conv2dImpl.class.isAssignableFrom(targetClass)) {
            return new LoraConv2d((Conv2dImpl) target, name);
        }
        if (Conv1dImpl.class.isAssignableFrom(targetClass)) {
            return new LoraConv1d((Conv1dImpl) target, name);
        }
        if (Conv3dImpl.class.isAssignableFrom(targetClass)) {
            return new LoraConv3d((Conv3dImpl) target, name);
        }
        if (MultiheadAttentionImpl.class.isAssignableFrom(targetClass)) {
            return new LoraMHA((MultiheadAttentionImpl) target, name);
        }
        // ParamWrapper fallback
        TensorHolder th = new TensorHolder(target);
        return new LoraParamWrapper(name, th.firstParam());
    }

    /** Tiny helper to fetch the first parameter of a module without exposing nn.Parameter type. */
    public static class TensorHolder {
        private final Module m;
        public TensorHolder(Module m) { this.m = m; }
        public org.bytedeco.pytorch.Tensor firstParam() {
            try {
                if (m.parameters() != null && m.parameters().size() > 0) {
                    return m.parameters().get(0);
                }
            } catch (Exception ignored) {}
            return null;
        }
    }

    /** Matchers for forward signature dispatch. */
    public static final java.util.regex.Pattern MATCH_LINEAR = java.util.regex.Pattern.compile("linear|lora_.*linear");
    public static final java.util.regex.Pattern MATCH_EMBED = java.util.regex.Pattern.compile("embed");
    public static final java.util.regex.Pattern MATCH_CONV1D = java.util.regex.Pattern.compile("conv1d");
    public static final java.util.regex.Pattern MATCH_CONV2D = java.util.regex.Pattern.compile("conv2d");
    public static final java.util.regex.Pattern MATCH_CONV3D = java.util.regex.Pattern.compile("conv3d");
    public static final java.util.regex.Pattern MATCH_MHA = java.util.regex.Pattern.compile("multihead");
    public static final java.util.regex.Pattern MATCH_PARAM = java.util.regex.Pattern.compile(".*");
}