/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 * You may not use this file except in compliance with the License.
 */
package org.bytedeco.pytorch.llm.peft;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Base configuration for prompt-learning PEFT methods (Prompt Tuning, Prefix Tuning,
 * P-Tuning, Multitask Prompt Tuning, CPT). Mirrors HuggingFace
 * {@code peft.config.PromptLearningConfig}.
 *
 * <p>Concrete classes must use {@link Builder} pattern and call
 * {@link #postBuild(Object)} to validate derived fields.
 */
public abstract class PromptLearningConfig extends PeftConfig {

    protected long numVirtualTokens = 0;
    protected long tokenDim = 0;
    protected long numLayers = 0;
    protected long numAttentionHeads = 0;
    protected Object layersToTransform = null;
    protected java.util.List<String> layersPattern = null;
    protected java.util.List<String> modulesToSave = null;
    protected boolean inferenceMode = false;

    protected PromptLearningConfig(Builder<?, ?> b) {
        super((PeftConfig.Builder<?>) b);
        this.numVirtualTokens = b.numVirtualTokens;
        this.tokenDim = b.tokenDim;
        this.numLayers = b.numLayers;
        this.numAttentionHeads = b.numAttentionHeads;
        this.layersToTransform = b.layersToTransform;
        this.layersPattern = b.layersPattern;
        this.modulesToSave = b.modulesToSave;
        this.inferenceMode = b.inferenceMode;
    }

    public long numVirtualTokens() { return numVirtualTokens; }
    public void numVirtualTokens(long v) { this.numVirtualTokens = v; }
    public long tokenDim() { return tokenDim; }
    public void tokenDim(long v) { this.tokenDim = v; }
    public long numLayers() { return numLayers; }
    public void numLayers(long v) { this.numLayers = v; }
    public long numAttentionHeads() { return numAttentionHeads; }
    public void numAttentionHeads(long v) { this.numAttentionHeads = v; }
    public Object layersToTransform() { return layersToTransform; }
    public void layersToTransform(Object v) { this.layersToTransform = v; }
    public java.util.List<String> layersPattern() { return layersPattern; }
    public void layersPattern(java.util.List<String> v) { this.layersPattern = v; }
    public java.util.List<String> modulesToSave() { return modulesToSave; }
    public void modulesToSave(java.util.List<String> v) { this.modulesToSave = v; }
    public boolean inferenceMode() { return inferenceMode; }
    public void inferenceMode(boolean v) { this.inferenceMode = v; }

    /** Translate a python-style regex/suffix pattern key into a regex String. */
    public static java.util.regex.Pattern patternFromKey(String key) {
        if (key == null || key.isEmpty()) return java.util.regex.Pattern.compile(".*");
        String regex = key.replace(".", "\\.").replace("**", ".*").replace("*", "[^.]*");
        return java.util.regex.Pattern.compile(regex);
    }

    /** Return a deep-copy of this config via the Builder (HF {@code copy()}). */
    public abstract PromptLearningConfig copy();

    @Override
    public Map<String, Object> toDict() {
        Map<String, Object> base = super.toDict();
        base.put("num_virtual_tokens", numVirtualTokens);
        base.put("token_dim", tokenDim);
        base.put("num_layers", numLayers);
        base.put("num_attention_heads", numAttentionHeads);
        if (layersToTransform != null) base.put("layers_to_transform", layersToTransform);
        if (layersPattern != null) base.put("layers_pattern", layersPattern);
        if (modulesToSave != null) base.put("modules_to_save", modulesToSave);
        return base;
    }

    static long toLong(Object o) {
        if (o instanceof Number) return ((Number) o).longValue();
        return Long.parseLong(String.valueOf(o));
    }

    @SuppressWarnings("unchecked")
    static java.util.List<String> toStringList(Object o) {
        if (o instanceof java.util.List) return new java.util.ArrayList<>((java.util.List<String>) o);
        if (o instanceof String) return new java.util.ArrayList<>(java.util.Arrays.asList((String) o));
        return new java.util.ArrayList<>();
    }

    /** Common builder for prompt-learning configs. */
    public abstract static class Builder<SELF extends Builder<SELF, C>, C extends PromptLearningConfig>
            extends PeftConfig.Builder<SELF> {
        protected long numVirtualTokens = 0;
        protected long tokenDim = 0;
        protected long numLayers = 0;
        protected long numAttentionHeads = 0;
        protected Object layersToTransform = null;
        protected java.util.List<String> layersPattern = null;
        protected java.util.List<String> modulesToSave = null;
        protected boolean inferenceMode = false;

        @Override
        @SuppressWarnings("unchecked")
        public SELF self() { return (SELF) this; }

        public SELF numVirtualTokens(long v) { this.numVirtualTokens = v; return self(); }
        public SELF tokenDim(long v) { this.tokenDim = v; return self(); }
        public SELF numLayers(long v) { this.numLayers = v; return self(); }
        public SELF numAttentionHeads(long v) { this.numAttentionHeads = v; return self(); }
        public SELF layersToTransform(Object v) { this.layersToTransform = v; return self(); }
        public SELF layersPattern(java.util.List<String> v) { this.layersPattern = v; return self(); }
        public SELF modulesToSave(java.util.List<String> v) { this.modulesToSave = v; return self(); }
        public SELF inferenceMode(boolean v) { this.inferenceMode = v; return self(); }

        @SuppressWarnings("unchecked")
        @Override
        public C build() {
            throw new UnsupportedOperationException(
                "Subclasses of PromptLearningConfig must override build()");
        }
    }

    /** Convert a String|List<String> into a normalised list of patterns. */
    public static java.util.List<String> normaliseStringOrList(Object o) {
        if (o == null) return null;
        if (o instanceof java.util.List) return new java.util.ArrayList<>((java.util.List<String>) o);
        return new java.util.ArrayList<>(java.util.Arrays.asList(String.valueOf(o)));
    }

    /** Resolve the effective layer-transformation index list from {@link #layersToTransform}. */
    public java.util.List<Integer> resolveLayersToTransform(int numTotalLayers) {
        java.util.List<Integer> result = new java.util.ArrayList<>();
        if (layersToTransform == null) {
            for (int i = 0; i < numTotalLayers; i++) result.add(i);
            return result;
        }
        if (layersToTransform instanceof Number) {
            int k = ((Number) layersToTransform).intValue();
            for (int i = 0; i < numTotalLayers; i++) {
                if (i >= k) result.add(i);
            }
            return result;
        }
        if (layersToTransform instanceof java.util.List) {
            for (Object v : (java.util.List<?>) layersToTransform) result.add(((Number) v).intValue());
            return result;
        }
        throw new IllegalArgumentException("layersToTransform must be int, list, or null");
    }
}