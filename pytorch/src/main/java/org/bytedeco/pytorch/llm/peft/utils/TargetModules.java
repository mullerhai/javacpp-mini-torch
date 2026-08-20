/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.utils;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.constants}.
 *
 * <p>Centralises filenames, layer-name patterns, and other constants used across
 * PEFT. The transformer-specific target-module mappings live in {@link TargetModules}.
 */
public final class TargetModules {

    private TargetModules() {}

    public static final String LORA_PREFIX = "lora_";
    public static final String IA3_PREFIX = "ia3_";
    public static final String OFT_PREFIX = "oft_";
    public static final String BOFT_PREFIX = "boft_";
    public static final String HADA_PREFIX = "hada_";
    public static final String LOKR_PREFIX = "lokr_";
    public static final String VERA_PREFIX = "vera_lambda_";

    /** HuggingFace-compatible transformers target-modules mapping (analog of
     *  TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING). */
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING;
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING;
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING;
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING;
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING;
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_HRA_TARGET_MODULES_MAPPING;
    public static final Map<String, List<String>> TRANSFORMERS_MODELS_TO_BOFT_TARGET_MODULES_MAPPING;

    static {
        Map<String, List<String>> lora = new LinkedHashMap<>();
        lora.put("t5",           java.util.Arrays.asList("q", "k", "v", "o", "wi_0", "wi_1", "wo"));
        lora.put("mt5",          java.util.Arrays.asList("q", "k", "v", "o", "wi_0", "wi_1", "wo"));
        lora.put("bart",         java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"));
        lora.put("mbart",        java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"));
        lora.put("gpt2",         java.util.Arrays.asList("c_attn", "c_proj", "c_fc"));
        lora.put("bloom",        java.util.Arrays.asList("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"));
        lora.put("llama",        java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("mistral",      java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("mixtral",      java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"));
        lora.put("falcon",       java.util.Arrays.asList("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"));
        lora.put("gemma",        java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("phi",          java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"));
        lora.put("phi3",         java.util.Arrays.asList("qkv_proj", "o_proj", "gate_up_proj", "down_proj"));
        lora.put("gpt_neox",     java.util.Arrays.asList("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"));
        lora.put("mpt",          java.util.Arrays.asList("qkv_proj", "out_proj", "up_proj", "down_proj"));
        lora.put("opt",          java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"));
        lora.put("roberta",      java.util.Arrays.asList("query", "key", "value", "dense"));
        lora.put("bert",         java.util.Arrays.asList("query", "key", "value", "dense"));
        lora.put("deberta",      java.util.Arrays.asList("query_proj", "key_proj", "value_proj", "dense"));
        lora.put("deberta-v2",   java.util.Arrays.asList("query_proj", "key_proj", "value_proj", "dense"));
        lora.put("starcoder2",   java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "c_proj"));
        lora.put("qwen2",        java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("qwen2_moe",    java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("codegen",      java.util.Arrays.asList("qkv_proj", "out_proj", "fc_in", "fc_out"));
        lora.put("llama_moe",    java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("dbrx",         java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "v", "wi"));
        lora.put("internlm",     java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("internlm2",    java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("chatglm",      java.util.Arrays.asList("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"));
        lora.put("baichuan",     java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("xglm",         java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "out_proj"));
        lora.put("xllmxglm",     java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "out_proj"));
        lora.put("paligemma",    java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("minicpm",      java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        lora.put("mistral",      java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = lora;

        Map<String, List<String>> ia3 = new LinkedHashMap<>();
        ia3.put("llama",        java.util.Arrays.asList("k_proj", "v_proj", "down_proj"));
        ia3.put("mistral",      java.util.Arrays.asList("k_proj", "v_proj", "down_proj"));
        ia3.put("mixtral",      java.util.Arrays.asList("w2", "w1", "w3"));
        ia3.put("gemma",        java.util.Arrays.asList("k_proj", "v_proj", "down_proj"));
        ia3.put("qwen2",        java.util.Arrays.asList("k_proj", "v_proj", "down_proj"));
        ia3.put("falcon",       java.util.Arrays.asList("query_key_value", "dense_4h_to_h"));
        ia3.put("bloom",        java.util.Arrays.asList("query_key_value", "dense_4h_to_h"));
        ia3.put("gpt2",         java.util.Arrays.asList("c_attn", "c_proj"));
        ia3.put("t5",           java.util.Arrays.asList("k", "o", "wi_1"));
        ia3.put("bart",         java.util.Arrays.asList("k_proj", "out_proj", "fc_out"));
        ia3.put("roberta",      java.util.Arrays.asList("key", "value", "dense"));
        ia3.put("bert",         java.util.Arrays.asList("key", "value", "dense"));
        TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING = ia3;

        Map<String, List<String>> ia3FF = new LinkedHashMap<>();
        ia3FF.put("llama",       java.util.Arrays.asList("down_proj"));
        ia3FF.put("mistral",     java.util.Arrays.asList("down_proj"));
        ia3FF.put("qwen2",       java.util.Arrays.asList("down_proj"));
        TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING = ia3FF;

        Map<String, List<String>> oft = new LinkedHashMap<>();
        oft.put("llama",        java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING = oft;

        Map<String, List<String>> vera = new LinkedHashMap<>();
        vera.put("llama",       java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING = vera;

        Map<String, List<String>> hra = new LinkedHashMap<>();
        hra.put("llama",        java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        TRANSFORMERS_MODELS_TO_HRA_TARGET_MODULES_MAPPING = hra;

        Map<String, List<String>> boft = new LinkedHashMap<>();
        boft.put("llama",       java.util.Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        TRANSFORMERS_MODELS_TO_BOFT_TARGET_MODULES_MAPPING = boft;
    }
}
