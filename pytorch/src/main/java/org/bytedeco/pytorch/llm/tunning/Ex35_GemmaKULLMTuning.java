/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning-Tutorial authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.Arrays;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;

/**
 * Ex35 — Gemma-2B fine-tuning on KULLM-V2 dataset.
 *
 * <p>Maps to: LoRA + QLoRA + SFTConfig configuration
 */
public final class Ex35_GemmaKULLMTuning {

    public static void run() {
        TunningSupport.banner(35, "Gemma-2B + KULLM-V2 LoRA Fine-tuning");

        // Tokenizer
        FastTokenizer tok = TunningSupport.tokenizerFor("beomi/gemma-ko-2b");

        // Model config
        PretrainedConfig cfg = PretrainedConfig.builder()
                .torchDtype("bfloat16")
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        // BnB config for 4-bit quantization (QLoRA)
        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .bnb4BitComputeDtype("bfloat16")
                .build();

        // LoRA config
        LoraConfig loraCfg = LoraConfig.builder()
                .taskType("CAUSAL_LM")
                .r(16)
                .alpha(32)
                .dropout(0.05)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj"))
                .build();

        PeftModel peft = PeftModel.getPeftModel(model, loraCfg);
        peft.printTrainableParameters();

        // SFTConfig
        SFTConfig sftConfig = SFTConfig.builder()
                .learningRate(2e-4)
                .maxSteps(3000)
                .loggingSteps(20)
                .gradientAccumulationSteps(8)
                .bf16(true)
                .maxSeqLength(600)
                .build();

        System.out.println("[Ex35] Gemma-2B + QLoRA + KULLM-V2 configuration ready");
        System.out.println("[Ex35] LoRA config: " + loraCfg);
        System.out.println("[Ex35] SFT config: " + sftConfig);
    }

    private Ex35_GemmaKULLMTuning() {}

    public static void main(String[] args) {
        run();
    }
}
