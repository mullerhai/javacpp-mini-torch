/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

/**
 * Ex07 — Falcon-7B QLoRA instruction tuning.
 */
public final class Ex07_Falcon7BQLoRA {

    private Ex07_Falcon7BQLoRA() {}

    public static void main(String[] args) {
        run();
    }

    public static void run() {
        TunningSupport.banner(7, "Falcon-7B QLoRA Instruction Tuning");

        // 1. QLoRA NF4 quantization config
        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .bnb4BitUseDoubleQuant(true)
                .bnb4BitComputeDtype("bfloat16")
                .build();

        // 2. Base model (stub in offline mode)
        CausalLM model = CausalLM.fromConfig(PretrainedConfig.builder()
                .torchDtype("bfloat16")
                .build());

        // 3. LoRA config
        LoraConfig loraConfig = LoraConfig.builder()
                .r(16)
                .alpha(32)
                .dropout(0.05)
                .targetModules(Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"))
                .bias("none")
                .taskType("CAUSAL_LM")
                .build();

        // 4. Wrap base model with PEFT
        PeftModel peft = PeftModel.getPeftModel(model, loraConfig);
        peft.printTrainableParameters();

        // 5. Tokenizer
        FastTokenizer tok = TunningSupport.tokenizerFor("tiiuae/falcon-7b-instruct");

        // 6. SFTConfig
        SFTConfig sftConfig = SFTConfig.builder()
                .maxSeqLength(2048)
                .maxSteps(500)
                .learningRate(2e-4)
                .warmupRatio(0.03)
                .lrSchedulerType("cosine")
                .loggingSteps(10)
                .bf16(true)
                .packing(false)
                .datasetTextField("text")
                .build();

        // 7. Dataset (in-memory fallback)
        List<Map<String, Object>> trainData = TunningSupport.loadDataset("yahma/alpaca-cleaned", null, "train", 512);

        // 8. Create SFT Trainer
        SFTTrainer trainer = SFTTrainer.of(model, sftConfig, trainData, loraConfig, tok, "text");

        // 9. Run training
        try {
            trainer.train();
            System.out.println("Training completed. Steps: " + trainer.globalStep());
        } catch (Exception e) {
            System.err.println("Training error (expected in stub mode): " + e.getMessage());
        }

        // 10. Save adapter
        try {
            peft.savePretrained(System.getProperty("java.io.tmpdir") + "/falcon_qlora_adapter");
            System.out.println("Adapter saved.");
        } catch (Exception e) {
            System.out.println("Save skipped (stub mode).");
        }

        trainer.close();
    }
}
