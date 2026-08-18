/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;

import org.bytedeco.pytorch.llm.bitsandbytes.BitsAndBytes;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.trl.trainer.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

/** Ex06 — Facebook OPT-6.1B QLoRA with LoRA + SFTTrainer. */
public final class Ex06_OptLoRA {
    public static final String NAME = "Ex06_OptLoRA";

    public static void run(FastTokenizer tokenizer) throws IOException {
        TunningSupport.banner(6, "OPT-6.1B QLoRA");
        File out = new File("build/ex06_outputs"); out.mkdirs();
        HfHub hub = HfHub.fromEnv();
        FastTokenizer tok = AutoTokenizer.fromPretrainedTokenizerOnly("facebook/opt-6.7b", hub);
        PretrainedConfig cfg = PretrainedConfig.builder()
                .vocabSize(50272)
                .hiddenSize(4096)
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        LoraConfig peft = LoraConfig.builder()
                .r(16).alpha(32).dropout(0.05)
                .targetModules("q_proj", "v_proj").bias("none").taskType("CAUSAL_LM").build();

        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .build();
        BitsAndBytes.prepareModelForKBitTraining(model);

        SFTConfig sftConfig = SFTConfig.builder()
                .maxSteps(200)
                .perDeviceTrainBatchSize(4)
                .gradientAccumulationSteps(4)
                .numTrainEpochs(1)
                .learningRate(2e-4)
                .fp16(true)
                .warmupRatio(0.03)
                .lrSchedulerType("constant")
                .optim("paged_adamw_32bit")
                .maxSeqLength(512)
                .datasetTextField("text")
                .build();

        java.util.List<java.util.Map<String, Object>> trainData =
                TunningSupport.loadDataset("yahma/alpaca-cleaned", null, "train", 1024);
        Function<java.util.Map<String, Object>, java.util.Map<String, Object>> fmt =
                TunningSupport.sftFormattingFunc(TunningSupport::alpacaPrompt, tok, sftConfig.maxSeqLength(), true);
        java.util.List<java.util.Map<String, Object>> tokenizedData = trainData.stream().map(fmt::apply).toList();

        SFTTrainer trainer = SFTTrainer.of(model, sftConfig, tokenizedData, peft, tok, "text");
        trainer.train();
        PeftModel.getPeftModel(model, peft).savePretrained(out);
        System.out.println("Trained " + trainer.globalStep() + " steps");
    }

    public static void main(String[] args) throws IOException { run(TunningSupport.tokenizerFor("facebook/opt-6.7b")); }
}
