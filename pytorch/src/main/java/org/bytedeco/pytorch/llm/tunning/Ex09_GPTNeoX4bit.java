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
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.trl.trainer.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

/** Ex09 — GPT-NeoX-20B 4-bit QLoRA. */
public final class Ex09_GPTNeoX4bit {
    public static final String NAME = "Ex09_GPTNeoX4bit";

    public static void run(FastTokenizer tokenizer) throws IOException {
        TunningSupport.banner(9, "GPT-NeoX-20B 4-bit");
        File out = new File("build/ex09_outputs"); out.mkdirs();
        FastTokenizer tok = AutoTokenizer.fromPretrainedTokenizerOnly("EleutherAI/gpt-neox-20b");
        PretrainedConfig cfg = PretrainedConfig.builder()
                .vocabSize(50304)
                .hiddenSize(6144)
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        LoraConfig peft = LoraConfig.builder()
                .r(8).alpha(16).dropout(0.05)
                .targetModules("query_key_value", "dense").bias("none").taskType("CAUSAL_LM").build();

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

    public static void main(String[] args) throws IOException { run(TunningSupport.tokenizerFor("EleutherAI/gpt-neox-20b")); }
}
