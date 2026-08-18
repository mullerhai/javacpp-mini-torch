/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.io.File;
import java.io.IOException;
import java.util.Map;
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

/** Ex08 — StableVicuna 13B QLoRA + StableVicuna chat template. */
public final class Ex08_StableVicuna {
    public static final String NAME = "Ex08_StableVicuna";

    public static void run(FastTokenizer tokenizer) throws IOException {
        TunningSupport.banner(8, "StableVicuna 13B QLoRA");
        File out = new File("build/ex08_outputs"); out.mkdirs();
        FastTokenizer tok = AutoTokenizer.fromPretrainedTokenizerOnly("CarperAI/StableVicuna-13B");
        PretrainedConfig cfg = PretrainedConfig.builder()
                .vocabSize(32000)
                .hiddenSize(4096)
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        LoraConfig peft = LoraConfig.builder()
                .r(16).alpha(32).dropout(0.05)
                .targetModules("q_proj", "k_proj", "v_proj", "o_proj")
                .bias("none").taskType("CAUSAL_LM").build();

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
                .saveSteps(50)
                .loggingSteps(10)
                .maxSeqLength(512)
                .datasetTextField("text")
                .build();

        java.util.List<Map<String, Object>> trainData =
                TunningSupport.loadDataset("yahma/alpaca-cleaned", null, "train", 1024);
        java.util.List<Map<String, Object>> formattedData = trainData.stream().map(row -> {
            String t = TunningSupport.stableVicunaPrompt(TunningSupport.chatMessages(
                    "system", "You are a helpful assistant.",
                    "user", String.valueOf(row.get("instruction")),
                    "assistant", String.valueOf(row.get("output"))));
            Map<String, Object> r = new java.util.LinkedHashMap<>(row);
            r.put("text", t);
            return r;
        }).toList();
        Function<Map<String, Object>, Map<String, Object>> fmt =
                TunningSupport.sftFormattingFunc(row -> String.valueOf(row.get("text")), tok, sftConfig.maxSeqLength(), true);
        java.util.List<Map<String, Object>> tokenizedData = formattedData.stream().map(fmt::apply).toList();

        SFTTrainer trainer = SFTTrainer.of(model, sftConfig, tokenizedData, peft, tok, "text");
        trainer.train();
        PeftModel.getPeftModel(model, peft).savePretrained(out);
        System.out.println("Trained " + trainer.globalStep() + " steps");
    }

    public static void main(String[] args) throws IOException { run(TunningSupport.tokenizerFor("CarperAI/StableVicuna-13B")); }
}
