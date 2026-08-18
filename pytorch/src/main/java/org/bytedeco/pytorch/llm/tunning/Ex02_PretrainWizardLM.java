/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
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

/**
 * Ex02 — Pretrain WizardLM Causal LM from scratch.
 */
public final class Ex02_PretrainWizardLM {

    public static final String NAME = "Ex02_PretrainWizardLM";

    public static void run(FastTokenizer tokenizer) throws IOException {
        TunningSupport.banner(2, "Pretrain WizardLM causal LM");
        File out = new File("build/ex02_outputs"); out.mkdirs();

        HfHub hub = HfHub.fromEnv();
        FastTokenizer tok = AutoTokenizer.fromPretrainedTokenizerOnly("WizardLM/WizardLM-7B", hub);
        PretrainedConfig cfg = PretrainedConfig.builder()
                .vocabSize(32000)
                .hiddenSize(4096)
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        LoraConfig peftConfig = LoraConfig.builder()
                .r(8).alpha(16).dropout(0.05)
                .targetModules("q_proj", "v_proj")
                .bias("none")
                .taskType("CAUSAL_LM")
                .build();

        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .build();
        BitsAndBytes.prepareModelForKBitTraining(model);

        SFTConfig sftConfig = SFTConfig.builder()
                .maxSteps(1500)
                .perDeviceTrainBatchSize(8)
                .gradientAccumulationSteps(8)
                .numTrainEpochs(3)
                .saveSteps(200)
                .loggingSteps(20)
                .learningRate(3e-4)
                .fp16(true)
                .warmupRatio(0.03)
                .lrSchedulerType("linear")
                .maxSeqLength(1024)
                .datasetTextField("text")
                .build();

        List<Map<String, Object>> trainData = TunningSupport.loadDataset("WizardLMTeam/WizardLM_evol_instruct_70k",
                null, "train", 1024);
        Function<Map<String, Object>, Map<String, Object>> fmt =
                TunningSupport.sftFormattingFunc(TunningSupport::alpacaPrompt, tok, 1024, true);
        List<Map<String, Object>> tokenizedData = trainData.stream().map(fmt::apply).toList();

        SFTTrainer trainer = SFTTrainer.of(model, sftConfig, tokenizedData, peftConfig, tok, "text");
        trainer.train();

        PeftModel peft = PeftModel.getPeftModel(model, peftConfig);
        peft.savePretrained(out);
        System.out.println("Trained " + trainer.globalStep() + " steps, saved to " + out);
    }

    public static void main(String[] args) throws IOException {
        run(TunningSupport.tokenizerFor("WizardLM/WizardLM-7B"));
    }
}
