/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.io.File;
import java.io.IOException;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.llm.trl.config.RewardConfig;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;

/**
 * Ex11 — RLHF (Reward Modeling + PPO + DPO) end-to-end configuration.
 */
public final class Ex11_RLHFPPO {
    public static final String NAME = "Ex11_RLHFPPO";

    public static void run(FastTokenizer tokenizer) throws IOException {
        TunningSupport.banner(11, "RLHF (Reward Modeling + PPO + DPO)");
        File out = new File("build/ex11_outputs"); out.mkdirs();
        FastTokenizer tok = AutoTokenizer.fromPretrainedTokenizerOnly("EleutherAI/gpt-neox-20b");

        PretrainedConfig cfg = PretrainedConfig.builder()
                .vocabSize(50432)
                .hiddenSize(1024)
                .build();
        CausalLM base = CausalLM.fromConfig(cfg);

        LoraConfig peft = LoraConfig.builder()
                .r(8).alpha(16).dropout(0.05)
                .targetModules("q_proj", "v_proj").bias("none").taskType("CAUSAL_LM").build();
        PeftModel peftBase = PeftModel.getPeftModel(base, peft);

        // Stage 1: Reward modeling config
        RewardConfig rewardArgs = RewardConfig.builder()
                .learningRate(2e-5).perDeviceTrainBatchSize(4)
                .gradientAccumulationSteps(1).numTrainEpochs(1)
                .outputDir(out.toString() + "/reward").build();

        // Stage 2: PPO config
        PPOConfig ppoArgs = PPOConfig.builder()
                .learningRate(1.4e-5).perDeviceTrainBatchSize(2)
                .gradientAccumulationSteps(2).numTrainEpochs(1)
                .outputDir(out.toString() + "/ppo").build();

        // Stage 3: DPO config
        DPOConfig dpoArgs = DPOConfig.builder()
                .learningRate(5e-7).perDeviceTrainBatchSize(2)
                .gradientAccumulationSteps(2).numTrainEpochs(1)
                .outputDir(out.toString() + "/dpo").build();

        // Stage 4: SFT config
        SFTConfig sftArgs = SFTConfig.builder()
                .learningRate(2e-4).perDeviceTrainBatchSize(4)
                .maxSteps(200).outputDir(out.toString() + "/sft").build();

        System.out.println("RLHF Configs:");
        System.out.println("  Reward: " + rewardArgs);
        System.out.println("  PPO: " + ppoArgs);
        System.out.println("  DPO: " + dpoArgs);
        System.out.println("  SFT: " + sftArgs);
        System.out.println("PEFT base: " + peftBase);
    }

    public static void main(String[] args) throws IOException { run(TunningSupport.tokenizerFor("EleutherAI/gpt-neox-20b")); }
}
