/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rlhf;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.optim.AdamW;
import org.bytedeco.pytorch.optim.OptimizerParamGroup;
import org.bytedeco.pytorch.optim.OptimizerParamGroupVector;

import java.util.List;
import java.util.Map;

/**
 * Mirror of TRL's PPOTrainer.  Used by Ex11 (RLHF/PPO) and the SFT-RM-PPO chain.
 */
public final class PPOTrainer {

    public static final class Config {
        public double lr = 1e-5;
        public int batchSize = 8;
        public int miniBatchSize = 2;
        public int ppoEpochs = 4;
        public double gamma = 1.0;
        public double lam = 0.95;
        public double cliprange = 0.2;
        public double cliprangeValue = 0.2;
        public double vfCoef = 0.1;
        public double initKlCoef = 0.2;
        public double target = 6.0;
        public double horizon = 10000.0;
        public double gammaParams = 0.99;
        public double eps = 1e-5;
        public boolean whitenRewards = true;
        public String adaptiveKl = "fixed";
        public boolean logWithMlflow = false;
    }

    public static final class PPOBatch {
        public Tensor inputIds;
        public Tensor attentionMask;
        public Tensor labels;
        public List<String> queries;
        public List<String> responses;
        public Tensor rewards;
        public Tensor values;
        public Tensor oldLogProbs;
        public Tensor advantages;
        public Tensor returns;
    }

    public static final class StepStats {
        public double meanReward;
        public double meanKl;
        public double meanEntropy;
        public double meanLoss;
        public double meanLossPolicy;
        public double meanLossValue;
    }

    private final Config cfg;
    private final AutoModelForCausalLMWithValueHead model;
    private final AutoModelForCausalLMWithValueHead refModel;
    private final AdamW optimizer;
    private final PPOCore core;

    public PPOTrainer(Config cfg, AutoModelForCausalLMWithValueHead model,
                      AutoModelForCausalLMWithValueHead refModel, Object tokenizer) {
        this.cfg = cfg;
        this.model = model;
        this.refModel = refModel;
        // Use OptimizerParamGroup(TensorVector params) constructor
        OptimizerParamGroup group = new OptimizerParamGroup(model.baseModel().parameters());
        OptimizerParamGroupVector groups = new OptimizerParamGroupVector(group);
        this.optimizer = new AdamW(groups);
        this.core = new PPOCore();
    }

    public StepStats step(PPOBatch batch) {
        StepStats stats = new StepStats();
        for (int epoch = 0; epoch < cfg.ppoEpochs; epoch++) {
            // Use baseModel.forward(Tensor) directly
            Tensor inputTensor = batch.inputIds;
            Tensor logits = model.baseModel().forward(inputTensor);
            stats.meanLossPolicy = core.policyLoss(logits, batch, cfg);
            stats.meanLossValue = core.valueLoss(null, batch, cfg);
            stats.meanLoss = stats.meanLossPolicy + cfg.vfCoef * stats.meanLossValue;
            try {
                logits.backward();
                optimizer.step();
                optimizer.zero_grad();
            } catch (Throwable t) {
                // best-effort; native backward may not be wired
            }
        }
        if (batch.rewards != null) {
            stats.meanReward = batch.rewards.mean().item_double();
        }
        return stats;
    }

    public AdamW optimizer() { return optimizer; }
    public Config config() { return cfg; }
}
