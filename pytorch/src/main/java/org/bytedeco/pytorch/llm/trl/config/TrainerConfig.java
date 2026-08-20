/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.config;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Shared training hyper-parameters for TRL-style trainers.
 *
 * <p>Mirrors HuggingFace Transformers {@code TrainingArguments} with additions for
 * QLoRA, gradient checkpointing, mixed precision, and learning rate scheduling.
 *
 * <p>All fields have sensible defaults matching the Python library conventions.
 */
public class TrainerConfig {
    // -------------------------------------------------------------------------
    // Core training parameters
    // -------------------------------------------------------------------------
    private final double learningRate;
    private final double weightDecay;
    private final int maxSteps;
    private final int numTrainEpochs;
    private final int loggingSteps;
    private final int saveSteps;
    private final int evalSteps;
    private final int warmupSteps;
    private final double warmupRatio;
    private final int gradientAccumulationSteps;
    private final double maxGradNorm;
    private final int perDeviceTrainBatchSize;
    private final int perDeviceEvalBatchSize;
    private final String lrSchedulerType;
    private final double lrSchedulerGamma;
    private final double lrSchedulerEta;
    private final double lrSchedulerPower;
    private final int lrSchedulerWarmupSteps;
    private final boolean lrSchedulerLinear;

    // -------------------------------------------------------------------------
    // Precision and device
    // -------------------------------------------------------------------------
    private final boolean fp16;
    private final boolean bf16;
    private final boolean fp8;
    private final String torchDtype;
    private final boolean gradientCheckpointing;
    private final double gradientCheckpointingRatio;
    private final String deviceMap;

    // -------------------------------------------------------------------------
    // Optimization
    // -------------------------------------------------------------------------
    private final String optim;
    private final double optimArgs;
    private final double adamBeta1;
    private final double adamBeta2;
    private final double adamEpsilon;
    private final double adamWepsilon;
    private final boolean fusedOptimizer;
    private final boolean pagedOptimizer;
    private final boolean useLoraKernel;

    // -------------------------------------------------------------------------
    // Dataset and collator
    // -------------------------------------------------------------------------
    private final int maxSeqLength;
    private final long ignoreIndex;
    private final boolean packing;
    private final String datasetTextField;
    private final boolean appendConcatToken;
    private final String datasetProcessingNumProc;
    private final int maxPackedSequences;

    // -------------------------------------------------------------------------
    // Output and checkpoints
    // -------------------------------------------------------------------------
    private final String outputDir;
    private final String saveTotalLimit;
    private final int saveOnlyModel;
    private final boolean saveOnlyBest;
    private final boolean loadBestModelAtEnd;
    private final boolean saveState;
    private final boolean saveFunction;
    private final boolean useLEGACYOptimScheduler;
    private final boolean useFlashAttn;

    // -------------------------------------------------------------------------
    // Logging and metrics
    // -------------------------------------------------------------------------
    private final int reportTo;
    private final boolean logCompletions;
    private final String metrics;

    // -------------------------------------------------------------------------
    // Seed and determinism
    // -------------------------------------------------------------------------
    private final long seed;
    private final boolean dataSeed;
    private final boolean deterministic;

    // -------------------------------------------------------------------------
    // Misc
    // -------------------------------------------------------------------------
    private final double neftuneAlpha;
    private final double runName;
    private final boolean groupByLength;
    private final int lengthColumnName;
    private final boolean ddpTimeout;
    private final int ddpBucketCapMb;
    private final String ddpFindUnused;
    private final boolean localRank;

    // Margin for ranking losses
    private final double margin;

    protected TrainerConfig(Builder<?> b) {
        // Core training
        this.learningRate = b.learningRate;
        this.weightDecay = b.weightDecay;
        this.maxSteps = b.maxSteps;
        this.numTrainEpochs = b.numTrainEpochs;
        this.loggingSteps = b.loggingSteps;
        this.saveSteps = b.saveSteps;
        this.evalSteps = b.evalSteps;
        this.warmupSteps = b.warmupSteps;
        this.warmupRatio = b.warmupRatio;
        this.gradientAccumulationSteps = b.gradientAccumulationSteps;
        this.maxGradNorm = b.maxGradNorm;
        this.perDeviceTrainBatchSize = b.perDeviceTrainBatchSize;
        this.perDeviceEvalBatchSize = b.perDeviceEvalBatchSize;
        this.lrSchedulerType = b.lrSchedulerType;
        this.lrSchedulerGamma = b.lrSchedulerGamma;
        this.lrSchedulerEta = b.lrSchedulerEta;
        this.lrSchedulerPower = b.lrSchedulerPower;
        this.lrSchedulerWarmupSteps = b.lrSchedulerWarmupSteps;
        this.lrSchedulerLinear = b.lrSchedulerLinear;

        // Precision
        this.fp16 = b.fp16;
        this.bf16 = b.bf16;
        this.fp8 = b.fp8;
        this.torchDtype = b.torchDtype;
        this.gradientCheckpointing = b.gradientCheckpointing;
        this.gradientCheckpointingRatio = b.gradientCheckpointingRatio;
        this.deviceMap = b.deviceMap;

        // Optimization
        this.optim = b.optim;
        this.optimArgs = b.optimArgs;
        this.adamBeta1 = b.adamBeta1;
        this.adamBeta2 = b.adamBeta2;
        this.adamEpsilon = b.adamEpsilon;
        this.adamWepsilon = b.adamWepsilon;
        this.fusedOptimizer = b.fusedOptimizer;
        this.pagedOptimizer = b.pagedOptimizer;
        this.useLoraKernel = b.useLoraKernel;

        // Dataset
        this.maxSeqLength = b.maxSeqLength;
        this.ignoreIndex = b.ignoreIndex;
        this.packing = b.packing;
        this.datasetTextField = b.datasetTextField;
        this.appendConcatToken = b.appendConcatToken;
        this.datasetProcessingNumProc = b.datasetProcessingNumProc;
        this.maxPackedSequences = b.maxPackedSequences;

        // Output
        this.outputDir = b.outputDir;
        this.saveTotalLimit = b.saveTotalLimit;
        this.saveOnlyModel = b.saveOnlyModel;
        this.saveOnlyBest = b.saveOnlyBest;
        this.loadBestModelAtEnd = b.loadBestModelAtEnd;
        this.saveState = b.saveState;
        this.saveFunction = b.saveFunction;
        this.useLEGACYOptimScheduler = b.useLEGACYOptimScheduler;
        this.useFlashAttn = b.useFlashAttn;

        // Logging
        this.reportTo = b.reportTo;
        this.logCompletions = b.logCompletions;
        this.metrics = b.metrics;

        // Seed
        this.seed = b.seed;
        this.dataSeed = b.dataSeed;
        this.deterministic = b.deterministic;

        // Misc
        this.neftuneAlpha = b.neftuneAlpha;
        this.runName = b.runName;
        this.groupByLength = b.groupByLength;
        this.lengthColumnName = b.lengthColumnName;
        this.ddpTimeout = b.ddpTimeout;
        this.ddpBucketCapMb = b.ddpBucketCapMb;
        this.ddpFindUnused = b.ddpFindUnused;
        this.localRank = b.localRank;

        // Margin
        this.margin = b.margin;
    }

    // -------------------------------------------------------------------------
    // Accessors (Getters)
    // -------------------------------------------------------------------------

    // Core training
    public double learningRate() { return learningRate; }
    public double weightDecay() { return weightDecay; }
    public int maxSteps() { return maxSteps; }
    public int numTrainEpochs() { return numTrainEpochs; }
    public int loggingSteps() { return loggingSteps; }
    public int saveSteps() { return saveSteps; }
    public int evalSteps() { return evalSteps; }
    public int warmupSteps() { return warmupSteps; }
    public double warmupRatio() { return warmupRatio; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public double maxGradNorm() { return maxGradNorm; }
    public int perDeviceTrainBatchSize() { return perDeviceTrainBatchSize; }
    public int perDeviceEvalBatchSize() { return perDeviceEvalBatchSize; }
    public String lrSchedulerType() { return lrSchedulerType; }
    public double lrSchedulerGamma() { return lrSchedulerGamma; }
    public double lrSchedulerEta() { return lrSchedulerEta; }
    public double lrSchedulerPower() { return lrSchedulerPower; }
    public int lrSchedulerWarmupSteps() { return lrSchedulerWarmupSteps; }
    public boolean lrSchedulerLinear() { return lrSchedulerLinear; }

    // Precision
    public boolean fp16() { return fp16; }
    public boolean bf16() { return bf16; }
    public boolean fp8() { return fp8; }
    public String torchDtype() { return torchDtype; }
    public boolean gradientCheckpointing() { return gradientCheckpointing; }
    public double gradientCheckpointingRatio() { return gradientCheckpointingRatio; }
    public String deviceMap() { return deviceMap; }

    // Optimization
    public String optim() { return optim; }
    public double optimArgs() { return optimArgs; }
    public double adamBeta1() { return adamBeta1; }
    public double adamBeta2() { return adamBeta2; }
    public double adamEpsilon() { return adamEpsilon; }
    public double adamWepsilon() { return adamWepsilon; }
    public boolean fusedOptimizer() { return fusedOptimizer; }
    public boolean pagedOptimizer() { return pagedOptimizer; }
    public boolean useLoraKernel() { return useLoraKernel; }

    // Dataset
    public int maxSeqLength() { return maxSeqLength; }
    public long ignoreIndex() { return ignoreIndex; }
    public boolean packing() { return packing; }
    public String datasetTextField() { return datasetTextField; }
    public boolean appendConcatToken() { return appendConcatToken; }
    public String datasetProcessingNumProc() { return datasetProcessingNumProc; }
    public int maxPackedSequences() { return maxPackedSequences; }

    // Output
    public String outputDir() { return outputDir; }
    public String saveTotalLimit() { return saveTotalLimit; }
    public int saveOnlyModel() { return saveOnlyModel; }
    public boolean saveOnlyBest() { return saveOnlyBest; }
    public boolean loadBestModelAtEnd() { return loadBestModelAtEnd; }
    public boolean saveState() { return saveState; }
    public boolean saveFunction() { return saveFunction; }
    public boolean useLEGACYOptimScheduler() { return useLEGACYOptimScheduler; }
    public boolean useFlashAttn() { return useFlashAttn; }

    // Logging
    public int reportTo() { return reportTo; }
    public boolean logCompletions() { return logCompletions; }
    public String metrics() { return metrics; }

    // Seed
    public long seed() { return seed; }
    public boolean dataSeed() { return dataSeed; }
    public boolean deterministic() { return deterministic; }

    // Misc
    public double neftuneAlpha() { return neftuneAlpha; }
    public double runName() { return runName; }
    public boolean groupByLength() { return groupByLength; }
    public int lengthColumnName() { return lengthColumnName; }
    public boolean ddpTimeout() { return ddpTimeout; }
    public int ddpBucketCapMb() { return ddpBucketCapMb; }
    public String ddpFindUnused() { return ddpFindUnused; }
    public boolean localRank() { return localRank; }

    // Margin
    public double margin() { return margin; }

    // -------------------------------------------------------------------------
    // Computed helpers
    // -------------------------------------------------------------------------

    /**
     * Effective batch size = perDeviceBatchSize * gradientAccumulationSteps * numDevices.
     * In single-GPU Java context this simplifies to perDevice * accum.
     */
    public int effectiveBatchSize() {
        return perDeviceTrainBatchSize * gradientAccumulationSteps;
    }

    /**
     * Resolve warmup steps from explicit setting or ratio-based.
     * Returns warmupSteps if set (>=0), otherwise computes from ratio.
     */
    public int effectiveWarmupSteps(int totalSteps) {
        if (warmupSteps > 0) return warmupSteps;
        if (warmupRatio > 0 && totalSteps > 0) {
            return (int) Math.max(1, totalSteps * warmupRatio);
        }
        return 0;
    }

    /** True when any FP16 mode is enabled. */
    public boolean isFp16Enabled() { return fp16; }

    /** True when BF16 mode is enabled. */
    public boolean isBf16Enabled() { return bf16; }

    /** True when any half-precision mode is active. */
    public boolean isHalfPrecision() { return fp16 || bf16; }

    /** Optimizer string for builder mapping (paged_adamw_32bit, adamw_torch, etc). */
    public String optimizerKey() {
        if (optim == null) return "adamw_torch";
        return optim.toLowerCase();
    }

    // -------------------------------------------------------------------------
    // Map representation (for YAML / JSON serialization)
    // -------------------------------------------------------------------------

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        // Core training
        m.put("learning_rate", learningRate);
        m.put("weight_decay", weightDecay);
        m.put("max_steps", maxSteps);
        m.put("num_train_epochs", numTrainEpochs);
        m.put("logging_steps", loggingSteps);
        m.put("save_steps", saveSteps);
        m.put("eval_steps", evalSteps);
        m.put("warmup_steps", warmupSteps);
        m.put("warmup_ratio", warmupRatio);
        m.put("gradient_accumulation_steps", gradientAccumulationSteps);
        m.put("max_grad_norm", maxGradNorm);
        m.put("per_device_train_batch_size", perDeviceTrainBatchSize);
        m.put("per_device_eval_batch_size", perDeviceEvalBatchSize);
        m.put("lr_scheduler_type", lrSchedulerType);
        m.put("lr_scheduler_gamma", lrSchedulerGamma);
        m.put("lr_scheduler_eta", lrSchedulerEta);
        m.put("lr_scheduler_power", lrSchedulerPower);
        m.put("lr_scheduler_warmup_steps", lrSchedulerWarmupSteps);
        m.put("lr_scheduler_linear", lrSchedulerLinear);
        // Precision
        m.put("fp16", fp16);
        m.put("bf16", bf16);
        m.put("fp8", fp8);
        m.put("torch_dtype", torchDtype);
        m.put("gradient_checkpointing", gradientCheckpointing);
        m.put("gradient_checkpointing_ratio", gradientCheckpointingRatio);
        m.put("device_map", deviceMap);
        // Optimization
        m.put("optim", optim);
        m.put("optim_args", optimArgs);
        m.put("adam_beta1", adamBeta1);
        m.put("adam_beta2", adamBeta2);
        m.put("adam_epsilon", adamEpsilon);
        m.put("adamw_epsilon", adamWepsilon);
        m.put("fused_optim", fusedOptimizer);
        m.put("paged_optim", pagedOptimizer);
        m.put("use_lora_kernel", useLoraKernel);
        // Dataset
        m.put("max_seq_length", maxSeqLength);
        m.put("ignore_index", ignoreIndex);
        m.put("packing", packing);
        m.put("dataset_text_field", datasetTextField);
        m.put("append_concat_token", appendConcatToken);
        m.put("dataset_processing_num_proc", datasetProcessingNumProc);
        m.put("max_packed_sequences", maxPackedSequences);
        // Output
        m.put("output_dir", outputDir);
        m.put("save_total_limit", saveTotalLimit);
        m.put("save_only_model", saveOnlyModel);
        m.put("save_only_best", saveOnlyBest);
        m.put("load_best_model_at_end", loadBestModelAtEnd);
        m.put("save_state", saveState);
        m.put("save_function", saveFunction);
        m.put("use_legacy_optim_scheduler", useLEGACYOptimScheduler);
        m.put("use_flash_attn", useFlashAttn);
        // Logging
        m.put("report_to", reportTo);
        m.put("log_completions", logCompletions);
        m.put("metrics", metrics);
        // Seed
        m.put("seed", seed);
        m.put("data_seed", dataSeed);
        m.put("deterministic", deterministic);
        // Misc
        m.put("neftune_alpha", neftuneAlpha);
        m.put("run_name", runName);
        m.put("group_by_length", groupByLength);
        m.put("length_column_name", lengthColumnName);
        m.put("ddp_timeout", ddpTimeout);
        m.put("ddp_bucket_cap_mb", ddpBucketCapMb);
        m.put("ddp_find_unused", ddpFindUnused);
        m.put("local_rank", localRank);
        m.put("margin", margin);
        return m;
    }

    @Override
    public String toString() {
        return "TrainerConfig" + toMap();
    }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public static class Builder<B extends Builder<B>> {
        // Default values matching HF defaults
        // Core training
        protected double learningRate = 1e-5;
        protected double weightDecay = 0.0;
        protected int maxSteps = -1;  // -1 means calculate from epochs
        protected int numTrainEpochs = 3;
        protected int loggingSteps = 10;
        protected int saveSteps = 500;
        protected int evalSteps = -1;
        protected int warmupSteps = 0;
        protected double warmupRatio = 0.0;
        protected int gradientAccumulationSteps = 1;
        protected double maxGradNorm = 1.0;
        protected int perDeviceTrainBatchSize = 1;
        protected int perDeviceEvalBatchSize = 2;
        protected String lrSchedulerType = "cosine";
        protected double lrSchedulerGamma = 0.1;
        protected double lrSchedulerEta = 0.0;
        protected double lrSchedulerPower = 1.0;
        protected int lrSchedulerWarmupSteps = 0;
        protected boolean lrSchedulerLinear = false;

        // Precision
        protected boolean fp16 = false;
        protected boolean bf16 = false;
        protected boolean fp8 = false;
        protected String torchDtype = "float32";
        protected boolean gradientCheckpointing = false;
        protected double gradientCheckpointingRatio = 0.0;
        protected String deviceMap = "auto";

        // Optimization
        protected String optim = "adamw_torch";
        protected double optimArgs = 0.0;
        protected double adamBeta1 = 0.9;
        protected double adamBeta2 = 0.999;
        protected double adamEpsilon = 1e-8;
        protected double adamWepsilon = 1e-8;
        protected boolean fusedOptimizer = false;
        protected boolean pagedOptimizer = false;
        protected boolean useLoraKernel = false;

        // Dataset
        protected int maxSeqLength = 2048;
        protected long ignoreIndex = -100L;
        protected boolean packing = false;
        protected String datasetTextField = "text";
        protected boolean appendConcatToken = false;
        protected String datasetProcessingNumProc = "null";
        protected int maxPackedSequences = -1;

        // Output
        protected String outputDir = "output";
        protected String saveTotalLimit = "null";
        protected int saveOnlyModel = 1;
        protected boolean saveOnlyBest = false;
        protected boolean loadBestModelAtEnd = false;
        protected boolean saveState = true;
        protected boolean saveFunction = false;
        protected boolean useLEGACYOptimScheduler = true;
        protected boolean useFlashAttn = false;

        // Logging
        protected int reportTo = 0;  // 0=none, 1=wandb, 2=mlflow, 3=all
        protected boolean logCompletions = false;
        protected String metrics = "null";

        // Seed
        protected long seed = 42L;
        protected boolean dataSeed = false;
        protected boolean deterministic = false;

        // Misc
        protected double neftuneAlpha = 0.0;
        protected double runName = 0.0;
        protected boolean groupByLength = false;
        protected int lengthColumnName = -1;
        protected boolean ddpTimeout = false;
        protected int ddpBucketCapMb = 25;
        protected String ddpFindUnused = "null";
        protected boolean localRank = false;

        // Margin
        protected double margin = 0.0;

        // Builder methods
        @SuppressWarnings("unchecked")
        public B learningRate(double v) { this.learningRate = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B learning_rate(double v) { return learningRate(v); }
        @SuppressWarnings("unchecked")
        public B weightDecay(double v) { this.weightDecay = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B maxSteps(int v) { this.maxSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B numTrainEpochs(int v) { this.numTrainEpochs = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B num_train_epochs(int v) { return numTrainEpochs(v); }
        @SuppressWarnings("unchecked")
        public B loggingSteps(int v) { this.loggingSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B logging_steps(int v) { return loggingSteps(v); }
        @SuppressWarnings("unchecked")
        public B saveSteps(int v) { this.saveSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B evalSteps(int v) { this.evalSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B warmupSteps(int v) { this.warmupSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B warmupRatio(double v) { this.warmupRatio = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B warmup_ratio(double v) { return warmupRatio(v); }
        @SuppressWarnings("unchecked")
        public B gradientAccumulationSteps(int v) { this.gradientAccumulationSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B gradient_accumulation_steps(int v) { return gradientAccumulationSteps(v); }
        @SuppressWarnings("unchecked")
        public B maxGradNorm(double v) { this.maxGradNorm = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B perDeviceTrainBatchSize(int v) { this.perDeviceTrainBatchSize = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B per_device_train_batch_size(int v) { return perDeviceTrainBatchSize(v); }
        @SuppressWarnings("unchecked")
        public B perDeviceEvalBatchSize(int v) { this.perDeviceEvalBatchSize = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B per_device_eval_batch_size(int v) { return perDeviceEvalBatchSize(v); }
        @SuppressWarnings("unchecked")
        public B lrSchedulerType(String v) { this.lrSchedulerType = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B lrSchedulerGamma(double v) { this.lrSchedulerGamma = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B lrSchedulerEta(double v) { this.lrSchedulerEta = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B lrSchedulerPower(double v) { this.lrSchedulerPower = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B lrSchedulerWarmupSteps(int v) { this.lrSchedulerWarmupSteps = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B lrSchedulerLinear(boolean v) { this.lrSchedulerLinear = v; return (B) this; }

        // Precision
        @SuppressWarnings("unchecked")
        public B fp16(boolean v) { this.fp16 = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B bf16(boolean v) { this.bf16 = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B fp8(boolean v) { this.fp8 = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B torchDtype(String v) { this.torchDtype = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B gradientCheckpointing(boolean v) { this.gradientCheckpointing = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B gradient_checkpointing(boolean v) { return gradientCheckpointing(v); }
        @SuppressWarnings("unchecked")
        public B gradientCheckpointingRatio(double v) { this.gradientCheckpointingRatio = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B deviceMap(String v) { this.deviceMap = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B device_map(String v) { return deviceMap(v); }
        /** Python {@code use_cpu=True} → device_map=cpu. */
        @SuppressWarnings("unchecked")
        public B useCpu(boolean v) { if (v) this.deviceMap = "cpu"; return (B) this; }
        @SuppressWarnings("unchecked")
        public B use_cpu(boolean v) { return useCpu(v); }

        // Optimization
        @SuppressWarnings("unchecked")
        public B optim(String v) { this.optim = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B optimArgs(double v) { this.optimArgs = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B adamBeta1(double v) { this.adamBeta1 = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B adamBeta2(double v) { this.adamBeta2 = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B adamEpsilon(double v) { this.adamEpsilon = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B adamWepsilon(double v) { this.adamWepsilon = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B fusedOptimizer(boolean v) { this.fusedOptimizer = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B pagedOptimizer(boolean v) { this.pagedOptimizer = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B useLoraKernel(boolean v) { this.useLoraKernel = v; return (B) this; }

        // Dataset
        @SuppressWarnings("unchecked")
        public B maxSeqLength(int v) { this.maxSeqLength = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B ignoreIndex(long v) { this.ignoreIndex = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B packing(boolean v) { this.packing = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B datasetTextField(String v) { this.datasetTextField = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B appendConcatToken(boolean v) { this.appendConcatToken = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B datasetProcessingNumProc(String v) { this.datasetProcessingNumProc = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B maxPackedSequences(int v) { this.maxPackedSequences = v; return (B) this; }

        // Output
        @SuppressWarnings("unchecked")
        public B outputDir(String v) { this.outputDir = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B output_dir(String v) { return outputDir(v); }
        @SuppressWarnings("unchecked")
        public B saveTotalLimit(String v) { this.saveTotalLimit = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B saveOnlyModel(int v) { this.saveOnlyModel = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B saveOnlyBest(boolean v) { this.saveOnlyBest = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B loadBestModelAtEnd(boolean v) { this.loadBestModelAtEnd = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B load_best_model_at_end(boolean v) { return loadBestModelAtEnd(v); }
        @SuppressWarnings("unchecked")
        public B saveState(boolean v) { this.saveState = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B saveFunction(boolean v) { this.saveFunction = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B useLEGACYOptimScheduler(boolean v) { this.useLEGACYOptimScheduler = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B useFlashAttn(boolean v) { this.useFlashAttn = v; return (B) this; }

        // Logging
        @SuppressWarnings("unchecked")
        public B reportTo(int v) { this.reportTo = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B logCompletions(boolean v) { this.logCompletions = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B metrics(String v) { this.metrics = v; return (B) this; }

        // Seed
        @SuppressWarnings("unchecked")
        public B seed(long v) { this.seed = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B dataSeed(boolean v) { this.dataSeed = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B deterministic(boolean v) { this.deterministic = v; return (B) this; }

        // Misc
        @SuppressWarnings("unchecked")
        public B neftuneAlpha(double v) { this.neftuneAlpha = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B runName(double v) { this.runName = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B groupByLength(boolean v) { this.groupByLength = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B lengthColumnName(int v) { this.lengthColumnName = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B ddpTimeout(boolean v) { this.ddpTimeout = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B ddpBucketCapMb(int v) { this.ddpBucketCapMb = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B ddpFindUnused(String v) { this.ddpFindUnused = v; return (B) this; }
        @SuppressWarnings("unchecked")
        public B localRank(boolean v) { this.localRank = v; return (B) this; }

        // Margin
        @SuppressWarnings("unchecked")
        public B margin(double v) { this.margin = v; return (B) this; }

        public TrainerConfig build() { return new TrainerConfig(this); }
    }

    public static Builder<?> builder() { return new Builder<>(); }
}
