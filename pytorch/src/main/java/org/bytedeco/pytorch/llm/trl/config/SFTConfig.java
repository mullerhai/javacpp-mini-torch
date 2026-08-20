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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Supervised Fine-Tuning configuration (HuggingFace TRL {@code SFTTrainer} / {@code SFTConfig}).
 *
 * <p>Mirrors the Python API and adds the missing SFT-specific attributes:
 * <pre>{@code
 * SFTConfig(
 *     output_dir="output",
 *     num_train_epochs=3,
 *     per_device_train_batch_size=4,
 *     gradient_accumulation_steps=4,
 *     optim="paged_adamw_32bit",
 *     learning_rate=2e-4,
 *     weight_decay=0.001,
 *     fp16=True,
 *     bf16=False,
 *     max_grad_norm=0.3,
 *     warmup_ratio=0.03,
 *     lr_scheduler_type="linear",
 *     logging_steps=20,
 *     save_steps=100,
 *     save_total_limit=2,
 *     max_seq_length=2048,
 *     dataset_text_field="text",
 *     packing=False,
 *     group_by_length=True,
 *     gradient_checkpointing=True,
 *     gradient_checkpointing_ratio=0.5,
 *     neftune_alpha=0,
 *     use_flash_attn=False,
 *     ddp_timeout=1800,
 *     warmup_steps=0,
 *     eval_steps=0,
 *     logging_strategy="steps",
 *     save_strategy="steps",
 *     report_to="none",
 * )
 * }</pre>
 */
public final class SFTConfig extends TrainerConfig {

    // -------------------------------------------------------------------------
    // Model-related (TRL SFT-specific)
    // -------------------------------------------------------------------------

    /** Keyword args forwarded to {@code AutoModelForCausalLM.from_pretrained}. */
    private final Map<String, Object> modelInitKwargs;

    /** Allow loading models / tokenizers with custom Python code. */
    private final boolean trustRemoteCode;

    /** Coefficient of the load-balancing auxiliary loss (MoE only). */
    private final double routerAuxLossCoef;

    /** Path to a tokenizer or Jinja chat template. */
    private final String chatTemplatePath;

    // -------------------------------------------------------------------------
    // Data preprocessing
    // -------------------------------------------------------------------------

    /** Whether to shuffle the dataset before training. */
    private final boolean shuffleDataset;

    /** Truncation mode: "keep_start" | "keep_end". */
    private final String truncationMode;

    /** Optional EOS token to override tokenizer default. */
    private final String eosToken;

    /** Optional PAD token to override tokenizer default. */
    private final String padToken;

    /** {@code max_length} alias (Python default 1024). Mirrors {@code maxSeqLength}. */
    private final int maxLength;

    /** Optional dictionary of dataset-prep kwargs (e.g. {@code skip_prepare_dataset}). */
    private final Map<String, Object> datasetKwargs;

    /** Packing strategy: "bfd" | "bfd_split" | "wrapped". */
    private final String packingStrategy;

    /** Whether to flatten batches without padding (requires FlashAttention 2/3). */
    private final boolean paddingFree;

    /** Pad sequence length to a multiple of this value. */
    private final int padToMultipleOf;

    /** Whether to pack the eval dataset (defaults to {@code packing}). */
    private final boolean evalPacking;

    // -------------------------------------------------------------------------
    // Loss computation
    // -------------------------------------------------------------------------

    /** {@code true}/{@code false}/{@code null}. When null, auto-inferred from dataset shape. */
    private final Boolean completionOnlyLoss;

    /** Compute loss only on assistant responses (conversational datasets). */
    private final boolean assistantOnlyLoss;

    /** Loss type: "nll" | "dft" | "chunked_nll". */
    private final String lossType;

    /** Whether to offload activations to CPU. */
    private final boolean activationOffloading;

    // -------------------------------------------------------------------------
    // Legacy SFT fields
    // -------------------------------------------------------------------------

    /** Padding token id used when packing sequences. */
    private final long ignoreIndex;

    /** Whether to pack multiple short sequences into one sample. */
    private final boolean packing;

    /** Name of the text field in dataset to use for training. */
    private final String datasetTextField;

    /** Tokenizer vocab size (auto-populated from tokenizer). */
    private final int vocabSize;

    /** Whether to append separator token between concatenated samples. */
    private final boolean appendConcatToken;

    /** NEFTune noise alpha (0 disables NEFTune). */
    private final double neftuneAlpha;

    /** Whether to use Flash Attention 2. */
    private final boolean useFlashAttn;

    /** DDP timeout in seconds. */
    private final int ddpTimeoutSeconds;

    /** Module to format prompts (class name or null). */
    private final String datasetFormatter;

    /** Additional dataset mixins (e.g. chat template). */
    private final String datasetMixins;

    /** Whether to preprocess columns while loading dataset. */
    private final boolean datasetProcessingWarning;

    /** Truncation strategy: "only_first" | "only_second" | "longest_first" | "do_not_truncate". */
    private final String truncation;

    /** Padding strategy: "longest" | "max_length". */
    private final String padding;

    /** Number of processors for dataset preprocessing (string form to match parent). */
    private final String datasetProcessingNumProc;

    /** Maximum number of packed sequences per sample. */
    private final int maxPackedSequences;

    /** Dataset cache directory. */
    private final String datasetCacheDir;

    /** Optional HuggingFace dataset split. */
    private final String datasetSplit;

    /** Verification token (for private datasets). */
    private final String token;

    /** Map function to apply to dataset. */
    private final String mapFun;

    /** Whether to include prompt in labels (for SFT loss). */
    private final boolean taskInstructions;

    /** Split train/eval ratio. */
    private final double trainSplitRatio;

    /** Special tokens configuration JSON. */
    private final String specialTokens;

    // HuggingFace TrainingArguments fields used by the SFT tutorial notebooks
    private final String evalStrategy;
    private final String saveStrategy;
    private final String loggingStrategy;
    private final boolean doEval;
    private final boolean pushToHub;
    private final boolean removeUnusedColumns;
    private final List<String> labelNames;
    private final String reportToName;

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    private SFTConfig(Builder b) {
        super(b);
        // Model-related
        this.modelInitKwargs = b.modelInitKwargs == null ? null
                : new LinkedHashMap<>(b.modelInitKwargs);
        this.trustRemoteCode = b.trustRemoteCode;
        this.routerAuxLossCoef = b.routerAuxLossCoef;
        this.chatTemplatePath = b.chatTemplatePath;

        // Data preprocessing
        this.shuffleDataset = b.shuffleDataset;
        this.truncationMode = b.truncationMode;
        this.eosToken = b.eosToken;
        this.padToken = b.padToken;
        this.maxLength = b.maxLength > 0 ? b.maxLength : b.maxSeqLength;
        this.datasetKwargs = b.datasetKwargs == null ? null
                : new LinkedHashMap<>(b.datasetKwargs);
        this.packingStrategy = b.packingStrategy;
        this.paddingFree = b.paddingFree;
        this.padToMultipleOf = b.padToMultipleOf;
        this.evalPacking = b.evalPacking;

        // Loss
        this.completionOnlyLoss = b.completionOnlyLoss;
        this.assistantOnlyLoss = b.assistantOnlyLoss;
        this.lossType = b.lossType;
        this.activationOffloading = b.activationOffloading;

        // Legacy SFT fields
        this.ignoreIndex = b.ignoreIndex;
        this.packing = b.packing;
        this.datasetTextField = b.datasetTextField;
        this.vocabSize = b.vocabSize;
        this.appendConcatToken = b.appendConcatToken;
        this.neftuneAlpha = b.neftuneAlpha;
        this.useFlashAttn = b.useFlashAttn;
        this.ddpTimeoutSeconds = b.ddpTimeout;
        this.datasetFormatter = b.datasetFormatter;
        this.datasetMixins = b.datasetMixins;
        this.datasetProcessingWarning = b.datasetProcessingWarning;
        this.truncation = b.truncation;
        this.padding = b.padding;
        this.datasetProcessingNumProc = b.datasetProcessingNumProc;
        this.maxPackedSequences = b.maxPackedSequences;
        this.datasetCacheDir = b.datasetCacheDir;
        this.datasetSplit = b.datasetSplit;
        this.token = b.token;
        this.mapFun = b.mapFun;
        this.taskInstructions = b.taskInstructions;
        this.trainSplitRatio = b.trainSplitRatio;
        this.specialTokens = b.specialTokens;
        this.evalStrategy = b.evalStrategy;
        this.saveStrategy = b.saveStrategy;
        this.loggingStrategy = b.loggingStrategy;
        this.doEval = b.doEval;
        this.pushToHub = b.pushToHub;
        this.removeUnusedColumns = b.removeUnusedColumns;
        this.labelNames = b.labelNames == null ? List.of() : Collections.unmodifiableList(new ArrayList<>(b.labelNames));
        this.reportToName = b.reportToName;
    }

    // -------------------------------------------------------------------------
    // Model-related accessors
    // -------------------------------------------------------------------------

    public Map<String, Object> modelInitKwargs() { return modelInitKwargs; }
    public boolean trustRemoteCode() { return trustRemoteCode; }
    public double routerAuxLossCoef() { return routerAuxLossCoef; }
    public String chatTemplatePath() { return chatTemplatePath; }

    // -------------------------------------------------------------------------
    // Data-preprocessing accessors
    // -------------------------------------------------------------------------

    public boolean shuffleDataset() { return shuffleDataset; }
    public String truncationMode() { return truncationMode; }
    public String eosToken() { return eosToken; }
    public String padToken() { return padToken; }
    public int maxLength() { return maxLength; }
    public Map<String, Object> datasetKwargs() { return datasetKwargs; }
    public String packingStrategy() { return packingStrategy; }
    public boolean paddingFree() { return paddingFree; }
    public int padToMultipleOf() { return padToMultipleOf; }
    public boolean evalPacking() { return evalPacking; }

    // -------------------------------------------------------------------------
    // Loss accessors
    // -------------------------------------------------------------------------

    public Boolean completionOnlyLoss() { return completionOnlyLoss; }
    public boolean assistantOnlyLoss() { return assistantOnlyLoss; }
    public String lossType() { return lossType; }
    public boolean activationOffloading() { return activationOffloading; }

    // -------------------------------------------------------------------------
    // Legacy SFT accessors
    // -------------------------------------------------------------------------

    public int maxSeqLength() { return maxLength; }
    public long ignoreIndex() { return ignoreIndex; }
    public boolean packing() { return packing; }
    public String datasetTextField() { return datasetTextField; }
    public int vocabSize() { return vocabSize; }
    public boolean appendConcatToken() { return appendConcatToken; }
    public double neftuneAlpha() { return neftuneAlpha; }
    public boolean useFlashAttn() { return useFlashAttn; }
    public int ddpTimeoutSeconds() { return ddpTimeoutSeconds; }
    public String datasetFormatter() { return datasetFormatter; }
    public String datasetMixins() { return datasetMixins; }
    public boolean datasetProcessingWarning() { return datasetProcessingWarning; }
    public String truncation() { return truncation; }
    public String padding() { return padding; }
    public String datasetProcessingNumProc() { return datasetProcessingNumProc; }
    public int datasetProcessingNumProcAsInt() {
        try {
            return Integer.parseInt(datasetProcessingNumProc);
        } catch (NumberFormatException e) {
            return -1;
        }
    }
    public int maxPackedSequences() { return maxPackedSequences; }
    public String datasetCacheDir() { return datasetCacheDir; }
    public String datasetSplit() { return datasetSplit; }
    public String token() { return token; }
    public String mapFun() { return mapFun; }
    public boolean taskInstructions() { return taskInstructions; }
    public double trainSplitRatio() { return trainSplitRatio; }
    public String specialTokens() { return specialTokens; }

    public String evalStrategy() { return evalStrategy; }
    public String saveStrategy() { return saveStrategy; }
    public String loggingStrategy() { return loggingStrategy; }
    public boolean doEval() { return doEval; }
    public boolean pushToHub() { return pushToHub; }
    public boolean removeUnusedColumns() { return removeUnusedColumns; }
    public List<String> labelNames() { return labelNames; }
    /** {@code "none"} / {@code "tensorboard"} / {@code "wandb"} (Python {@code report_to}). */
    public String reportToName() { return reportToName; }

    // -------------------------------------------------------------------------
    // Computed helpers
    // -------------------------------------------------------------------------

    /** Effective batch size = perDeviceTrainBatchSize * gradientAccumulationSteps. */
    public int effectiveBatchSize() {
        return perDeviceTrainBatchSize() * gradientAccumulationSteps();
    }

    /** Whether NEFTune is enabled (alpha > 0). */
    public boolean neftuneEnabled() {
        return neftuneAlpha > 0.0;
    }

    /** Effective warmup steps: prefers explicit warmup_steps, falls back to warmup_ratio. */
    public int effectiveWarmupSteps(int totalSteps) {
        if (warmupSteps() > 0) return warmupSteps();
        if (warmupRatio() > 0.0 && totalSteps > 0) {
            return (int) Math.max(1, totalSteps * warmupRatio());
        }
        return 0;
    }

    /** Whether completion-only loss is enabled (effective). */
    public boolean isCompletionOnlyLoss() {
        if (completionOnlyLoss != null) return completionOnlyLoss;
        return false;
    }

    /** Whether to pad to a multiple. */
    public boolean hasPadToMultipleOf() { return padToMultipleOf > 0; }

    // -------------------------------------------------------------------------
    // Map representation
    // -------------------------------------------------------------------------

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>(super.toMap());
        // Model
        if (modelInitKwargs != null) m.put("model_init_kwargs", modelInitKwargs);
        m.put("trust_remote_code", trustRemoteCode);
        m.put("router_aux_loss_coef", routerAuxLossCoef);
        m.put("chat_template_path", chatTemplatePath);
        // Data preprocessing
        m.put("max_length", maxLength);
        m.put("shuffle_dataset", shuffleDataset);
        m.put("truncation_mode", truncationMode);
        m.put("eos_token", eosToken);
        m.put("pad_token", padToken);
        if (datasetKwargs != null) m.put("dataset_kwargs", datasetKwargs);
        m.put("packing_strategy", packingStrategy);
        m.put("padding_free", paddingFree);
        m.put("pad_to_multiple_of", padToMultipleOf);
        m.put("eval_packing", evalPacking);
        // Loss
        m.put("completion_only_loss", completionOnlyLoss);
        m.put("assistant_only_loss", assistantOnlyLoss);
        m.put("loss_type", lossType);
        m.put("activation_offloading", activationOffloading);
        // Legacy SFT
        m.put("max_seq_length", maxLength);
        m.put("ignore_index", ignoreIndex);
        m.put("packing", packing);
        m.put("dataset_text_field", datasetTextField);
        m.put("vocab_size", vocabSize);
        m.put("append_concat_token", appendConcatToken);
        m.put("neftune_alpha", neftuneAlpha);
        m.put("use_flash_attn", useFlashAttn);
        m.put("ddp_timeout", ddpTimeoutSeconds);
        m.put("dataset_formatter", datasetFormatter);
        m.put("dataset_mixins", datasetMixins);
        m.put("dataset_processing_warning", datasetProcessingWarning);
        m.put("truncation", truncation);
        m.put("padding", padding);
        m.put("dataset_processing_num_proc", datasetProcessingNumProc);
        m.put("max_packed_sequences", maxPackedSequences);
        m.put("dataset_cache_dir", datasetCacheDir);
        m.put("dataset_split", datasetSplit);
        m.put("token", token);
        m.put("map_fun", mapFun);
        m.put("task_instruct", taskInstructions);
        m.put("train_split_ratio", trainSplitRatio);
        m.put("special_tokens", specialTokens);
        m.put("eval_strategy", evalStrategy);
        m.put("save_strategy", saveStrategy);
        m.put("logging_strategy", loggingStrategy);
        m.put("do_eval", doEval);
        m.put("push_to_hub", pushToHub);
        m.put("remove_unused_columns", removeUnusedColumns);
        m.put("label_names", new ArrayList<>(labelNames));
        m.put("report_to", reportToName);
        return m;
    }

    @Override
    public String toString() {
        return "SFTConfig" + toMap();
    }

    // -------------------------------------------------------------------------
    // Factory defaults
    // -------------------------------------------------------------------------

    /** Return a pre-configured builder for QLoRA fine-tuning. */
    public static Builder defaultQLoRA() {
        return builder()
                .fp16(true)
                .optim("paged_adamw_32bit")
                .gradientCheckpointing(true)
                .perDeviceTrainBatchSize(1)
                .gradientAccumulationSteps(4)
                .warmupRatio(0.03)
                .lrSchedulerLinear(true)
                .maxSeqLength(2048)
                .packing(true)
                .packingStrategy("bfd")
                .lossType("nll");
    }

    /** Return a pre-configured builder for full fine-tuning. */
    public static Builder defaultFullSFT() {
        return builder()
                .fp16(true)
                .optim("adamw_torch")
                .perDeviceTrainBatchSize(2)
                .gradientAccumulationSteps(8)
                .warmupRatio(0.03)
                .lrSchedulerLinear(true)
                .maxSeqLength(2048)
                .lossType("chunked_nll");
    }

    // -------------------------------------------------------------------------
    // Builder
    // -------------------------------------------------------------------------

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {

        // Model-related defaults
        private Map<String, Object> modelInitKwargs = null;
        private boolean trustRemoteCode = false;
        private double routerAuxLossCoef = 0.001;
        private String chatTemplatePath = null;

        // Data preprocessing defaults
        private boolean shuffleDataset = false;
        private String truncationMode = "keep_start";
        private String eosToken = null;
        private String padToken = null;
        private int maxLength = -1; // -1 means "use maxSeqLength"
        private Map<String, Object> datasetKwargs = null;
        private String packingStrategy = "bfd";
        private boolean paddingFree = false;
        private int padToMultipleOf = -1;
        private boolean evalPacking = false;

        // Loss defaults
        private Boolean completionOnlyLoss = null;
        private boolean assistantOnlyLoss = false;
        private String lossType = null; // auto-detected in post-init
        private boolean activationOffloading = false;

        // Legacy SFT defaults
        private int maxSeqLength = 1024; // Python default (overrides parent 2048)
        private long ignoreIndex = -100L;
        private boolean packing = false;
        private String datasetTextField = "text";
        private int vocabSize = -1;
        private boolean appendConcatToken = false;
        private double neftuneAlpha = 0.0;
        private boolean useFlashAttn = false;
        private int ddpTimeout = 1800;
        private String datasetFormatter = null;
        private String datasetMixins = null;
        private boolean datasetProcessingWarning = true;
        private String truncation = "longest_first";
        private String padding = "longest";
        private String datasetProcessingNumProc = "null";
        private int maxPackedSequences = -1;
        private String datasetCacheDir = null;
        private String datasetSplit = null;
        private String token = null;
        private String mapFun = null;
        private boolean taskInstructions = false;
        private double trainSplitRatio = 0.0;
        private String specialTokens = null;
        private String evalStrategy = "no";
        private String saveStrategy = "steps";
        private String loggingStrategy = "steps";
        private boolean doEval = false;
        private boolean pushToHub = false;
        private boolean removeUnusedColumns = true;
        private List<String> labelNames = new ArrayList<>();
        private String reportToName = "none";

        // Override parent defaults for SFT (Python TRL convention)
        {
            maxSteps(-1);
            numTrainEpochs(3);
            loggingSteps(10);  // SFT default, differs from parent's 500
            saveSteps(500);
            learningRate(2e-5); // SFT default, differs from parent's 1e-5
            maxGradNorm(1.0);
            perDeviceTrainBatchSize(1);
            gradientAccumulationSteps(1);
            warmupRatio(0.0);
            warmupSteps(0);
            fp16(false);
            bf16(false);
            optim("adamw_torch");
            gradientCheckpointing(true); // SFT default, differs from parent's false
            groupByLength(false);
            seed(42L);
            outputDir("output");
        }

        // ---- Model-related ----

        public Builder modelInitKwargs(Map<String, Object> v) { this.modelInitKwargs = v; return this; }
        public Builder model_init_kwargs(Map<String, Object> v) { return modelInitKwargs(v); }
        public Builder trustRemoteCode(boolean v) { this.trustRemoteCode = v; return this; }
        public Builder trust_remote_code(boolean v) { return trustRemoteCode(v); }
        public Builder routerAuxLossCoef(double v) { this.routerAuxLossCoef = v; return this; }
        public Builder router_aux_loss_coef(double v) { return routerAuxLossCoef(v); }
        public Builder chatTemplatePath(String v) { this.chatTemplatePath = v; return this; }
        public Builder chat_template_path(String v) { return chatTemplatePath(v); }

        // ---- Data preprocessing ----

        public Builder shuffleDataset(boolean v) { this.shuffleDataset = v; return this; }
        public Builder shuffle_dataset(boolean v) { return shuffleDataset(v); }

        public Builder truncationMode(String v) {
            if (v != null && !v.equals("keep_start") && !v.equals("keep_end")) {
                throw new IllegalArgumentException(
                        "truncation_mode must be 'keep_start' or 'keep_end', got: " + v);
            }
            this.truncationMode = v;
            return this;
        }
        public Builder truncation_mode(String v) { return truncationMode(v); }

        public Builder eosToken(String v) { this.eosToken = v; return this; }
        public Builder eos_token(String v) { return eosToken(v); }

        public Builder padToken(String v) { this.padToken = v; return this; }
        public Builder pad_token(String v) { return padToken(v); }

        public Builder maxLength(int v) {
            if (v <= 0) {
                throw new IllegalArgumentException("max_length must be > 0");
            }
            this.maxLength = v;
            this.maxSeqLength = v; // keep in sync
            return this;
        }
        public Builder max_length(int v) { return maxLength(v); }

        public Builder datasetKwargs(Map<String, Object> v) { this.datasetKwargs = v; return this; }
        public Builder dataset_kwargs(Map<String, Object> v) { return datasetKwargs(v); }

        public Builder packingStrategy(String v) {
            if (v != null && !v.equals("bfd") && !v.equals("bfd_split") && !v.equals("wrapped")) {
                throw new IllegalArgumentException(
                        "packing_strategy must be 'bfd', 'bfd_split', or 'wrapped', got: " + v);
            }
            this.packingStrategy = v == null ? "bfd" : v;
            return this;
        }
        public Builder packing_strategy(String v) { return packingStrategy(v); }

        public Builder paddingFree(boolean v) { this.paddingFree = v; return this; }
        public Builder padding_free(boolean v) { return paddingFree(v); }

        public Builder padToMultipleOf(int v) {
            if (v <= 0) {
                throw new IllegalArgumentException("pad_to_multiple_of must be > 0");
            }
            this.padToMultipleOf = v;
            return this;
        }
        public Builder pad_to_multiple_of(int v) { return padToMultipleOf(v); }

        public Builder evalPacking(boolean v) { this.evalPacking = v; return this; }
        public Builder eval_packing(boolean v) { return evalPacking(v); }

        // ---- Loss ----

        public Builder completionOnlyLoss(Boolean v) { this.completionOnlyLoss = v; return this; }
        public Builder completion_only_loss(Boolean v) { return completionOnlyLoss(v); }
        public Builder completionOnlyLoss(boolean v) { this.completionOnlyLoss = v; return this; }
        public Builder completion_only_loss(boolean v) { return completionOnlyLoss(v); }

        public Builder assistantOnlyLoss(boolean v) { this.assistantOnlyLoss = v; return this; }
        public Builder assistant_only_loss(boolean v) { return assistantOnlyLoss(v); }

        public Builder lossType(String v) {
            if (v != null && !v.equals("nll") && !v.equals("dft") && !v.equals("chunked_nll")) {
                throw new IllegalArgumentException(
                        "loss_type must be 'nll', 'dft', or 'chunked_nll', got: " + v);
            }
            this.lossType = v;
            return this;
        }
        public Builder loss_type(String v) { return lossType(v); }

        public Builder activationOffloading(boolean v) { this.activationOffloading = v; return this; }
        public Builder activation_offloading(boolean v) { return activationOffloading(v); }

        // ---- Legacy SFT ----

        public Builder maxSeqLength(int v) {
            this.maxSeqLength = v;
            if (this.maxLength <= 0) this.maxLength = v;
            return this;
        }
        public Builder max_seq_length(int v) { return maxSeqLength(v); }

        public Builder ignoreIndex(long v) { this.ignoreIndex = v; return this; }
        public Builder ignore_index(long v) { return ignoreIndex(v); }

        public Builder packing(boolean v) { this.packing = v; return this; }

        public Builder datasetTextField(String v) { this.datasetTextField = v; return this; }
        public Builder dataset_text_field(String v) { return datasetTextField(v); }

        public Builder vocabSize(int v) { this.vocabSize = v; return this; }
        public Builder vocab_size(int v) { return vocabSize(v); }

        public Builder appendConcatToken(boolean v) { this.appendConcatToken = v; return this; }
        public Builder append_concat_token(boolean v) { return appendConcatToken(v); }

        public Builder neftuneAlpha(double v) { this.neftuneAlpha = v; return this; }
        public Builder neftune_alpha(double v) { return neftuneAlpha(v); }

        public Builder useFlashAttn(boolean v) { this.useFlashAttn = v; return this; }
        public Builder use_flash_attn(boolean v) { return useFlashAttn(v); }

        public Builder ddpTimeout(int v) { this.ddpTimeout = v; return this; }
        public Builder ddp_timeout(int v) { return ddpTimeout(v); }

        public Builder datasetFormatter(String v) { this.datasetFormatter = v; return this; }
        public Builder dataset_formatter(String v) { return datasetFormatter(v); }

        public Builder datasetMixins(String v) { this.datasetMixins = v; return this; }
        public Builder dataset_mixins(String v) { return datasetMixins(v); }

        public Builder datasetProcessingWarning(boolean v) { this.datasetProcessingWarning = v; return this; }
        public Builder dataset_processing_warning(boolean v) { return datasetProcessingWarning(v); }

        public Builder truncation(String v) { this.truncation = v; return this; }

        public Builder padding(String v) { this.padding = v; return this; }

        public Builder datasetProcessingNumProc(int v) {
            this.datasetProcessingNumProc = Integer.toString(v);
            return this;
        }
        public Builder datasetProcessingNumProc(String v) { this.datasetProcessingNumProc = v; return this; }
        public Builder dataset_processing_num_proc(int v) { return datasetProcessingNumProc(v); }
        public Builder dataset_processing_num_proc(String v) { return datasetProcessingNumProc(v); }

        public Builder maxPackedSequences(int v) { this.maxPackedSequences = v; return this; }
        public Builder max_packed_sequences(int v) { return maxPackedSequences(v); }

        public Builder datasetCacheDir(String v) { this.datasetCacheDir = v; return this; }
        public Builder dataset_cache_dir(String v) { return datasetCacheDir(v); }

        public Builder datasetSplit(String v) { this.datasetSplit = v; return this; }
        public Builder dataset_split(String v) { return datasetSplit(v); }

        public Builder token(String v) { this.token = v; return this; }

        public Builder mapFun(String v) { this.mapFun = v; return this; }
        public Builder map_fun(String v) { return mapFun(v); }

        public Builder taskInstructions(boolean v) { this.taskInstructions = v; return this; }
        public Builder task_instructions(boolean v) { return taskInstructions(v); }

        public Builder trainSplitRatio(double v) { this.trainSplitRatio = v; return this; }
        public Builder train_split_ratio(double v) { return trainSplitRatio(v); }

        public Builder specialTokens(String v) { this.specialTokens = v; return this; }
        public Builder special_tokens(String v) { return specialTokens(v); }

        public Builder evalStrategy(String v) { this.evalStrategy = v == null ? "no" : v; return this; }
        public Builder eval_strategy(String v) { return evalStrategy(v); }
        public Builder saveStrategy(String v) { this.saveStrategy = v == null ? "steps" : v; return this; }
        public Builder save_strategy(String v) { return saveStrategy(v); }
        public Builder loggingStrategy(String v) { this.loggingStrategy = v == null ? "steps" : v; return this; }
        public Builder logging_strategy(String v) { return loggingStrategy(v); }
        public Builder doEval(boolean v) { this.doEval = v; return this; }
        public Builder do_eval(boolean v) { return doEval(v); }
        public Builder pushToHub(boolean v) { this.pushToHub = v; return this; }
        public Builder push_to_hub(boolean v) { return pushToHub(v); }
        public Builder removeUnusedColumns(boolean v) { this.removeUnusedColumns = v; return this; }
        public Builder remove_unused_columns(boolean v) { return removeUnusedColumns(v); }
        public Builder labelNames(List<String> v) {
            this.labelNames = v == null ? new ArrayList<>() : new ArrayList<>(v);
            return this;
        }
        public Builder label_names(List<String> v) { return labelNames(v); }
        public Builder labelNames(String... v) {
            this.labelNames = v == null ? new ArrayList<>() : new ArrayList<>(List.of(v));
            return this;
        }
        public Builder label_names(String... v) { return labelNames(v); }
        /**
         * Python {@code report_to="tensorboard"|"wandb"|"none"|None}. Also maps
         * onto the parent int {@code reportTo} (0=none, 1=wandb, 2=mlflow, 3=all).
         */
        public Builder reportToName(String v) {
            if (v == null || "none".equalsIgnoreCase(v) || "null".equalsIgnoreCase(v)) {
                this.reportToName = "none";
                super.reportTo(0);
            } else if ("tensorboard".equalsIgnoreCase(v)) {
                this.reportToName = "tensorboard";
                super.reportTo(0);
            } else if ("wandb".equalsIgnoreCase(v)) {
                this.reportToName = "wandb";
                super.reportTo(1);
            } else if ("mlflow".equalsIgnoreCase(v)) {
                this.reportToName = "mlflow";
                super.reportTo(2);
            } else if ("all".equalsIgnoreCase(v)) {
                this.reportToName = "all";
                super.reportTo(3);
            } else {
                this.reportToName = v;
            }
            return this;
        }
        public Builder report_to(String v) { return reportToName(v); }
        public Builder report_to(int v) { super.reportTo(v); return this; }

        public Builder learning_rate(double v) { return learningRate(v); }
        public Builder output_dir(String v) { return outputDir(v); }
        public Builder per_device_eval_batch_size(int v) { return perDeviceEvalBatchSize(v); }
        public Builder gradient_accumulation_steps(int v) { return gradientAccumulationSteps(v); }
        public Builder eval_steps(int v) { return evalSteps(v); }
        public Builder max_grad_norm(double v) { return maxGradNorm(v); }
        public Builder bf16(boolean v) { return super.bf16(v); }
        public Builder fp16(boolean v) { return super.fp16(v); }

        // Snake_case aliases for parent
        public Builder warmup_ratio(double v) { return warmupRatio(v); }
        public Builder warmup_steps(int v) { return warmupSteps(v); }
        public Builder gradient_checkpointing(boolean v) { return gradientCheckpointing(v); }
        public Builder per_device_train_batch_size(int v) { return perDeviceTrainBatchSize(v); }
        public Builder num_train_epochs(int v) { return numTrainEpochs(v); }
        public Builder save_steps(int v) { return saveSteps(v); }
        public Builder logging_steps(int v) { return loggingSteps(v); }
        /**
         * Always persist the scheduler name. {@code "linear"} additionally
         * flips {@link #lrSchedulerLinear(boolean)}; other values (cosine,
         * constant, …) used to be silently dropped — that was a bug.
         */
        public Builder lr_scheduler_type(String v) {
            if (v == null) return this;
            lrSchedulerType(v);
            if ("linear".equalsIgnoreCase(v)) return lrSchedulerLinear(true);
            return lrSchedulerLinear(false);
        }
        public Builder group_by_length(boolean v) { return groupByLength(v); }

        /**
         * Post-init validation: defaults that depend on other fields
         * (e.g. loss_type when not explicitly set).
         */
        private void postInit() {
            // Default loss_type to "chunked_nll" (matches TRL 0.x default).
            if (lossType == null) {
                lossType("chunked_nll");
            }
            // packing implies padding_free with bfd strategy.
            if (packing && "bfd".equals(packingStrategy) && !paddingFree) {
                paddingFree(true);
            }
        }

        @Override
        public SFTConfig build() {
            postInit();
            return new SFTConfig(this);
        }
    }
}