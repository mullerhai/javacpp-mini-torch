/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.bytedeco.pytorch.llm.bitsandbytes.BitsAndBytes;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

/**
 * Ex01 — Efficiently train Large Language Models with LoRA and Hugging Face.
 *
 * <p>Mirrors the 1st Python tutorial:
 * <pre>{@code
 *   dataset = load_dataset("yelp_review_full", split="train")
 *   tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
 *   lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
 *   model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", quantization_config=BitsAndBytesConfig(...), device_map="auto")
 *   model = get_peft_model(model, lora_config)
 *   train_args = TrainingArguments(output_dir=..., per_device_train_batch_size=4, ...)
 *   trainer = SFTTrainer(model=model, args=train_args, train_dataset=ds, peft_config=lora_config,
 *                        dataset_text_field="text", max_seq_length=512, tokenizer=tokenizer)
 *   trainer.train()
 *   lora_loaded = LoraConfig.from_pretrained('outputs')
 *   model = get_peft_model(model, lora_loaded)
 * }</pre>
 */
public final class Ex01_LoraFineTuning {

    public static final String NAME = "Ex01_LoraFineTuning";

    private Ex01_LoraFineTuning() {}

    public static void run(FastTokenizer tokenizer) throws IOException {
        TunningSupport.banner(1, "Efficiently train Large Language Models with LoRA");
        File outDir = new File("build/ex01_outputs"); outDir.mkdirs();

        // 1) Tokenizer (HF AutoTokenizer.from_pretrained)
        FastTokenizer tok = AutoTokenizer.fromPretrainedTokenizerOnly("bigscience/bloom-560m");
        System.out.println("Tokenizer pad=" + tok.padId() + " eos=" + tok.eosId());

        // 2) Build LoraConfig (HF-style with snake_case aliases)
        LoraConfig peftConfig = LoraConfig.builder()
                .r(16)
                .alpha(32)
                .dropout(0.05)
                .targetModules("q_proj", "k_proj", "v_proj", "o_proj")
                .bias("none")
                .taskType("CAUSAL_LM")
                .build();

        // 3) Base causal LM (stub when offline) — PretrainedConfig + CausalLM.fromConfig
        PretrainedConfig cfg = PretrainedConfig.builder()
                .vocabSize(50257)
                .hiddenSize(1024)
                .numAttentionHeads(16)
                .numHiddenLayers(24)
                .build();
        CausalLM model = CausalLM.fromConfig(cfg);

        // 4) BitsAndBytes 4-bit quantization + prepare model for k-bit training
        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .bnb4BitUseDoubleQuant(true)
                .bnb4BitComputeDtype("float16")
                .build();
        BitsAndBytes.prepareModelForKbitTraining(model);
        System.out.println("bnb: " + bnb);

        // 5) Wrap with PEFT — get_peft_model(model, lora_config)
        PeftModel peft = PeftModel.getPeftModel(model, peftConfig);
        peft.printTrainableParameters();

        // 6) Load dataset (in-memory fallback)
        List<Map<String, Object>> trainData = TunningSupport.loadDataset("yahma/alpaca-cleaned", null, "train", 1024);
        List<Map<String, Object>> evalData = TunningSupport.loadDataset("yahma/alpaca-cleaned", null, "test", 256);
        System.out.println("train rows=" + trainData.size() + " eval rows=" + evalData.size());

        // 7) format_prompts_func + map
        Function<Map<String, Object>, Map<String, Object>> fmt =
                TunningSupport.sftFormattingFunc(
                        TunningSupport::alpacaPrompt,
                        tok,
                        512,
                        true);
        List<Map<String, Object>> tokenizedTrain = trainData.stream().map(fmt::apply).toList();
        List<Map<String, Object>> tokenizedEval = evalData.stream().map(fmt::apply).toList();

        // 8) SFTConfig
        SFTConfig sftConfig = SFTConfig.builder()
                .maxSeqLength(512)
                .maxSteps(100)
                .perDeviceTrainBatchSize(4)
                .gradientAccumulationSteps(4)
                .optim("paged_adamw_32bit")
                .numTrainEpochs(1)
                .saveSteps(100)
                .loggingSteps(20)
                .learningRate(2e-4)
                .fp16(true)
                .maxGradNorm(0.3)
                .warmupRatio(0.03)
                .groupByLength(true)
                .lrSchedulerType("linear")
                .datasetTextField("text")
                .packing(false)
                .build();

        // 9) Create SFT Trainer
        SFTTrainer trainer = SFTTrainer.of(model, sftConfig, tokenizedTrain, peftConfig, tok, "text");

        // 10) Run training
        trainer.train();
        System.out.println("Trained " + trainer.globalStep() + " optimizer steps");

        // 11) Save adapter
        peft.savePretrained(outDir);
        System.out.println("Adapter saved to " + outDir);
    }

    /** Mirrors the simple batch feeder used by every SFTTrainer example. */
    private static BaseTrainer.BatchSupplier buildSupplier(List<Map<String, Object>> rows, int batchSize) {
        int[] cursor = {0};
        return () -> {
            if (cursor[0] >= rows.size()) return null;
            int end = Math.min(rows.size(), cursor[0] + batchSize);
            List<int[]> idsBatch = new ArrayList<>(end - cursor[0]);
            List<int[]> maskBatch = new ArrayList<>(end - cursor[0]);
            for (int i = cursor[0]; i < end; i++) {
                Map<String, Object> row = rows.get(i);
                idsBatch.add((int[]) row.get("input_ids"));
                maskBatch.add((int[]) row.get("attention_mask"));
            }
            cursor[0] = end;
            Map<String, org.bytedeco.pytorch.Tensor> out = new java.util.LinkedHashMap<>();
            out.put("input_ids", org.bytedeco.pytorch.llm.data.BatchBuilder.fromLongs(toLong(idsBatch)));
            out.put("attention_mask", org.bytedeco.pytorch.llm.data.BatchBuilder.fromLongs(toLong(maskBatch)));
            // labels = input_ids for SFT
            out.put("labels", out.get("input_ids"));
            return out;
        };
    }

    private static long[][] toLong(List<int[]> batch) {
        long[][] out = new long[batch.size()][];
        for (int i = 0; i < batch.size(); i++) {
            int[] r = batch.get(i);
            out[i] = new long[r.length];
            for (int j = 0; j < r.length; j++) out[i][j] = r[j];
        }
        return out;
    }

    public static void main(String[] args) {
        try (FastTokenizer tok = TunningSupport.tokenizerFor("bigscience/bloom-560m")) {
            run(tok);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}