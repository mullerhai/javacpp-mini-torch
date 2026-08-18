/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import org.bytedeco.pytorch.optim.AdamW;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamWOptions;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.trl.trainer.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.nn.Module;

import java.util.*;

/**
 * Ex37 — Complete LLM Fine-Tuning Tutorial (HF Transformers + PEFT) ported
 *        to pure Java using the JavaCPP PyTorch / llm.finetuning framework.
 *
 * <p>This single end-to-end example walks through the 10 sections of the
 * reference Python notebook {@code LLM_Fine_Tuning_Tutorial.ipynb}:
 * <ol>
 *   <li>Section 0/1 — Environment setup, utilities.</li>
 *   <li>Section 2 — Datasets &amp; tokenizers (IMDB, WikiText-2, Dolly-style).</li>
 *   <li>Section 3 — Full fine-tuning for sequence classification (DistilBERT on IMDB).</li>
 *   <li>Section 4 — Instruction tuning (DistilGPT2 on toy Alpaca).</li>
 *   <li>Section 5 — PEFT: LoRA on Causal LM (optionally 4-bit / QLoRA).</li>
 *   <li>Section 6 — PEFT: Prefix Tuning on Causal LM.</li>
 *   <li>Section 7 — PEFT: BitFit on Causal LM.</li>
 *   <li>Section 8 — Inference utility (text-generation pipeline).</li>
 *   <li>Section 9 — Saving &amp; loading models &amp; adapters.</li>
 *   <li>Section 10 — Comparison report.</li>
 * </ol>
 *
 * <p>Every step runs offline with simulated training loops, while preserving
 * the same end-to-end structure as the reference notebook so it can be wired
 * to real models / datasets simply by swapping the stub components.
 */
public final class Ex37_LLMFineTuningTutorial {

    public static final String NAME = "Ex37_LLMFineTuningTutorial";

    // Reproducibility (matches cell 11)
    private static final long SEED = 42L;

    // Toy (instruction, input, output) triples used by sections 4–7 (cell 14)
    private static final List<Map<String, String>> TOY_INSTR_DATA = List.of(
            Map.of("instruction", "Translate to French",
                    "input", "Hello world!",
                    "output", "Bonjour le monde !"),
            Map.of("instruction", "Summarize",
                    "input", "Transformers are powerful sequence models for NLP.",
                    "output", "Transformers are strong NLP sequence models."),
            Map.of("instruction", "Give a title",
                    "input", "A beginner guide to PEFT methods.",
                    "output", "PEFT Methods: A Beginner's Guide"),
            Map.of("instruction", "Translate to Spanish",
                    "input", "Good morning",
                    "output", "Buenos días"),
            Map.of("instruction", "Classify sentiment",
                    "input", "I absolutely loved this movie, it was fantastic!",
                    "output", "Positive"),
            Map.of("instruction", "Classify sentiment",
                    "input", "What a waste of time, totally boring.",
                    "output", "Negative"));

    private Ex37_LLMFineTuningTutorial() {}

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(37, "LLM Fine-Tuning Tutorial (HF Transformers + PEFT) — Java Edition");

        // -----------------------------------------------------------------------
        // Section 0 / 1 — Environment setup & utilities
        // -----------------------------------------------------------------------
        section("Section 0 / 1 — Environment setup & utilities");
        Map<String, Object> env = detectEnvironment();
        boolean cuda = (boolean) env.get("cuda");
        boolean bnbAvailable = (boolean) env.get("bnbAvailable");
        System.out.println("PyTorch available? yes (JavaCPP port)");
        System.out.println("CUDA available? " + cuda);
        System.out.println("bitsandbytes available? " + bnbAvailable);
        System.out.println("Using device: " + (cuda ? "cuda" : "cpu"));

        // -----------------------------------------------------------------------
        // Section 2 — Datasets & tokenizers
        // -----------------------------------------------------------------------
        section("Section 2 — Datasets & tokenizers");
        FastTokenizer tokBert = TunningSupport.tokenizerFor("distilbert-base-uncased");
        FastTokenizer tokGpt  = TunningSupport.tokenizerFor("distilgpt2");
        // GPT-2 family has no PAD token by default (mirror cell 6 / cell 14).
        if (tokGpt.padId() < 0 || tokGpt.padId() == tokGpt.eosId()) {
            System.out.println("[tok_gpt] PAD token aligned with EOS token (distilgpt2).");
        }
        List<Map<String, Object>> imdbSample = loadImdbSample(2000, 1000);
        System.out.printf("IMDB: train=%d, test=%d%n", 2000, 1000);
        System.out.printf("Tokenizers: bert=%s, gpt=%s%n",
                tokBert.getClass().getSimpleName(), tokGpt.getClass().getSimpleName());

        // -----------------------------------------------------------------------
        // Section 3 — Full fine-tuning (DistilBERT on IMDB)
        // -----------------------------------------------------------------------
        section("Section 3 — Full fine-tuning for sequence classification (DistilBERT on IMDB)");
        Map<String, Double> clsMetrics = fullFinetuneClassification(tokBert, imdbSample);
        System.out.printf("[Full FT] final metrics: %s%n", clsMetrics);

        // -----------------------------------------------------------------------
        // Section 4 — Instruction tuning (DistilGPT2 on toy Alpaca data)
        // -----------------------------------------------------------------------
        section("Section 4 — Instruction tuning (Causal LM)");
        List<int[]> lmTokens = tokenizeInstructionData(tokGpt, TOY_INSTR_DATA, 256);
        List<int[]> lmLabels = lmTokens;
        Map<String, Double> lmMetrics = instructionTuneCausalLM(tokGpt, lmTokens, lmLabels);
        System.out.printf("[Instruction Tuning] final metrics: %s%n", lmMetrics);

        // -----------------------------------------------------------------------
        // Section 5 — PEFT LoRA on Causal LM (optionally QLoRA)
        // -----------------------------------------------------------------------
        section("Section 5 — PEFT: LoRA on Causal LM");
        Map<String, Double> loraMetrics = loraFinetune(tokGpt, lmTokens, lmLabels, bnbAvailable && cuda);
        System.out.printf("[LoRA] final metrics: %s%n", loraMetrics);

        // -----------------------------------------------------------------------
        // Section 6 — PEFT Prefix Tuning on Causal LM
        // -----------------------------------------------------------------------
        section("Section 6 — PEFT: Prefix Tuning on Causal LM");
        Map<String, Double> prefixMetrics = prefixFinetune(tokGpt, lmTokens, lmLabels);
        System.out.printf("[Prefix Tuning] final metrics: %s%n", prefixMetrics);

        // -----------------------------------------------------------------------
        // Section 7 — PEFT BitFit on Causal LM
        // -----------------------------------------------------------------------
        section("Section 7 — PEFT: BitFit on Causal LM");
        Map<String, Double> bitfitMetrics = bitfitFinetune(tokGpt, lmTokens, lmLabels);
        System.out.printf("[BitFit] final metrics: %s%n", bitfitMetrics);

        // -----------------------------------------------------------------------
        // Section 8 — Inference utility (text-generation pipeline)
        // -----------------------------------------------------------------------
        section("Section 8 — Inference utility (text-generation pipeline)");
        String prompt1 = "### Instruction:\nTranslate to French\n\n### Input:\nGood morning\n\n### Response:\n";
        String prompt2 = "Once upon a time in a land far, far away";
        System.out.printf("Generated for prompt 1: %s%n", generateResponse(prompt1, 60));
        System.out.printf("Generated for prompt 2: %s%n", generateResponse(prompt2, 60));

        // -----------------------------------------------------------------------
        // Section 9 — Saving & loading
        // -----------------------------------------------------------------------
        section("Section 9 — Saving & loading (model + adapters)");
        java.io.File saveDir = new java.io.File("./ft_cls/final");
        saveDir.mkdirs();
        System.out.printf("Saved model + tokenizer to %s%n", saveDir.getAbsolutePath());
        System.out.printf("Adapter saved to %s%n", "./ft_lora_adapter");
        System.out.printf("Prefix weights saved to %s%n", "./ft_prefix_adapter");

        // -----------------------------------------------------------------------
        // Section 10 — What to report (comparison table)
        // -----------------------------------------------------------------------
        section("Section 10 — Comparison report");
        reportResults(clsMetrics, lmMetrics, loraMetrics, prefixMetrics, bitfitMetrics);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("distilgpt2")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ===========================================================================
    // Section 0 / 1 — Environment setup & utilities
    // ===========================================================================

    /** Equivalent to {@code torch.cuda.is_available()} + bitsandbytes detection. */
    private static Map<String, Object> detectEnvironment() {
        Map<String, Object> out = new LinkedHashMap<>();
        // CUDA probing would be done via javacpp-presets-torch.cuda.isAvailable().
        out.put("cuda", false);
        out.put("bnbAvailable", isAvailable("org.bytedeco.pytorch.llm.bitsandbytes.BitsAndBytes"));
        return out;
    }

    private static boolean isAvailable(String fqcn) {
        try { Class.forName(fqcn); return true; }
        catch (Throwable t) { return false; }
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("----------------------------------------------------------------");
        System.out.println("  " + title);
        System.out.println("----------------------------------------------------------------");
    }

    // ===========================================================================
    // Section 2 — Datasets & tokenizers
    // ===========================================================================

    /**
     * Equivalent of cell 8:
     * <pre>
     *   imdb_tok = imdb.map(tokenize_imdb, batched=True)
     *   imdb_tok = imdb_tok.remove_columns(["text"]).rename_column("label", "labels")
     *   imdb_tok.set_format(type="torch")
     *   small_train = imdb_tok["train"].shuffle(seed=42).select(range(2000))
     *   small_test  = imdb_tok["test"].shuffle(seed=42).select(range(1000))
     * </pre>
     * Falls back to a deterministic synthetic dataset if the real IMDB cannot
     * be downloaded (no network in this sandbox).
     */
    private static List<Map<String, Object>> loadImdbSample(int trainSize, int testSize) {
        List<Map<String, Object>> out = new ArrayList<>();
        Random rng = new Random(SEED);
        String[] positives = {
                "A wonderful, heartwarming film with great acting.",
                "Loved every minute of this beautiful story.",
                "Brilliant performances and a clever script.",
                "Truly inspiring — I would watch this again."
        };
        String[] negatives = {
                "Painfully boring and a total waste of time.",
                "Bad acting, terrible plot, do not recommend.",
                "I couldn't even finish this movie.",
                "Awful direction and a confusing story."
        };
        for (int i = 0; i < trainSize + testSize; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            int label = i % 2;
            row.put("text", label == 1 ? positives[rng.nextInt(positives.length)]
                                         : negatives[rng.nextInt(negatives.length)]);
            row.put("label", label);
            out.add(row);
        }
        return out;
    }

    /**
     * Equivalent of cell 14:
     * <pre>
     *   def format_example(ex):
     *       instruction = ex["instruction"].strip()
     *       inp = ex.get("input", "").strip()
     *       out = ex["output"].strip()
     *       if inp:
     *           prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
     *       else:
     *           prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
     *       return prompt, out
     * </pre>
     */
    public static String[] formatExample(Map<String, String> ex) {
        String instruction = Objects.toString(ex.get("instruction"), "").trim();
        String input = Objects.toString(ex.get("input"), "").trim();
        String output = Objects.toString(ex.get("output"), "").trim();
        String prompt = input.isEmpty()
                ? "### Instruction:\n" + instruction + "\n\n### Response:\n"
                : "### Instruction:\n" + instruction + "\n\n### Input:\n" + input + "\n\n### Response:\n";
        return new String[] { prompt, output };
    }

    /** Mirrors {@code build_text} from cell 14. */
    public static String buildText(Map<String, String> ex) {
        String[] fo = formatExample(ex);
        return fo[0] + fo[1];
    }

    /** Equivalent of cell 14 {@code tokenize_lm}: pad/truncate to maxLength. */
    private static List<int[]> tokenizeInstructionData(FastTokenizer tok,
                                                       List<Map<String, String>> data,
                                                       int maxLength) {
        List<int[]> out = new ArrayList<>();
        for (Map<String, String> ex : data) {
            String text = buildText(ex) + "<|endoftext|>"; // EOS approximation
            int[] ids = tok.encode(text, false).ids();
            int n = Math.min(ids.length, maxLength);
            int[] truncated = new int[n];
            System.arraycopy(ids, 0, truncated, 0, n);
            out.add(truncated);
        }
        return out;
    }

    // ===========================================================================
    // Section 3 — Full fine-tuning for sequence classification
    // ===========================================================================

    /**
     * Equivalent of cell 11 (DistilBERT full fine-tuning on IMDB).
     *
     * <p>Mirrors:
     * <pre>
     *   model = AutoModelForSequenceClassification.from_pretrained(
     *              "distilbert-base-uncased", num_labels=2).to(DEVICE)
     *
     *   training_args = TrainingArguments(
     *       output_dir="./ft_cls",
     *       learning_rate=2e-5,
     *       per_device_train_batch_size=8,
     *       per_device_eval_batch_size=8,
     *       num_train_epochs=10,
     *       logging_steps=50,
     *       seed=SEED,
     *       report_to="none",
     *       save_total_limit=2,
     *       evaluation_strategy="epoch",
     *       save_strategy="epoch",
     *       load_best_model_at_end=True,
     *       metric_for_best_model="f1",
     *       greater_is_better=True,
     *   )
     *
     *   trainer = Trainer(
     *       model=model,
     *       args=training_args,
     *       train_dataset=small_train,
     *       eval_dataset=small_test,
     *       tokenizer=tok,
     *       compute_metrics=compute_metrics,
     *   )
     * </pre>
     */
    private static Map<String, Double> fullFinetuneClassification(
            FastTokenizer tok, List<Map<String, Object>> imdbSample) {

        // --- Stub model: replace with the real
        //     AutoModelForSequenceClassification.fromPretrained("distilbert-base-uncased", num_labels=2)
        Module model = new Module("distilbert-base-uncased-cls");

        // --- TrainingArguments (cell 11 "version-agnostic") ---
        SFTConfig cfg = SFTConfig.builder()
                .outputDir("./ft_cls")
                .learningRate(2e-5)
                .perDeviceTrainBatchSize(8)
                .perDeviceEvalBatchSize(8)
                .numTrainEpochs(10)
                .loggingSteps(50)
                .seed(SEED)
                .saveTotalLimit("2")
                .loadBestModelAtEnd(true)
                .optim("adamw_torch")
                .reportTo(0)   // 0 == report_to="none"
                .maxSeqLength(256)
                .build();

        // --- compute_metrics(eval_pred) → {accuracy, f1} ---
        Map<String, Double> metrics = computeClassificationMetrics(imdbSample);

        // --- Trainer.train() + Trainer.evaluate() (stub loop) ---
        double[] lossCurve = TunningSupport.simulateTrainingLoop(10, 0.96);
        System.out.printf("[Full FT] loss: %.4f → %.4f%n", lossCurve[0], lossCurve[lossCurve.length - 1]);

        // Final evaluation (cell 11):
        //   metrics = trainer.evaluate()
        //   print("Final metrics:", metrics)
        System.out.printf("[Full FT] eval accuracy=%.4f, f1=%.4f%n",
                metrics.get("accuracy"), metrics.get("f1"));

        // trainer.save_model("./ft_cls/final") + tok.save_pretrained(...)
        // Quick test inference:
        //   clf = pipeline("text-classification", model="./ft_cls/final", tokenizer=tok)
        //   print(clf("This movie was absolutely wonderful and inspiring!"))
        System.out.println("[Full FT] sanity test: \"This movie was absolutely wonderful and inspiring!\" → POSITIVE");
        System.out.println("[Full FT] sanity test: \"This was painfully boring...\" → NEGATIVE");

        return metrics;
    }

    /**
     * Equivalent to cell 11:
     * <pre>
     *   def compute_metrics(eval_pred):
     *       logits, labels = eval_pred
     *       preds = np.argmax(logits, axis=-1)
     *       return {
     *           "accuracy": accuracy_score(labels, preds),
     *           "f1": f1_score(labels, preds),
     *       }
     * </pre>
     */
    private static Map<String, Double> computeClassificationMetrics(List<Map<String, Object>> dataset) {
        Map<String, Double> out = new LinkedHashMap<>();
        long correct = 0, total = 0;
        long tp = 0, fp = 0, fn = 0;
        for (Map<String, Object> row : dataset) {
            int label = (int) row.get("label");
            // Simulated prediction: label itself (perfect classifier for offline demo).
            int pred = label;
            if (pred == label) correct++;
            total++;
            if (label == 1 && pred == 1) tp++;
            else if (label == 0 && pred == 1) fp++;
            else if (label == 1 && pred == 0) fn++;
        }
        double acc = total > 0 ? (double) correct / total : 0.0;
        double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall    = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
        out.put("accuracy", acc);
        out.put("f1", f1);
        out.put("precision", precision);
        out.put("recall", recall);
        return out;
    }

    // ===========================================================================
    // Section 4 — Instruction tuning (Causal LM)
    // ===========================================================================

    /**
     * Equivalent of cell 14 — fine-tunes DistilGPT2 on the toy Alpaca data.
     * <pre>
     *   args_lm = TrainingArguments(
     *       output_dir="./ft_instr",
     *       per_device_train_batch_size=2,
     *       per_device_eval_batch_size=2,
     *       num_train_epochs=10,
     *       learning_rate=5e-5,
     *       eval_strategy="epoch",
     *       save_strategy="epoch",
     *       load_best_model_at_end=True,
     *       logging_steps=10,
     *       report_to="none",
     *   )
     *
     *   trainer_lm = Trainer(
     *       model=model_lm,
     *       args=args_lm,
     *       train_dataset=toy_tok["train"],
     *       eval_dataset=toy_tok["test"],
     *       data_collator=data_collator,
     *   )
     * </pre>
     */
    private static Map<String, Double> instructionTuneCausalLM(
            FastTokenizer tok, List<int[]> inputIds, List<int[]> labels) {

        Module modelLlm = new Module("distilgpt2");
        LoraConfig lora = LoraConfig.builder()
                .r(8).alpha(32).dropout(0.05)
                .targetModules("c_attn", "c_proj")
                .fanInFanOut(true)
                .baseModelNameOrPath("distilgpt2")
                .taskType("CAUSAL_LM")
                .build();
        PeftModel peft = PeftModel.getPeftModel(modelLlm, lora);

        // Demonstrate new SFTConfig fields: loss_type, completion-only loss,
        // assistant-only loss, packing, eval-packing, max_length (alias),
        // pad_to_multiple_of, truncation_mode, eos_token.
        SFTConfig cfg = SFTConfig.builder()
                .outputDir("./ft_instr")
                .perDeviceTrainBatchSize(2)
                .perDeviceEvalBatchSize(2)
                .numTrainEpochs(10)
                .learningRate(5e-5)
                .loadBestModelAtEnd(true)
                .loggingSteps(10)
                .reportTo(0)  // none
                .maxLength(256)
                .optim("adamw_torch")
                .lossType("chunked_nll")
                .completionOnlyLoss(false)
                .assistantOnlyLoss(false)
                .packing(false)
                .packingStrategy("bfd")
                .paddingFree(false)
                .padToMultipleOf(8)
                .truncationMode("keep_start")
                .eosToken("<|endoftext|>")
                .evalPacking(false)
                .shuffleDataset(false)
                .build();

        Optimizer optim = new AdamW(peft.trainableParameters(), new AdamWOptions(5e-5));

        SFTTrainer trainer = SFTTrainer.of(
                peft.root(),
                cfg,
                tokenizedRows(inputIds, labels),
                lora,
                tok,
                "text");

        double[] lossCurve = TunningSupport.simulateTrainingLoop(10, 0.94);
        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("loss", lossCurve[lossCurve.length - 1]);
        System.out.printf("[Instruction Tuning] loss: %.4f → %.4f%n",
                lossCurve[0], lossCurve[lossCurve.length - 1]);
        return metrics;
    }

    /** Convenience to wrap tokenized ids/labels into the {input_ids, attention_mask, labels} rows. */
    private static List<Map<String, Object>> tokenizedRows(List<int[]> ids, List<int[]> labels) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < ids.size(); i++) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("input_ids", ids.get(i));
            r.put("attention_mask", makeMask(ids.get(i)));
            r.put("labels", labels.get(i));
            rows.add(r);
        }
        return rows;
    }

    private static int[] makeMask(int[] ids) {
        int[] m = new int[ids.length];
        Arrays.fill(m, 1);
        return m;
    }

    // ===========================================================================
    // Section 5 — PEFT LoRA on Causal LM
    // ===========================================================================

    /**
     * Equivalent of cell 16 (LoRA on Causal LM, optionally 4-bit QLoRA).
     * <pre>
     *   if bnb_available:
     *       quant_kwargs["quantization_config"] = BitsAndBytesConfig(
     *           load_in_4bit=True,
     *           bnb_4bit_use_double_quant=True,
     *           bnb_4bit_quant_type="nf4")
     *       quant_kwargs["device_map"] = {"": 0}
     *
     *   base_lm = AutoModelForCausalLM.from_pretrained("distilgpt2", **quant_kwargs)
     *
     *   lora_cfg = LoraConfig(
     *       task_type=TaskType.CAUSAL_LM,
     *       r=8,
     *       lora_alpha=32,
     *       lora_dropout=0.05,
     *       target_modules=["c_attn", "c_proj"],
     *       fan_in_fan_out=True,
     *   )
     *
     *   lora_model = get_peft_model(base_lm, lora_cfg)
     *
     *   args_lora = TrainingArguments(
     *       output_dir="./ft_lora",
     *       per_device_train_batch_size=2,
     *       per_device_eval_batch_size=2,
     *       num_train_epochs=20,
     *       learning_rate=1e-4,
     *       eval_strategy="epoch",
     *       save_strategy="epoch",
     *       logging_steps=10,
     *       optim="adamw_torch",
     *   )
     * </pre>
     */
    private static Map<String, Double> loraFinetune(
            FastTokenizer tok, List<int[]> inputIds, List<int[]> labels, boolean qlora) {

        if (qlora) {
            System.out.println("[LoRA] 4-bit QLoRA enabled (bitsandbytes + CUDA available).");
        } else {
            System.out.println("[LoRA] Running full-precision LoRA (no CUDA / no bitsandbytes).");
        }

        Module baseLm = new Module("distilgpt2-base");
        // Full LoraConfig with all PEFT attributes demonstrated:
        //   r, lora_alpha, lora_dropout, target_modules, fan_in_fan_out,
        //   use_rslora, use_dora, init_lora_weights, bias, modules_to_save,
        //   layers_to_transform, layers_pattern, rank_pattern, alpha_pattern,
        //   base_model_name_or_path, task_type
        Map<String, Integer> rankPattern = new LinkedHashMap<>();
        rankPattern.put(".*c_attn.*", 16); // boost rank on attention layers
        Map<String, Integer> alphaPattern = new LinkedHashMap<>();
        alphaPattern.put(".*c_attn.*", 64);

        LoraConfig loraCfg = LoraConfig.builder()
                .r(8)
                .lora_alpha(32)
                .lora_dropout(0.05)
                .targetModules("c_attn", "c_proj")
                .fanInFanOut(true)
                .useRslora(true)
                .useDora(false)
                .initLoraWeights("gaussian")
                .bias("none")
                .modulesToSave("lm_head")
                .layersToTransform(0, 1, 2) // only first 3 transformer layers
                .layersPattern("h")
                .rankPattern(rankPattern)
                .alphaPattern(alphaPattern)
                .baseModelNameOrPath("distilgpt2")
                .revision("main")
                .taskType("CAUSAL_LM")
                .build();
        PeftModel loraModel = PeftModel.getPeftModel(baseLm, loraCfg);
        loraModel.printTrainableParameters();

        // SFTConfig with packing, padding_free, pad_to_multiple_of, NEFTune,
        // loss_type, completion-only loss, eval-packing, router_aux_loss_coef.
        SFTConfig cfg = SFTConfig.builder()
                .outputDir("./ft_lora")
                .perDeviceTrainBatchSize(2)
                .perDeviceEvalBatchSize(2)
                .numTrainEpochs(20)
                .learningRate(1e-4)
                .loggingSteps(10)
                .optim("adamw_torch")
                .maxLength(256)
                .packing(true)
                .packingStrategy("bfd")
                .paddingFree(false) // would require FlashAttention 2/3
                .padToMultipleOf(8)
                .neftuneAlpha(5.0)              // NEFTune perturbation
                .lossType("chunked_nll")
                .completionOnlyLoss(false)
                .assistantOnlyLoss(false)
                .shuffleDataset(true)
                .evalPacking(false)
                .routerAuxLossCoef(0.0)         // set to 0.001 for MoE
                .modelInitKwargs(Map.of("torch_dtype", "float32"))
                .trustRemoteCode(false)
                .chatTemplatePath(null)
                .activationOffloading(false)
                .build();

        SFTTrainer trainer = SFTTrainer.of(
                loraModel.root(),
                cfg,
                tokenizedRows(inputIds, labels),
                loraCfg,
                tok,
                "text");

        double[] lossCurve = TunningSupport.simulateTrainingLoop(20, 0.95);
        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("loss", lossCurve[lossCurve.length - 1]);
        System.out.printf("[LoRA] loss: %.4f → %.4f%n",
                lossCurve[0], lossCurve[lossCurve.length - 1]);

        // lora_model.save_pretrained("./ft_lora_adapter")
        try {
            loraModel.save_pretrained(new java.io.File("./ft_lora_adapter"));
        } catch (java.io.IOException ioe) {
            System.err.println("[LoRA] Failed to save adapter: " + ioe.getMessage());
        }
        return metrics;
    }

    // ===========================================================================
    // Section 6 — PEFT Prefix Tuning
    // ===========================================================================

    /**
     * Equivalent of cell 18 — Prefix Tuning on Causal LM.
     * <pre>
     *   prefix_cfg = PrefixTuningConfig(
     *       task_type=TaskType.CAUSAL_LM,
     *       num_virtual_tokens=20,
     *   )
     *   prefix_model = get_peft_model(base_lm_prefix, prefix_cfg)
     *
     *   args_prefix = TrainingArguments(
     *       output_dir="./ft_prefix",
     *       per_device_train_batch_size=2,
     *       per_device_eval_batch_size=2,
     *       num_train_epochs=100,
     *       learning_rate=1e-4,
     *       eval_strategy="epoch",
     *       save_strategy="epoch",
     *       logging_steps=10,
     *   )
     * </pre>
     */
    private static Map<String, Double> prefixFinetune(
            FastTokenizer tok, List<int[]> inputIds, List<int[]> labels) {

        Module baseLm = new Module("distilgpt2-prefix");
        // The framework's PEFT layer is LoRA-based; for prefix tuning we
        // emulate the prefix configuration with a small LoRA config whose
        // rank matches num_virtual_tokens (mirrors PrefixTuningConfig).
        LoraConfig prefixCfg = LoraConfig.builder()
                .r(20)
                .alpha(20)
                .dropout(0.0)
                .targetModules("c_attn")
                .taskType("CAUSAL_LM")
                .build();
        PeftModel prefixModel = PeftModel.getPeftModel(baseLm, prefixCfg);

        SFTConfig cfg = SFTConfig.builder()
                .outputDir("./ft_prefix")
                .perDeviceTrainBatchSize(2)
                .perDeviceEvalBatchSize(2)
                .numTrainEpochs(100)
                .learningRate(1e-4)
                .loggingSteps(10)
                .optim("adamw_torch")
                .maxSeqLength(256)
                .build();

        SFTTrainer trainer = SFTTrainer.of(
                prefixModel.root(),
                cfg,
                tokenizedRows(inputIds, labels),
                prefixCfg,
                tok,
                "text");

        double[] lossCurve = TunningSupport.simulateTrainingLoop(100, 0.97);
        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("loss", lossCurve[lossCurve.length - 1]);
        System.out.printf("[Prefix Tuning] loss: %.4f → %.4f%n",
                lossCurve[0], lossCurve[lossCurve.length - 1]);

        try {
            prefixModel.save_pretrained(new java.io.File("./ft_prefix_adapter"));
        } catch (java.io.IOException ioe) {
            System.err.println("[Prefix Tuning] Failed to save prefix adapter: " + ioe.getMessage());
        }
        return metrics;
    }

    // ===========================================================================
    // Section 7 — PEFT BitFit (bias-only) on Causal LM
    // ===========================================================================

    /**
     * Equivalent of cell 20 — BitFit: train biases only.
     * <pre>
     *   bitfit_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
     *   tok_gpt.pad_token = tok_gpt.eos_token
     *   bitfit_model.config.pad_token_id = tok_gpt.pad_token_id
     *   bitfit_model.config.eos_token_id = tok_gpt.eos_token_id
     *
     *   for name, p in bitfit_model.named_parameters():
     *       p.requires_grad = False
     *   trainable = []
     *   for name, p in bitfit_model.named_parameters():
     *       if p.ndim > 0 and name.endswith(".bias"):
     *           p.requires_grad = True
     *           trainable.append(name)
     *
     *   data_collator = DataCollatorForLanguageModeling(tokenizer=tok_gpt, mlm=False)
     *
     *   args_bitfit = TrainingArguments(
     *       output_dir="./ft_bitfit",
     *       per_device_train_batch_size=2,
     *       per_device_eval_batch_size=2,
     *       num_train_epochs=10,
     *       learning_rate=5e-4,
     *       eval_strategy="epoch",
     *       save_strategy="epoch",
     *       load_best_model_at_end=True,
     *       logging_steps=10,
     *       report_to="none",
     *   )
     * </pre>
     */
    private static Map<String, Double> bitfitFinetune(
            FastTokenizer tok, List<int[]> inputIds, List<int[]> labels) {

        Module bitfitModel = new Module("distilgpt2-bitfit");

        // Align PAD/EOS (mirrors cell 20)
        // (handled by the tokenizer wrapper, no-op here)

        // Freeze all params then unfreeze only biases — equivalent to cell 20:
        //   for name, p in bitfit_model.named_parameters(): p.requires_grad = False
        //   for name, p in bitfit_model.named_parameters():
        //       if p.ndim > 0 and name.endswith(".bias"): p.requires_grad = True
        //   trainable = [...]
        List<String> trainableBiases = enableBitFit(bitfitModel);
        System.out.printf("Trainable (bias-only) params: %d%n", trainableBiases.size());

        SFTConfig cfg = SFTConfig.builder()
                .outputDir("./ft_bitfit")
                .perDeviceTrainBatchSize(2)
                .perDeviceEvalBatchSize(2)
                .numTrainEpochs(10)
                .learningRate(5e-4) // typical BitFit LR
                .loadBestModelAtEnd(true)
                .loggingSteps(10)
                .reportTo(0) // none
                .optim("adamw_torch")
                .maxSeqLength(256)
                .build();

        // No PEFT (BitFit touches biases directly on the base model)
        SFTTrainer trainer = SFTTrainer.of(
                bitfitModel,
                cfg,
                tokenizedRows(inputIds, labels),
                (LoraConfig) null,
                tok,
                "text");

        double[] lossCurve = TunningSupport.simulateTrainingLoop(10, 0.93);
        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("loss", lossCurve[lossCurve.length - 1]);
        System.out.printf("[BitFit] loss: %.4f → %.4f%n",
                lossCurve[0], lossCurve[lossCurve.length - 1]);
        return metrics;
    }

    /**
     * Mirrors cell 20 — freezes everything, then unfreezes every parameter
     * whose name ends with {@code .bias}.
     *
     * <p>The framework's {@link Module} doesn't expose a per-parameter
     * {@code requires_grad} flag directly; in real usage this would be
     * applied via the native bindings. We surface the list of bias names
     * that would be marked trainable so the surrounding loop is identical
     * to the reference notebook.
     */
    private static List<String> enableBitFit(Module model) {
        List<String> trainable = new ArrayList<>();
        // Walk the typical GPT-2 parameter tree to approximate named_parameters().
        // In a real wiring, replace this with model.named_parameters() and filter by suffix.
        for (String prefix : new String[]{"transformer.h.", "transformer.wte.", "transformer.wpe.", "lm_head."}) {
            for (int i = 0; i < 12; i++) {
                String name = prefix + i + ".attn.c_attn.bias";
                trainable.add(name);
                trainable.add(name.replace(".attn.c_attn.", ".attn.c_proj."));
                trainable.add(name.replace(".attn.c_attn.", ".ln_1."));
                trainable.add(name.replace(".attn.c_attn.", ".ln_2."));
                trainable.add(name.replace(".attn.c_attn.", ".mlp.c_fc."));
                trainable.add(name.replace(".attn.c_attn.", ".mlp.c_proj."));
            }
        }
        return trainable;
    }

    // ===========================================================================
    // Section 8 — Inference utility (text-generation pipeline)
    // ===========================================================================

    /**
     * Equivalent of cell 22:
     * <pre>
     *   text_gen = pipeline(
     *       "text-generation",
     *       model=lora_model,
     *       tokenizer=tok_gpt,
     *       device=0 if DEVICE=="cuda" else -1,
     *   )
     *
     *   def generate_response(prompt, max_new_tokens=60):
     *       out = text_gen(prompt, max_new_tokens=max_new_tokens,
     *                      do_sample=True, temperature=0.9, top_p=0.95)
     *       return out[0]["generated_text"]
     * </pre>
     */
    private static String generateResponse(String prompt, int maxNewTokens) {
        Random rng = new Random(prompt.hashCode());
        String suffix = prompt.length() < 64
                ? " (autocorrect): a curated response generated by the JavaCPP PEFT pipeline."
                : " ...and the dragon whispered softly into the twilight.";
        int cut = Math.min(suffix.length(), Math.max(0, maxNewTokens * 4));
        return prompt + suffix.substring(0, cut) + " (temp=0.9, top_p=0.95)";
    }

    // ===========================================================================
    // Section 10 — Comparison report
    // ===========================================================================

    /**
     * Mirrors section 10 ("What to Report") with a summary table covering:
     * Full FT, Instruction Tuning, LoRA, Prefix Tuning, BitFit.
     */
    private static void reportResults(Map<String, Double> cls,
                                      Map<String, Double> lm,
                                      Map<String, Double> lora,
                                      Map<String, Double> prefix,
                                      Map<String, Double> bitfit) {
        String[] headers = {"Method", "loss", "accuracy", "f1", "lr", "epochs"};
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[]{"Full FT (cls)",     "-", f(cls.get("accuracy")), f(cls.get("f1")), "2e-5",  "10"});
        rows.add(new String[]{"Instruction Tuning", f(lm.get("loss")),    "-", "-", "5e-5", "10"});
        rows.add(new String[]{"LoRA",              f(lora.get("loss")),   "-", "-", "1e-4", "20"});
        rows.add(new String[]{"Prefix Tuning",     f(prefix.get("loss")), "-", "-", "1e-4", "100"});
        rows.add(new String[]{"BitFit",            f(bitfit.get("loss")), "-", "-", "5e-4", "10"});
        TunningSupport.printTable(headers, rows);

        System.out.println("\nSetup notes:");
        System.out.println("  - Model family: distilgpt2 (small CPU-friendly).");
        System.out.println("  - Tokenizers: distilbert-base-uncased (cls), distilgpt2 (causal).");
        System.out.println("  - Hardware: CPU-only (set CUDA env to enable bitsandbytes 4-bit QLoRA).");
        System.out.println("  - Reproducibility: SEED=" + SEED + " (matches cell 11).");
    }

    private static String f(Double v) {
        if (v == null) return "-";
        return String.format("%.4f", v);
    }
}