/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.llamafactory;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * YAML config + CLI launcher mirror of LlamaFactory.
 */
public final class LlamaFactoryConfig {

    public static final class Builder {
        Map<String, Object> cfg = new LinkedHashMap<>();
        public Builder llamaPath(String s) { cfg.put("model_name_or_path", s); return this; }
        public Builder stage(String s) { cfg.put("stage", s); return this; }
        public Builder doRA(String s) { cfg.put("do_raised", s); return this; }
        public Builder doRFT(String s) { cfg.put("do_rft", s); return this; }
        public Builder dataset(String s) { cfg.put("dataset", s); return this; }
        public Builder template(String s) { cfg.put("template", s); return this; }
        public Builder outputDir(String s) { cfg.put("output_dir", s); return this; }
        public Builder loraRank(int r) { cfg.put("lora_rank", r); return this; }
        public Builder loraAlpha(double a) { cfg.put("lora_alpha", a); return this; }
        public Builder loraDropout(double d) { cfg.put("lora_dropout", d); return this; }
        public Builder batchSize(int bs) { cfg.put("per_device_train_batch_size", bs); return this; }
        public Builder gradAccum(int ga) { cfg.put("gradient_accumulation_steps", ga); return this; }
        public Builder epochs(int e) { cfg.put("num_train_epochs", e); return this; }
        public Builder lr(double l) { cfg.put("learning_rate", l); return this; }
        public Builder scheduler(String s) { cfg.put("lr_scheduler_type", s); return this; }
        public Builder warmupSteps(int s) { cfg.put("warmup_steps", s); return this; }
        public Builder weightDecay(double wd) { cfg.put("weight_decay", wd); return this; }
        public Builder optimizer(String s) { cfg.put("optim", s); return this; }
        public Builder fp16(boolean v) { cfg.put("fp16", v); return this; }
        public Builder bf16(boolean v) { cfg.put("bf16", v); return this; }
        public Builder bnbConfig(BitsAndBytesConfig b) {
            int bits = b.isLoadIn4Bit() ? 4 : (b.isLoadIn8Bit() ? 8 : 16);
            cfg.put("quantization_bit", bits);
            return this;
        }
        public Builder gradientCheckpointing(boolean v) { cfg.put("gradient_checkpointing", v); return this; }
        public Builder flashAttn(String s) { cfg.put("flash_attn", s); return this; }
        public Builder seed(int s) { cfg.put("seed", s); return this; }
        public Map<String, Object> build() { return cfg; }
    }

    public static Builder builder() { return new Builder(); }

    public static String serializeYaml(Map<String, Object> c) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Object> e : c.entrySet()) {
            sb.append(e.getKey()).append(": ").append(e.getValue()).append("\n");
        }
        return sb.toString();
    }

    public static FastTokenizer selectTokenizer(String modelPath) {
        try {
            return FastTokenizer.fromFile(java.nio.file.Path.of(modelPath, "tokenizer.json"));
        } catch (Exception e) {
            return FastTokenizer.builder().build();
        }
    }

    /** Resolve a scheduler by name (stub). */
    public static Object resolveScheduler(String name, double lr, int warmup, int total) {
        // Stub: return scheduler name
        return name;
    }

    /** Pretend to invoke the CLI. */
    public static boolean run(Map<String, Object> cfg) {
        return true;
    }
}
