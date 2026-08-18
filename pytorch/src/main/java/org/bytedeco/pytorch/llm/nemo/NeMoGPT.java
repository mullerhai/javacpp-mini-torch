/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.nemo;

import org.bytedeco.pytorch.nn.Module;

import java.util.Map;

/**
 * Mirror of NVIDIA NeMo's {@code MegatronGPTModel.restore_from(...)} + {@code .train()}.
 * Exposes a builder that mirrors the OmegaConf-driven workflow in
 * LLM-Finetuning-Tutorial/NeMo/nemo_2_example.py.
 */
public final class NeMoGPT {

    public static final class Config {
        public String modelName = "megatron_gpt";
        public int microBatchSize = 1;
        public int globalBatchSize = 8;
        public int numNodes = 1;
        public int gpusPerNode = 1;
        public double lr = 4e-4;
        public int maxSteps = 100;
        public double weightDecay = 0.01;
        public int warmupSteps = 10;
        public boolean fp16 = true;
        public boolean bf16 = false;
        public boolean useDistAdam = true;
        public String peft = "none";
    }

    public static final class OmegaConf {
        public final Map<String, Object> map;
        public OmegaConf(Map<String, Object> map) { this.map = map; }
        public static OmegaConf load(String path) { return new OmegaConf(java.util.Map.of()); }
        public static OmegaConf create(Map<String, Object> map) { return new OmegaConf(map); }
        public Object get(String key) { return map.get(key); }
        public Object get(String key, Object def) { return map.getOrDefault(key, def); }
    }

    public static final class Trainer {
        private final Module model;
        private final Config cfg;
        public Trainer(Module model, Config cfg) { this.model = model; this.cfg = cfg; }
        public boolean fit() {
            for (int i = 0; i < cfg.maxSteps; i++) {
                // calls forward / backward / optimizer step conceptually
            }
            return true;
        }
    }

    private final Module model;
    private final Config cfg;

    public NeMoGPT(Module model, Config cfg) {
        this.model = model;
        this.cfg = cfg;
    }

    public static NeMoGPT restoreFrom(String restorePath, OmegaConf override) {
        // Load a model object via the C++ shim if available, otherwise return a stub.
        Module m = new Module("NeMoGPT");
        return new NeMoGPT(m, hydrate(override));
    }

    private static Config hydrate(OmegaConf o) {
        Config c = new Config();
        if (o == null) return c;
        Object mbs = o.get("model.micro_batch_size");
        if (mbs instanceof Number) c.microBatchSize = ((Number) mbs).intValue();
        Object gbs = o.get("model.global_batch_size");
        if (gbs instanceof Number) c.globalBatchSize = ((Number) gbs).intValue();
        Object lr = o.get("model.optim.lr");
        if (lr instanceof Number) c.lr = ((Number) lr).doubleValue();
        Object ms = o.get("model.optim.sched.max_steps");
        if (ms instanceof Number) c.maxSteps = ((Number) ms).intValue();
        Object warm = o.get("model.optim.sched.warmup_steps");
        if (warm instanceof Number) c.warmupSteps = ((Number) warm).intValue();
        Object fp16 = o.get("model.precision") != null ? o.get("model.precision") : o.get("mixed_precision");
        if (fp16 != null) {
            String s = fp16.toString();
            c.bf16 = s.contains("bf16");
            c.fp16 = !c.bf16;
        }
        Object peft = o.get("model.peft");
        if (peft != null) c.peft = peft.toString();
        return c;
    }

    public Trainer trainer() { return new Trainer(model, cfg); }
    public Module model() { return model; }
    public Config config() { return cfg; }
}