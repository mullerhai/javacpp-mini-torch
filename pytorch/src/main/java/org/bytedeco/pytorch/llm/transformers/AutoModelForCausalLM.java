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
package org.bytedeco.pytorch.llm.transformers;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.tokenizers.DirectoryTokenizerLoader;
import org.bytedeco.pytorch.llm.transformers.modeling.GlmForCausalLM;
import org.bytedeco.pytorch.llm.transformers.modeling.LlamaForCausalLM;
import org.bytedeco.pytorch.llm.transformers.modeling.Qwen2ForCausalLM;
import org.bytedeco.pytorch.llm.transformers.modeling.Qwen3ForCausalLM;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
import org.bytedeco.pytorch.llm.bitsandbytes.BitsAndBytes;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.generation.Generator;
import org.bytedeco.pytorch.llm.transformers.loading.SnapshotFiles;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.llm.transformers.mapping.WeightMap;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code AutoModelForCausalLM.from_pretrained} entry point.
 *
 * <p>Resolves architecture via {@link ModelRegistry}, zero-copy-binds safetensors
 * via {@link WeightLoader}, and pairs with {@link FastTokenizer}.
 *
 * <pre>{@code
 * AutoModelForCausalLM.Bundle b = AutoModelForCausalLM.fromPretrained(
 *     "Qwen/Qwen2-0.5B-Instruct", hub);
 * String reply = b.chat(List.of(
 *     Map.of("role","user","content","What is 2+2?")));
 * }</pre>
 */
public final class AutoModelForCausalLM {

    private AutoModelForCausalLM() {}

    /** Loaded model + tokenizer + configs + load report (+ optional bnb quant state). */
    public static final class Bundle {
        private final Module model;
        private final FastTokenizer tokenizer;
        private final PretrainedConfig config;
        private final GenerationConfig generationConfig;
        private final Path snapshot;
        private final WeightLoader.LoadReport loadReport;
        private final ChatTemplate chatTemplate;
        private final BitsAndBytesConfig quantizationConfig;
        private final BitsAndBytes.QuantizedModel quantizedModel;

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      GenerationConfig generationConfig, Path snapshot,
                      WeightLoader.LoadReport loadReport, ChatTemplate chatTemplate) {
            this(model, tokenizer, config, generationConfig, snapshot, loadReport, chatTemplate, null, null);
        }

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      GenerationConfig generationConfig, Path snapshot,
                      WeightLoader.LoadReport loadReport, ChatTemplate chatTemplate,
                      BitsAndBytesConfig quantizationConfig,
                      BitsAndBytes.QuantizedModel quantizedModel) {
            this.model = Objects.requireNonNull(model);
            this.tokenizer = Objects.requireNonNull(tokenizer);
            this.config = Objects.requireNonNull(config);
            this.generationConfig = generationConfig == null ? GenerationConfig.greedy() : generationConfig;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
            this.chatTemplate = chatTemplate == null ? ChatTemplate.qwen() : chatTemplate;
            this.quantizationConfig = quantizationConfig;
            this.quantizedModel = quantizedModel;
        }

        public Module model() { return model; }
        public FastTokenizer tokenizer() { return tokenizer; }
        public PretrainedConfig config() { return config; }
        public GenerationConfig generationConfig() { return generationConfig; }
        public Path snapshot() { return snapshot; }
        public WeightLoader.LoadReport loadReport() { return loadReport; }
        public ChatTemplate chatTemplate() { return chatTemplate; }
        public BitsAndBytesConfig quantizationConfig() { return quantizationConfig; }
        public BitsAndBytes.QuantizedModel quantizedModel() { return quantizedModel; }
        public boolean isQuantized() { return quantizedModel != null; }

        /** Encode raw prompt and greedy/sample generate. */
        public String generate(String prompt, GenerationConfig gen) {
            var enc = tokenizer.encode(prompt, true);
            GenerationConfig g = mergeGen(gen);
            int[] out = Generator.generate(model, enc.ids(), g, config.maxPositionEmbeddings());
            // decode only the newly generated portion when possible
            int promptLen = enc.ids().length;
            if (out.length > promptLen) {
                int[] neu = new int[out.length - promptLen];
                System.arraycopy(out, promptLen, neu, 0, neu.length);
                return tokenizer.decode(neu, true);
            }
            return tokenizer.decode(out, true);
        }

        public String generate(String prompt, int maxNewTokens) {
            return generate(prompt, generationConfig.toBuilder().maxNewTokens(maxNewTokens).build());
        }

        /**
         * Convenience: greedy generation with default config.
         */
        public String generateGreedy(String prompt, int maxNewTokens) {
            GenerationConfig g = GenerationConfig.greedy().toBuilder()
                    .maxNewTokens(maxNewTokens)
                    .eosTokenId(config.eosTokenId())
                    .build();
            return generate(prompt, g);
        }

        /**
         * Convenience: sampling generation with default config.
         */
        public String generateSample(String prompt, int maxNewTokens, double temperature, int topK, double topP) {
            GenerationConfig g = GenerationConfig.sample(temperature, topK).toBuilder()
                    .maxNewTokens(maxNewTokens)
                    .topP(topP)
                    .eosTokenId(config.eosTokenId())
                    .build();
            return generate(prompt, g);
        }

        /**
         * Generate token ids only (no decoding) for advanced use cases.
         */
        public int[] generateIds(String prompt, int maxNewTokens) {
            var enc = tokenizer.encode(prompt, true);
            GenerationConfig g = generationConfig.toBuilder()
                    .maxNewTokens(maxNewTokens)
                    .build();
            return Generator.generate(model, enc.ids(), g, config.maxPositionEmbeddings());
        }

        /**
         * Generate token ids with full GenerationConfig.
         */
        public int[] generateIds(String prompt, GenerationConfig gen) {
            var enc = tokenizer.encode(prompt, true);
            GenerationConfig g = mergeGen(gen);
            return Generator.generate(model, enc.ids(), g, config.maxPositionEmbeddings());
        }

        /**
         * Compute next-token cross-entropy loss on a given text (eval/perplexity).
         */
        public double computeLoss(String text) {
            var enc = tokenizer.encode(text, true);
            int[] ids = enc.ids();
            // Shift: input = [0..N-2], labels = [1..N-1]
            int[] inputIds = new int[Math.max(0, ids.length - 1)];
            int[] labels = new int[Math.max(0, ids.length - 1)];
            System.arraycopy(ids, 0, inputIds, 0, inputIds.length);
            System.arraycopy(ids, 1, labels, 0, labels.length);

            long[][] inputB = new long[1][inputIds.length];
            long[][] labelB = new long[1][labels.length];
            for (int i = 0; i < inputIds.length; i++) {
                inputB[0][i] = inputIds[i];
                labelB[0][i] = labels[i];
            }
            // Forward to get logits
            try {
                var inputTensor = org.bytedeco.pytorch.global.torch.tensor(inputB);
                Tensor output = model.forward(inputTensor);
                // ComputeCrossEntropy
                Tensor ce = computeCrossEntropy(output, labels);
                double loss = ce.item_double();
                ce.close();
                output.close();
                inputTensor.close();
                return loss;
            } catch (Exception e) {
                System.err.println("computeLoss failed: " + e.getMessage());
                return Double.NaN;
            }
        }

        /**
         * Compute perplexity on a text (exp(average loss)).
         */
        public double computePerplexity(String text) {
            double loss = computeLoss(text);
            return Math.exp(loss);
        }

        private Tensor computeCrossEntropy(Tensor logits, int[] labels) {
            // logits shape: [B, T, V]; labels: [B, T]
            long V = logits.size(logits.dim() - 1);
            long T = logits.size(1);
            long B = logits.size(0);
            var reshaped = logits.view(-1, V);
            var labelTensor = org.bytedeco.pytorch.global.torch.tensor(toLong2D(labels, (int) B, (int) T)).view(-1);
            return org.bytedeco.pytorch.global.torch.cross_entropy_loss(reshaped, labelTensor);
        }

        private static long[][] toLong2D(int[] labels, int B, int T) {
            long[][] out = new long[B][T];
            for (int i = 0; i < B; i++) {
                for (int j = 0; j < T; j++) {
                    out[i][j] = labels[i * T + j];
                }
            }
            return out;
        }

        /** Apply chat template then generate (Instruct models). */
        public String chat(List<Map<String, String>> messages, GenerationConfig gen) {
            String prompt = chatTemplate.apply(messages, /*addGenerationPrompt=*/true);
            // Template already embeds BOS/specials — do not double-add via post-processor.
            return generateEncoded(prompt, gen, /*addSpecialTokens=*/false);
        }

        /** Encode + generate with explicit add_special_tokens control. */
        public String generateEncoded(String prompt, GenerationConfig gen, boolean addSpecialTokens) {
            var enc = tokenizer.encode(prompt, addSpecialTokens);
            GenerationConfig g = mergeGen(gen);
            int[] out = Generator.generate(model, enc.ids(), g, config.maxPositionEmbeddings());
            int promptLen = enc.ids().length;
            if (out.length > promptLen) {
                int[] neu = new int[out.length - promptLen];
                System.arraycopy(out, promptLen, neu, 0, neu.length);
                return tokenizer.decode(neu, true);
            }
            return tokenizer.decode(out, true);
        }

        public String chat(List<Map<String, String>> messages) {
            return chat(messages, generationConfig);
        }

        private GenerationConfig mergeGen(GenerationConfig gen) {
            GenerationConfig base = generationConfig;
            if (gen == null) gen = base;
            GenerationConfig.Builder b = gen.toBuilder();
            if (gen.eosTokenIds.isEmpty()) {
                b.eosTokenId(config.eosTokenId());
                for (int id : base.eosTokenIds) b.eosTokenId(id);
            }
            return b.build();
        }
    }

    public static final class LoadOptions {
        public WeightLoader.BindMode bindMode = WeightLoader.BindMode.ZERO_COPY;
        public boolean strict = true;
        public boolean zeroCopyMmap = true;
        public boolean loadWeights = true;
        /** Optional HF-style BitsAndBytes quantization (4/8-bit). */
        public BitsAndBytesConfig quantizationConfig;
        /** Freeze base weights after quant (QLoRA prepare). Default true when quant is set. */
        public boolean prepareForKbitTraining = true;
        /** HF revision (commit hash, branch, or tag). "main" by default. */
        public String revision = "main";
        /** HF repo type ("models" by default; "datasets" for datasets). */
        public String repoType = "models";
        /** Trust remote code (allows custom architectures). */
        public boolean trustRemoteCode = false;
        /** Force re-download even if cached. */
        public boolean forceDownload = false;
        /** Cache directory override (otherwise from HfHub). */
        public String cacheDir = null;
        /** HF token (otherwise from HfHub). */
        public String token = null;
        /** Use the safetensors index even when single-file weights exist. */
        public boolean preferSafetensors = true;
        /** Model-type override for ModelRegistry ("auto", "llama", "qwen2", etc.). */
        public String modelType = null;

        public LoadOptions bindMode(WeightLoader.BindMode m) { this.bindMode = m; return this; }
        public LoadOptions strict(boolean v) { this.strict = v; return this; }
        public LoadOptions zeroCopyMmap(boolean v) { this.zeroCopyMmap = v; return this; }
        public LoadOptions loadWeights(boolean v) { this.loadWeights = v; return this; }
        public LoadOptions quantizationConfig(BitsAndBytesConfig cfg) {
            this.quantizationConfig = cfg;
            return this;
        }
        /** Snake alias matching Python {@code quantization_config=}. */
        public LoadOptions quantization_config(BitsAndBytesConfig cfg) {
            return quantizationConfig(cfg);
        }
        public LoadOptions prepareForKbitTraining(boolean v) {
            this.prepareForKbitTraining = v;
            return this;
        }
        public LoadOptions revision(String v) { this.revision = v; return this; }
        public LoadOptions repoType(String v) { this.repoType = v; return this; }
        public LoadOptions trustRemoteCode(boolean v) { this.trustRemoteCode = v; return this; }
        public LoadOptions forceDownload(boolean v) { this.forceDownload = v; return this; }
        public LoadOptions cacheDir(String v) { this.cacheDir = v; return this; }
        public LoadOptions token(String v) { this.token = v; return this; }
        public LoadOptions preferSafetensors(boolean v) { this.preferSafetensors = v; return this; }
        public LoadOptions modelType(String v) { this.modelType = v; return this; }
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, new LoadOptions());
    }

    public static Bundle fromPretrained(String modelId, HfHub hub, LoadOptions opts) throws IOException {
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(hub, "hub");
        Path snap = hub.snapshotDownload(modelId);
        return fromDirectory(snap, opts);
    }

    /**
     * Convenience: load from a HF model id with the default Hub (env-driven).
     */
    public static Bundle fromPretrainedDefault(String modelId) throws IOException {
        HfHub hub = HfHub.fromEnv();
        return fromPretrained(modelId, hub);
    }

    /**
     * Convenience: load from a HF model id with a specific revision (commit hash, branch, or tag).
     */
    public static Bundle fromPretrained(String modelId, String revision, HfHub hub) throws IOException {
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(hub, "hub");
        Path snap = hub.snapshotDownload(modelId, revision);
        return fromDirectory(snap);
    }

    /**
     * Convenience: load from a local path (directory or config.json file).
     */
    public static Bundle fromLocal(String localPath) throws IOException {
        return fromDirectory(Path.of(localPath));
    }

    /**
     * Convenience: load from a local path with explicit options.
     */
    public static Bundle fromLocal(String localPath, LoadOptions opts) throws IOException {
        return fromDirectory(Path.of(localPath), opts);
    }

    /**
     * Convenience: load from a local Path with default options.
     */
    public static Bundle fromLocal(Path dir) throws IOException {
        return fromDirectory(dir);
    }

    /**
     * Convenience: load from a local Path with explicit options.
     */
    public static Bundle fromLocal(Path dir, LoadOptions opts) throws IOException {
        return fromDirectory(dir, opts);
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, new LoadOptions());
    }

    public static Bundle fromDirectory(Path dir, LoadOptions opts) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (opts == null) opts = new LoadOptions();
        if (!Files.isDirectory(dir)) {
            throw new IOException("Not a model directory: " + dir);
        }

        PretrainedConfig cfg = readConfig(dir);
        Module model = ModelRegistry.create(cfg);
        model.eval();

        // Convert model to the dtype specified in config (e.g., bfloat16 from HF safetensors)
        // Must be done BEFORE loading weights so storage sizes match
        String dtypeStr = cfg.torchDtype();
        System.out.println("[DEBUG] config torch_dtype = " + dtypeStr);
        boolean needsDtypeConversion = dtypeStr != null && !dtypeStr.isEmpty() && !"float32".equals(dtypeStr);
        if (needsDtypeConversion) {
            try {
                var scalarType = switch (dtypeStr.toLowerCase()) {
                    case "bfloat16", "bf16" -> org.bytedeco.pytorch.global.torch.ScalarType.BFloat16;
                    case "float16", "fp16", "half" -> org.bytedeco.pytorch.global.torch.ScalarType.Half;
                    case "float", "float32" -> org.bytedeco.pytorch.global.torch.ScalarType.Float;
                    case "double", "float64" -> org.bytedeco.pytorch.global.torch.ScalarType.Double;
                    case "int32" -> org.bytedeco.pytorch.global.torch.ScalarType.Int;
                    case "int64", "long" -> org.bytedeco.pytorch.global.torch.ScalarType.Long;
                    default -> null;
                };
                System.out.println("[DEBUG] converting to scalarType = " + scalarType);
                if (scalarType != null) {
                    SnapshotFiles.toDtype(model, scalarType);
                }
            } catch (Throwable t) {
                System.out.println("[DEBUG] dtype conversion failed: " + t.getMessage());
                // dtype conversion failed, continue with default dtype
            }
        }

        WeightLoader.LoadReport report = null;
        if (opts.loadWeights) {
            WeightMap map = ModelRegistry.weightMap(cfg);
            // When dtype conversion was needed, we must use COPY mode since ZERO_COPY
            // cannot rebind storage between tensors of different dtypes
            var bindMode = needsDtypeConversion ? WeightLoader.BindMode.COPY : opts.bindMode;
            if (needsDtypeConversion) {
                System.out.println("[DEBUG] Using COPY mode due to dtype conversion");
            }
            // When tie_word_embeddings is true, lm_head.weight is tied to embed_tokens.weight
            // so we allow strict to pass even if lm_head.weight is "missing"
            boolean allowTiedMissing = cfg.tieWordEmbeddings();
            try {
                report = WeightLoader.loadAndBind(model, dir, map, bindMode, opts.strict && !allowTiedMissing, opts.zeroCopyMmap);
            } catch (IOException e) {
                if (opts.strict) throw e;
                // no weights — leave random init
                report = new WeightLoader.LoadReport(
                        List.of(), List.of("(no safetensors)"), List.of(), List.of(), 0, 0, bindMode);
            }
            // Always print bind stats — silent miss + tie_word_embeddings (strict off)
            // previously left Qwen3-VL on random weights producing garbage tokens.
            System.out.println("[AutoModelForCausalLM] " + report
                    + " map=" + (map == null ? "null" : map.getClass().getSimpleName())
                    + " model=" + model.getClass().getSimpleName()
                    + " type=" + cfg.modelType()
                    + " d=" + cfg.hiddenSize()
                    + " L=" + cfg.numHiddenLayers()
                    + " H=" + cfg.numAttentionHeads()
                    + " headDim=" + cfg.headDim()
                    + " rope=" + cfg.ropeTheta());
            if (report.matchedCount() == 0) {
                throw new IllegalStateException(
                        "Weight bind matched 0 parameters for " + dir
                                + " — check WeightMap / language_model prefix / layers.N→layers/N. "
                                + report);
            }
            // Soft guard: language towers should match most module params
            if (report.matchedCount() < 50 && report.missing.size() > report.matchedCount()) {
                System.out.println("[AutoModelForCausalLM] WARNING low match rate — missing e.g. "
                        + (report.missing.isEmpty() ? "[]" : report.missing.subList(0, Math.min(5, report.missing.size()))));
            }
            // After weight load, re-apply tie_word_embeddings.
            // ZERO_COPY rebinds wte/embed storage; COPY replaces values — both break a
            // constructor-time set_() share, so always re-tie when requested.
            // Uses dtype-aware logic: set_() when dtypes match, copy_() when they differ.
            if (cfg.tieWordEmbeddings()) {
                try {
                    if (model instanceof CausalLM clm) {
                        if (clm.retieWordEmbeddings()) {
                            System.out.println("[DEBUG] Re-tied CausalLM lm_head ← wte");
                        }
                    } else if (model instanceof LlamaForCausalLM llama) {
                        if (llama.retieWordEmbeddings()) {
                            System.out.println("[DEBUG] Re-tied LlamaForCausalLM lm_head ← embed_tokens");
                        }
                    } else if (model instanceof GlmForCausalLM glm) {
                        if (glm.retieWordEmbeddings()) {
                            System.out.println("[DEBUG] Re-tied Glm lm_head ← embed_tokens");
                        }
                    } else if (model instanceof Qwen2ForCausalLM qwen) {
                        if (qwen.retieWordEmbeddings()) {
                            System.out.println("[DEBUG] Re-tied Qwen2 lm_head ← embed_tokens");
                        }
                    } else if (model instanceof Qwen3ForCausalLM qwen3) {
                        if (qwen3.retieWordEmbeddings()) {
                            System.out.println("[DEBUG] Re-tied Qwen3 lm_head ← embed_tokens");
                        }
                    }
                } catch (Throwable t) {
                    System.out.println("[DEBUG] Failed to re-apply tie: " + t.getMessage());
                }
            }
        }

        FastTokenizer tok = readTokenizer(dir, cfg);
        GenerationConfig genCfg = readGenerationConfig(dir, cfg);
        ChatTemplate template = ChatTemplate.detect(dir, cfg);

        BitsAndBytes.QuantizedModel qm = applyQuantization(model, opts);
        return new Bundle(model, tok, cfg, genCfg, dir, report, template,
                opts.quantizationConfig, qm);
    }

    /** Random-init tiny model for offline unit tests (no weights). */
    public static Bundle tiny(String kind) {
        return tiny(kind, null);
    }

    /** Random-init tiny model with optional BitsAndBytes quantization (QLoRA offline path). */
    public static Bundle tiny(String kind, BitsAndBytesConfig bnb) {
        PretrainedConfig cfg = switch (kind == null ? "qwen" : kind.toLowerCase()) {
            case "llama", "mistral" -> PretrainedConfig.tinyLlama();
            case "qwen", "qwen2" -> PretrainedConfig.tinyQwen();
            case "qwen3" -> PretrainedConfig.tinyQwen3();
            case "gemma" -> PretrainedConfig.tinyGemma();
            case "gemma2" -> PretrainedConfig.tinyGemma2();
            case "gemma3" -> PretrainedConfig.tinyGemma3();
            case "phi3" -> PretrainedConfig.tinyPhi3();
            case "mixtral" -> PretrainedConfig.tinyMixtral();
            default -> PretrainedConfig.tinyGpt2();
        };
        Module model = ModelRegistry.create(cfg);
        BitsAndBytes.QuantizedModel qm = null;
        if (bnb != null && bnb.isQuantized()) {
            LoadOptions opts = new LoadOptions()
                    .quantizationConfig(bnb)
                    .prepareForKbitTraining(true);
            qm = applyQuantization(model, opts);
        }
        FastTokenizer tok = FastTokenizer.whitespace()
                .modelMaxLength(cfg.maxPositionEmbeddings())
                .build();
        GenerationConfig gen = GenerationConfig.builder()
                .maxNewTokens(16)
                .eosTokenId(cfg.eosTokenId())
                .build();
        return new Bundle(model, tok, cfg, gen, null, null,
                ChatTemplate.forModelType(cfg.modelType()), bnb, qm);
    }

    /**
     * Apply BitsAndBytes 4/8-bit quantization to a loaded model when
     * {@link LoadOptions#quantizationConfig} is set.
     *
     * <p>Collects HF-named linears from {@link CausalLM},
     * {@link Qwen2ForCausalLM},
     * and {@link LlamaForCausalLM},
     * then quantize→materialize→freeze (QLoRA prepare).
     */
    public static BitsAndBytes.QuantizedModel applyQuantization(Module model, LoadOptions opts) {
        if (opts == null || opts.quantizationConfig == null || !opts.quantizationConfig.isQuantized()) {
            return null;
        }
        BitsAndBytesConfig bnb = opts.quantizationConfig;
        BitsAndBytes.QuantizedModel qm = null;
        try {
            Map<String, LinearImpl> linears = collectQuantizableLinears(model);
            if (!linears.isEmpty()) {
                qm = BitsAndBytes.prepareForQLoRA(linears, bnb);
            } else if (opts.prepareForKbitTraining) {
                BitsAndBytes.prepareModelForKbitTraining(model);
            }
        } catch (Exception e) {
            System.out.println("[bnb] quantization skipped: " + e.getMessage());
        }
        if (opts.prepareForKbitTraining && model != null) {
            try {
                BitsAndBytes.prepareModelForKbitTraining(model);
            } catch (Exception ignored) {}
        }
        return qm;
    }

    /**
     * Collect quantizable LinearImpls (excludes lm_head) from known causal LM layouts.
     *
     * <p>This implementation uses the structural assumption of the Llama/Qwen/Gemma/Phi3
     * family: each layer exposes a {@code self_attn} sub-module with {@code q_proj},
     * {@code k_proj}, {@code v_proj}, {@code o_proj} linears, and an {@code mlp} sub-module
     * with {@code gate_proj}, {@code up_proj}, {@code down_proj} linears.
     *
     * <p>For unknown architectures, this method falls back to flat reflection over
     * all named parameters and returns any {@link LinearImpl} whose name does not
     * end with {@code lm_head.weight}.
     */
    public static Map<String, LinearImpl> collectQuantizableLinears(Module model) {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        if (model == null) return m;
        if (model instanceof CausalLM clm) {
            return clm.quantizableLinears();
        }
        if (model instanceof Qwen2ForCausalLM qwen) {
            return collectFromLayerList(qwen.model().layers, "model.layers.", qwen);
        }
        if (model instanceof LlamaForCausalLM || model instanceof Qwen3ForCausalLM
                || model instanceof GlmForCausalLM) {
            try {
                Class<?> cls = model.getClass();
                var modelMethod = cls.getMethod("model");
                Object inner = modelMethod.invoke(model);
                @SuppressWarnings("unchecked")
                var layers = (java.util.List<?>) inner.getClass().getField("layers").get(inner);
                return collectFromLayerList(layers, "model.layers.", model);
            } catch (Exception ignored) {
                // fall through to generic
            }
        }
        // Generic fallback: walk named parameters and pick linears
        return collectAllNamedLinears(model, "lm_head");
    }

    /**
     * Generic helper that collects linears from a list of layer objects using field reflection.
     */
    private static Map<String, LinearImpl> collectFromLayerList(java.util.List<?> layers, String prefix, Object parent) {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        for (int i = 0; i < layers.size(); i++) {
            Object layer = layers.get(i);
            String p = prefix + i;
            try {
                Object attn = layer.getClass().getField("self_attn").get(layer);
                Object mlp = layer.getClass().getField("mlp").get(layer);
                m.put(p + ".self_attn.q_proj", (LinearImpl) attn.getClass().getField("q_proj").get(attn));
                m.put(p + ".self_attn.k_proj", (LinearImpl) attn.getClass().getField("k_proj").get(attn));
                m.put(p + ".self_attn.v_proj", (LinearImpl) attn.getClass().getField("v_proj").get(attn));
                m.put(p + ".self_attn.o_proj", (LinearImpl) attn.getClass().getField("o_proj").get(attn));
                m.put(p + ".mlp.gate_proj", (LinearImpl) mlp.getClass().getField("gate_proj").get(mlp));
                m.put(p + ".mlp.up_proj", (LinearImpl) mlp.getClass().getField("up_proj").get(mlp));
                m.put(p + ".mlp.down_proj", (LinearImpl) mlp.getClass().getField("down_proj").get(mlp));
            } catch (Exception ignored) {
                // layer shape mismatch — skip this layer
            }
        }
        return m;
    }

    /**
     * Collect all named linear layers that are not in the excluded set.
     * Used as a fallback for unknown architectures.
     */
    public static Map<String, LinearImpl> collectAllNamedLinears(Module model, String... excludeLeafNames) {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        if (model == null) return m;
        java.util.Set<String> excluded = new java.util.HashSet<>();
        for (String e : excludeLeafNames) excluded.add(e);

        try {
            // Try the model's named_linears() method
            var method = model.getClass().getMethod("named_linears");
            @SuppressWarnings("unchecked")
            java.util.Map<String, LinearImpl> named = (java.util.Map<String, LinearImpl>) method.invoke(model);
            for (java.util.Map.Entry<String, LinearImpl> e : named.entrySet()) {
                String leaf = e.getKey();
                int idx = leaf.lastIndexOf('.');
                String leafName = idx >= 0 ? leaf.substring(idx + 1) : leaf;
                if (!excluded.contains(leafName)) {
                    m.put(e.getKey(), e.getValue());
                }
            }
        } catch (Exception ignored) {
            // fall through
        }

        if (m.isEmpty()) {
            // Try named_parameters
            try {
                var method = model.getClass().getMethod("named_parameters", boolean.class);
                @SuppressWarnings("unchecked")
                java.util.Map<String, org.bytedeco.pytorch.Tensor> params = 
                        (java.util.Map<String, org.bytedeco.pytorch.Tensor>) method.invoke(model, true);
            } catch (Exception ignored) {}
        }
        return m;
    }

    private static PretrainedConfig readConfig(Path dir) throws IOException {
        Path cfg = SnapshotFiles.configJson(dir);
        if (Files.isRegularFile(cfg)) {
            return PretrainedConfig.fromFile(cfg);
        }
        throw new IOException("Missing config.json in " + dir);
    }

    private static GenerationConfig readGenerationConfig(Path dir, PretrainedConfig cfg) {
        Path p = SnapshotFiles.generationConfigJson(dir);
        try {
            if (Files.isRegularFile(p)) {
                GenerationConfig g = GenerationConfig.fromFile(p);
                if (g.eosTokenIds.isEmpty()) {
                    return g.toBuilder().eosTokenId(cfg.eosTokenId()).build();
                }
                return g;
            }
        } catch (IOException ignored) {}
        return GenerationConfig.builder()
                .maxNewTokens(64)
                .eosTokenId(cfg.eosTokenId())
                .padTokenId(cfg.padTokenId())
                .bosTokenId(cfg.bosTokenId())
                .build();
    }

    private static FastTokenizer readTokenizer(Path dir, PretrainedConfig cfg) throws IOException {
        // Full HF tokenizer.json / vocab+merges / whitespace fallback
        FastTokenizer tok = DirectoryTokenizerLoader.load(dir);
        if (cfg != null && cfg.maxPositionEmbeddings() > 0
                && tok.modelMaxLength() <= 0) {
            return tok.withTruncation(FastTokenizer.Truncation.of(cfg.maxPositionEmbeddings()));
        }
        return tok;
    }
}
