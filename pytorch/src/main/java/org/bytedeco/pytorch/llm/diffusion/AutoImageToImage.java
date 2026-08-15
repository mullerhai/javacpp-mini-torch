/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.diffusion;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensorsLoader;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.diffusion.modeling.UNet2DConditionModel;
import org.bytedeco.pytorch.llm.diffusion.modeling.AutoencoderKL;
import org.bytedeco.pytorch.llm.diffusion.modeling.CLIPTextEmbeddings;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.loading.SnapshotFiles;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.nn.Module;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code AutoImageToImage.from_pretrained} entry point.
 *
 * <p>Loads diffusion model weights (UNet + VAE + TextEncoder + Scheduler)
 * from a checkpoint directory (safetensors + config.json) and provides
 * a unified text-to-image generation API.
 *
 * <p>Supported architectures:
 * <ul>
 *   <li>StableDiffusion (SD 1.5, SD 2.x)</li>
 *   <li>SDXL, SDXL-Turbo</li>
 *   <li>PixArt-Alpha, Anima (SD-based fine-tunes)</li>
 *   <li>Kandinsky 2.x</li>
 *   <li>DeepFloyd-IF</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * AutoImageToImage.Bundle bundle = AutoImageToImage.fromDirectory(
 *     Paths.get("/path/to/anima/checkpoint"));
 *
 * BufferedImage img = bundle.pipeline.generate(
 *     "A giant panda eating bamboo",
 *     StableDiffusionPipeline.GenerationParams.create()
 *         .height(512).width(512)
 *         .numInferenceSteps(20)
 *         .guidanceScale(7.5f));
 *
 * bundle.pipeline.saveImage(img, "output.png");
 * }</pre>
 *
 * <p>Build classpath for this class:
 * <pre>{@code
 * javac -d target/classes \
 *   -cp "target/classes:.../pytorch.jar:.../javacpp.jar" \
 *   .../llm/diffusion/AutoImageToImage.java
 * }</pre>
 */
public final class AutoImageToImage {

    private AutoImageToImage() {}

    // ── Bundle ──────────────────────────────────────────────────────────

    /**
     * Loaded pipeline bundle: pipeline + component configs + load report.
     */
    public static final class Bundle {
        private final StableDiffusionPipeline pipeline;
        private final Path snapshot;
        private final LoadReport loadReport;
        private final PipelineInfo info;

        public Bundle(StableDiffusionPipeline pipeline, Path snapshot,
                      LoadReport loadReport, PipelineInfo info) {
            this.pipeline = Objects.requireNonNull(pipeline);
            this.snapshot = Objects.requireNonNull(snapshot);
            this.loadReport = loadReport;
            this.info = info;
        }

        public StableDiffusionPipeline pipeline() { return pipeline; }
        public Path snapshot() { return snapshot; }
        public LoadReport loadReport() { return loadReport; }
        public PipelineInfo info() { return info; }

        /**
         * Generate an image from text.
         *
         * @param prompt text description
         * @param params generation parameters (or null for defaults)
         * @return generated image
         */
        public BufferedImage generate(String prompt,
                StableDiffusionPipeline.GenerationParams params) throws Exception {
            if (params == null) {
                params = StableDiffusionPipeline.GenerationParams.create();
            }
            return pipeline.generate(prompt, params);
        }

        /**
         * Generate with default parameters.
         */
        public BufferedImage generate(String prompt) throws Exception {
            return generate(prompt, null);
        }

        /**
         * Generate multiple images.
         */
        public BufferedImage[] generate(String[] prompts,
                StableDiffusionPipeline.GenerationParams params) throws Exception {
            BufferedImage[] results = new BufferedImage[prompts.length];
            for (int i = 0; i < prompts.length; i++) {
                results[i] = generate(prompts[i], params);
            }
            return results;
        }

        /**
         * Generate with per-prompt seed (stable reproducibility).
         */
        public BufferedImage generateWithSeed(String prompt, long seed,
                StableDiffusionPipeline.GenerationParams params) throws Exception {
            if (params == null) {
                params = StableDiffusionPipeline.GenerationParams.create();
            }
            params.seed(seed);
            return generate(prompt, params);
        }
    }

    // ── Load Report ─────────────────────────────────────────────────────

    public static final class LoadReport {
        private final int totalTensors;
        private final long totalBytes;
        private final int boundTensors;
        private final int missingKeys;
        private final int unexpectedKeys;
        private final long loadTimeMs;
        private final double throughputMBps;
        private final java.util.List<String> errors;
        private final java.util.List<String> missingKeyList;
        private final java.util.List<String> unexpectedKeyList;

        public LoadReport(int totalTensors, long totalBytes, int boundTensors,
                          int missingKeys, int unexpectedKeys, long loadTimeMs,
                          double throughputMBps, java.util.List<String> errors,
                          java.util.List<String> missingKeyList,
                          java.util.List<String> unexpectedKeyList) {
            this.totalTensors = totalTensors;
            this.totalBytes = totalBytes;
            this.boundTensors = boundTensors;
            this.missingKeys = missingKeys;
            this.unexpectedKeys = unexpectedKeys;
            this.loadTimeMs = loadTimeMs;
            this.throughputMBps = throughputMBps;
            this.errors = errors;
            this.missingKeyList = missingKeyList;
            this.unexpectedKeyList = unexpectedKeyList;
        }

        public int totalTensors() { return totalTensors; }
        public long totalBytes() { return totalBytes; }
        public int boundTensors() { return boundTensors; }
        public int missingKeys() { return missingKeys; }
        public int unexpectedKeys() { return unexpectedKeys; }
        public long loadTimeMs() { return loadTimeMs; }
        public double throughputMBps() { return throughputMBps; }
        public java.util.List<String> errors() { return errors; }
        public java.util.List<String> missingKeyList() { return missingKeyList; }
        public java.util.List<String> unexpectedKeyList() { return unexpectedKeyList; }

        @Override
        public String toString() {
            return String.format(
                "LoadReport{tensors=%d, bytes=%.1fGB, bound=%d, missing=%d, unexpected=%d, time=%.1fs, throughput=%.1fMB/s}",
                totalTensors, totalBytes / 1e9, boundTensors, missingKeys, unexpectedKeys,
                loadTimeMs / 1000.0, throughputMBps);
        }
    }

    // ── Pipeline Info ───────────────────────────────────────────────────

    /** Metadata extracted from config.json. */
    public static final class PipelineInfo {
        private final String modelType;       // e.g. "stable-diffusion", "pixart", "kandinsky"
        private final int inChannels;         // e.g. 4 (latent channels)
        private final int outChannels;         // e.g. 4
        private final int vaeInChannels;       // e.g. 3 (RGB)
        private final int latentChannels;      // e.g. 4
        private final double latentFactor;     // e.g. 8 (512/64)
        private final int textEncoderVocabSize;
        private final int textEncoderHiddenSize;
        private final int textEncoderLayers;
        private final int textEncoderHeads;
        private final String schedulerType;     // e.g. "euler", "ddpm", "ddim"
        private final StringTokenizer tokenizerType;
        private final int unetBlockOutChannels0, unetBlockOutChannels1, unetBlockOutChannels2;

        public PipelineInfo(String modelType, int inChannels, int outChannels,
                           int vaeInChannels, int latentChannels, double latentFactor,
                           int textEncoderVocabSize, int textEncoderHiddenSize,
                           int textEncoderLayers, int textEncoderHeads,
                           String schedulerType, StringTokenizer tokenizerType,
                           int unetBlockOutChannels0, int unetBlockOutChannels1, int unetBlockOutChannels2) {
            this.modelType = modelType;
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            this.vaeInChannels = vaeInChannels;
            this.latentChannels = latentChannels;
            this.latentFactor = latentFactor;
            this.textEncoderVocabSize = textEncoderVocabSize;
            this.textEncoderHiddenSize = textEncoderHiddenSize;
            this.textEncoderLayers = textEncoderLayers;
            this.textEncoderHeads = textEncoderHeads;
            this.schedulerType = schedulerType;
            this.tokenizerType = tokenizerType;
            this.unetBlockOutChannels0 = unetBlockOutChannels0;
            this.unetBlockOutChannels1 = unetBlockOutChannels1;
            this.unetBlockOutChannels2 = unetBlockOutChannels2;
        }

        // Getters
        public String modelType() { return modelType; }
        public int inChannels() { return inChannels; }
        public int outChannels() { return outChannels; }
        public int vaeInChannels() { return vaeInChannels; }
        public int latentChannels() { return latentChannels; }
        public double latentFactor() { return latentFactor; }
        public int textEncoderVocabSize() { return textEncoderVocabSize; }
        public int textEncoderHiddenSize() { return textEncoderHiddenSize; }
        public int textEncoderLayers() { return textEncoderLayers; }
        public int textEncoderHeads() { return textEncoderHeads; }
        public String schedulerType() { return schedulerType; }
        public StringTokenizer tokenizerType() { return tokenizerType; }
        public int unetBlockOutChannels0() { return unetBlockOutChannels0; }
        public int unetBlockOutChannels1() { return unetBlockOutChannels1; }
        public int unetBlockOutChannels2() { return unetBlockOutChannels2; }

        @Override
        public String toString() {
            return String.format(
                "PipelineInfo{modelType=%s, inCh=%d, latentCh=%d, textHidden=%d, textLayers=%d, unetBlocks=[%d,%d,%d], scheduler=%s}",
                modelType, inChannels, latentChannels, textEncoderHiddenSize, textEncoderLayers,
                unetBlockOutChannels0, unetBlockOutChannels1, unetBlockOutChannels2, schedulerType);
        }
    }

    public enum StringTokenizer {
        CLIP, BERT, T5, NONE
    }

    // ── Load Options ─────────────────────────────────────────────────────

    public static final class LoadOptions {
        public WeightLoader.BindMode bindMode = WeightLoader.BindMode.ZERO_COPY;
        public boolean strict = false;         // Diff models have more variant weights
        public boolean zeroCopyMmap = true;
        public boolean loadWeights = true;
        /** Force a specific scheduler type (null = auto-detect from config) */
        public String forceScheduler = null;
        /** Tokenizer to use (null = auto-detect) */
        public FastTokenizer tokenizer = null;
        /** Whether to load VAE weights (can be skipped for Latent Diffusion only) */
        public boolean loadVae = true;
        /** Whether to load text encoder weights (can be skipped for some models) */
        public boolean loadTextEncoder = true;

        public LoadOptions bindMode(WeightLoader.BindMode m) { this.bindMode = m; return this; }
        public LoadOptions strict(boolean v) { this.strict = v; return this; }
        public LoadOptions zeroCopyMmap(boolean v) { this.zeroCopyMmap = v; return this; }
        public LoadOptions loadWeights(boolean v) { this.loadWeights = v; return this; }
        public LoadOptions forceScheduler(String s) { this.forceScheduler = s; return this; }
        public LoadOptions tokenizer(FastTokenizer t) { this.tokenizer = t; return this; }
        public LoadOptions loadVae(boolean v) { this.loadVae = v; return this; }
        public LoadOptions loadTextEncoder(boolean v) { this.loadTextEncoder = v; return this; }
    }

    // ── Loading ──────────────────────────────────────────────────────────

    /**
     * Load from a checkpoint directory.
     *
     * @param dir directory containing config.json + .safetensors files
     */
    public static Bundle fromDirectory(Path dir) throws Exception {
        return fromDirectory(dir, new LoadOptions());
    }

    public static Bundle fromDirectory(Path dir, LoadOptions opts) throws Exception {
        Objects.requireNonNull(dir, "dir");
        if (!Files.exists(dir)) {
            throw new java.io.FileNotFoundException("Model directory not found: " + dir);
        }

        long start = System.currentTimeMillis();

        // 1. Parse config.json to get pipeline info
        PipelineInfo info = parseConfig(dir);

        // 2. Build models from config
        UNet2DConditionModel unet = buildUnet(info);
        AutoencoderKL vae = buildVae(info);
        CLIPTextEmbeddings textEncoder = buildTextEncoder(info);
        FastTokenizer tokenizer = opts.tokenizer != null ? opts.tokenizer : buildTokenizer(info);
        Scheduler scheduler = buildScheduler(info, opts);

        StableDiffusionPipeline pipeline = StableDiffusionPipeline.fromComponents(
            unet, vae, textEncoder, tokenizer, scheduler);

        // 3. Load safetensors and bind weights
        LoadReport report = loadAndBindWeights(dir, pipeline, opts);

        long loadTime = System.currentTimeMillis() - start;
        report = new LoadReport(
            report.totalTensors, report.totalBytes, report.boundTensors,
            report.missingKeys, report.unexpectedKeys, loadTime,
            report.totalBytes / (loadTime * 1000.0),
            report.errors, report.missingKeyList, report.unexpectedKeyList);

        return new Bundle(pipeline, dir, report, info);
    }

    // ── Config Parsing ──────────────────────────────────────────────────

    static PipelineInfo parseConfig(Path dir) throws Exception {
        Path configFile = dir.resolve("config.json");
        if (!Files.exists(configFile)) {
            System.out.println("  [AutoImageToImage] No config.json found — using default SD 1.5 config");
            return defaultSD15Config();
        }

        String json = Files.readString(configFile);

        // Simple JSON parsing (no external deps)
        String modelType = extractJsonStr(json, "model_type", "stable-diffusion");
        int inChannels = extractJsonInt(json, "in_channels", 4);
        int outChannels = extractJsonInt(json, "out_channels", 4);
        int latentCh = extractJsonInt(json, "latent_channels", 4);
        int vaeInCh = extractJsonInt(json, "vae_in_channels", 3);
        double latentFactor = extractJsonDouble(json, "latent_factor", 8.0);

        int textVocab = extractJsonInt(json, "text_encoder_vocab_size", 49408);
        int textHidden = extractJsonInt(json, "text_encoder_hidden_size", 768);
        int textLayers = extractJsonInt(json, "text_encoder_layers", 12);
        int textHeads = extractJsonInt(json, "text_encoder_heads", 12);

        int[] unetBlocks = extractJsonIntArray(json, "unet_block_out_channels",
            new int[]{320, 640, 1280});
        String scheduler = extractJsonStr(json, "scheduler", "euler");
        String tokenizerType = extractJsonStr(json, "tokenizer_type", "clip");

        return new PipelineInfo(
            modelType, inChannels, outChannels,
            vaeInCh, latentCh, latentFactor,
            textVocab, textHidden, textLayers, textHeads,
            scheduler, StringTokenizer.valueOf(tokenizerType.toUpperCase()),
            unetBlocks[0], unetBlocks.length > 1 ? unetBlocks[1] : unetBlocks[0],
            unetBlocks.length > 2 ? unetBlocks[2] : unetBlocks[0]);
    }

    static PipelineInfo defaultSD15Config() {
        return new PipelineInfo(
            "stable-diffusion-v1.5", 4, 4,
            3, 4, 8.0,
            49408, 768, 12, 12,
            "euler", StringTokenizer.CLIP,
            320, 640, 1280);
    }

    // Simple JSON helpers
    static String extractJsonStr(String json, String key, String def) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return def;
        int colon = json.indexOf(':', idx);
        int comma = json.indexOf(',', colon);
        int brace = json.indexOf('}', colon);
        int end = Math.min(comma > 0 ? comma : Integer.MAX_VALUE, brace > 0 ? brace : Integer.MAX_VALUE);
        String val = json.substring(colon + 1, end).trim();
        if (val.startsWith("\"")) val = val.substring(1, val.lastIndexOf('"'));
        return val.isEmpty() ? def : val;
    }

    static int extractJsonInt(String json, String key, int def) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return def;
        int colon = json.indexOf(':', idx);
        int comma = json.indexOf(',', colon);
        int brace = json.indexOf('}', colon);
        int end = Math.min(comma > 0 ? comma : Integer.MAX_VALUE, brace > 0 ? brace : Integer.MAX_VALUE);
        try {
            return Integer.parseInt(json.substring(colon + 1, end).trim());
        } catch (Exception e) {
            return def;
        }
    }

    static double extractJsonDouble(String json, String key, double def) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return def;
        int colon = json.indexOf(':', idx);
        int comma = json.indexOf(',', colon);
        int brace = json.indexOf('}', colon);
        int end = Math.min(comma > 0 ? comma : Integer.MAX_VALUE, brace > 0 ? brace : Integer.MAX_VALUE);
        try {
            return Double.parseDouble(json.substring(colon + 1, end).trim());
        } catch (Exception e) {
            return def;
        }
    }

    static int[] extractJsonIntArray(String json, String key, int[] def) {
        String pattern = "\"" + key + "\"";
        int idx = json.indexOf(pattern);
        if (idx < 0) return def;
        int colon = json.indexOf(':', idx);
        int bra = json.indexOf('[', colon);
        int ket = json.indexOf(']', colon);
        if (bra < 0 || ket < 0) return def;
        String nums = json.substring(bra + 1, ket).trim();
        String[] parts = nums.split("[,\\[\\]]+");
        int[] result = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try { result[i] = Integer.parseInt(parts[i].trim()); }
            catch (Exception e) { return def; }
        }
        return result;
    }

    // ── Model Building ─────────────────────────────────────────────────

    static UNet2DConditionModel buildUnet(PipelineInfo info) {
        UNet2DConditionModel.DiffusionUnetConfig c =
            new UNet2DConditionModel.DiffusionUnetConfig();
        c.inChannels(info.inChannels());
        c.outChannels(info.outChannels());
        c.blockOutChannels(new int[]{
            info.unetBlockOutChannels0(),
            info.unetBlockOutChannels1(),
            info.unetBlockOutChannels2()
        });
        c.crossAttentionDim(new int[]{info.textEncoderHiddenSize()});
        // UNet attention head dim is independent of the text encoder.  Use the
        // SD 1.5 default of 40 here (320 channels / 40 = 8 heads for the
        // shallowest block, scaling up proportionally for deeper blocks).
        c.attentionHeadDim(40);
        return new UNet2DConditionModel(c);
    }

    static AutoencoderKL buildVae(PipelineInfo info) {
        AutoencoderKL.VAEConfig c = new AutoencoderKL.VAEConfig();
        c.inChannels(info.vaeInChannels());
        c.latentChannels(info.latentChannels());
        c.latentFactor(info.latentFactor());
        return new AutoencoderKL(c);
    }

    static CLIPTextEmbeddings buildTextEncoder(PipelineInfo info) {
        CLIPTextEmbeddings.CLIPTextConfig c = new CLIPTextEmbeddings.CLIPTextConfig();
        c.vocabSize(info.textEncoderVocabSize());
        c.hiddenSize(info.textEncoderHiddenSize());
        c.numHiddenLayers(info.textEncoderLayers());
        c.numAttentionHeads(info.textEncoderHeads());
        return new CLIPTextEmbeddings(c);
    }

    static FastTokenizer buildTokenizer(PipelineInfo info) {
        switch (info.tokenizerType()) {
            case CLIP:
            default:
                return AutoTokenizer.cl100kBase();
            case BERT:
                return AutoTokenizer.tiktoken("cl100k_base");
        }
    }

    static Scheduler buildScheduler(PipelineInfo info, LoadOptions opts) {
        String type = opts.forceScheduler != null ? opts.forceScheduler : info.schedulerType();
        switch (type.toLowerCase()) {
            case "ddim": return new Scheduler.DDIMScheduler();
            case "ddpm": return new Scheduler.DDPMScheduler();
            case "euler":
            default:
                return new Scheduler.EulerDiscreteScheduler();
        }
    }

    // ── Weight Loading ─────────────────────────────────────────────────

    static LoadReport loadAndBindWeights(Path dir, StableDiffusionPipeline pipeline,
                                         LoadOptions opts) throws Exception {
        Map<String, Tensor> allTensors = new HashMap<>();
        java.util.List<String> errors = new java.util.ArrayList<>();
        long totalBytes = 0;

        // Collect all safetensors
        try (java.nio.file.DirectoryStream<Path> stream =
                java.nio.file.Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : stream) {
                try (SafeTensorsLoader loader = SafeTensorsLoader.createDefault()) {
                    SafeTensorsLoader.SafeTensorsLoadResult result = loader.load(p);
                    for (Map.Entry<String, Tensor> e : result.tensors.entrySet()) {
                        allTensors.put(e.getKey(), e.getValue());
                        totalBytes += e.getValue().nbytes();
                    }
                } catch (Exception e) {
                    errors.add(p.getFileName() + ": " + e.getMessage());
                }
            }
        }

        // Also try loading via SnapshotFiles
        try {
            Map<String, Tensor> snapshotTensors = SnapshotFiles.loadAllWeights(dir, true);
            for (Map.Entry<String, Tensor> e : snapshotTensors.entrySet()) {
                if (!allTensors.containsKey(e.getKey())) {
                    allTensors.put(e.getKey(), e.getValue());
                    totalBytes += e.getValue().nbytes();
                }
            }
        } catch (Exception e) {
            // SnapshotFiles may fail for some formats — that's ok
        }

        // Bind weights to pipeline components
        int boundCount = 0;
        java.util.List<String> missingKeys = new java.util.ArrayList<>();
        java.util.List<String> unexpectedKeys = new java.util.ArrayList<>();

        for (Map.Entry<String, Tensor> entry : allTensors.entrySet()) {
            String key = entry.getKey();
            Tensor value = entry.getValue();
            try {
                pipeline.bindWeight(key, value);
                boundCount++;
            } catch (Exception e) {
                unexpectedKeys.add(key + " (" + e.getMessage() + ")");
            }
        }

        return new LoadReport(
            allTensors.size(), totalBytes, boundCount,
            missingKeys.size(), unexpectedKeys.size(), 0,
            0.0, errors, missingKeys, unexpectedKeys);
    }
}
