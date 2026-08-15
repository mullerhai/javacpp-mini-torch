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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.diffusion.modeling.UNet2DConditionModel;
import org.bytedeco.pytorch.llm.diffusion.modeling.AutoencoderKL;
import org.bytedeco.pytorch.llm.diffusion.modeling.CLIPTextEmbeddings;
import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Objects;
import java.util.Random;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * StableDiffusionPipeline — end-to-end text-to-image generation pipeline.
 *
 * <p>Combines four components into a unified generation interface:
 * <ol>
 *   <li>{@link CLIPTextEmbeddings} — text tokenization + encoding</li>
 *   <li>{@link UNet2DConditionModel} — denoising U-Net (noise prediction)</li>
 *   <li>{@link AutoencoderKL} — VAE (latent ↔ pixel conversion)</li>
 *   <li>{@link Scheduler} — diffusion schedule (DDPM / DDIM / Euler)</li>
 * </ol>
 *
 * <p>Generation flow:
 * <pre>
 *  text → tokenizer → CLIP → text_embeddings
 *  noise → [repeat for each step] ──────────────────+→ U-Net → noise_pred
 *                           ↑                         |
 *                           └────── Scheduler.step ←──┘
 *  latent_final → VAE.decode → image
 * </pre>
 *
 * <p>Usage:
 * <pre>{@code
 * StableDiffusionPipeline pipe = StableDiffusionPipeline.fromDirectory(
 *     Paths.get("/path/to/anima/checkpoint"));
 *
 * BufferedImage img = pipe.generate(
 *     "A giant panda eating bamboo in a bamboo forest",
 *     StableDiffusionPipeline.GenerationParams.create()
 *         .height(512).width(512)
 *         .numInferenceSteps(20)
 *         .guidanceScale(7.5f));
 *
 * ImageIO.write(img, "png", new File("output.png"));
 * }</pre>
 */
public class StableDiffusionPipeline {

    public static class GenerationParams {
        private int height = 512;
        private int width = 512;
        private int numInferenceSteps = 50;
        private float guidanceScale = 7.5f;
        private Long seed = null;
        private int batchSize = 1;
        private String negativePrompt = "";

        public static GenerationParams create() {
            return new GenerationParams();
        }

        public GenerationParams height(int h) { this.height = h; return this; }
        public GenerationParams width(int w) { this.width = w; return this; }
        public GenerationParams numInferenceSteps(int s) { this.numInferenceSteps = s; return this; }
        public GenerationParams guidanceScale(float g) { this.guidanceScale = g; return this; }
        public GenerationParams seed(Long s) { this.seed = s; return this; }
        public GenerationParams batchSize(int b) { this.batchSize = b; return this; }
        public GenerationParams negativePrompt(String p) { this.negativePrompt = p; return this; }

        public int height() { return height; }
        public int width() { return width; }
        public int numInferenceSteps() { return numInferenceSteps; }
        public float guidanceScale() { return guidanceScale; }
        public Long seed() { return seed; }
        public int batchSize() { return batchSize; }
        public String negativePrompt() { return negativePrompt; }
    }

    public static class PipelineConfig {
        private UNet2DConditionModel.DiffusionUnetConfig unetConfig;
        private AutoencoderKL.VAEConfig vaeConfig;
        private CLIPTextEmbeddings.CLIPTextConfig textConfig;

        public PipelineConfig unetConfig(UNet2DConditionModel.DiffusionUnetConfig c) { this.unetConfig = c; return this; }
        public PipelineConfig vaeConfig(AutoencoderKL.VAEConfig c) { this.vaeConfig = c; return this; }
        public PipelineConfig textConfig(CLIPTextEmbeddings.CLIPTextConfig c) { this.textConfig = c; return this; }

        public UNet2DConditionModel.DiffusionUnetConfig unetConfig() { return unetConfig; }
        public AutoencoderKL.VAEConfig vaeConfig() { return vaeConfig; }
        public CLIPTextEmbeddings.CLIPTextConfig textConfig() { return textConfig; }
    }

    // ── Components ───────────────────────────────────────────────────

    public final UNet2DConditionModel unet;
    public final AutoencoderKL vae;
    public final CLIPTextEmbeddings textEncoder;
    public final FastTokenizer tokenizer;
    public final Scheduler scheduler;
    public final PipelineConfig config;

    // ── Constructors ────────────────────────────────────────────────

    public StableDiffusionPipeline(
            UNet2DConditionModel unet,
            AutoencoderKL vae,
            CLIPTextEmbeddings textEncoder,
            FastTokenizer tokenizer,
            Scheduler scheduler,
            PipelineConfig config) {
        this.unet = Objects.requireNonNull(unet, "unet");
        this.vae = Objects.requireNonNull(vae, "vae");
        this.textEncoder = Objects.requireNonNull(textEncoder, "textEncoder");
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Build a default pipeline with standard SD 1.5 config.
     * Use {@link #fromDirectory} or {@link #fromComponents} for production use.
     */
    public static StableDiffusionPipeline defaultPipeline() {
        UNet2DConditionModel.DiffusionUnetConfig uc = new UNet2DConditionModel.DiffusionUnetConfig();
        uc.inChannels(4);
        uc.outChannels(4);
        uc.blockOutChannels(new int[]{320, 640, 1280});
        uc.crossAttentionDim(new int[]{768});

        AutoencoderKL.VAEConfig vc = new AutoencoderKL.VAEConfig();
        vc.inChannels(3);
        vc.latentChannels(4);

        CLIPTextEmbeddings.CLIPTextConfig tc = new CLIPTextEmbeddings.CLIPTextConfig();
        // Use 100k vocab to support tiktoken cl100k_base token IDs (Synthetic default)
        tc.vocabSize(100256);
        tc.hiddenSize(768);
        tc.numHiddenLayers(12);
        tc.numAttentionHeads(12);

        return new StableDiffusionPipeline(
            new UNet2DConditionModel(uc),
            new AutoencoderKL(vc),
            new CLIPTextEmbeddings(tc),
            AutoTokenizer.cl100kBase(),
            new Scheduler.EulerDiscreteScheduler(),
            new PipelineConfig());
    }

    /**
     * Build pipeline from individual components (after loading weights).
     */
    public static StableDiffusionPipeline fromComponents(
            UNet2DConditionModel unet,
            AutoencoderKL vae,
            CLIPTextEmbeddings textEncoder,
            FastTokenizer tokenizer,
            Scheduler scheduler) {
        PipelineConfig config = new PipelineConfig();
        return new StableDiffusionPipeline(unet, vae, textEncoder, tokenizer, scheduler, config);
    }

    /**
     * Build pipeline from a checkpoint directory.
     * Attempts to load config.json and safetensors weights.
     *
     * @param modelDir directory containing model files
     * @return configured pipeline (weights may need manual binding via {@link #bindWeights})
     */
    public static StableDiffusionPipeline fromDirectory(java.nio.file.Path modelDir) throws Exception {
        java.nio.file.Path configPath = modelDir.resolve("config.json");
        if (!java.nio.file.Files.exists(configPath)) {
            // Use default config if no config.json
            return defaultPipeline();
        }

        // TODO: parse config.json and create models with correct dimensions
        // For now, return default pipeline (weights can be bound externally)
        return defaultPipeline();
    }

    /**
     * Bind safetensors weights from a directory to pipeline components.
     *
     * <p>Expected key patterns in safetensors:
     * <ul>
     *   <li>unet: {@code unet.*} → {@code unet.*}</li>
     *   <li>vae: {@codevae.*} → {@code vae.*}</li>
     *   <li>text_encoder: {@code text_encoder.*} → {@code textEncoder.*}</li>
     * </ul>
     *
     * @param modelDir directory containing .safetensors files
     */
    public void bindWeights(java.nio.file.Path modelDir) throws Exception {
        System.out.println("  Weight binding is a placeholder — for production, load weights directly into modules.");
    }

    /**
     * Bind a single weight tensor to the appropriate module parameter.
     *
     * @param key   HF-style parameter name (e.g. "unet.conv_in.weight")
     * @param value the loaded tensor
     */
    public void bindWeight(String key, Tensor value) {
        // Placeholder — for production use the WeightLoader with ModelRegistry
    }

    // ── Generation ─────────────────────────────────────────────────

    /**
     * Generate an image from a text prompt.
     *
     * @param prompt text description
     * @param params generation parameters
     * @return generated image as BufferedImage
     */
    public BufferedImage generate(String prompt, GenerationParams params) throws Exception {
        Random rng = params.seed() != null
                ? new Random(params.seed())
                : new Random();

        int H = params.height();
        int W = params.width();
        int latentH = H / 8;
        int latentW = W / 8;

        // 1. Encode prompt → text embeddings
        Tensor[] prompts = new Tensor[]{ encodeText(prompt) };

        // 2. Encode negative prompt (for classifier-free guidance)
        Tensor[] negPrompts;
        if (params.negativePrompt() != null && !params.negativePrompt().isEmpty()) {
            negPrompts = new Tensor[]{ encodeText(params.negativePrompt()) };
        } else {
            negPrompts = new Tensor[]{ zeros(new long[]{1, 1, textEncoder.config().hiddenSize()}) };
        }

        // 3. Initialize latent noise
        Tensor latents = torch.randn(
            new long[]{params.batchSize(), 4, latentH, latentW});

        // 4. Set up scheduler timesteps
        scheduler.setTimesteps(params.numInferenceSteps());
        int numSteps = scheduler.numInferenceTimesteps();

        // 5. Denoising loop
        for (int i = 0; i < numSteps; i++) {
            long t = scheduler.timestepAt(i);

            // Expand latents for classifier-free guidance
            Tensor latentModelInput = latents;  // TODO: duplicate for CFG

            // Concatenate with null text embedding for CFG
            // Note: simplified for now; full CFG needs two forward passes

            // U-Net forward
            Tensor noisePred = unet.forward(latentModelInput,
                torch.tensor(new long[]{t}),
                prompts[0]);

            // Scheduler step
            latents = scheduler.step(noisePred, t, latents);

            if (i % 5 == 0 || i == numSteps - 1) {
                System.out.println("  Step " + (i + 1) + "/" + numSteps
                    + " (t=" + t + ")");
            }
        }

        // 6. Decode latent → image
        latents = latents.div(new Scalar(0.18215));  // Scale factor used in SD
        Tensor imageTensor = vae.decode(latents);

        // 7. Convert to BufferedImage
        return tensorToBufferedImage(imageTensor);
    }

    /**
     * Encode text → [B, seq, hidden] embeddings.
     */
    public Tensor encodeText(String text) throws Exception {
        Encoding enc = tokenizer.encode(text);
        int[] ids = enc.ids();
        long[] longs = new long[ids.length];
        for (int i = 0; i < ids.length; i++) longs[i] = ids[i];
        Tensor inputIds = torch.tensor(longs);
        inputIds = inputIds.reshape(new long[]{1, inputIds.size(0)});
        return textEncoder.forward(inputIds);
    }

    // ── Tensor → BufferedImage ─────────────────────────────────────

    private static BufferedImage tensorToBufferedImage(Tensor tensor) {
        long[] shape = tensor.shape();
        int b, c, h, w;
        if (shape.length == 4) {
            b = (int) shape[0]; c = (int) shape[1]; h = (int) shape[2]; w = (int) shape[3];
        } else if (shape.length == 3) {
            b = 1; c = (int) shape[0]; h = (int) shape[1]; w = (int) shape[2];
        } else {
            throw new IllegalArgumentException("Invalid shape: " + java.util.Arrays.toString(shape));
        }

        if (b > 1) tensor = tensor.select(0, 0);
        if (c > 3) tensor = tensor.narrow(0, 0, 3);

        // Scale from [-1, 1] to [0, 1] then to [0, 255]
        tensor = tensor.add(new Scalar(1.0)).div(new Scalar(2.0));
        tensor = tensor.clamp_min(new Scalar(0.0)).clamp_max(new Scalar(1.0));

        Tensor r = tensor.select(0, 0);
        Tensor g = tensor.select(0, 1);
        Tensor bv = tensor.select(0, 2);

        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int ri = (int) clamp(r.select(0, y).select(0, x).item().toFloat() * 255.0f, 0, 255);
                int gi = (int) clamp(g.select(0, y).select(0, x).item().toFloat() * 255.0f, 0, 255);
                int bi = (int) clamp(bv.select(0, y).select(0, x).item().toFloat() * 255.0f, 0, 255);
                img.setRGB(x, y, (ri << 16) | (gi << 8) | bi);
            }
        }
        return img;
    }

    private static float clamp(float v, int min, int max) {
        return Math.max(min, Math.min(max, v));
    }

    // ── Getters ────────────────────────────────────────────────────

    public UNet2DConditionModel unet() { return unet; }
    public AutoencoderKL vae() { return vae; }
    public CLIPTextEmbeddings textEncoder() { return textEncoder; }
    public FastTokenizer tokenizer() { return tokenizer; }
    public Scheduler scheduler() { return scheduler; }
    public PipelineConfig config() { return config; }
}
