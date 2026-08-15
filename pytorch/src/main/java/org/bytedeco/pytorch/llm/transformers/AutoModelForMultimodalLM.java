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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers;

import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.loading.SnapshotFiles;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.llm.transformers.mapping.WeightMap;
import org.bytedeco.pytorch.llm.transformers.processor.*;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.vllm.multimodal.CompositeMultimodalProcessor;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;
import org.bytedeco.pytorch.llm.vllm.multimodal.MultimodalPrompt;
import org.bytedeco.pytorch.nn.Module;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Unified entry point for multimodal large language models.
 *
 * <p>Supports:
 * <ul>
 *   <li>Text generation from text-only inputs</li>
 *   <li>Image-to-text (vision-language models)</li>
 *   <li>Video-to-text (video understanding)</li>
 *   <li>Audio-to-text (speech recognition)</li>
 *   <li>Text-to-image generation</li>
 *   <li>Text-to-video generation</li>
 *   <li>Text-to-audio generation</li>
 *   <li>Any-to-any multimodal models (Qwen2.5-Omni, Gemma4, etc.)</li>
 * </ul>
 *
 * <p>Reference: HuggingFace AutoModelForMultimodalLM, AutoModelForImageTextToText,
 * Pipeline with "any-to-any" task
 *
 * <pre>{@code
 * // Load multimodal model
 * AutoModelForMultimodalLM.Bundle bundle = AutoModelForMultimodalLM.fromPretrained(
 *     "Qwen/Qwen2.5-Omni-3B", hub);
 *
 * // Use chat interface with multimodal inputs
 * List<Map<String, Object>> messages = List.of(
 *     Map.of("role", "user", "content", List.of(
 *         Map.of("type", "image", "image", imageData),
 *         Map.of("type", "text", "text", "Describe this image")
 *     ))
 * );
 *
 * String response = bundle.chat(messages);
 * }</pre>
 */
public class AutoModelForMultimodalLM {

    private AutoModelForMultimodalLM() {}

    /**
     * Generation output type.
     */
    public enum OutputType {
        TEXT,
        IMAGE,
        VIDEO,
        AUDIO,
        MULTIMODAL
    }

    /**
     * Loaded model + processor + configs + load report.
     */
    public static final class Bundle {
        private final Module model;
        private final Processor processor;
        private final FastTokenizer tokenizer;
        private final PretrainedConfig config;
        private final GenerationConfig generationConfig;
        private final Path snapshot;
        private final WeightLoader.LoadReport loadReport;
        private final ChatTemplate chatTemplate;
        private final OutputType supportedOutputType;
        private final Set<Processor.Modality> supportedInputModalities;
        private final CompositeMultimodalProcessor compositeProcessor;

        public Bundle(Module model, Processor processor, FastTokenizer tokenizer,
                      PretrainedConfig config, GenerationConfig generationConfig,
                      Path snapshot, WeightLoader.LoadReport loadReport,
                      ChatTemplate chatTemplate, OutputType supportedOutputType,
                      Set<Processor.Modality> supportedInputModalities) {
            this(model, processor, tokenizer, config, generationConfig, snapshot, loadReport,
                    chatTemplate, supportedOutputType, supportedInputModalities, null);
        }

        public Bundle(Module model, Processor processor, FastTokenizer tokenizer,
                      PretrainedConfig config, GenerationConfig generationConfig,
                      Path snapshot, WeightLoader.LoadReport loadReport,
                      ChatTemplate chatTemplate, OutputType supportedOutputType,
                      Set<Processor.Modality> supportedInputModalities,
                      CompositeMultimodalProcessor compositeProcessor) {
            this.model = Objects.requireNonNull(model, "model");
            this.processor = processor;
            this.tokenizer = tokenizer;
            this.config = Objects.requireNonNull(config, "config");
            this.generationConfig = generationConfig == null ? GenerationConfig.greedy() : generationConfig;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
            this.chatTemplate = chatTemplate == null ? ChatTemplate.qwen() : chatTemplate;
            this.supportedOutputType = supportedOutputType != null ? supportedOutputType : OutputType.TEXT;
            this.supportedInputModalities = supportedInputModalities != null ?
                    EnumSet.copyOf(supportedInputModalities) : EnumSet.of(Processor.Modality.TEXT);
            this.compositeProcessor = compositeProcessor;
        }

        public Module model() { return model; }
        public Processor processor() { return processor; }
        public FastTokenizer tokenizer() { return tokenizer; }
        public PretrainedConfig config() { return config; }
        public GenerationConfig generationConfig() { return generationConfig; }
        public Path snapshot() { return snapshot; }
        public WeightLoader.LoadReport loadReport() { return loadReport; }
        public ChatTemplate chatTemplate() { return chatTemplate; }
        public OutputType supportedOutputType() { return supportedOutputType; }
        public Set<Processor.Modality> supportedInputModalities() { return supportedInputModalities; }
        public CompositeMultimodalProcessor compositeProcessor() { return compositeProcessor; }

        /**
         * Check if this model supports a specific input modality.
         */
        public boolean supportsInput(Processor.Modality modality) {
            return supportedInputModalities.contains(modality);
        }

        /**
         * Check if this model supports a specific output type.
         */
        public boolean supportsOutput(OutputType outputType) {
            return this.supportedOutputType == outputType ||
                   this.supportedOutputType == OutputType.MULTIMODAL;
        }

        /**
         * Chat with multimodal messages using the vLLM CompositeMultimodalProcessor
         * when available — this routes text/image/audio/video parts into a single
         * token stream with media encoders (DINOv2/CLIP/SmolVLM for image, Whisper
         * for ASR, dedicated VideoEncoder).
         */
        public String chatViaComposite(List<Map<String, Object>> messages, GenerationConfig gen) {
            if (compositeProcessor == null) {
                // Fallback: build a MultimodalPrompt directly and use the simple path
                MultimodalPrompt prompt = buildPrompt(messages);
                return chatSimple(prompt, messages, gen);
            }
            MultimodalPrompt prompt = buildPrompt(messages);

            // Flatten messages to text-only for chat template
            List<Map<String, String>> textMessages = flattenMessages(messages);
            int[] inputIds = compositeProcessor.process(prompt, textMessages);

            if (gen == null) gen = generationConfig;
            GenerationConfig.Builder b = gen.toBuilder();
            if (gen.eosTokenIds.isEmpty()) {
                b.eosTokenId(config.eosTokenId());
                for (int id : generationConfig.eosTokenIds) b.eosTokenId(id);
            }
            GenerationConfig g = b.build();

            int[] out = org.bytedeco.pytorch.llm.transformers.generation.Generator.generate(
                    model, inputIds, g, config.maxPositionEmbeddings());

            int promptLen = inputIds.length;
            int[] generated = out.length > promptLen
                    ? Arrays.copyOfRange(out, promptLen, out.length)
                    : out;

            return tokenizer == null ? "" : tokenizer.decode(generated, true);
        }

        private String chatSimple(MultimodalPrompt prompt, List<Map<String, Object>> messages,
                                   GenerationConfig gen) {
            List<Map<String, String>> textMessages = flattenMessages(messages);
            String chat = chatTemplate.apply(textMessages, true);
            if (tokenizer == null) return chat;
            int[] ids = tokenizer.encode(chat, false).ids();
            if (gen == null) gen = generationConfig;
            int[] out = org.bytedeco.pytorch.llm.transformers.generation.Generator.generate(
                    model, ids, gen, config.maxPositionEmbeddings());
            int promptLen = ids.length;
            int[] generated = out.length > promptLen
                    ? Arrays.copyOfRange(out, promptLen, out.length)
                    : out;
            return tokenizer.decode(generated, true);
        }

        private static MultimodalPrompt buildPrompt(List<Map<String, Object>> messages) {
            List<MediaInput> parts = new ArrayList<>();
            for (Map<String, Object> msg : messages) {
                Object content = msg.get("content");
                if (content instanceof List) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        String type = String.valueOf(item.get("type"));
                        switch (type) {
                            case "text" -> parts.add(MediaInput.text(String.valueOf(item.get("text"))));
                            case "image" -> parts.add(toImageMediaInput(item.get("image")));
                            case "audio" -> parts.add(toAudioMediaInput(item.get("audio")));
                            case "video" -> parts.add(toVideoMediaInput(item.get("video")));
                            default -> {}
                        }
                    }
                } else if (content instanceof String) {
                    parts.add(MediaInput.text(String.valueOf(content)));
                }
            }
            return new MultimodalPrompt(parts);
        }

        private static MediaInput toImageMediaInput(Object obj) {
            if (obj instanceof java.nio.file.Path p) {
                return MediaInput.image(p);
            }
            if (obj instanceof byte[] bytes) {
                return MediaInput.imageBytes(bytes, 0, 0);
            }
            if (obj instanceof org.bytedeco.pytorch.Tensor t) {
                return MediaInput.builder().type(MediaType.IMAGE).tensor(t).build();
            }
            return MediaInput.text("[image]");
        }

        private static MediaInput toAudioMediaInput(Object obj) {
            if (obj instanceof java.nio.file.Path p) {
                return MediaInput.audio(p);
            }
            if (obj instanceof org.bytedeco.pytorch.Tensor t) {
                return MediaInput.builder().type(MediaType.AUDIO).tensor(t).build();
            }
            return MediaInput.text("[audio]");
        }

        private static MediaInput toVideoMediaInput(Object obj) {
            if (obj instanceof java.nio.file.Path p) {
                return MediaInput.video(p);
            }
            if (obj instanceof org.bytedeco.pytorch.Tensor t) {
                return MediaInput.builder().type(MediaType.VIDEO).tensor(t).build();
            }
            return MediaInput.text("[video]");
        }

        private static List<Map<String, String>> flattenMessages(List<Map<String, Object>> messages) {
            List<Map<String, String>> flat = new ArrayList<>();
            for (Map<String, Object> msg : messages) {
                String role = String.valueOf(msg.getOrDefault("role", "user"));
                Object content = msg.get("content");
                String text;
                if (content instanceof List) {
                    StringBuilder sb = new StringBuilder();
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        if ("text".equals(String.valueOf(item.get("type")))) {
                            if (sb.length() > 0) sb.append('\n');
                            sb.append(item.get("text"));
                        }
                    }
                    text = sb.toString();
                } else {
                    text = String.valueOf(content);
                }
                flat.add(Map.of("role", role, "content", text));
            }
            return flat;
        }

        /**
         * Generate text from text-only input.
         */
        public String generate(String prompt, GenerationConfig gen) {
            Objects.requireNonNull(prompt, "prompt");

            var enc = tokenizer.encode(prompt, true);
            GenerationConfig g = mergeGen(gen);
            int[] out = org.bytedeco.pytorch.llm.transformers.generation.Generator.generate(
                    model, enc.ids(), g, config.maxPositionEmbeddings());

            int promptLen = enc.ids().length;
            if (out.length > promptLen) {
                int[] neu = new int[out.length - promptLen];
                System.arraycopy(out, promptLen, neu, 0, neu.length);
                return tokenizer.decode(neu, true);
            }
            return tokenizer.decode(out, true);
        }

        /**
         * Chat with text-only messages.
         */
        public String chat(List<Map<String, Object>> messages, GenerationConfig gen) {
            if (messages == null || messages.isEmpty()) {
                throw new IllegalArgumentException("messages cannot be null or empty");
            }

            String prompt = chatTemplate.applyMultimodal(messages, true);
            return generate(prompt, gen);
        }

        /**
         * Chat with multimodal messages (images, videos, audio).
         */
        public MultimodalOutput chatMultimodal(List<Map<String, Object>> messages, GenerationConfig gen) {
            if (messages == null || messages.isEmpty()) {
                throw new IllegalArgumentException("messages cannot be null or empty");
            }

            // Apply chat template and process inputs
            Processor.ProcessingInput input = prepareMultimodalInput(messages);
            Processor.ProcessorOutput processed = processor.process(input);

            // Generate
            GenerationConfig g = mergeGen(gen);
            int[] inputIds = extractIntArray(processed.inputIds());

            int[] output = org.bytedeco.pytorch.llm.transformers.generation.Generator.generate(
                    model, inputIds, g, config.maxPositionEmbeddings());

            // Decode
            int promptLen = inputIds.length;
            int[] generatedIds;
            if (output.length > promptLen) {
                generatedIds = Arrays.copyOfRange(output, promptLen, output.length);
            } else {
                generatedIds = output;
            }

            String text = tokenizer.decode(generatedIds, true);

            // Parse response based on output type
            return parseResponse(text, processed);
        }

        /**
         * Chat with multimodal messages using default generation config.
         */
        public MultimodalOutput chatMultimodal(List<Map<String, Object>> messages) {
            return chatMultimodal(messages, generationConfig);
        }

        /**
         * Chat with text-only messages using default generation config.
         */
        public String chat(List<Map<String, Object>> messages) {
            return chat(messages, generationConfig);
        }

        /**
         * Describe an image with text.
         */
        public String describeImage(Object image, String question, GenerationConfig gen) {
            List<Map<String, Object>> messages = List.of(
                    Map.of("role", "user", "content", List.of(
                            Map.of("type", "image", "image", image),
                            Map.of("type", "text", "text", question != null ? question : "Describe this image")
                    ))
            );
            return chat(messages, gen).trim();
        }

        /**
         * Describe a video with text.
         */
        public String describeVideo(Object video, String question, GenerationConfig gen) {
            List<Map<String, Object>> messages = List.of(
                    Map.of("role", "user", "content", List.of(
                            Map.of("type", "video", "video", video),
                            Map.of("type", "text", "text", question != null ? question : "Describe this video")
                    ))
            );
            return chat(messages, gen).trim();
        }

        /**
         * Transcribe audio to text.
         */
        public String transcribe(Object audio, GenerationConfig gen) {
            List<Map<String, Object>> messages = List.of(
                    Map.of("role", "user", "content", List.of(
                            Map.of("type", "audio", "audio", audio)
                    ))
            );
            return chat(messages, gen).trim();
        }

        /**
         * Generate image from text (for image generation models).
         */
        public Object generateImage(String prompt, GenerationConfig gen) {
            if (!supportsOutput(OutputType.IMAGE) && !supportsOutput(OutputType.MULTIMODAL)) {
                throw new UnsupportedOperationException("This model does not support image generation");
            }

            String fullPrompt = chatTemplate.apply(Collections.singletonList(
                    Map.of("role", "user", "content", prompt)), true);

            int[] ids = tokenizer.encode(fullPrompt, true).ids();
            GenerationConfig g = mergeGen(gen);

            int[] output = org.bytedeco.pytorch.llm.transformers.generation.Generator.generate(
                    model, ids, g, config.maxPositionEmbeddings());

            // For image generation models, decode to image tensor
            return decodeImageOutput(output, ids.length);
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

        private Processor.ProcessingInput prepareMultimodalInput(List<Map<String, Object>> messages) {
            Processor.ProcessingInput input = Processor.ProcessingInput.of("");
            List<Object> images = new ArrayList<>();
            List<Object> videos = new ArrayList<>();
            List<Object> audios = new ArrayList<>();
            StringBuilder textBuilder = new StringBuilder();

            for (Map<String, Object> msg : messages) {
                Object content = msg.get("content");
                if (content instanceof List) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        String type = String.valueOf(item.get("type"));
                        switch (type) {
                            case "image" -> {
                                images.add(item.get("image"));
                                textBuilder.append("<image>");
                            }
                            case "video" -> {
                                videos.add(item.get("video"));
                                textBuilder.append("<video>");
                            }
                            case "audio" -> {
                                audios.add(item.get("audio"));
                                textBuilder.append("<audio>");
                            }
                            case "text" -> {
                                textBuilder.append(item.get("text"));
                            }
                        }
                    }
                } else if (content instanceof String) {
                    textBuilder.append(content);
                }
            }

            // Use factory method to create input with all components
            if (!images.isEmpty() && !videos.isEmpty()) {
                input = Processor.ProcessingInput.ofVideo(textBuilder.toString(), videos);
                input.images(images);
                input.audios(audios);
            } else if (!images.isEmpty()) {
                input = Processor.ProcessingInput.ofImages(textBuilder.toString(), images);
                input.audios(audios);
            } else if (!videos.isEmpty()) {
                input = Processor.ProcessingInput.ofVideo(textBuilder.toString(), videos);
                input.images(images);
                input.audios(audios);
            } else if (!audios.isEmpty()) {
                input = Processor.ProcessingInput.ofAudio(textBuilder.toString(), audios);
            } else {
                input = Processor.ProcessingInput.of(textBuilder.toString());
            }

            return input;
        }

        private int[] extractIntArray(org.bytedeco.pytorch.Tensor tensor) {
            if (tensor == null || tensor.numel() == 0) {
                return new int[0];
            }
            long numel = tensor.numel();
            int[] result = new int[(int) numel];
            for (int i = 0; i < numel; i++) {
                result[i] = (int) tensor.get(i).item().toLong();
            }
            return result;
        }

        private MultimodalOutput parseResponse(String text, Processor.ProcessorOutput processed) {
            // Parse generated text to determine output type
            // This is model-specific and would need customization per model
            return new MultimodalOutput(text, null, OutputType.TEXT);
        }

        private Object decodeImageOutput(int[] tokenIds, int promptLen) {
            // For image generation models, decode token IDs to image
            // Placeholder - actual implementation depends on model architecture
            return null;
        }
    }

    /**
     * Multimodal generation output.
     */
    public static class MultimodalOutput {
        private final String text;
        private final Object media;
        private final OutputType outputType;

        public MultimodalOutput(String text, Object media, OutputType outputType) {
            this.text = text;
            this.media = media;
            this.outputType = outputType;
        }

        public String text() { return text; }
        public Object media() { return media; }
        public OutputType outputType() { return outputType; }

        public String toString() {
            if (outputType == OutputType.TEXT) {
                return text;
            }
            return String.format("MultimodalOutput{type=%s, text='%s', hasMedia=%s}",
                    outputType, text, media != null);
        }
    }

    /**
     * Load options for multimodal models.
     */
    public static final class LoadOptions {
        public WeightLoader.BindMode bindMode = WeightLoader.BindMode.ZERO_COPY;
        public boolean strict = true;
        public boolean zeroCopyMmap = true;
        public boolean loadWeights = true;
        public boolean loadProcessor = true;
        public String processorType;  // Force specific processor type

        public LoadOptions bindMode(WeightLoader.BindMode m) { this.bindMode = m; return this; }
        public LoadOptions strict(boolean v) { this.strict = v; return this; }
        public LoadOptions zeroCopyMmap(boolean v) { this.zeroCopyMmap = v; return this; }
        public LoadOptions loadWeights(boolean v) { this.loadWeights = v; return this; }
        public LoadOptions loadProcessor(boolean v) { this.loadProcessor = v; return this; }
        public LoadOptions processorType(String type) { this.processorType = type; return this; }
    }

    // ============= Factory Methods =============

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, new LoadOptions());
    }

    public static Bundle fromPretrained(String modelId, HfHub hub, LoadOptions opts) throws IOException {
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(hub, "hub");
        Path snap = hub.snapshotDownload(modelId);
        return fromDirectory(snap, opts);
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

        // Read config
        PretrainedConfig cfg = readConfig(dir);

        // Determine model architecture
        ModelInfo info = detectModelArchitecture(dir, cfg);

        // Create model
        Module model = ModelRegistry.create(cfg);
        model.eval();

        // Handle dtype conversion
        String dtypeStr = cfg.torchDtype();
        boolean needsDtypeConversion = dtypeStr != null && !dtypeStr.isEmpty() && !"float32".equals(dtypeStr);
        if (needsDtypeConversion) {
            try {
                var scalarType = parseDtype(dtypeStr);
                if (scalarType != null) {
                    SnapshotFiles.toDtype(model, scalarType);
                }
            } catch (Exception e) {
                System.out.println("[DEBUG] dtype conversion failed: " + e.getMessage());
            }
        }

        // Load weights
        WeightLoader.LoadReport report = null;
        if (opts.loadWeights) {
            WeightMap map = ModelRegistry.weightMap(cfg);
            var bindMode = needsDtypeConversion ? WeightLoader.BindMode.COPY : opts.bindMode;
            try {
                report = WeightLoader.loadAndBind(model, dir, map, bindMode, opts.strict, opts.zeroCopyMmap);
            } catch (IOException e) {
                if (opts.strict) throw e;
                report = new WeightLoader.LoadReport(
                        List.of(), List.of("(no safetensors)"), List.of(), List.of(), 0, 0, bindMode);
            }

            System.out.println("[AutoModelForMultimodalLM] " + report
                    + " model=" + model.getClass().getSimpleName()
                    + " type=" + cfg.modelType()
                    + " arch=" + info.architecture);
        }

        // Load processor
        Processor processor = null;
        CompositeMultimodalProcessor composite = null;
        FastTokenizer tok = readTokenizer(dir, cfg);
        ChatTemplate template = ChatTemplate.detect(dir, cfg);

        if (opts.loadProcessor) {
            processor = loadProcessor(dir, cfg, opts.processorType);
            // Also compose a vLLM-style composite for runtime media-encoder routing
            try {
                composite = CompositeMultimodalProcessor.of(tok, template);
            } catch (Throwable t) {
                System.out.println("[DEBUG] compositeProcessor init failed: " + t.getMessage());
            }
        }

        // Load generation config
        GenerationConfig genCfg = readGenerationConfig(dir, cfg);

        return new Bundle(model, processor, tok, cfg, genCfg, dir, report, template,
                info.outputType, info.inputModalities, composite);
    }

    // ============= Helper Methods =============

    private static PretrainedConfig readConfig(Path dir) throws IOException {
        Path cfg = SnapshotFiles.configJson(dir);
        if (Files.isRegularFile(cfg)) {
            return PretrainedConfig.fromFile(cfg);
        }
        throw new IOException("Missing config.json in " + dir);
    }

    private static FastTokenizer readTokenizer(Path dir, PretrainedConfig cfg) throws IOException {
        try {
            return org.bytedeco.pytorch.llm.tokenizers.DirectoryTokenizerLoader.load(dir);
        } catch (IOException e) {
            System.out.println("[WARNING] Could not load tokenizer: " + e.getMessage());
            return null;
        }
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
                .maxNewTokens(256)
                .eosTokenId(cfg.eosTokenId())
                .padTokenId(cfg.padTokenId())
                .bosTokenId(cfg.bosTokenId())
                .build();
    }

    private static org.bytedeco.pytorch.global.torch.ScalarType parseDtype(String dtypeStr) {
        return switch (dtypeStr.toLowerCase()) {
            case "bfloat16", "bf16" -> org.bytedeco.pytorch.global.torch.ScalarType.BFloat16;
            case "float16", "fp16", "half" -> org.bytedeco.pytorch.global.torch.ScalarType.Half;
            case "float", "float32" -> org.bytedeco.pytorch.global.torch.ScalarType.Float;
            case "double", "float64" -> org.bytedeco.pytorch.global.torch.ScalarType.Double;
            default -> null;
        };
    }

    /**
     * Detect model architecture and capabilities from config.
     */
    private static ModelInfo detectModelArchitecture(Path dir, PretrainedConfig cfg) {
        Map<String, Object> extra = cfg.extra();
        String modelType = String.valueOf(extra.getOrDefault("model_type", cfg.modelType().name())).toLowerCase();
        String architectures = "";
        Object archs = extra.get("architectures");
        if (archs instanceof List<?> list) {
            architectures = list.stream()
                    .map(String::valueOf)
                    .reduce("", (a, b) -> a + "," + b).toLowerCase();
        } else if (archs instanceof String s) {
            architectures = s.toLowerCase();
        }

        // Check for multimodal architectures
        if (architectures.contains("qwen2_5_omni") || modelType.contains("qwen2_5_omni")) {
            return new ModelInfo("qwen2.5-omni", OutputType.MULTIMODAL,
                    EnumSet.of(Processor.Modality.TEXT, Processor.Modality.IMAGE, Processor.Modality.AUDIO));
        }
        if (architectures.contains("qwen2_vl") || modelType.contains("qwen2_vl") || modelType.contains("qwen2.5_vl")) {
            return new ModelInfo("qwen2-vl", OutputType.TEXT,
                    EnumSet.of(Processor.Modality.TEXT, Processor.Modality.IMAGE, Processor.Modality.VIDEO));
        }
        if (architectures.contains("qwen3_vl") || modelType.contains("qwen3_vl")) {
            return new ModelInfo("qwen3-vl", OutputType.TEXT,
                    EnumSet.of(Processor.Modality.TEXT, Processor.Modality.IMAGE, Processor.Modality.VIDEO));
        }
        if (architectures.contains("gemma") && (architectures.contains("multimodal") || architectures.contains("vision"))) {
            return new ModelInfo("gemma-multimodal", OutputType.TEXT,
                    EnumSet.of(Processor.Modality.TEXT, Processor.Modality.IMAGE, Processor.Modality.VIDEO, Processor.Modality.AUDIO));
        }
        if (architectures.contains("minimax_vl") || modelType.contains("minimax")) {
            return new ModelInfo("minimax-vl", OutputType.TEXT,
                    EnumSet.of(Processor.Modality.TEXT, Processor.Modality.IMAGE));
        }
        if (architectures.contains("llava") || modelType.contains("llava")) {
            return new ModelInfo("llava", OutputType.TEXT,
                    EnumSet.of(Processor.Modality.TEXT, Processor.Modality.IMAGE));
        }
        if (architectures.contains("whisper") || modelType.contains("whisper")) {
            return new ModelInfo("whisper", OutputType.TEXT,
                    EnumSet.of(Processor.Modality.AUDIO));
        }
        if (architectures.contains("stable_diffusion") || modelType.contains("stable_diffusion")) {
            return new ModelInfo("stable-diffusion", OutputType.IMAGE,
                    EnumSet.of(Processor.Modality.TEXT));
        }

        // Default: text-only model
        return new ModelInfo(modelType, OutputType.TEXT,
                EnumSet.of(Processor.Modality.TEXT));
    }

    /**
     * Load appropriate processor for the model.
     */
    private static Processor loadProcessor(Path dir, PretrainedConfig cfg, String forcedType) {
        try {
            if (forcedType != null) {
                return AutoProcessor.create(forcedType, dir);
            }
            return AutoProcessor.fromPretrained(dir);
        } catch (IOException e) {
            System.out.println("[WARNING] Could not load processor: " + e.getMessage());
            return new TextOnlyProcessorForMultimodal(null);
        }
    }

    /**
     * Model information for architecture detection.
     */
    private static class ModelInfo {
        final String architecture;
        final OutputType outputType;
        final Set<Processor.Modality> inputModalities;

        ModelInfo(String architecture, OutputType outputType, Set<Processor.Modality> inputModalities) {
            this.architecture = architecture;
            this.outputType = outputType;
            this.inputModalities = inputModalities;
        }
    }

    /**
     * Simple text-only processor for fallback.
     */
    private static class TextOnlyProcessorForMultimodal implements Processor {
        private final FastTokenizer tokenizer;

        TextOnlyProcessorForMultimodal(FastTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        @Override public String version() { return "1.0"; }
        @Override public FastTokenizer tokenizer() { return tokenizer; }
        @Override public List<Processor.Modality> supportedModalities() { return List.of(Processor.Modality.TEXT); }
        @Override public Processor.TextOutput processText(String text, boolean addSpecialTokens) {
            if (tokenizer != null) {
                return new Processor.TextOutput(tokenizer.encode(text, addSpecialTokens).ids(), null, 0);
            }
            return new Processor.TextOutput(new int[0], null, 0);
        }
        @Override public Processor.TextOutput processTextBatch(List<String> texts) {
            return processText(texts.isEmpty() ? "" : String.join(" ", texts), true);
        }
        @Override public Processor.ImageOutput processImage(Object image) {
            throw new UnsupportedOperationException("Not supported");
        }
        @Override public List<Processor.ImageOutput> processImageBatch(List<?> images) { return List.of(); }
        @Override public Processor.AudioOutput processAudio(Object audio) {
            throw new UnsupportedOperationException("Not supported");
        }
        @Override public Processor.VideoOutput processVideo(Object video) {
            throw new UnsupportedOperationException("Not supported");
        }
        @Override public Processor.ProcessorOutput process(Processor.ProcessingInput input) {
            int[] ids = processText(input.text(), true).inputIds();
            return Processor.ProcessorOutput.builder()
                    .inputIds(org.bytedeco.pytorch.global.torch.tensor(ids))
                    .build();
        }
        @Override public Processor.ProcessorStats getStats() { return new Processor.ProcessorStats(0, 0, 0, 0, 0, 0); }
        @Override public void resetStats() {}
        @Override public boolean isClosed() { return false; }
        @Override public void close() {}
    }
}
