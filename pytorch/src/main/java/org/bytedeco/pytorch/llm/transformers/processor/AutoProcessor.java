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
package org.bytedeco.pytorch.llm.transformers.processor;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Factory for automatically loading multimodal processors based on model configuration.
 *
 * <p>Supports:
 * <ul>
 *   <li>Auto-detection of processor type from config</li>
 *   <li>Text-only models (no processing)</li>
 *   <li>Vision-Language models (Qwen2-VL, Qwen3-VL, MiniMax-VL, LLaVA)</li>
 *   <li>Extensible processor registration</li>
 * </ul>
 *
 * <p>Reference: HuggingFace AutoProcessor
 *
 * <pre>{@code
 * // Auto-detect and load processor
 * try (Processor processor = AutoProcessor.fromPretrained("Qwen/Qwen2-VL-7B-Instruct")) {
 *     ProcessorOutput output = processor.process(input);
 * }
 *
 * // Force specific processor type
 * try (Processor processor = AutoProcessor.create("qwen2_vl", config)) {
 *     // ...
 * }
 * }</pre>
 */
public class AutoProcessor {

    private static final Map<String, ProcessorFactory> REGISTERED_FACTORIES = new ConcurrentHashMap<>();

    static {
        // Register default factories
        registerFactory("qwen2_vl", Qwen2VLProcessor::builder);
        registerFactory("qwen3_vl", Qwen2VLProcessor::builder);  // Qwen3-VL uses same processor
        registerFactory("qwen_vl", Qwen2VLProcessor::builder);
        registerFactory("minimax_vl", MiniMaxVLProcessor::builder);
        registerFactory("llava", LlavaProcessor::builder);
        registerFactory("cogvlm", CogVLMProcessor::builder);
        registerFactory("idefics", IdeficsProcessor::builder);
        registerFactory("fuyu", FuyuProcessor::builder);
        registerFactory("paligemma", PaliGemmaProcessor::builder);
    }

    /**
     * Functional interface for processor creation.
     */
    @FunctionalInterface
    public interface ProcessorFactory {
        Builder create();
    }

    /**
     * Abstract builder for processors.
     */
    public abstract static class Builder implements AutoCloseable {
        protected FastTokenizer tokenizer;
        protected Path modelPath;

        public abstract Builder tokenizer(FastTokenizer tokenizer);
        public abstract Builder modelPath(Path modelPath);
        public abstract Processor build();

        public Builder tokenizer(String tokenizerPath) {
            try {
                return tokenizer(FastTokenizer.fromFile(tokenizerPath));
            } catch (IOException e) {
                throw new RuntimeException("Failed to load tokenizer: " + tokenizerPath, e);
            }
        }

        public Builder modelPath(String modelPath) {
            return modelPath(Path.of(modelPath));
        }
    }

    /**
     * Register a processor factory for a model type.
     */
    public static void registerFactory(String modelType, ProcessorFactory factory) {
        REGISTERED_FACTORIES.put(modelType.toLowerCase(), factory);
    }

    /**
     * Load processor from pretrained model path.
     */
    public static Processor fromPretrained(String modelPath) throws IOException {
        return fromPretrained(Path.of(modelPath));
    }

    /**
     * Load processor from pretrained model path.
     */
    public static Processor fromPretrained(Path modelPath) throws IOException {
        Path configPath = modelPath.resolve("config.json");
        Path processorConfigPath = modelPath.resolve("preprocessor_config.json");

        // Try to load processor config first
        if (Files.exists(processorConfigPath)) {
            return loadFromProcessorConfig(processorConfigPath, modelPath);
        }

        // Fall back to model config
        if (Files.exists(configPath)) {
            return loadFromModelConfig(configPath, modelPath);
        }

        throw new IOException("No config found at: " + modelPath);
    }

    /**
     * Load processor from preprocessor_config.json.
     */
    @SuppressWarnings("unchecked")
    private static Processor loadFromProcessorConfig(Path configPath, Path modelPath) throws IOException {
        String json = Files.readString(configPath);
        Map<String, Object> config = Json.decodeObject(json);

        // Determine processor type
        String processorClass = (String) config.getOrDefault("processor_class", "");
        String modelType = (String) config.getOrDefault("model_type", "");
        String autoMap = (String) config.getOrDefault("auto_map", "");

        // Check for vision-language models
        if (processorClass.toLowerCase().contains("qwen2_vl") ||
            modelType.toLowerCase().contains("qwen2_vl")) {
            return createQwen2VL(modelPath, config);
        }

        if (processorClass.toLowerCase().contains("qwen3_vl") ||
            modelType.toLowerCase().contains("qwen3_vl")) {
            return createQwen3VL(modelPath, config);
        }

        if (processorClass.toLowerCase().contains("minimax") ||
            autoMap.toLowerCase().contains("minimax")) {
            return createMiniMaxVL(modelPath, config);
        }

        if (processorClass.toLowerCase().contains("llava")) {
            return createLlava(modelPath, config);
        }

        if (processorClass.toLowerCase().contains("cogvlm")) {
            return createCogVLM(modelPath, config);
        }

        // Default: try registered factories
        String type = detectModelType(config);
        ProcessorFactory factory = REGISTERED_FACTORIES.get(type);
        if (factory != null) {
            Builder builder = factory.create();
            builder.modelPath(modelPath);
            return builder.build();
        }

        throw new IOException("Unknown processor type: " + processorClass + " / " + modelType);
    }

    /**
     * Load processor from model config.json.
     */
    @SuppressWarnings("unchecked")
    private static Processor loadFromModelConfig(Path configPath, Path modelPath) throws IOException {
        String json = Files.readString(configPath);
        Map<String, Object> config = Json.decodeObject(json);

        String modelType = (String) config.getOrDefault("model_type", "");

        // Check for vision config (VL models have vision_config or is_vision)
        boolean isVision = config.containsKey("vision_config") ||
                          config.containsKey("is_vision_model") ||
                          config.getOrDefault("is_vision_model", false).equals(true);

        if (isVision) {
            return loadFromProcessorConfig(
                modelPath.resolve("preprocessor_config.json"),
                modelPath
            );
        }

        // Text-only model - use base processor
        return createTextOnly(modelPath, config);
    }

    /**
     * Detect model type from config.
     */
    private static String detectModelType(Map<String, Object> config) {
        String modelType = (String) config.getOrDefault("model_type", "");
        String architecture = (String) config.getOrDefault("architectures", "");

        modelType = modelType.toLowerCase();
        architecture = architecture.toLowerCase();

        if (modelType.contains("qwen2_vl") || architecture.contains("qwen2_vl")) {
            return "qwen2_vl";
        }
        if (modelType.contains("qwen3_vl") || architecture.contains("qwen3_vl")) {
            return "qwen3_vl";
        }
        if (modelType.contains("minimax") || architecture.contains("minimax")) {
            return "minimax_vl";
        }
        if (modelType.contains("llava") || architecture.contains("llava")) {
            return "llava";
        }
        if (modelType.contains("cogvlm") || architecture.contains("cogvlm")) {
            return "cogvlm";
        }
        if (modelType.contains("idefics") || architecture.contains("idefics")) {
            return "idefics";
        }

        return modelType;
    }

    // ============= Processor creators =============

    private static Processor createQwen2VL(Path modelPath, Map<String, Object> config) throws IOException {
        Qwen2VLProcessor.Builder builder = Qwen2VLProcessor.builder();

        // Load tokenizer
        Path tokenizerPath = modelPath.resolve("tokenizer.json");
        if (Files.exists(tokenizerPath)) {
            builder.tokenizer(FastTokenizer.fromFile(tokenizerPath.toString()));
        }

        // Configure from vision config
        Object visionConfig = config.get("vision_config");
        if (visionConfig instanceof Map) {
            Map<String, Object> vc = (Map<String, Object>) visionConfig;
            builder.spatialMergeSize(((Number) vc.getOrDefault("spatial_merge_size", 14)).intValue());
        }

        return builder.build();
    }

    private static Processor createQwen3VL(Path modelPath, Map<String, Object> config) throws IOException {
        // Qwen3-VL uses same processor as Qwen2-VL
        return createQwen2VL(modelPath, config);
    }

    private static Processor createMiniMaxVL(Path modelPath, Map<String, Object> config) throws IOException {
        MiniMaxVLProcessor.Builder builder = MiniMaxVLProcessor.builder();

        // Load tokenizer
        Path tokenizerPath = modelPath.resolve("tokenizer.json");
        if (Files.exists(tokenizerPath)) {
            builder.tokenizer(FastTokenizer.fromFile(tokenizerPath.toString()));
        }

        return builder.build();
    }

    private static Processor createLlava(Path modelPath, Map<String, Object> config) {
        // Llava processor (placeholder for full implementation)
        return new LlavaProcessor(modelPath.toString());
    }

    private static Processor createCogVLM(Path modelPath, Map<String, Object> config) {
        // CogVLM processor (placeholder for full implementation)
        return new CogVLMProcessor(modelPath.toString());
    }

    @SuppressWarnings("unchecked")
    private static Processor createTextOnly(Path modelPath, Map<String, Object> config) throws IOException {
        Path tokenizerPath = modelPath.resolve("tokenizer.json");
        if (Files.exists(tokenizerPath)) {
            return new TextOnlyProcessor(FastTokenizer.fromFile(tokenizerPath.toString()));
        }
        throw new IOException("Tokenizer not found at: " + tokenizerPath);
    }

    /**
     * Create processor by type name.
     */
    public static Processor create(String processorType, Path modelPath) throws IOException {
        ProcessorFactory factory = REGISTERED_FACTORIES.get(processorType.toLowerCase());
        if (factory == null) {
            throw new IOException("Unknown processor type: " + processorType);
        }

        Builder builder = factory.create();
        builder.modelPath(modelPath);
        return builder.build();
    }

    /**
     * Get list of supported processor types.
     */
    public static List<String> supportedProcessorTypes() {
        return List.copyOf(REGISTERED_FACTORIES.keySet());
    }

    // ============= Placeholder processors =============

    /**
     * Placeholder Llava processor.
     */
    public static class LlavaProcessor implements Processor {
        private volatile boolean closed;
        private final FastTokenizer tokenizer;

        public LlavaProcessor(String modelPath) {
            this.tokenizer = null;  // Would load from modelPath
        }

        @Override public String version() { return "1.0"; }
        @Override public FastTokenizer tokenizer() { return tokenizer; }
        @Override public List<Modality> supportedModalities() { return List.of(Modality.TEXT, Modality.IMAGE); }
        @Override public TextOutput processText(String text, boolean addSpecialTokens) { return new TextOutput(new int[0], null, 0); }
        @Override public TextOutput processTextBatch(List<String> texts) { return new TextOutput(new int[0], null, 0); }
        @Override public ImageOutput processImage(Object image) { return null; }
        @Override public List<ImageOutput> processImageBatch(List<?> images) { return List.of(); }
        @Override public AudioOutput processAudio(Object audio) { return null; }
        @Override public VideoOutput processVideo(Object video) { return null; }
        @Override public ProcessorOutput process(ProcessingInput input) { return ProcessorOutput.builder().build(); }
        @Override public ProcessorStats getStats() { return new ProcessorStats(0, 0, 0, 0, 0, 0); }
        @Override public void resetStats() {}
        @Override public boolean isClosed() { return closed; }
        @Override public void close() { closed = true; }
    }

    /**
     * Placeholder CogVLM processor.
     */
    public static class CogVLMProcessor implements Processor {
        private volatile boolean closed;
        private final FastTokenizer tokenizer;

        public CogVLMProcessor(String modelPath) {
            this.tokenizer = null;
        }

        @Override public String version() { return "1.0"; }
        @Override public FastTokenizer tokenizer() { return tokenizer; }
        @Override public List<Modality> supportedModalities() { return List.of(Modality.TEXT, Modality.IMAGE); }
        @Override public TextOutput processText(String text, boolean addSpecialTokens) { return new TextOutput(new int[0], null, 0); }
        @Override public TextOutput processTextBatch(List<String> texts) { return new TextOutput(new int[0], null, 0); }
        @Override public ImageOutput processImage(Object image) { return null; }
        @Override public List<ImageOutput> processImageBatch(List<?> images) { return List.of(); }
        @Override public AudioOutput processAudio(Object audio) { return null; }
        @Override public VideoOutput processVideo(Object video) { return null; }
        @Override public ProcessorOutput process(ProcessingInput input) { return ProcessorOutput.builder().build(); }
        @Override public ProcessorStats getStats() { return new ProcessorStats(0, 0, 0, 0, 0, 0); }
        @Override public void resetStats() {}
        @Override public boolean isClosed() { return closed; }
        @Override public void close() { closed = true; }
    }

    /**
     * Placeholder Idefics processor.
     */
    public static class IdeficsProcessor implements Processor {
        private volatile boolean closed;
        private final FastTokenizer tokenizer;

        public IdeficsProcessor(String modelPath) {
            this.tokenizer = null;
        }

        @Override public String version() { return "1.0"; }
        @Override public FastTokenizer tokenizer() { return tokenizer; }
        @Override public List<Modality> supportedModalities() { return List.of(Modality.TEXT, Modality.IMAGE); }
        @Override public TextOutput processText(String text, boolean addSpecialTokens) { return new TextOutput(new int[0], null, 0); }
        @Override public TextOutput processTextBatch(List<String> texts) { return new TextOutput(new int[0], null, 0); }
        @Override public ImageOutput processImage(Object image) { return null; }
        @Override public List<ImageOutput> processImageBatch(List<?> images) { return List.of(); }
        @Override public AudioOutput processAudio(Object audio) { return null; }
        @Override public VideoOutput processVideo(Object video) { return null; }
        @Override public ProcessorOutput process(ProcessingInput input) { return ProcessorOutput.builder().build(); }
        @Override public ProcessorStats getStats() { return new ProcessorStats(0, 0, 0, 0, 0, 0); }
        @Override public void resetStats() {}
        @Override public boolean isClosed() { return closed; }
        @Override public void close() { closed = true; }
    }

    /**
     * Placeholder Fuyu processor.
     */
    public static class FuyuProcessor implements Processor {
        private volatile boolean closed;
        private final FastTokenizer tokenizer;

        public FuyuProcessor(String modelPath) {
            this.tokenizer = null;
        }

        @Override public String version() { return "1.0"; }
        @Override public FastTokenizer tokenizer() { return tokenizer; }
        @Override public List<Modality> supportedModalities() { return List.of(Modality.TEXT, Modality.IMAGE); }
        @Override public TextOutput processText(String text, boolean addSpecialTokens) { return new TextOutput(new int[0], null, 0); }
        @Override public TextOutput processTextBatch(List<String> texts) { return new TextOutput(new int[0], null, 0); }
        @Override public ImageOutput processImage(Object image) { return null; }
        @Override public List<ImageOutput> processImageBatch(List<?> images) { return List.of(); }
        @Override public AudioOutput processAudio(Object audio) { return null; }
        @Override public VideoOutput processVideo(Object video) { return null; }
        @Override public ProcessorOutput process(ProcessingInput input) { return ProcessorOutput.builder().build(); }
        @Override public ProcessorStats getStats() { return new ProcessorStats(0, 0, 0, 0, 0, 0); }
        @Override public void resetStats() {}
        @Override public boolean isClosed() { return closed; }
        @Override public void close() { closed = true; }
    }

    /**
     * Placeholder PaliGemma processor.
     */
    public static class PaliGemmaProcessor implements Processor {
        private volatile boolean closed;
        private final FastTokenizer tokenizer;

        public PaliGemmaProcessor(String modelPath) {
            this.tokenizer = null;
        }

        @Override public String version() { return "1.0"; }
        @Override public FastTokenizer tokenizer() { return tokenizer; }
        @Override public List<Modality> supportedModalities() { return List.of(Modality.TEXT, Modality.IMAGE); }
        @Override public TextOutput processText(String text, boolean addSpecialTokens) { return new TextOutput(new int[0], null, 0); }
        @Override public TextOutput processTextBatch(List<String> texts) { return new TextOutput(new int[0], null, 0); }
        @Override public ImageOutput processImage(Object image) { return null; }
        @Override public List<ImageOutput> processImageBatch(List<?> images) { return List.of(); }
        @Override public AudioOutput processAudio(Object audio) { return null; }
        @Override public VideoOutput processVideo(Object video) { return null; }
        @Override public ProcessorOutput process(ProcessingInput input) { return ProcessorOutput.builder().build(); }
        @Override public ProcessorStats getStats() { return new ProcessorStats(0, 0, 0, 0, 0, 0); }
        @Override public void resetStats() {}
        @Override public boolean isClosed() { return closed; }
        @Override public void close() { closed = true; }
    }

    /**
     * Text-only processor for language models.
     */
    public static class TextOnlyProcessor implements Processor {
        private volatile boolean closed;
        private final FastTokenizer tokenizer;
        private long textProcessed = 0;
        private long processingTimeMs = 0;

        public TextOnlyProcessor(FastTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        @Override
        public String version() { return "1.0"; }

        @Override
        public FastTokenizer tokenizer() { return tokenizer; }

        @Override
        public List<Modality> supportedModalities() { return List.of(Modality.TEXT); }

        @Override
        public TextOutput processText(String text, boolean addSpecialTokens) {
            long start = System.currentTimeMillis();
            int[] ids = tokenizer.encode(text, addSpecialTokens).ids();
            processingTimeMs += System.currentTimeMillis() - start;
            textProcessed++;
            return new TextOutput(ids, null, 0);
        }

        @Override
        public TextOutput processTextBatch(List<String> texts) {
            long start = System.currentTimeMillis();
            int[] ids = tokenizer.encodeBatch(texts).ids();
            processingTimeMs += System.currentTimeMillis() - start;
            textProcessed += texts.size();
            return new TextOutput(ids, null, 0);
        }

        @Override
        public ImageOutput processImage(Object image) {
            throw new UnsupportedOperationException("TextOnlyProcessor does not support images");
        }

        @Override
        public List<ImageOutput> processImageBatch(List<?> images) {
            throw new UnsupportedOperationException("TextOnlyProcessor does not support images");
        }

        @Override
        public AudioOutput processAudio(Object audio) {
            throw new UnsupportedOperationException("TextOnlyProcessor does not support audio");
        }

        @Override
        public VideoOutput processVideo(Object video) {
            throw new UnsupportedOperationException("TextOnlyProcessor does not support video");
        }

        @Override
        public ProcessorOutput process(ProcessingInput input) {
            return ProcessorOutput.builder()
                    .inputIds(torch.tensor(processText(input.text(), true).inputIds()))
                    .build();
        }

        @Override
        public ProcessorStats getStats() {
            return new ProcessorStats(textProcessed, 0, 0, 0, processingTimeMs, 0);
        }

        @Override
        public void resetStats() {
            textProcessed = 0;
            processingTimeMs = 0;
        }

        @Override
        public boolean isClosed() { return closed; }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
            System.out.printf("[TextOnlyProcessor] Closed: textProcessed=%d, time=%.2fs%n",
                    textProcessed, processingTimeMs / 1000.0);
        }

        private static org.bytedeco.pytorch.Tensor tensor(int[] ids) {
            return org.bytedeco.pytorch.global.torch.tensor(ids);
        }
    }
}
