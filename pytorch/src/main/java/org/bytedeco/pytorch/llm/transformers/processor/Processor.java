/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file file except in compliance with the License.
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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.io.Closeable;
import java.util.List;
import java.util.Map;

/**
 * Base interface for all multimodal processors.
 *
 * <p>Processors handle:
 * <ul>
 *   <li>Text encoding/decoding</li>
 *   <li>Image processing and encoding</li>
 *   <li>Audio processing and encoding</li>
 *   <li>Video processing and encoding</li>
 *   <li>Modality fusion</li>
 * </ul>
 *
 * <p>Reference: HuggingFace transformers Processor, LLaVA, Qwen-VL, MiniMax-VL
 *
 * <pre>{@code
 * // Example: Process text + image for Qwen2-VL
 * try (Processor processor = Qwen2VLProcessor.fromPretrained("Qwen/Qwen2-VL")) {
 *     ProcessorOutput output = processor.process(
 *         text: "Describe this image",
 *         images: List.of(image1, image2)
 *     );
 *
 *     // Use output for model forward
 *     Tensor inputIds = output.inputIds();
 *     Tensor pixelValues = output.pixelValues();
 *     Tensor imageGridTHW = output.imageGridTHW();
 * }
 * }</pre>
 */
public interface Processor extends AutoCloseable {

    /**
     * Get processor version.
     */
    String version();

    /**
     * Get the associated tokenizer.
     */
    FastTokenizer tokenizer();

    /**
     * Get supported modalities.
     */
    List<Modality> supportedModalities();

    /**
     * Check if this processor supports a modality.
     */
    default boolean supports(Modality modality) {
        return supportedModalities().contains(modality);
    }

    /**
     * Process text input.
     */
    default TextOutput processText(String text) {
        return processText(text, true);
    }

    /**
     * Process text input with optional special tokens.
     */
    TextOutput processText(String text, boolean addSpecialTokens);

    /**
     * Process multiple texts (batch).
     */
    TextOutput processTextBatch(List<String> texts);

    /**
     * Process single image.
     */
    ImageOutput processImage(Object image);

    /**
     * Process multiple images (batch).
     */
    List<ImageOutput> processImageBatch(List<?> images);

    /**
     * Process audio.
     */
    AudioOutput processAudio(Object audio);

    /**
     * Process video.
     */
    VideoOutput processVideo(Object video);

    /**
     * Combined multimodal processing.
     */
    ProcessorOutput process(ProcessingInput input);

    /**
     * Decode output tensor to text.
     */
    default String decode(int[] tokenIds) {
        return tokenizer().decode(tokenIds, true);
    }

    /**
     * Decode batch output tensors to texts.
     */
    default List<String> decodeBatch(List<int[]> tokenIdsBatch) {
        return tokenIdsBatch.stream()
                .map(ids -> decode(ids))
                .toList();
    }

    /**
     * Get processor statistics.
     */
    ProcessorStats getStats();

    /**
     * Reset statistics.
     */
    void resetStats();

    /**
     * Check if processor is closed.
     */
    boolean isClosed();

    @Override
    void close();

    // ============= Nested types =============

    /**
     * Supported modalities.
     */
    enum Modality {
        TEXT,
        IMAGE,
        VIDEO,
        AUDIO,
        EMBEDDING
    }

    /**
     * Processing input for multimodal data.
     */
    final class ProcessingInput {
        private String text;
        private List<String> texts;
        private List<Object> images;
        private List<Object> audios;
        private List<Object> videos;
        private Map<String, Object> kwargs;

        public static ProcessingInput of(String text) {
            ProcessingInput input = new ProcessingInput();
            input.text = text;
            return input;
        }

        public static ProcessingInput ofImages(String text, List<Object> images) {
            ProcessingInput input = new ProcessingInput();
            input.text = text;
            input.images = images;
            return input;
        }

        public static ProcessingInput ofVideo(String text, List<Object> videos) {
            ProcessingInput input = new ProcessingInput();
            input.text = text;
            input.videos = videos;
            return input;
        }

        public static ProcessingInput ofAudio(String text, List<Object> audios) {
            ProcessingInput input = new ProcessingInput();
            input.text = text;
            input.audios = audios;
            return input;
        }

        public ProcessingInput texts(List<String> texts) {
            this.texts = texts;
            return this;
        }

        public ProcessingInput images(List<Object> images) {
            this.images = images;
            return this;
        }

        public ProcessingInput audios(List<Object> audios) {
            this.audios = audios;
            return this;
        }

        public ProcessingInput videos(List<Object> videos) {
            this.videos = videos;
            return this;
        }

        public ProcessingInput kwargs(Map<String, Object> kwargs) {
            this.kwargs = kwargs;
            return this;
        }

        // Getters
        public String text() { return text; }
        public List<String> texts() { return texts; }
        public List<Object> images() { return images; }
        public List<Object> audios() { return audios; }
        public List<Object> videos() { return videos; }
        public Map<String, Object> kwargs() { return kwargs; }
    }

    /**
     * Combined processor output.
     */
    final class ProcessorOutput {
        private final Tensor inputIds;
        private final Tensor attentionMask;
        private final Tensor pixelValues;         // Image tensors
        private final Tensor imageGridTHW;       // Image grid for VL
        private final Tensor imageBoundings;     // Bounding boxes for VL
        private final Tensor videoPixelValues;   // Video tensors
        private final Tensor videoGridTHW;       // Video grid
        private final Tensor audioFeatures;      // Audio features
        private final Tensor audioGridTHW;       // Audio grid
        private final Map<String, Object> metadata;

        private ProcessorOutput(Builder builder) {
            this.inputIds = builder.inputIds;
            this.attentionMask = builder.attentionMask;
            this.pixelValues = builder.pixelValues;
            this.imageGridTHW = builder.imageGridTHW;
            this.imageBoundings = builder.imageBoundings;
            this.videoPixelValues = builder.videoPixelValues;
            this.videoGridTHW = builder.videoGridTHW;
            this.audioFeatures = builder.audioFeatures;
            this.audioGridTHW = builder.audioGridTHW;
            this.metadata = builder.metadata;
        }

        public static Builder builder() { return new Builder(); }

        public Tensor inputIds() { return inputIds; }
        public Tensor attentionMask() { return attentionMask; }
        public Tensor pixelValues() { return pixelValues; }
        public Tensor imageGridTHW() { return imageGridTHW; }
        public Tensor imageBoundings() { return imageBoundings; }
        public Tensor videoPixelValues() { return videoPixelValues; }
        public Tensor videoGridTHW() { return videoGridTHW; }
        public Tensor audioFeatures() { return audioFeatures; }
        public Tensor audioGridTHW() { return audioGridTHW; }
        public Map<String, Object> metadata() { return metadata; }

        public boolean hasImages() { return pixelValues != null; }
        public boolean hasVideo() { return videoPixelValues != null; }
        public boolean hasAudio() { return audioFeatures != null; }

        public static class Builder {
            private Tensor inputIds;
            private Tensor attentionMask;
            private Tensor pixelValues;
            private Tensor imageGridTHW;
            private Tensor imageBoundings;
            private Tensor videoPixelValues;
            private Tensor videoGridTHW;
            private Tensor audioFeatures;
            private Tensor audioGridTHW;
            private Map<String, Object> metadata;

            public Builder inputIds(Tensor inputIds) { this.inputIds = inputIds; return this; }
            public Builder attentionMask(Tensor attentionMask) { this.attentionMask = attentionMask; return this; }
            public Builder pixelValues(Tensor pixelValues) { this.pixelValues = pixelValues; return this; }
            public Builder imageGridTHW(Tensor imageGridTHW) { this.imageGridTHW = imageGridTHW; return this; }
            public Builder imageBoundings(Tensor imageBoundings) { this.imageBoundings = imageBoundings; return this; }
            public Builder videoPixelValues(Tensor videoPixelValues) { this.videoPixelValues = videoPixelValues; return this; }
            public Builder videoGridTHW(Tensor videoGridTHW) { this.videoGridTHW = videoGridTHW; return this; }
            public Builder audioFeatures(Tensor audioFeatures) { this.audioFeatures = audioFeatures; return this; }
            public Builder audioGridTHW(Tensor audioGridTHW) { this.audioGridTHW = audioGridTHW; return this; }
            public Builder metadata(Map<String, Object> metadata) { this.metadata = metadata; return this; }

            public ProcessorOutput build() { return new ProcessorOutput(this); }
        }
    }

    /**
     * Text processing output.
     */
    final class TextOutput {
        private final int[] inputIds;
        private final int[] attentionMask;
        private final int numImageTokens;

        public TextOutput(int[] inputIds, int[] attentionMask, int numImageTokens) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
            this.numImageTokens = numImageTokens;
        }

        public static TextOutput of(int[] inputIds) {
            return new TextOutput(inputIds, null, 0);
        }

        public int[] inputIds() { return inputIds; }
        public int[] attentionMask() { return attentionMask; }
        public int numImageTokens() { return numImageTokens; }
        public int length() { return inputIds.length; }
    }

    /**
     * Image processing output.
     */
    final class ImageOutput {
        private final Tensor pixelValues;
        private final int height;
        private final int width;
        private final int gridHeight;
        private final int gridWidth;
        private final int numPatches;
        private final long[] originalSize;
        private final long[] processedSize;

        public ImageOutput(Tensor pixelValues, int height, int width,
                         int gridHeight, int gridWidth, int numPatches,
                         long[] originalSize, long[] processedSize) {
            this.pixelValues = pixelValues;
            this.height = height;
            this.width = width;
            this.gridHeight = gridHeight;
            this.gridWidth = gridWidth;
            this.numPatches = numPatches;
            this.originalSize = originalSize;
            this.processedSize = processedSize;
        }

        public Tensor pixelValues() { return pixelValues; }
        public int height() { return height; }
        public int width() { return width; }
        public int gridHeight() { return gridHeight; }
        public int gridWidth() { return gridWidth; }
        public int numPatches() { return numPatches; }
        public long[] originalSize() { return originalSize; }
        public long[] processedSize() { return processedSize; }
    }

    /**
     * Audio processing output.
     */
    final class AudioOutput {
        private final Tensor features;
        private final int sampleRate;
        private final int numFrames;
        private final int numMelBins;
        private final float[] waveform;

        public AudioOutput(Tensor features, int sampleRate, int numFrames,
                         int numMelBins, float[] waveform) {
            this.features = features;
            this.sampleRate = sampleRate;
            this.numFrames = numFrames;
            this.numMelBins = numMelBins;
            this.waveform = waveform;
        }

        public Tensor features() { return features; }
        public int sampleRate() { return sampleRate; }
        public int numFrames() { return numFrames; }
        public int numMelBins() { return numMelBins; }
        public float[] waveform() { return waveform; }
    }

    /**
     * Video processing output.
     */
    final class VideoOutput {
        private final Tensor pixelValues;
        private final int numFrames;
        private final int height;
        private final int width;
        private final int numPatches;
        private final float frameRate;

        public VideoOutput(Tensor pixelValues, int numFrames, int height,
                          int width, int numPatches, float frameRate) {
            this.pixelValues = pixelValues;
            this.numFrames = numFrames;
            this.height = height;
            this.width = width;
            this.numPatches = numPatches;
            this.frameRate = frameRate;
        }

        public Tensor pixelValues() { return pixelValues; }
        public int numFrames() { return numFrames; }
        public int height() { return height; }
        public int width() { return width; }
        public int numPatches() { return numPatches; }
        public float frameRate() { return frameRate; }
    }

    /**
     * Processor statistics.
     */
    final class ProcessorStats {
        public final long textProcessedCount;
        public final long imageProcessedCount;
        public final long audioProcessedCount;
        public final long videoProcessedCount;
        public final long totalProcessingTimeMs;
        public final long totalTokensProcessed;

        public ProcessorStats(long textProcessedCount, long imageProcessedCount,
                           long audioProcessedCount, long videoProcessedCount,
                           long totalProcessingTimeMs, long totalTokensProcessed) {
            this.textProcessedCount = textProcessedCount;
            this.imageProcessedCount = imageProcessedCount;
            this.audioProcessedCount = audioProcessedCount;
            this.videoProcessedCount = videoProcessedCount;
            this.totalProcessingTimeMs = totalProcessingTimeMs;
            this.totalTokensProcessed = totalTokensProcessed;
        }

        public double avgTextTimeMs() {
            return textProcessedCount > 0 ? (double) totalProcessingTimeMs / textProcessedCount : 0;
        }

        public double avgImageTimeMs() {
            return imageProcessedCount > 0 ? (double) totalProcessingTimeMs / imageProcessedCount : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "ProcessorStats{text=%d, images=%d, audio=%d, video=%d, " +
                    "time=%.2fs, tokens=%d}",
                    textProcessedCount, imageProcessedCount, audioProcessedCount,
                    videoProcessedCount, totalProcessingTimeMs / 1000.0, totalTokensProcessed);
        }
    }
}
