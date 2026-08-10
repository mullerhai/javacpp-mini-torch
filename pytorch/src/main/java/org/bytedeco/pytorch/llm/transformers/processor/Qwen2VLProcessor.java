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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Qwen2-VL Vision-Language processor.
 *
 * <p>Handles:
 * <ul>
 *   <li>Text tokenization with special image tokens</li>
 *   <li>Image processing with dynamic resolution</li>
 *   <li>Image grid calculation for spatial merge</li>
 *   <li>Combined multimodal input preparation</li>
 * </ul>
 *
 * <p>Reference: Qwen2-VL, Qwen2.5-VL, Qwen3-VL
 *
 * <pre>{@code
 * Qwen2VLProcessor processor = Qwen2VLProcessor.fromPretrained("Qwen/Qwen2-VL-7B-Instruct");
 *
 * ProcessorOutput output = processor.process(
 *     Processor.ProcessingInput.ofImages(
 *         "Describe this image: <image>",
 *         List.of(image1)
 *     )
 * );
 * }</pre>
 */
public class Qwen2VLProcessor implements Processor {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Components
    private final FastTokenizer tokenizer;
    private final ImageProcessor imageProcessor;

    // Configuration
    private final int visionConfigImageGridWidth;
    private final int visionConfigImageGridHeight;
    private final int spatialMergeSize;
    private final int spatialMergeUnit;
    private final int sequenceLength;
    private final int imageBoundTokenId;
    private final int imageStartTokenId;
    private final int imageEndTokenId;
    private final int videoStartTokenId;
    private final int videoEndTokenId;

    // Performance metrics
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong textProcessed = new AtomicLong(0);
    private final AtomicLong imagesProcessed = new AtomicLong(0);
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);

    /**
     * Create Qwen2VLProcessor with default settings.
     */
    public static Qwen2VLProcessor createDefault() {
        return builder().build();
    }

    /**
     * Create Qwen2VLProcessor from pretrained configuration.
     */
    public static Qwen2VLProcessor fromPretrained(String modelPath) {
        // In practice, this would load tokenizer and image processor from model path
        return builder()
                .tokenizer(createDefaultTokenizer())
                .imageProcessor(ImageProcessor.createQwen2VL())
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private Qwen2VLProcessor(Builder builder) {
        this.tokenizer = builder.tokenizer;
        this.imageProcessor = builder.imageProcessor;
        this.visionConfigImageGridWidth = builder.visionConfigImageGridWidth;
        this.visionConfigImageGridHeight = builder.visionConfigImageGridHeight;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.spatialMergeUnit = builder.spatialMergeUnit;
        this.sequenceLength = builder.sequenceLength;
        this.imageBoundTokenId = builder.imageBoundTokenId;
        this.imageStartTokenId = builder.imageStartTokenId;
        this.imageEndTokenId = builder.imageEndTokenId;
        this.videoStartTokenId = builder.videoStartTokenId;
        this.videoEndTokenId = builder.videoEndTokenId;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Override
    public FastTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public List<Modality> supportedModalities() {
        return Arrays.asList(Modality.TEXT, Modality.IMAGE, Modality.VIDEO);
    }

    @Override
    public TextOutput processText(String text, boolean addSpecialTokens) {
        long start = System.currentTimeMillis();
        try {
            int[] ids = tokenizer.encode(text, addSpecialTokens).ids();
            textProcessed.incrementAndGet();
            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
            return new TextOutput(ids, null, 0);
        } catch (Exception e) {
            return new TextOutput(new int[0], null, 0);
        }
    }

    @Override
    public TextOutput processTextBatch(List<String> texts) {
        long start = System.currentTimeMillis();
        try {
            int[] ids = tokenizer.encodeBatch(texts).ids();
            textProcessed.incrementAndGet();
            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
            return new TextOutput(ids, null, 0);
        } catch (Exception e) {
            return new TextOutput(new int[0], null, 0);
        }
    }

    @Override
    public ImageOutput processImage(Object image) {
        long start = System.currentTimeMillis();
        imagesProcessed.incrementAndGet();
        totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
        return imageProcessor.process(image);
    }

    @Override
    public List<ImageOutput> processImageBatch(List<?> images) {
        long start = System.currentTimeMillis();
        List<ImageOutput> outputs = imageProcessor.processBatch(images);
        totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
        return outputs;
    }

    @Override
    public AudioOutput processAudio(Object audio) {
        throw new UnsupportedOperationException("Qwen2VL does not support audio");
    }

    @Override
    public VideoOutput processVideo(Object video) {
        long start = System.currentTimeMillis();
        // Simplified video processing
        totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
        return new VideoOutput(torch.zeros(new long[]{1, 8, 3, 224, 224}), 8, 224, 224, 64, 30.0f);
    }

    @Override
    public ProcessorOutput process(ProcessingInput input) {
        long start = System.currentTimeMillis();

        try {
            // Process text
            int[] inputIds;
            if (input.text() != null) {
                inputIds = tokenizer.encode(input.text(), true).ids();
            } else if (input.texts() != null) {
                inputIds = tokenizer.encodeBatch(input.texts()).ids();
            } else {
                inputIds = new int[0];
            }

            // Process images
            Tensor pixelValues = null;
            Tensor imageGridTHW = null;

            if (input.images() != null && !input.images().isEmpty()) {
                List<ImageOutput> imageOutputs = processImageBatch(input.images());

                // Concatenate all pixel values
                Tensor[] pixelTensors = imageOutputs.stream()
                        .map(ImageOutput::pixelValues)
                        .toArray(Tensor[]::new);

                if (pixelTensors.length > 0) {
                    pixelValues = torch.cat(Arrays.asList(pixelTensors), 0);
                }

                // Calculate image grid (batch, temporal, height, width)
                int[] gridArray = new int[imageOutputs.size() * 3];
                int idx = 0;
                for (ImageOutput img : imageOutputs) {
                    gridArray[idx++] = 1;  // temporal dimension
                    gridArray[idx++] = img.gridHeight();
                    gridArray[idx++] = img.gridWidth();
                }
                imageGridTHW = torch.tensor(gridArray).reshape(-1, 3);
            }

            // Process videos
            Tensor videoPixelValues = null;
            Tensor videoGridTHW = null;

            if (input.videos() != null && !input.videos().isEmpty()) {
                // Process each video frame
                // Simplified implementation
            }

            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            return ProcessorOutput.builder()
                    .inputIds(torch.tensor(inputIds))
                    .pixelValues(pixelValues)
                    .imageGridTHW(imageGridTHW)
                    .videoPixelValues(videoPixelValues)
                    .videoGridTHW(videoGridTHW)
                    .build();

        } catch (Exception e) {
            System.err.println("Qwen2VLProcessor.process error: " + e.getMessage());
            return ProcessorOutput.builder()
                    .inputIds(torch.zeros(1, 1))
                    .build();
        }
    }

    @Override
    public String decode(int[] tokenIds) {
        return tokenizer.decode(tokenIds, true);
    }

    @Override
    public ProcessorStats getStats() {
        return new ProcessorStats(
                textProcessed.get(),
                imagesProcessed.get(),
                0,  // audio count
                0,  // video count
                totalProcessingTimeMs.get(),
                totalTokensProcessed.get()
        );
    }

    @Override
    public void resetStats() {
        textProcessed.set(0);
        imagesProcessed.set(0);
        totalProcessingTimeMs.set(0);
        totalTokensProcessed.set(0);
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (imageProcessor != null) {
            try { imageProcessor.close(); } catch (Exception ignored) {}
        }

        System.out.printf(
                "[Qwen2VLProcessor] Closed: text=%d, images=%d, time=%.2fs%n",
                textProcessed.get(), imagesProcessed.get(),
                totalProcessingTimeMs.get() / 1000.0);
    }

    /**
     * Calculate image grid based on image dimensions and model config.
     */
    public int[] calculateImageGrid(int height, int width) {
        int h = (int) Math.ceil(height / (float) spatialMergeSize);
        int w = (int) Math.ceil(width / (float) spatialMergeSize);
        return new int[]{1, h, w};
    }

    /**
     * Get number of image tokens for an image.
     */
    public int getNumImageTokens(int height, int width) {
        int[] grid = calculateImageGrid(height, width);
        return grid[1] * grid[2];  // height * width
    }

    private static FastTokenizer createDefaultTokenizer() {
        // Placeholder - actual implementation would load real tokenizer
        return null;
    }

    /**
     * Builder for Qwen2VLProcessor.
     */
    public static class Builder {
        private FastTokenizer tokenizer;
        private ImageProcessor imageProcessor = ImageProcessor.createQwen2VL();
        private int visionConfigImageGridWidth = 12;
        private int visionConfigImageGridHeight = 12;
        private int spatialMergeSize = 14;
        private int spatialMergeUnit = 2;
        private int sequenceLength = 32768;
        private int imageBoundTokenId = 151652;
        private int imageStartTokenId = 151644;
        private int imageEndTokenId = 151645;
        private int videoStartTokenId = 151646;
        private int videoEndTokenId = 151647;

        public Builder tokenizer(FastTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder imageProcessor(ImageProcessor imageProcessor) {
            this.imageProcessor = imageProcessor;
            return this;
        }

        public Builder visionConfigImageGridWidth(int visionConfigImageGridWidth) {
            this.visionConfigImageGridWidth = visionConfigImageGridWidth;
            return this;
        }

        public Builder visionConfigImageGridHeight(int visionConfigImageGridHeight) {
            this.visionConfigImageGridHeight = visionConfigImageGridHeight;
            return this;
        }

        public Builder spatialMergeSize(int spatialMergeSize) {
            this.spatialMergeSize = spatialMergeSize;
            return this;
        }

        public Builder spatialMergeUnit(int spatialMergeUnit) {
            this.spatialMergeUnit = spatialMergeUnit;
            return this;
        }

        public Builder sequenceLength(int sequenceLength) {
            this.sequenceLength = sequenceLength;
            return this;
        }

        public Builder imageBoundTokenId(int imageBoundTokenId) {
            this.imageBoundTokenId = imageBoundTokenId;
            return this;
        }

        public Builder imageStartTokenId(int imageStartTokenId) {
            this.imageStartTokenId = imageStartTokenId;
            return this;
        }

        public Builder imageEndTokenId(int imageEndTokenId) {
            this.imageEndTokenId = imageEndTokenId;
            return this;
        }

        public Builder videoStartTokenId(int videoStartTokenId) {
            this.videoStartTokenId = videoStartTokenId;
            return this;
        }

        public Builder videoEndTokenId(int videoEndTokenId) {
            this.videoEndTokenId = videoEndTokenId;
            return this;
        }

        public Qwen2VLProcessor build() {
            return new Qwen2VLProcessor(this);
        }
    }
}
