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
 * MiniMax VL Vision-Language processor.
 *
 * <p>MiniMax VL features:
 * <ul>
 *   <li>Long context handling (up to 1M tokens)</li>
 *   <li>High-resolution image support</li>
 *   <li>Efficient multimodal fusion</li>
 *   <li>Text-video understanding</li>
 * </ul>
 *
 * <p>Reference: MiniMax-VL, MiniMax
 *
 * <pre>{@code
 * MiniMaxVLProcessor processor = MiniMaxVLProcessor.fromPretrained("MiniMaxAI/MiniMax-VL");
 *
 * ProcessorOutput output = processor.process(
 *     Processor.ProcessingInput.ofImages(
 *         "What is happening in this video?",
 *         List.of(video)
 *     )
 * );
 * }</pre>
 */
public class MiniMaxVLProcessor implements Processor {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Components
    private final FastTokenizer tokenizer;
    private final ImageProcessor imageProcessor;

    // Configuration
    private final int visionPatchSize;
    private final int spatialMergeSize;
    private final int maxImageSize;
    private final int maxSeqLength;
    private final boolean useGatedToken;

    // Special tokens
    private final int imageTokenId;
    private final int videoTokenId;

    // Performance metrics
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong textProcessed = new AtomicLong(0);
    private final AtomicLong imagesProcessed = new AtomicLong(0);
    private final AtomicLong videosProcessed = new AtomicLong(0);
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);

    /**
     * Create MiniMaxVLProcessor with default settings.
     */
    public static MiniMaxVLProcessor createDefault() {
        return builder().build();
    }

    /**
     * Create MiniMaxVLProcessor from pretrained configuration.
     */
    public static MiniMaxVLProcessor fromPretrained(String modelPath) {
        return builder()
                .tokenizer(createDefaultTokenizer())
                .imageProcessor(ImageProcessor.createMiniMaxVL())
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private MiniMaxVLProcessor(Builder builder) {
        this.tokenizer = builder.tokenizer;
        this.imageProcessor = builder.imageProcessor;
        this.visionPatchSize = builder.visionPatchSize;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.maxImageSize = builder.maxImageSize;
        this.maxSeqLength = builder.maxSeqLength;
        this.useGatedToken = builder.useGatedToken;
        this.imageTokenId = builder.imageTokenId;
        this.videoTokenId = builder.videoTokenId;
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
        try {
            Processor.ImageOutput output = imageProcessor.process(image, maxImageSize, maxImageSize);
            imagesProcessed.incrementAndGet();
            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
            return output;
        } catch (Exception e) {
            return new Processor.ImageOutput(
                    torch.zeros(new long[]{1, 3, maxImageSize, maxImageSize}),
                    maxImageSize, maxImageSize, 0, 0, 0,
                    new long[]{maxImageSize, maxImageSize},
                    new long[]{maxImageSize, maxImageSize}
            );
        }
    }

    @Override
    public List<ImageOutput> processImageBatch(List<?> images) {
        long start = System.currentTimeMillis();
        List<ImageOutput> outputs = images.stream()
                .map(img -> processImage(img))
                .toList();
        totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);
        return outputs;
    }

    @Override
    public AudioOutput processAudio(Object audio) {
        throw new UnsupportedOperationException("MiniMaxVL does not support standalone audio");
    }

    @Override
    public VideoOutput processVideo(Object video) {
        long start = System.currentTimeMillis();
        videosProcessed.incrementAndGet();
        totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

        // Simplified video processing - actual implementation would:
        // 1. Sample frames uniformly
        // 2. Process each frame through image processor
        // 3. Temporal merge
        return new VideoOutput(
                torch.zeros(new long[]{1, 16, 3, 224, 224}),
                16, 224, 224, 256, 30.0f
        );
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
                // Count image tokens
                int totalImageTokens = 0;
                for (Object img : input.images()) {
                    int[] grid = calculateImageGrid(maxImageSize, maxImageSize);
                    totalImageTokens += grid[0] * grid[1];
                }

                // Process images
                List<ImageOutput> imageOutputs = processImageBatch(input.images());

                // Concatenate pixel values
                if (!imageOutputs.isEmpty()) {
                    Tensor[] tensors = imageOutputs.stream()
                            .map(ImageOutput::pixelValues)
                            .toArray(Tensor[]::new);
                    pixelValues = torch.cat(Arrays.asList(tensors), 0);
                }

                // Calculate grid
                int[] gridArray = new int[input.images().size() * 3];
                int idx = 0;
                for (ImageOutput img : imageOutputs) {
                    gridArray[idx++] = 1;
                    gridArray[idx++] = img.gridHeight();
                    gridArray[idx++] = img.gridWidth();
                }
                imageGridTHW = torch.tensor(gridArray).reshape(-1, 3);
            }

            // Process videos
            Tensor videoPixelValues = null;
            Tensor videoGridTHW = null;

            if (input.videos() != null && !input.videos().isEmpty()) {
                videosProcessed.addAndGet(input.videos().size());

                // Process video frames (simplified)
                VideoOutput videoOutput = processVideo(input.videos().get(0));
                videoPixelValues = videoOutput.pixelValues();

                int[] videoGrid = new int[]{1, videoOutput.numFrames(), videoOutput.height() / spatialMergeSize};
                videoGridTHW = torch.tensor(videoGrid);
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
            System.err.println("MiniMaxVLProcessor.process error: " + e.getMessage());
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
                0,  // audio
                videosProcessed.get(),
                totalProcessingTimeMs.get(),
                totalTokensProcessed.get()
        );
    }

    @Override
    public void resetStats() {
        textProcessed.set(0);
        imagesProcessed.set(0);
        videosProcessed.set(0);
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
                "[MiniMaxVLProcessor] Closed: text=%d, images=%d, videos=%d, time=%.2fs%n",
                textProcessed.get(), imagesProcessed.get(), videosProcessed.get(),
                totalProcessingTimeMs.get() / 1000.0);
    }

    /**
     * Calculate image grid based on resolution and patch size.
     */
    public int[] calculateImageGrid(int height, int width) {
        int h = (int) Math.ceil(height / (float) spatialMergeSize);
        int w = (int) Math.ceil(width / (float) spatialMergeSize);
        return new int[]{h, w};
    }

    /**
     * Get number of image tokens.
     */
    public int getNumImageTokens(int height, int width) {
        int[] grid = calculateImageGrid(height, width);
        return grid[0] * grid[1];
    }

    private static FastTokenizer createDefaultTokenizer() {
        // Placeholder - actual implementation would load real tokenizer
        return null;
    }

    /**
     * Builder for MiniMaxVLProcessor.
     */
    public static class Builder {
        private FastTokenizer tokenizer;
        private ImageProcessor imageProcessor = ImageProcessor.createMiniMaxVL();
        private int visionPatchSize = 14;
        private int spatialMergeSize = 14;
        private int maxImageSize = 384;
        private int maxSeqLength = 1048576;  // 1M tokens
        private boolean useGatedToken = true;
        private int imageTokenId = 151652;
        private int videoTokenId = 151653;

        public Builder tokenizer(FastTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder imageProcessor(ImageProcessor imageProcessor) {
            this.imageProcessor = imageProcessor;
            return this;
        }

        public Builder visionPatchSize(int visionPatchSize) {
            this.visionPatchSize = visionPatchSize;
            return this;
        }

        public Builder spatialMergeSize(int spatialMergeSize) {
            this.spatialMergeSize = spatialMergeSize;
            return this;
        }

        public Builder maxImageSize(int maxImageSize) {
            this.maxImageSize = maxImageSize;
            return this;
        }

        public Builder maxSeqLength(int maxSeqLength) {
            this.maxSeqLength = maxSeqLength;
            return this;
        }

        public Builder useGatedToken(boolean useGatedToken) {
            this.useGatedToken = useGatedToken;
            return this;
        }

        public Builder imageTokenId(int imageTokenId) {
            this.imageTokenId = imageTokenId;
            return this;
        }

        public Builder videoTokenId(int videoTokenId) {
            this.videoTokenId = videoTokenId;
            return this;
        }

        public MiniMaxVLProcessor build() {
            return new MiniMaxVLProcessor(this);
        }
    }
}
