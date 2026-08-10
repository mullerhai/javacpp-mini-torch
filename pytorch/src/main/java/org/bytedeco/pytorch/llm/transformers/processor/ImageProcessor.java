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

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise-grade image processor for vision-language models.
 *
 * <p>Features:
 * <ul>
 *   <li>Multiple image processing modes (resize, center crop, padding)</li>
 *   <li>Normalization (ImageNet, CLIP, custom)</li>
 *   <li>Dynamic resolution support (Qwen2-VL style)</li>
 *   <li>Batch processing</li>
 *   <li>Performance monitoring</li>
 * </ul>
 *
 * <p>Reference: HuggingFace image_transforms, Qwen-VL, LLaVA
 *
 * <pre>{@code
 * ImageProcessor processor = ImageProcessor.builder()
 *     .imageMean(new float[]{0.485f, 0.456f, 0.406f})
 *     .imageStd(new float[]{0.229f, 0.224f, 0.225f})
 *     .doNormalize(true)
 *     .build();
 *
 * ImageOutput output = processor.process(image, 224, 224);
 * }</pre>
 */
public class ImageProcessor implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final ResizeMode resizeMode;
    private final int targetSize;
    private final int imageMean[];
    private final int imageStd[];
    private final boolean doNormalize;
    private final boolean doRescale;
    private final DataFormat outputFormat;
    private final int maxImageSize;
    private final int minImageSize;

    // Dynamic resolution support (for Qwen2-VL style)
    private final int spatialMergeSize;
    private final int spatialMergeUnit;
    private final boolean useDynamicResolution;

    // Performance metrics
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong imagesProcessed = new AtomicLong(0);
    private final AtomicLong totalPixelsProcessed = new AtomicLong(0);
    private final AtomicReference<String> lastError = new AtomicReference<>(null);

    // Standard normalization values
    public static final float[] IMAGENET_STANDARD_MEAN = {0.5f, 0.5f, 0.5f};
    public static final float[] IMAGENET_STANDARD_STD = {0.5f, 0.5f, 0.5f};
    public static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    public static final float[] CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};

    /**
     * Resize mode for images.
     */
    public enum ResizeMode {
        /** Resize to exact dimensions (may distort aspect ratio). */
        SQUARE,
        /** Center crop to target size. */
        CENTER_CROP,
        /** Resize shortest side, then crop/pad to target. */
        BOTTOM_RIGHT_CROP,
        /** Pad to target size (keep aspect ratio). */
        PAD,
        /** Resize shortest side to match target, keep aspect ratio. */
        KEEP_ASPECT_RATIO
    }

    /**
     * Output data format.
     */
    public enum DataFormat {
        NCHW,  // (batch, channels, height, width)
        NHWC   // (batch, height, width, channels)
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create ImageProcessor with ImageNet standard normalization.
     */
    public static ImageProcessor createImageNet() {
        return builder()
                .resizeMode(ResizeMode.SQUARE)
                .targetSize(224)
                .imageMean(new int[]{123.675f, 116.28f, 103.53f})
                .imageStd(new int[]{58.395f, 57.12f, 57.375f})
                .doNormalize(true)
                .doRescale(true)
                .build();
    }

    /**
     * Create ImageProcessor with CLIP normalization.
     */
    public static ImageProcessor createCLIP() {
        return builder()
                .resizeMode(ResizeMode.SQUARE)
                .targetSize(336)
                .imageMean(new int[]{0.48145466f, 0.4578275f, 0.40821073f})
                .imageStd(new int[]{0.26862954f, 0.26130258f, 0.27577711f})
                .doNormalize(true)
                .doRescale(true)
                .build();
    }

    /**
     * Create ImageProcessor for Qwen2-VL.
     */
    public static ImageProcessor createQwen2VL() {
        return builder()
                .resizeMode(ResizeMode.KEEP_ASPECT_RATIO)
                .targetSize(1280)
                .imageMean(new int[]{0.48145466f, 0.4578275f, 0.40821073f})
                .imageStd(new int[]{0.26862954f, 0.26130258f, 0.27577711f})
                .doNormalize(true)
                .doRescale(true)
                .useDynamicResolution(true)
                .spatialMergeSize(14)
                .spatialMergeUnit(2)
                .build();
    }

    /**
     * Create ImageProcessor for MiniMax VL.
     */
    public static ImageProcessor createMiniMaxVL() {
        return builder()
                .resizeMode(ResizeMode.KEEP_ASPECT_RATIO)
                .targetSize(384)
                .imageMean(new int[]{0.485f, 0.456f, 0.406f})
                .imageStd(new int[]{0.229f, 0.224f, 0.225f})
                .doNormalize(true)
                .doRescale(true)
                .maxImageSize(384)
                .build();
    }

    private ImageProcessor(Builder builder) {
        this.resizeMode = builder.resizeMode;
        this.targetSize = builder.targetSize;
        this.imageMean = builder.imageMean != null ? builder.imageMean.clone() : null;
        this.imageStd = builder.imageStd != null ? builder.imageStd.clone() : null;
        this.doNormalize = builder.doNormalize;
        this.doRescale = builder.doRescale;
        this.outputFormat = builder.outputFormat;
        this.maxImageSize = builder.maxImageSize;
        this.minImageSize = builder.minImageSize;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.spatialMergeUnit = builder.spatialMergeUnit;
        this.useDynamicResolution = builder.useDynamicResolution;
    }

    /**
     * Process an image (placeholder - actual implementation requires image library).
     */
    public Processor.ImageOutput process(Object image) {
        return process(image, targetSize, targetSize);
    }

    /**
     * Process an image to specific dimensions.
     */
    public Processor.ImageOutput process(Object image, int height, int width) {
        long start = System.currentTimeMillis();
        try {
            // This is a simplified implementation
            // Actual implementation would use Java image libraries (BufferedImage, etc.)
            // or native bindings to image processing libraries

            if (image == null) {
                lastError.set("Image is null");
                return createEmptyOutput(height, width);
            }

            // Create a placeholder tensor - actual implementation would process the image
            Tensor pixelValues = createPixelValues(height, width);

            imagesProcessed.incrementAndGet();
            totalPixelsProcessed.addAndGet(height * width);
            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            // Calculate grid dimensions for VL models
            int gridHeight = height / spatialMergeSize;
            int gridWidth = width / spatialMergeSize;

            return new Processor.ImageOutput(
                    pixelValues, height, width,
                    gridHeight, gridWidth,
                    gridHeight * gridWidth,
                    new long[]{height, width},
                    new long[]{height, width}
            );

        } catch (Exception e) {
            lastError.set(e.getMessage());
            return createEmptyOutput(height, width);
        }
    }

    /**
     * Process batch of images.
     */
    public java.util.List<Processor.ImageOutput> processBatch(java.util.List<?> images) {
        return images.stream()
                .map(img -> process(img))
                .toList();
    }

    /**
     * Resize image while keeping aspect ratio.
     */
    public int[] resizeKeepAspect(int srcWidth, int srcHeight, int maxSize) {
        if (srcWidth <= maxSize && srcHeight <= maxSize) {
            return new int[]{srcWidth, srcHeight};
        }

        float scale = Math.min((float) maxSize / srcWidth, (float) maxSize / srcHeight);
        int newWidth = Math.round(srcWidth * scale);
        int newHeight = Math.round(srcHeight * scale);

        // Ensure divisible by 2 (required by many vision models)
        newWidth = (newWidth / 2) * 2;
        newHeight = (newHeight / 2) * 2;

        return new int[]{newWidth, newHeight};
    }

    /**
     * Calculate image grid for dynamic resolution (Qwen2-VL style).
     */
    public int[] calculateImageGrid(int height, int width) {
        int gridHeight = height / spatialMergeSize;
        int gridWidth = width / spatialMergeSize;
        return new int[]{1, gridHeight, gridWidth};
    }

    /**
     * Normalize tensor with mean/std.
     */
    public Tensor normalize(Tensor input) {
        if (!doNormalize || imageMean == null || imageStd == null) {
            return input;
        }

        // Simple normalization: (x - mean) / std
        Tensor mean = torch.tensor(imageMean).reshape(1, 3, 1, 1).to(input.scalar_type());
        Tensor std = torch.tensor(imageStd).reshape(1, 3, 1, 1).to(input.scalar_type());

        Tensor result = input.sub(mean).div(std);

        mean.close();
        std.close();

        return result;
    }

    /**
     * Rescale tensor from [0, 255] to [0, 1].
     */
    public Tensor rescale(Tensor input) {
        if (!doRescale) {
            return input;
        }
        return input.div(255.0f);
    }

    /**
     * Get processing statistics.
     */
    public ImageProcessorStats getStats() {
        return new ImageProcessorStats(
                imagesProcessed.get(),
                totalPixelsProcessed.get(),
                totalProcessingTimeMs.get(),
                lastError.get()
        );
    }

    /**
     * Reset statistics.
     */
    public void resetStats() {
        totalProcessingTimeMs.set(0);
        imagesProcessed.set(0);
        totalPixelsProcessed.set(0);
        lastError.set(null);
    }

    /**
     * Check if closed.
     */
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[ImageProcessor] Closed: images=%d, pixels=%d, time=%.2fs%n",
                imagesProcessed.get(), totalPixelsProcessed.get(),
                totalProcessingTimeMs.get() / 1000.0);
    }

    // Getters
    public ResizeMode resizeMode() { return resizeMode; }
    public int targetSize() { return targetSize; }
    public boolean doNormalize() { return doNormalize; }
    public boolean doRescale() { return doRescale; }
    public DataFormat outputFormat() { return outputFormat; }
    public boolean useDynamicResolution() { return useDynamicResolution; }

    private Tensor createEmptyOutput(int height, int width) {
        Tensor empty = torch.zeros(new long[]{1, 3, height, width});
        return new Processor.ImageOutput(
                empty, height, width, 0, 0, 0,
                new long[]{height, width}, new long[]{height, width}
        );
    }

    private Tensor createPixelValues(int height, int width) {
        // Placeholder implementation
        // Actual implementation would process real image data
        return torch.zeros(new long[]{1, 3, height, width});
    }

    /**
     * Image processor statistics.
     */
    public static class ImageProcessorStats {
        public final long imagesProcessed;
        public final long totalPixelsProcessed;
        public final long totalProcessingTimeMs;
        public final String lastError;

        public ImageProcessorStats(long imagesProcessed, long totalPixelsProcessed,
                                long totalProcessingTimeMs, String lastError) {
            this.imagesProcessed = imagesProcessed;
            this.totalPixelsProcessed = totalPixelsProcessed;
            this.totalProcessingTimeMs = totalProcessingTimeMs;
            this.lastError = lastError;
        }

        public double avgTimeMs() {
            return imagesProcessed > 0 ? (double) totalProcessingTimeMs / imagesProcessed : 0;
        }

        public double avgPixelsPerImage() {
            return imagesProcessed > 0 ? (double) totalPixelsProcessed / imagesProcessed : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "ImageProcessorStats{images=%d, pixels=%d, avgTime=%.2fms, avgPixels=%.0f}",
                    imagesProcessed, totalPixelsProcessed, avgTimeMs(), avgPixelsPerImage());
        }
    }

    /**
     * Builder for ImageProcessor.
     */
    public static class Builder {
        private ResizeMode resizeMode = ResizeMode.SQUARE;
        private int targetSize = 224;
        private int[] imageMean;
        private int[] imageStd;
        private boolean doNormalize = true;
        private boolean doRescale = true;
        private DataFormat outputFormat = DataFormat.NCHW;
        private int maxImageSize = 4096;
        private int minImageSize = 28;
        private int spatialMergeSize = 14;
        private int spatialMergeUnit = 2;
        private boolean useDynamicResolution = false;

        public Builder resizeMode(ResizeMode resizeMode) {
            this.resizeMode = resizeMode;
            return this;
        }

        public Builder targetSize(int targetSize) {
            this.targetSize = targetSize;
            return this;
        }

        public Builder imageMean(int[] imageMean) {
            this.imageMean = imageMean;
            return this;
        }

        public Builder imageMean(float[] imageMean) {
            this.imageMean = new int[imageMean.length];
            for (int i = 0; i < imageMean.length; i++) {
                this.imageMean[i] = (int) imageMean[i];
            }
            return this;
        }

        public Builder imageStd(int[] imageStd) {
            this.imageStd = imageStd;
            return this;
        }

        public Builder imageStd(float[] imageStd) {
            this.imageStd = new int[imageStd.length];
            for (int i = 0; i < imageStd.length; i++) {
                this.imageStd[i] = (int) imageStd[i];
            }
            return this;
        }

        public Builder doNormalize(boolean doNormalize) {
            this.doNormalize = doNormalize;
            return this;
        }

        public Builder doRescale(boolean doRescale) {
            this.doRescale = doRescale;
            return this;
        }

        public Builder outputFormat(DataFormat outputFormat) {
            this.outputFormat = outputFormat;
            return this;
        }

        public Builder maxImageSize(int maxImageSize) {
            this.maxImageSize = maxImageSize;
            return this;
        }

        public Builder minImageSize(int minImageSize) {
            this.minImageSize = minImageSize;
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

        public Builder useDynamicResolution(boolean useDynamicResolution) {
            this.useDynamicResolution = useDynamicResolution;
            return this;
        }

        /**
         * Configure for ImageNet normalization.
         */
        public Builder imagenet() {
            this.imageMean = new int[]{123.675f, 116.28f, 103.53f};
            this.imageStd = new int[]{58.395f, 57.12f, 57.375f};
            return this;
        }

        /**
         * Configure for CLIP normalization.
         */
        public Builder clip() {
            this.imageMean = new int[]{0.48145466f, 0.4578275f, 0.40821073f};
            this.imageStd = new int[]{0.26862954f, 0.26130258f, 0.27577711f};
            return this;
        }

        public ImageProcessor build() {
            return new ImageProcessor(this);
        }
    }
}
