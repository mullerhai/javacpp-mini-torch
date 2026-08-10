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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Enterprise-grade video processor for multimodal models.
 *
 * <p>Features:
 * <ul>
 *   <li>Frame sampling (uniform, random, keyframe)</li>
 *   <li>Video normalization</li>
 *   <li>Temporal patch extraction</li>
 *   <li>Batch processing</li>
 *   <li>Performance monitoring</li>
 * </ul>
 *
 * <p>Reference: VideoLlama, LLaMA-VID, MiniGPT-Video
 *
 * <pre>{@code
 * VideoProcessor processor = VideoProcessor.builder()
 *     .numFrames(8)
 *     .frameSampling(SamplingMode.UNIFORM)
 *     .targetSize(224)
 *     .build();
 *
 * VideoOutput output = processor.process(videoFrames);
 * }</pre>
 */
public class VideoProcessor implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final int numFrames;
    private final SamplingMode samplingMode;
    private final int targetHeight;
    private final int targetWidth;
    private final float[] mean;
    private final float[] std;
    private final boolean doNormalize;
    private final int temporalPatchSize;
    private final int spatialPatchSize;
    private final int maxVideoLength;

    // Image processor for frame processing
    private final ImageProcessor imageProcessor;

    // Performance metrics
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong videosProcessed = new AtomicLong(0);
    private final AtomicLong framesExtracted = new AtomicLong(0);
    private final AtomicLong totalFramesProcessed = new AtomicLong(0);

    /**
     * Frame sampling modes.
     */
    public enum SamplingMode {
        /** Sample frames uniformly across the video. */
        UNIFORM,
        /** Sample frames randomly. */
        RANDOM,
        /** Sample only keyframes. */
        KEYFRAME,
        /** Sample first N frames. */
        FIRST,
        /** Sample last N frames. */
        LAST,
        /** Center crop sampling. */
        CENTER
    }

    /**
     * Create VideoProcessor with defaults for video understanding.
     */
    public static VideoProcessor createDefault() {
        return builder()
                .numFrames(8)
                .samplingMode(SamplingMode.UNIFORM)
                .targetSize(224)
                .build();
    }

    /**
     * Create VideoProcessor optimized for long video understanding.
     */
    public static VideoProcessor createForLongVideo() {
        return builder()
                .numFrames(16)
                .samplingMode(SamplingMode.UNIFORM)
                .targetSize(224)
                .maxVideoLength(300)  // 5 minutes at 1fps
                .build();
    }

    /**
     * Create VideoProcessor for video generation models.
     */
    public static VideoProcessor createForVideoGeneration() {
        return builder()
                .numFrames(16)
                .samplingMode(SamplingMode.UNIFORM)
                .targetSize(256)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private VideoProcessor(Builder builder) {
        this.numFrames = builder.numFrames;
        this.samplingMode = builder.samplingMode;
        this.targetHeight = builder.targetHeight;
        this.targetWidth = builder.targetWidth;
        this.mean = builder.mean != null ? builder.mean.clone() : null;
        this.std = builder.std != null ? builder.std.clone() : null;
        this.doNormalize = builder.doNormalize;
        this.temporalPatchSize = builder.temporalPatchSize;
        this.spatialPatchSize = builder.spatialPatchSize;
        this.maxVideoLength = builder.maxVideoLength;

        // Initialize image processor for frame processing
        this.imageProcessor = ImageProcessor.builder()
                .resizeMode(ImageProcessor.ResizeMode.KEEP_ASPECT_RATIO)
                .targetSize(targetHeight)
                .doNormalize(doNormalize)
                .imageMean(mean != null ? mean : new float[]{0.485f, 0.456f, 0.406f})
                .imageStd(std != null ? std : new float[]{0.229f, 0.224f, 0.225f})
                .build();
    }

    /**
     * Process a video from list of frames.
     *
     * @param frames List of frames (each frame as a tensor [C, H, W])
     * @return VideoOutput containing processed video tensor
     */
    public Processor.VideoOutput process(List<Tensor> frames) {
        long start = System.currentTimeMillis();

        try {
            if (frames == null || frames.isEmpty()) {
                return createEmptyOutput();
            }

            // Sample frames
            List<Tensor> sampledFrames = sampleFrames(frames);

            // Process each frame
            List<Tensor> processedFrames = new ArrayList<>();
            for (Tensor frame : sampledFrames) {
                Processor.ImageOutput imgOut = imageProcessor.process(frame, targetHeight, targetWidth);
                processedFrames.add(imgOut.pixelValues());
            }

            // Stack frames: [T, C, H, W]
            Tensor videoTensor;
            if (!processedFrames.isEmpty()) {
                videoTensor = torch.stack(processedFrames, 0);
            } else {
                videoTensor = torch.zeros(new long[]{numFrames, 3, targetHeight, targetWidth});
            }

            // Calculate grid
            int height = targetHeight;
            int width = targetWidth;
            int numPatches = (height / spatialPatchSize) * (width / spatialPatchSize);

            videosProcessed.incrementAndGet();
            framesExtracted.addAndGet(sampledFrames.size());
            totalFramesProcessed.addAndGet(processedFrames.size());
            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            return new Processor.VideoOutput(
                    videoTensor,
                    sampledFrames.size(),
                    height,
                    width,
                    numPatches,
                    30.0f  // Default frame rate
            );

        } catch (Exception e) {
            System.err.println("VideoProcessor.process error: " + e.getMessage());
            return createEmptyOutput();
        }
    }

    /**
     * Process a video tensor directly.
     *
     * @param videoTensor [batch, frames, channels, height, width] or [frames, channels, height, width]
     * @return VideoOutput
     */
    public Processor.VideoOutput process(Tensor videoTensor) {
        long start = System.currentTimeMillis();

        try {
            if (videoTensor == null || !videoTensor.defined()) {
                return createEmptyOutput();
            }

            // Handle different tensor shapes
            long numDims = videoTensor.dim();
            List<Tensor> frames = new ArrayList<>();

            if (numDims == 4) {
                // [T, C, H, W] - single video
                long T = videoTensor.size(0);
                for (long t = 0; t < T; t++) {
                    frames.add(videoTensor.select(0, t));
                }
            } else if (numDims == 5) {
                // [B, T, C, H, W] - batch of videos
                long B = videoTensor.size(0);
                long T = videoTensor.size(1);
                for (long b = 0; b < B; b++) {
                    for (long t = 0; t < T; t++) {
                        frames.add(videoTensor.select(0, b).select(0, t));
                    }
                }
            }

            videosProcessed.incrementAndGet();
            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            return process(frames);

        } catch (Exception e) {
            System.err.println("VideoProcessor.process tensor error: " + e.getMessage());
            return createEmptyOutput();
        }
    }

    /**
     * Sample frames according to sampling mode.
     */
    private List<Tensor> sampleFrames(List<Tensor> frames) {
        int totalFrames = frames.size();

        if (totalFrames <= numFrames) {
            // Not enough frames, return all
            return new ArrayList<>(frames);
        }

        List<Tensor> sampled = new ArrayList<>();

        switch (samplingMode) {
            case UNIFORM:
                sampled = sampleUniformly(frames, numFrames);
                break;
            case FIRST:
                for (int i = 0; i < Math.min(numFrames, totalFrames); i++) {
                    sampled.add(frames.get(i));
                }
                break;
            case LAST:
                for (int i = Math.max(0, totalFrames - numFrames); i < totalFrames; i++) {
                    sampled.add(frames.get(i));
                }
                break;
            case CENTER:
                int start = (totalFrames - numFrames) / 2;
                for (int i = start; i < start + numFrames && i < totalFrames; i++) {
                    sampled.add(frames.get(i));
                }
                break;
            case RANDOM:
                sampled = sampleRandomly(frames, numFrames);
                break;
            case KEYFRAME:
                sampled = sampleKeyframes(frames, numFrames);
                break;
        }

        return sampled;
    }

    /**
     * Sample frames uniformly.
     */
    private List<Tensor> sampleUniformly(List<Tensor> frames, int numSamples) {
        List<Tensor> sampled = new ArrayList<>();
        int total = frames.size();

        for (int i = 0; i < numSamples; i++) {
            int idx = (int) ((i * total) / numSamples);
            idx = Math.min(idx, total - 1);
            sampled.add(frames.get(idx));
        }

        return sampled;
    }

    /**
     * Sample frames randomly.
     */
    private List<Tensor> sampleRandomly(List<Tensor> frames, int numSamples) {
        List<Tensor> sampled = new ArrayList<>();
        int total = frames.size();
        List<Integer> indices = new ArrayList<>();

        for (int i = 0; i < total; i++) {
            indices.add(i);
        }

        // Shuffle
        for (int i = indices.size() - 1; i > 0; i--) {
            int j = (int) (Math.random() * (i + 1));
            int temp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, temp);
        }

        // Take first numSamples
        for (int i = 0; i < Math.min(numSamples, indices.size()); i++) {
            sampled.add(frames.get(indices.get(i)));
        }

        return sampled;
    }

    /**
     * Sample keyframes (simplified - uses uniform sampling as placeholder).
     */
    private List<Tensor> sampleKeyframes(List<Tensor> frames, int numSamples) {
        // Simplified keyframe extraction
        // Actual implementation would detect scene changes or use optical flow
        return sampleUniformly(frames, numSamples);
    }

    /**
     * Calculate temporal patches for video.
     */
    public int getNumTemporalPatches(int numFrames) {
        return (int) Math.ceil(numFrames / (float) temporalPatchSize);
    }

    /**
     * Calculate spatial patches for video frame.
     */
    public int getNumSpatialPatches(int height, int width) {
        int h = (int) Math.ceil(height / (float) spatialPatchSize);
        int w = (int) Math.ceil(width / (float) spatialPatchSize);
        return h * w;
    }

    /**
     * Calculate total video tokens.
     */
    public int getNumVideoTokens(int numFrames, int height, int width) {
        int temporalPatches = getNumTemporalPatches(numFrames);
        int spatialPatches = getNumSpatialPatches(height, width);
        return temporalPatches * spatialPatches;
    }

    /**
     * Get number of frames.
     */
    public int numFrames() { return numFrames; }

    /**
     * Get sampling mode.
     */
    public SamplingMode samplingMode() { return samplingMode; }

    /**
     * Get target height.
     */
    public int targetHeight() { return targetHeight; }

    /**
     * Get target width.
     */
    public int targetWidth() { return targetWidth; }

    /**
     * Get statistics.
     */
    public VideoProcessorStats getStats() {
        return new VideoProcessorStats(
                videosProcessed.get(),
                framesExtracted.get(),
                totalFramesProcessed.get(),
                totalProcessingTimeMs.get()
        );
    }

    /**
     * Reset statistics.
     */
    public void resetStats() {
        totalProcessingTimeMs.set(0);
        videosProcessed.set(0);
        framesExtracted.set(0);
        totalFramesProcessed.set(0);
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (imageProcessor != null) {
            try { imageProcessor.close(); } catch (Exception ignored) {}
        }

        System.out.printf(
                "[VideoProcessor] Closed: videos=%d, framesExtracted=%d, framesProcessed=%d, time=%.2fs%n",
                videosProcessed.get(), framesExtracted.get(),
                totalFramesProcessed.get(), totalProcessingTimeMs.get() / 1000.0);
    }

    private Processor.VideoOutput createEmptyOutput() {
        return new Processor.VideoOutput(
                torch.zeros(new long[]{numFrames, 3, targetHeight, targetWidth}),
                numFrames, targetHeight, targetWidth,
                getNumSpatialPatches(targetHeight, targetWidth),
                30.0f
        );
    }

    /**
     * Statistics.
     */
    public static class VideoProcessorStats {
        public final long videosProcessed;
        public final long framesExtracted;
        public final long totalFramesProcessed;
        public final long totalProcessingTimeMs;

        public VideoProcessorStats(long videosProcessed, long framesExtracted,
                             long totalFramesProcessed, long totalProcessingTimeMs) {
            this.videosProcessed = videosProcessed;
            this.framesExtracted = framesExtracted;
            this.totalFramesProcessed = totalFramesProcessed;
            this.totalProcessingTimeMs = totalProcessingTimeMs;
        }

        public double avgTimeMs() {
            return videosProcessed > 0 ? (double) totalProcessingTimeMs / videosProcessed : 0;
        }

        public double avgFramesPerVideo() {
            return videosProcessed > 0 ? (double) totalFramesProcessed / videosProcessed : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "VideoProcessorStats{videos=%d, frames=%d, avgFrames=%.1f, avgTime=%.2fms}",
                    videosProcessed, totalFramesProcessed, avgFramesPerVideo(), avgTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private int numFrames = 8;
        private SamplingMode samplingMode = SamplingMode.UNIFORM;
        private int targetHeight = 224;
        private int targetWidth = 224;
        private float[] mean;
        private float[] std;
        private boolean doNormalize = true;
        private int temporalPatchSize = 2;
        private int spatialPatchSize = 14;
        private int maxVideoLength = 300;  // Max frames

        public Builder numFrames(int numFrames) { this.numFrames = numFrames; return this; }
        public Builder samplingMode(SamplingMode samplingMode) { this.samplingMode = samplingMode; return this; }
        public Builder targetHeight(int targetHeight) { this.targetHeight = targetHeight; return this; }
        public Builder targetWidth(int targetWidth) { this.targetWidth = targetWidth; return this; }
        public Builder targetSize(int size) { this.targetHeight = size; this.targetWidth = size; return this; }
        public Builder mean(float[] mean) { this.mean = mean; return this; }
        public Builder std(float[] std) { this.std = std; return this; }
        public Builder doNormalize(boolean doNormalize) { this.doNormalize = doNormalize; return this; }
        public Builder temporalPatchSize(int temporalPatchSize) { this.temporalPatchSize = temporalPatchSize; return this; }
        public Builder spatialPatchSize(int spatialPatchSize) { this.spatialPatchSize = spatialPatchSize; return this; }
        public Builder maxVideoLength(int maxVideoLength) { this.maxVideoLength = maxVideoLength; return this; }

        /**
         * Configure for CLIP-style normalization.
         */
        public Builder clip() {
            this.mean = new float[]{0.48145466f, 0.4578275f, 0.40821073f};
            this.std = new float[]{0.26862954f, 0.26130258f, 0.27577711f};
            return this;
        }

        /**
         * Configure for ImageNet normalization.
         */
        public Builder imagenet() {
            this.mean = new float[]{0.485f, 0.456f, 0.406f};
            this.std = new float[]{0.229f, 0.224f, 0.225f};
            return this;
        }

        public VideoProcessor build() { return new VideoProcessor(this); }
    }
}
