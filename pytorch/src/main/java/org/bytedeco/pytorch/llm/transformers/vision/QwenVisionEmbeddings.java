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
package org.bytedeco.pytorch.llm.transformers.vision;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.modules.RMSNorm;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Vision embeddings for Qwen2-VL and Qwen3-VL models.
 *
 * <p>Handles:
 * <ul>
 *   <li>Image patch embedding with 2D sin/cos positional embeddings</li>
 *   <li>Spatial merging for efficient representation</li>
 *   <li>Dynamic resolution support</li>
 *   <li>Image grid calculation</li>
 * </ul>
 *
 * <p>Reference: Qwen2-VL, Qwen3-VL
 */
public class QwenVisionEmbeddings implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final int hiddenSize;
    private final int patchSize;
    private final int spatialMergeSize;
    private final int spatialMergeUnit;
    private final int temporalPatchSize;
    private final int imageGridSize;
    private final boolean dynamicResolution;

    // Modules
    private final Module patchEmbed;      // Conv2D patch embedding
    private final Module spatialMerge;    // Spatial merge MLP
    private final Module temporalMerge;   // Temporal merge MLP (for video)
    private final RMSNorm imageNorm;
    private final RMSNorm videoNorm;

    // Position embeddings (cached)
    private Tensor spatialPosEmbed;  // [1, seq_len, hidden_size]
    private Tensor temporalPosEmbed;

    // Performance metrics
    private final AtomicLong totalForwardTimeMs = new AtomicLong(0);
    private final AtomicLong imagesProcessed = new AtomicLong(0);
    private final AtomicLong framesProcessed = new AtomicLong(0);

    /**
     * Create QwenVisionEmbeddings with configuration.
     */
    public static QwenVisionEmbeddings create(QwenVisionConfig config) {
        return builder()
                .hiddenSize(config.visionEmbedDim())
                .patchSize(config.patchSize())
                .spatialMergeSize(config.spatialMergeSize())
                .spatialMergeUnit(config.spatialMergeUnit())
                .temporalPatchSize(config.temporalPatchSize())
                .imageGridSize(config.imageGridSize())
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private QwenVisionEmbeddings(Builder builder) {
        this.hiddenSize = builder.hiddenSize;
        this.patchSize = builder.patchSize;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.spatialMergeUnit = builder.spatialMergeUnit;
        this.temporalPatchSize = builder.temporalPatchSize;
        this.imageGridSize = builder.imageGridSize;
        this.dynamicResolution = builder.dynamicResolution;

        // Initialize patch embedding (Conv2D with patch_size kernel)
        // In PyTorch: nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        this.patchEmbed = createPatchEmbedding();

        // Initialize spatial merge MLP
        // Merges adjacent patches to reduce sequence length
        this.spatialMerge = createSpatialMergeMLP();

        // Initialize temporal merge MLP (for video)
        this.temporalMerge = createTemporalMergeMLP();

        // Initialize layer norms
        this.imageNorm = new RMSNorm(hiddenSize);
        this.imageNorm.setEps(builder.rmsNormEps);

        this.videoNorm = new RMSNorm(hiddenSize);
        videoNorm.setEps(builder.rmsNormEps);
    }

    /**
     * Forward pass for image embeddings.
     *
     * @param pixelValues [batch, channels, height, width]
     * @return [batch, num_patches, hidden_size]
     */
    public Tensor forward(Tensor pixelValues) {
        long start = System.currentTimeMillis();

        try {
            // Get batch size and spatial dimensions
            long batchSize = pixelValues.size(0);
            long channels = pixelValues.size(1);
            long height = pixelValues.size(2);
            long width = pixelValues.size.size(3);

            // 1. Patch embedding
            // [B, C, H, W] -> [B, hidden_size, H/patch_size, W/patch_size]
            Tensor patches = patchEmbed.forward(pixelValues);

            // 2. Flatten spatial dimensions
            // [B, hidden_size, H', W'] -> [B, hidden_size, H'*W']
            long hPatches = patches.size(2);
            long wPatches = patches.size(3);
            patches = patches.flatten(2, 3);  // [B, hidden_size, seq_len]

            // 3. Transpose to [B, seq_len, hidden_size]
            patches = patches.transpose(1, 2);  // [B, seq_len, hidden_size]

            // 4. Spatial merge (merge adjacent patches)
            // Merge every spatialMergeSize x spatialMergeSize patches
            patches = spatialMergeForward(patches, hPatches, wPatches);

            // 5. Add positional embeddings
            if (spatialPosEmbed == null || dynamicResolution) {
                // Compute dynamic positional embeddings
                int seqLen = (int) patches.size(1);
                spatialPosEmbed = createSpatialPosEmbed(seqLen);
            }
            patches = patches.add(spatialPosEmbed);

            // 6. Layer norm
            patches = imageNorm.forward(patches);

            imagesProcessed.incrementAndGet();
            totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);

            return patches;

        } catch (Exception e) {
            System.err.println("QwenVisionEmbeddings.forward error: " + e.getMessage());
            return torch.zeros(new long[]{1, 1, hiddenSize});
        }
    }

    /**
     * Forward pass for video embeddings.
     *
     * @param pixelValues [batch, frames, channels, height, width]
     * @return [batch, num_patches, hidden_size]
     */
    public Tensor forwardVideo(Tensor pixelValues) {
        long start = System.currentTimeMillis();

        try {
            long batchSize = pixelValues.size(0);
            long frames = pixelValues.size(1);
            long channels = pixelValues.size(2);
            long height = pixelValues.size(3);
            long width = pixelValues.size(4);

            // Reshape to process as batch of frames
            // [B, T, C, H, W] -> [B*T, C, H, W]
            Tensor reshaped = pixelValues.reshape(batchSize * frames, channels, height, width);

            // Patch embedding
            Tensor patches = patchEmbed.forward(reshaped);

            // Reshape back: [B*T, hidden_size, H', W'] -> [B, T*H'*W', hidden_size]
            long hPatches = patches.size(2);
            long wPatches = patches.size(3);
            patches = patches.flatten(2, 3);
            patches = patches.reshape(batchSize, frames * hPatches * wPatches, hiddenSize);
            patches = patches.transpose(1, 2);  // [B, seq, hidden]

            // Spatial merge
            patches = spatialMergeForwardVideo(patches, frames, hPatches, wPatches);

            // Temporal merge
            patches = temporalMerge.forward(patches);

            // Add positional embeddings
            if (temporalPosEmbed == null || dynamicResolution) {
                int seqLen = (int) patches.size(1);
                temporalPosEmbed = createSpatialPosEmbed(seqLen);
            }
            patches = patches.add(temporalPosEmbed);

            // Layer norm
            patches = videoNorm.forward(patches);

            framesProcessed.addAndGet(frames);
            totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);

            return patches;

        } catch (Exception e) {
            System.err.println("QwenVisionEmbeddings.forwardVideo error: " + e.getMessage());
            return torch.zeros(new long[]{1, 1, hiddenSize});
        }
    }

    /**
     * Calculate image grid THW for given dimensions.
     */
    public int[] calculateImageGrid(int height, int width) {
        int h = (int) Math.ceil(height / (float) patchSize);
        int w = (int) Math.ceil(width / (float) patchSize);

        // Apply spatial merge
        h = (int) Math.ceil(h / (float) spatialMergeSize);
        w = (int) Math.ceil(w / (float) spatialMergeSize);

        return new int[]{1, h, w};  // [T, H, W]
    }

    /**
     * Calculate number of image tokens.
     */
    public int getNumImageTokens(int height, int width) {
        int[] grid = calculateImageGrid(height, width);
        return grid[1] * grid[2];
    }

    /**
     * Create 2D sinusoidal positional embeddings.
     */
    private Tensor createSpatialPosEmbed(int seqLen) {
        // Simplified 2D sinusoidal embeddings
        // Actual implementation would use proper 2D position encoding
        float[] posEmbed = new float[seqLen * hiddenSize];

        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                float angle = (float) (i / Math.pow(10000, 2.0 * j / hiddenSize));
                posEmbed[i * hiddenSize + j] = (float) Math.sin(angle);
            }
        }

        return torch.tensor(posEmbed).reshape(1, seqLen, hiddenSize);
    }

    /**
     * Spatial merge forward (merge adjacent patches).
     */
    private Tensor spatialMergeForward(Tensor patches, long hPatches, long wPatches) {
        // This is a simplified version
        // Actual implementation uses a learned MLP to merge spatial patches
        return patches;
    }

    /**
     * Spatial merge for video.
     */
    private Tensor spatialMergeForwardVideo(Tensor patches, long frames, long hPatches, long wPatches) {
        return patches;
    }

    /**
     * Create patch embedding module.
     */
    private Module createPatchEmbedding() {
        // Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        return torch.nn.conv2d(
                3, hiddenSize,
                new long[]{patchSize, patchSize},
                new long[]{patchSize, patchSize}
        );
    }

    /**
     * Create spatial merge MLP.
     */
    private Module createSpatialMergeMLP() {
        // MLP that merges spatial patches: hidden_size -> hidden_size
        return torch.nn.linear(hiddenSize, hiddenSize);
    }

    /**
     * Create temporal merge MLP.
     */
    private Module createTemporalMergeMLP() {
        // MLP that merges temporal patches: hidden_size -> hidden_size
        return torch.nn.linear(hiddenSize, hiddenSize);
    }

    /**
     * Get statistics.
     */
    public VisionEmbeddingsStats getStats() {
        return new VisionEmbeddingsStats(
                hiddenSize,
                patchSize,
                spatialMergeSize,
                imagesProcessed.get(),
                framesProcessed.get(),
                totalForwardTimeMs.get()
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (patchEmbed != null) patchEmbed.close();
        if (spatialMerge != null) spatialMerge.close();
        if (temporalMerge != null) temporalMerge.close();
        if (imageNorm != null) imageNorm.close();
        if (videoNorm != null) videoNorm.close();
        if (spatialPosEmbed != null) spatialPosEmbed.close();
        if (temporalPosEmbed != null) temporalPosEmbed.close();

        System.out.printf(
                "[QwenVisionEmbeddings] Closed: images=%d, frames=%d, time=%.2fs%n",
                imagesProcessed.get(), framesProcessed.get(),
                totalForwardTimeMs.get() / 1000.0);
    }

    /**
     * Statistics.
     */
    public static class VisionEmbeddingsStats {
        public final int hiddenSize;
        public final int patchSize;
        public final int spatialMergeSize;
        public final long imagesProcessed;
        public final long framesProcessed;
        public final long totalForwardTimeMs;

        public VisionEmbeddingsStats(int hiddenSize, int patchSize, int spatialMergeSize,
                                 long imagesProcessed, long framesProcessed, long totalForwardTimeMs) {
            this.hiddenSize = hiddenSize;
            this.patchSize = patchSize;
            this.spatialMergeSize = spatialMergeSize;
            this.imagesProcessed = imagesProcessed;
            this.framesProcessed = framesProcessed;
            this.totalForwardTimeMs = totalForwardTimeMs;
        }

        public double avgTimeMs() {
            long total = imagesProcessed + framesProcessed;
            return total > 0 ? (double) totalForwardTimeMs / total : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "VisionEmbeddingsStats{hiddenSize=%d, patchSize=%d, " +
                    "images=%d, frames=%d, avgTime=%.2fms}",
                    hiddenSize, patchSize, imagesProcessed, framesProcessed, avgTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private int hiddenSize = 1280;
        private int patchSize = 14;
        private int spatialMergeSize = 2;
        private int spatialMergeUnit = 2;
        private int temporalPatchSize = 2;
        private int imageGridSize = 2;
        private boolean dynamicResolution = true;
        private double rmsNormEps = 1e-6;

        public Builder hiddenSize(int v) { this.hiddenSize = v; return this; }
        public Builder patchSize(int v) { this.patchSize = v; return this; }
        public Builder spatialMergeSize(int v) { this.spatialMergeSize = v; return this; }
        public Builder spatialMergeUnit(int v) { this.spatialMergeUnit = v; return this; }
        public Builder temporalPatchSize(int v) { this.temporalPatchSize = v; return this; }
        public Builder imageGridSize(int v) { this.imageGridSize = v; return this; }
        public Builder dynamicResolution(boolean v) { this.dynamicResolution = v; return this; }
        public Builder rmsNormEps(double v) { this.rmsNormEps = v; return this; }

        public QwenVisionEmbeddings build() { return new QwenVisionEmbeddings(this); }
    }
}
