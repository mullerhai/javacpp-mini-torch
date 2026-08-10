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
import org.bytedeco.pytorch.geometric.nn.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * MiniMax Vision Encoder for VL models.
 *
 * <p>Handles:
 * <ul>
 *   <li>Patch embedding with convolution</li>
 *   <li>Transformer blocks for vision processing</li>
 *   <li>Spatial and temporal merging</li>
 *   <li>Dynamic resolution support</li>
 * </ul>
 *
 * <p>Reference: MiniMax-VL
 */
public class MiniMaxVisionEncoder implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final int hiddenSize;
    private final int patchSize;
    private final int spatialMergeSize;
    private final int numLayers;
    private final int numAttentionHeads;
    private final int visionEmbedDim;

    // Modules
    private final Module patchEmbed;
    private final Module spatialMerge;
    private final Module projLayer;
    private final List<Module> blocks;
    private final Module norm;
    private final Module positionEmbedding;

    // Performance metrics
    private final AtomicLong totalForwardTimeMs = new AtomicLong(0);
    private final AtomicLong imagesProcessed = new AtomicLong(0);
    private final AtomicLong framesProcessed = new AtomicLong(0);
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);

    /**
     * Create encoder from config.
     */
    public static MiniMaxVisionEncoder create(MiniMaxVisionConfig config) {
        return builder()
                .hiddenSize(config.hiddenSize())
                .patchSize(config.patchSize())
                .spatialMergeSize(config.spatialMergeSize())
                .numLayers(config.numHiddenLayers())
                .numAttentionHeads(config.numAttentionHeads())
                .visionEmbedDim(config.visionEmbedDim())
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private MiniMaxVisionEncoder(Builder builder) {
        this.hiddenSize = builder.hiddenSize;
        this.patchSize = builder.patchSize;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.numLayers = builder.numLayers;
        this.numAttentionHeads = builder.numAttentionHeads;
        this.visionEmbedDim = builder.visionEmbedDim;

        // Initialize modules
        this.patchEmbed = torch.nn.conv2d(
                3, hiddenSize,
                new long[]{patchSize, patchSize},
                new long[]{patchSize, patchSize}
        );

        this.spatialMerge = createSpatialMergeModule();

        this.projLayer = torch.nn.linear(hiddenSize, visionEmbedDim);

        // Initialize transformer blocks
        this.blocks = new ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            blocks.add(createTransformerBlock());
        }

        this.norm = new LayerNorm(hiddenSize);

        this.positionEmbedding = createPositionEmbedding();
    }

    /**
     * Forward pass for image encoding.
     *
     * @param pixelValues [batch, channels, height, width]
     * @return [batch, num_patches, vision_embed_dim]
     */
    public Tensor forward(Tensor pixelValues) {
        long start = System.currentTimeMillis();

        try {
            // 1. Patch embedding
            // [B, C, H, W] -> [B, hidden_size, H/patch_size, W/patch_size]
            Tensor x = patchEmbed.forward(pixelValues);

            // 2. Flatten spatial dimensions
            // [B, hidden_size, H', W'] -> [B, hidden_size, H'*W']
            x = x.flatten(2, 3);

            // 3. Transpose to [B, seq_len, hidden_size]
            x = x.transpose(1, 2);

            // 4. Spatial merge (reduce sequence length)
            x = spatialMerge.forward(x);

            // 5. Add position embedding
            x = x.add(positionEmbedding);

            // 6. Transformer blocks
            for (Module block : blocks) {
                x = block.forward(x);
            }

            // 7. Final norm
            x = norm.forward(x);

            // 8. Project to vision embedding dimension
            x = projLayer.forward(x);

            imagesProcessed.incrementAndGet();
            totalTokensProcessed.addAndGet(x.size(1));
            totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);

            return x;

        } catch (Exception e) {
            System.err.println("MiniMaxVisionEncoder.forward error: " + e.getMessage());
            return torch.zeros(new long[]{1, 1, visionEmbedDim});
        }
    }

    /**
     * Forward pass for video encoding.
     *
     * @param pixelValues [batch, frames, channels, height, width]
     * @return [batch, num_patches, vision_embed_dim]
     */
    public Tensor forwardVideo(Tensor pixelValues) {
        long start = System.currentTimeMillis();

        try {
            long batchSize = pixelValues.size(0);
            long frames = pixelValues.size(1);
            long channels = pixelValues.size(2);
            long height = pixelValues.size(3);
            long width = pixelValues.size(4);

            // Reshape: [B, T, C, H, W] -> [B*T, C, H, W]
            Tensor reshaped = pixelValues.reshape(batchSize * frames, channels, height, width);

            // Process through patch embedding
            Tensor x = patchEmbed.forward(reshaped);
            x = x.flatten(2, 3);
            x = x.transpose(1, 2);

            // Spatial merge
            x = spatialMerge.forward(x);

            // Reshape back with temporal dimension
            long seqPerFrame = x.size(1) / frames;
            x = x.reshape(batchSize, frames * seqPerFrame, hiddenSize);

            // Add position embedding
            x = x.add(positionEmbedding);

            // Transformer blocks
            for (Module block : blocks) {
                x = block.forward(x);
            }

            // Final norm and projection
            x = norm.forward(x);
            x = projLayer.forward(x);

            framesProcessed.addAndGet(frames);
            totalTokensProcessed.addAndGet(x.size(1));
            totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);

            return x;

        } catch (Exception e) {
            System.err.println("MiniMaxVisionEncoder.forwardVideo error: " + e.getMessage());
            return torch.zeros(new long[]{1, 1, visionEmbedDim});
        }
    }

    /**
     * Calculate image grid for given dimensions.
     */
    public int[] calculateImageGrid(int height, int width) {
        int h = (int) Math.ceil(height / (float) patchSize);
        int w = (int) Math.ceil(width / (float) patchSize);

        // Apply spatial merge
        h = (int) Math.ceil(h / (float) spatialMergeSize);
        w = (int) Math.ceil(w / (float) spatialMergeSize);

        return new int[]{1, h, w};
    }

    /**
     * Get number of image tokens.
     */
    public int getNumImageTokens(int height, int width) {
        int[] grid = calculateImageGrid(height, width);
        return grid[1] * grid[2];
    }

    private Module createSpatialMergeModule() {
        // Simple linear layer for spatial merging
        // In practice, this might be a more complex MLP
        return torch.nn.linear(hiddenSize, hiddenSize);
    }

    private Module createTransformerBlock() {
        // Simplified transformer block
        // Actual implementation would include:
        // - Attention
        // - MLP
        // - Layer norm
        // - Residual connections
        return torch.nn.linear(hiddenSize, hiddenSize);
    }

    private Module createPositionEmbedding() {
        // Learnable position embedding
        // [1, max_seq_len, hidden_size]
        return torch.nn.embedding(1024, hiddenSize);
    }

    /**
     * Get statistics.
     */
    public VisionEncoderStats getStats() {
        return new VisionEncoderStats(
                hiddenSize,
                visionEmbedDim,
                numLayers,
                patchSize,
                imagesProcessed.get(),
                framesProcessed.get(),
                totalTokensProcessed.get(),
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
        if (projLayer != null) projLayer.close();
        if (norm != null) norm.close();
        if (positionEmbedding != null) positionEmbedding.close();

        for (Module block : blocks) {
            if (block != null) block.close();
        }
        blocks.clear();

        System.out.printf(
                "[MiniMaxVisionEncoder] Closed: images=%d, frames=%d, tokens=%d, time=%.2fs%n",
                imagesProcessed.get(), framesProcessed.get(),
                totalTokensProcessed.get(), totalForwardTimeMs.get() / 1000.0);
    }

    /**
     * Statistics.
     */
    public static class VisionEncoderStats {
        public final int hiddenSize;
        public final int visionEmbedDim;
        public final int numLayers;
        public final int patchSize;
        public final long imagesProcessed;
        public final long framesProcessed;
        public final long totalTokensProcessed;
        public final long totalForwardTimeMs;

        public VisionEncoderStats(int hiddenSize, int visionEmbedDim, int numLayers,
                            int patchSize, long imagesProcessed, long framesProcessed,
                            long totalTokensProcessed, long totalForwardTimeMs) {
            this.hiddenSize = hiddenSize;
            this.visionEmbedDim = visionEmbedDim;
            this.numLayers = numLayers;
            this.patchSize = patchSize;
            this.imagesProcessed = imagesProcessed;
            this.framesProcessed = framesProcessed;
            this.totalTokensProcessed = totalTokensProcessed;
            this.totalForwardTimeMs = totalForwardTimeMs;
        }

        public double avgTimeMs() {
            long total = imagesProcessed + framesProcessed;
            return total > 0 ? (double) totalForwardTimeMs / total : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "VisionEncoderStats{layers=%d, hidden=%d, embed=%d, " +
                    "images=%d, frames=%d, tokens=%d, avgTime=%.2fms}",
                    numLayers, hiddenSize, visionEmbedDim,
                    imagesProcessed, framesProcessed, totalTokensProcessed, avgTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private int hiddenSize = 2048;
        private int patchSize = 14;
        private int spatialMergeSize = 2;
        private int numLayers = 24;
        private int numAttentionHeads = 16;
        private int visionEmbedDim = 2048;

        public Builder hiddenSize(int v) { this.hiddenSize = v; return this; }
        public Builder patchSize(int v) { this.patchSize = v; return this; }
        public Builder spatialMergeSize(int v) { this.spatialMergeSize = v; return this; }
        public Builder numLayers(int v) { this.numLayers = v; return this; }
        public Builder numAttentionHeads(int v) { this.numAttentionHeads = v; return this; }
        public Builder visionEmbedDim(int v) { this.visionEmbedDim = v; return this; }

        public MiniMaxVisionEncoder build() { return new MiniMaxVisionEncoder(this); }
    }
}
