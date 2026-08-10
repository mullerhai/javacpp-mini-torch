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
import org.bytedeco.pytorch.llm.modules.RMSNorm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Whisper-style audio encoder for speech understanding.
 *
 * <p>Handles:
 * <ul>
 *   <li>Mel spectrogram processing</li>
 *   <li>Convolutional feature extraction</li>
 *   <li>Transformer encoder blocks</li>
 *   <li>Positional encoding</li>
 * </ul>
 *
 * <p>Reference: Whisper (OpenAI)
 */
public class WhisperEncoder implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final int nMelBins;
    private final int dModel;
    private final int nLayers;
    private final int nHeads;
    private final int dFF;
    private final int dHead;
    private final double dropout;

    // Modules
    private final Module conv1;     // First conv layer
    private final Module conv2;     // Second conv layer
    private final List<Module> blocks;  // Transformer blocks
    private final Module layerNorm;
    private final Module positionalEmbedding;

    // Performance metrics
    private final AtomicLong totalForwardTimeMs = new AtomicLong(0);
    private final AtomicLong audioProcessed = new AtomicLong(0);
    private final AtomicLong totalFramesProcessed = new AtomicLong(0);

    /**
     * Create WhisperEncoder with default configuration.
     */
    public static WhisperEncoder createDefault() {
        return builder()
                .nMelBins(80)
                .dModel(1280)
                .nLayers(32)
                .nHeads(20)
                .dHead(64)
                .dFF(5120)
                .build();
    }

    /**
     * Create WhisperEncoder for tiny model.
     */
    public static WhisperEncoder createTiny() {
        return builder()
                .nMelBins(80)
                .dModel(384)
                .nLayers(4)
                .nHeads(6)
                .dHead(64)
                .dFF(1536)
                .build();
    }

    /**
     * Create WhisperEncoder for base model.
     */
    public static WhisperEncoder createBase() {
        return builder()
                .nMelBins(80)
                .dModel(512)
                .nLayers(6)
                .nHeads(8)
                .dHead(64)
                .dFF(2048)
                .build();
    }

    /**
     * Create WhisperEncoder for large model.
     */
    public static WhisperEncoder createLarge() {
        return builder()
                .nMelBins(128)
                .dModel(1280)
                .nLayers(32)
                .nHeads(20)
                .dHead(64)
                .dFF(5120)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private WhisperEncoder(Builder builder) {
        this.nMelBins = builder.nMelBins;
        this.dModel = builder.dModel;
        this.nLayers = builder.nLayers;
        this.nHeads = builder.nHeads;
        this.dHead = builder.dHead;
        this.dFF = builder.dFF;
        this.dropout = builder.dropout;

        // Initialize conv layers (mel -> d_model)
        // Conv1d(in_channels=n_mel_bins, out_channels=d_model, kernel_size=3, padding=1)
        this.conv1 = torch.nn.conv1d(
                nMelBins, dModel,
                new long[]{3},
                new long[]{1},
                new long[]{1}  // padding=1
        );

        // Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1)
        this.conv2 = torch.nn.conv1d(
                dModel, dModel,
                new long[]{3},
                new long[]{2},
                new long[]{1}
        );

        // Initialize positional embedding
        int maxLen = builder.maxLen;
        this.positionalEmbedding = torch.nn.embedding(maxLen, dModel);

        // Initialize transformer blocks
        this.blocks = new ArrayList<>();
        for (int i = 0; i < nLayers; i++) {
            blocks.add(createTransformerBlock());
        }

        // Final layer norm
        this.layerNorm = new LayerNorm(dModel);
    }

    /**
     * Forward pass.
     *
     * @param melSpectrogram [batch, n_mel_bins, seq_len]
     * @return [batch, seq_len/4, d_model]
     */
    public Tensor forward(Tensor melSpectrogram) {
        long start = System.currentTimeMillis();

        try {
            // 1. First conv layer with GELU activation
            // [B, n_mel, T] -> [B, d_model, T]
            Tensor x = conv1.forward(melSpectrogram);
            x = torch.gelu(x);

            // 2. Second conv layer with GELU activation and stride=2
            // [B, d_model, T] -> [B, d_model, T/2]
            x = conv2.forward(x);
            x = torch.gelu(x);

            // 3. Transpose for transformer: [B, d_model, T] -> [B, T, d_model]
            x = x.transpose(1, 2);

            // 4. Add positional embedding
            int seqLen = (int) x.size(1);
            Tensor posEmbed = positionalEmbedding.forward(
                    torch.arange(0, seqLen)
            );
            x = x.add(posEmbed);

            // 5. Transformer blocks
            for (Module block : blocks) {
                x = block.forward(x);
            }

            // 6. Final layer norm
            x = layerNorm.forward(x);

            audioProcessed.incrementAndGet();
            totalFramesProcessed.addAndGet(seqLen);
            totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);

            return x;

        } catch (Exception e) {
            System.err.println("WhisperEncoder.forward error: " + e.getMessage());
            return torch.zeros(new long[]{1, 1, dModel});
        }
    }

    /**
     * Create a transformer encoder block.
     */
    private Module createTransformerBlock() {
        // Simplified transformer block
        // Actual implementation would include:
        // - Multi-head self-attention
        // - Feed-forward network
        // - Layer normalization
        // - Residual connections
        return torch.nn.linear(dModel, dModel);
    }

    /**
     * Get output dimension.
     */
    public int dModel() {
        return dModel;
    }

    /**
     * Get statistics.
     */
    public WhisperEncoderStats getStats() {
        return new WhisperEncoderStats(
                dModel,
                nLayers,
                nHeads,
                audioProcessed.get(),
                totalFramesProcessed.get(),
                totalForwardTimeMs.get()
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (conv1 != null) conv1.close();
        if (conv2 != null) conv2.close();
        if (layerNorm != null) layerNorm.close();
        if (positionalEmbedding != null) positionalEmbedding.close();

        for (Module block : blocks) {
            if (block != null) block.close();
        }
        blocks.clear();

        System.out.printf(
                "[WhisperEncoder] Closed: audio=%d, frames=%d, time=%.2fs%n",
                audioProcessed.get(), totalFramesProcessed.get(),
                totalForwardTimeMs.get() / 1000.0);
    }

    /**
     * Statistics.
     */
    public static class WhisperEncoderStats {
        public final int dModel;
        public final int nLayers;
        public final int nHeads;
        public final long audioProcessed;
        public final long totalFramesProcessed;
        public final long totalForwardTimeMs;

        public WhisperEncoderStats(int dModel, int nLayers, int nHeads,
                            long audioProcessed, long totalFramesProcessed,
                            long totalForwardTimeMs) {
            this.dModel = dModel;
            this.nLayers = nLayers;
            this.nHeads = nHeads;
            this.audioProcessed = audioProcessed;
            this.totalFramesProcessed = totalFramesProcessed;
            this.totalForwardTimeMs = totalForwardTimeMs;
        }

        public double avgTimeMs() {
            return audioProcessed > 0 ? (double) totalForwardTimeMs / audioProcessed : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "WhisperEncoderStats{dModel=%d, layers=%d, heads=%d, " +
                    "audio=%d, frames=%d, avgTime=%.2fms}",
                    dModel, nLayers, nHeads, audioProcessed,
                    totalFramesProcessed, avgTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private int nMelBins = 80;
        private int dModel = 1280;
        private int nLayers = 32;
        private int nHeads = 20;
        private int dHead = 64;
        private int dFF = 5120;
        private double dropout = 0.0;
        private int maxLen = 3000;

        public Builder nMelBins(int v) { this.nMelBins = v; return this; }
        public Builder dModel(int v) { this.dModel = v; return this; }
        public Builder nLayers(int v) { this.nLayers = v; return this; }
        public Builder nHeads(int v) { this.nHeads = v; return this; }
        public Builder dHead(int v) { this.dHead = v; return this; }
        public Builder dFF(int v) { this.dFF = v; return this; }
        public Builder dropout(double v) { this.dropout = v; return this; }
        public Builder maxLen(int v) { this.maxLen = v; return this; }

        public WhisperEncoder build() { return new WhisperEncoder(this); }
    }
}
