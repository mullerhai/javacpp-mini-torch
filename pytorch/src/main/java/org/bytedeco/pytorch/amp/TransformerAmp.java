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
package org.bytedeco.pytorch.amp;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mixed precision utilities for transformer models.
 *
 * <p>Provides:
 * <ul>
 *   <li>Automatic precision casting for transformer layers</li>
 *   <li>Memory-efficient mixed precision operations</li>
 *   <li>Precision-aware attention computation</li>
 *   <li>Multi-modal tensor management</li>
 * </ul>
 *
 * <p>Reference: Megatron-LM, DeepSpeed, and transformer mixed precision research
 */
public class TransformerAmp implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final AmpPrecision attentionPrecision;
    private final AmpPrecision mlpPrecision;
    private final AmpPrecision normPrecision;
    private final AmpPrecision embedPrecision;
    private final boolean useFlashAttention;
    private final boolean useMemoryEfficientAttention;

    // Autocast context
    private final AutocastContext autocast;

    // Performance metrics
    private final AtomicLong totalCasts = new AtomicLong(0);
    private final AtomicLong totalForwardTimeMs = new AtomicLong(0);
    private final AtomicLong totalMemorySavedBytes = new AtomicLong(0);

    /**
     * Create TransformerAmp with default settings.
     */
    public static TransformerAmp createDefault() {
        return builder().build();
    }

    /**
     * Create TransformerAmp optimized for BF16 training.
     */
    public static TransformerAmp createForBF16() {
        return builder()
                .attentionPrecision(AmpPrecision.BF16)
                .mlpPrecision(AmpPrecision.BF16)
                .normPrecision(AmpPrecision.FP32)
                .embedPrecision(AmpPrecision.BF16)
                .useFlashAttention(true)
                .build();
    }

    /**
     * Create TransformerAmp optimized for FP8 inference.
     */
    public static TransformerAmp createForFP8Inference() {
        return builder()
                .attentionPrecision(AmpPrecision.FP8_E4M3)
                .mlpPrecision(AmpPrecision.FP8_E4M3)
                .normPrecision(AmpPrecision.FP32)
                .embedPrecision(AmpPrecision.FP16)
                .useFlashAttention(true)
                .useMemoryEfficientAttention(true)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private TransformerAmp(Builder builder) {
        this.attentionPrecision = builder.attentionPrecision;
        this.mlpPrecision = builder.mlpPrecision;
        this.normPrecision = builder.normPrecision;
        this.embedPrecision = builder.embedPrecision;
        this.useFlashAttention = builder.useFlashAttention;
        this.useMemoryEfficientAttention = builder.useMemoryEfficientAttention;
        this.autocast = builder.autocast != null
                ? builder.autocast
                : AutocastContext.create(
                        builder.device != null ? builder.device : new Device("cuda"),
                        builder.defaultPrecision);
    }

    /**
     * Cast input tensor to attention precision.
     */
    public Tensor castForAttention(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(attentionPrecision);
        totalCasts.incrementAndGet();
        totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast input tensor to MLP precision.
     */
    public Tensor castForMlp(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(mlpPrecision);
        totalCasts.incrementAndGet();
        totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to FP32 for normalization (stability).
     */
    public Tensor castForNorm(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        totalCasts.incrementAndGet();
        totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(ScalarType.Float);
    }

    /**
     * Cast tensor to embedding precision.
     */
    public Tensor castForEmbed(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(embedPrecision);
        totalCasts.incrementAndGet();
        totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to FP32 for loss computation (stability).
     */
    public Tensor castForLoss(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        totalCasts.incrementAndGet();
        totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(ScalarType.Float);
    }

    /**
     * Compute scaled dot-product attention with mixed precision.
     */
    public Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value,
                                          Tensor attnMask, float scale) {
        if (query == null || !query.defined()) return null;

        long start = System.currentTimeMillis();

        // Cast to attention precision
        Tensor q = castForAttention(query);
        Tensor k = castForAttention(key);
        Tensor v = castForAttention(value);

        Tensor result;

        if (useFlashAttention && isFlashAttentionSupported()) {
            // Use flash attention for memory efficiency
            result = scaledDotProductAttentionFlash(q, k, v, attnMask, scale);
        } else {
            // Standard attention
            result = scaledDotProductAttentionStandard(q, k, v, attnMask, scale);
        }

        totalForwardTimeMs.addAndGet(System.currentTimeMillis() - start);
        return result;
    }

    /**
     * Standard scaled dot-product attention.
     */
    private Tensor scaledDotProductAttentionStandard(Tensor q, Tensor k, Tensor v,
                                                   Tensor attnMask, float scale) {
        // Q @ K^T
        Tensor scores = torch.matmul(q, k.transpose(-2, -1));
        if (scale != 1.0f) {
            scores = scores.mul(new Scalar(scale));
        }
        if (attnMask != null && attnMask.defined()) {
            scores = scores.add(attnMask);
        }
        Tensor attnWeights = torch.softmax(scores, -1);
        return torch.matmul(attnWeights, v);
    }

    /**
     * Flash attention (placeholder - requires CUDA kernel).
     */
    private Tensor scaledDotProductAttentionFlash(Tensor q, Tensor k, Tensor v,
                                                Tensor attnMask, float scale) {
        // Simplified flash attention - in practice this would use CUDA kernel
        // For now, fall back to standard attention
        return scaledDotProductAttentionStandard(q, k, v, attnMask, scale);
    }

    /**
     * Check if flash attention is supported.
     */
    private boolean isFlashAttentionSupported() {
        try {
            return torch.cuda_is_available();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get attention precision.
     */
    public AmpPrecision getAttentionPrecision() {
        return attentionPrecision;
    }

    /**
     * Get MLP precision.
     */
    public AmpPrecision getMlpPrecision() {
        return mlpPrecision;
    }

    /**
     * Get norm precision.
     */
    public AmpPrecision getNormPrecision() {
        return normPrecision;
    }

    /**
     * Get embed precision.
     */
    public AmpPrecision getEmbedPrecision() {
        return embedPrecision;
    }

    /**
     * Check if using flash attention.
     */
    public boolean isUsingFlashAttention() {
        return useFlashAttention;
    }

    /**
     * Get transformer AMP statistics.
     */
    public TransformerAmpStats getStats() {
        return new TransformerAmpStats(
                attentionPrecision.name(),
                mlpPrecision.name(),
                normPrecision.name(),
                embedPrecision.name(),
                useFlashAttention,
                useMemoryEfficientAttention,
                totalCasts.get(),
                totalForwardTimeMs.get(),
                totalMemorySavedBytes.get()
        );
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            autocast.close();
        } catch (Exception ignored) {}
        System.out.printf(
                "[TransformerAmp] Closed: precision=%s/%s/%s/%s, casts=%d, time=%.2fs%n",
                attentionPrecision, mlpPrecision, normPrecision, embedPrecision,
                totalCasts.get(), totalForwardTimeMs.get() / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Transformer AMP statistics.
     */
    public static class TransformerAmpStats {
        public final String attentionPrecision;
        public final String mlpPrecision;
        public final String normPrecision;
        public final String embedPrecision;
        public final boolean useFlashAttention;
        public final boolean useMemoryEfficientAttention;
        public final long totalCasts;
        public final long totalForwardTimeMs;
        public final long totalMemorySavedBytes;

        public TransformerAmpStats(String attentionPrecision, String mlpPrecision,
                               String normPrecision, String embedPrecision,
                               boolean useFlashAttention, boolean useMemoryEfficientAttention,
                               long totalCasts, long totalForwardTimeMs,
                               long totalMemorySavedBytes) {
            this.attentionPrecision = attentionPrecision;
            this.mlpPrecision = mlpPrecision;
            this.normPrecision = normPrecision;
            this.embedPrecision = embedPrecision;
            this.useFlashAttention = useFlashAttention;
            this.useMemoryEfficientAttention = useMemoryEfficientAttention;
            this.totalCasts = totalCasts;
            this.totalForwardTimeMs = totalForwardTimeMs;
            this.totalMemorySavedBytes = totalMemorySavedBytes;
        }

        public double memorySavingsGB() {
            return totalMemorySavedBytes / (1024.0 * 1024 * 1024);
        }

        @Override
        public String toString() {
            return String.format(
                    "TransformerAmpStats{precision=%s/%s/%s/%s, " +
                    "flashAttention=%b, memoryEfficient=%b, casts=%d, time=%.2fs, saved=%.2fGB}",
                    attentionPrecision, mlpPrecision, normPrecision, embedPrecision,
                    useFlashAttention, useMemoryEfficientAttention,
                    totalCasts, totalForwardTimeMs / 1000.0, memorySavingsGB());
        }
    }

    /**
     * Builder for TransformerAmp.
     */
    public static class Builder {
        private Device device;
        private AmpPrecision defaultPrecision = AmpPrecision.FP16;
        private AmpPrecision attentionPrecision = AmpPrecision.FP16;
        private AmpPrecision mlpPrecision = AmpPrecision.FP16;
        private AmpPrecision normPrecision = AmpPrecision.FP32;
        private AmpPrecision embedPrecision = AmpPrecision.FP16;
        private boolean useFlashAttention = false;
        private boolean useMemoryEfficientAttention = false;
        private AutocastContext autocast;

        public Builder device(Device device) {
            this.device = device;
            return this;
        }

        public Builder device(String device) {
            this.device = new Device(device);
            return this;
        }

        public Builder defaultPrecision(AmpPrecision defaultPrecision) {
            this.defaultPrecision = defaultPrecision;
            return this;
        }

        public Builder attentionPrecision(AmpPrecision attentionPrecision) {
            this.attentionPrecision = attentionPrecision;
            return this;
        }

        public Builder mlpPrecision(AmpPrecision mlpPrecision) {
            this.mlpPrecision = mlpPrecision;
            return this;
        }

        public Builder normPrecision(AmpPrecision normPrecision) {
            this.normPrecision = normPrecision;
            return this;
        }

        public Builder embedPrecision(AmpPrecision embedPrecision) {
            this.embedPrecision = embedPrecision;
            return this;
        }

        public Builder useFlashAttention(boolean useFlashAttention) {
            this.useFlashAttention = useFlashAttention;
            return this;
        }

        public Builder useMemoryEfficientAttention(boolean useMemoryEfficientAttention) {
            this.useMemoryEfficientAttention = useMemoryEfficientAttention;
            return this;
        }

        public Builder autocast(AutocastContext autocast) {
            this.autocast = autocast;
            return this;
        }

        /**
         * Configure for BF16 training (recommended).
         */
        public Builder bf16() {
            this.defaultPrecision = AmpPrecision.BF16;
            this.attentionPrecision = AmpPrecision.BF16;
            this.mlpPrecision = AmpPrecision.BF16;
            this.normPrecision = AmpPrecision.FP32;
            this.embedPrecision = AmpPrecision.BF16;
            return this;
        }

        /**
         * Configure for FP16 training.
         */
        public Builder fp16() {
            this.defaultPrecision = AmpPrecision.FP16;
            this.attentionPrecision = AmpPrecision.FP16;
            this.mlpPrecision = AmpPrecision.FP16;
            this.normPrecision = AmpPrecision.FP32;
            this.embedPrecision = AmpPrecision.FP16;
            return this;
        }

        /**
         * Configure for FP8 inference.
         */
        public Builder fp8() {
            this.defaultPrecision = AmpPrecision.FP8_E4M3;
            this.attentionPrecision = AmpPrecision.FP8_E4M3;
            this.mlpPrecision = AmpPrecision.FP8_E4M3;
            this.normPrecision = AmpPrecision.FP32;
            this.embedPrecision = AmpPrecision.FP16;
            return this;
        }

        public TransformerAmp build() {
            return new TransformerAmp(this);
        }
    }
}
