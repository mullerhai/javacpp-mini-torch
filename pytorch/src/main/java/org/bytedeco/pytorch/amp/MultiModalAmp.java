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
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mixed precision support for multi-modal models.
 *
 * <p>Handles different precision requirements for:
 * <ul>
 *   <li>Text/LLM components (FP16/BF16/FP8)</li>
 *   <li>Vision encoders (FP16/BF16)</li>
 *   <li>Audio processing (FP16)</li>
 *   <li>Cross-modal attention (mixed precision)</li>
 *   <li>Fusion layers (FP32 for stability)</li>
 * </ul>
 *
 * <p>Reference: LLaVA, CogVLM, and multi-modal training research
 */
public class MultiModalAmp implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Modal-specific precision
    private final AmpPrecision textPrecision;
    private final AmpPrecision visionPrecision;
    private final AmpPrecision audioPrecision;
    private final AmpPrecision fusionPrecision;
    private final AmpPrecision embeddingPrecision;

    // Vision encoder specific
    private final AmpPrecision visionEncoderPrecision;
    private final AmpPrecision visionProjectionPrecision;

    // Cross-modal attention precision
    private final AmpPrecision crossAttentionPrecision;

    // Performance metrics
    private final AtomicLong textCasts = new AtomicLong(0);
    private final AtomicLong visionCasts = new AtomicLong(0);
    private final AtomicLong audioCasts = new AtomicLong(0);
    private final AtomicLong fusionCasts = new AtomicLong(0);
    private final AtomicLong crossModalCasts = new AtomicLong(0);
    private final AtomicLong totalTimeMs = new AtomicLong(0);

    /**
     * Modal types supported.
     */
    public enum Modal {
        TEXT,
        VISION,
        AUDIO,
        VIDEO,
        EMBEDDING,
        FUSION,
        CROSS_ATTENTION
    }

    /**
     * Create MultiModalAmp for LLaVA-style models.
     */
    public static MultiModalAmp createForLLaVA() {
        return builder()
                .textPrecision(AmpPrecision.FP16)
                .visionPrecision(AmpPrecision.FP16)
                .visionEncoderPrecision(AmpPrecision.FP16)
                .visionProjectionPrecision(AmpPrecision.FP16)
                .crossAttentionPrecision(AmpPrecision.FP16)
                .fusionPrecision(AmpPrecision.FP32)
                .embeddingPrecision(AmpPrecision.FP16)
                .build();
    }

    /**
     * Create MultiModalAmp for BF16 training.
     */
    public static MultiModalAmp createForBF16() {
        return builder()
                .textPrecision(AmpPrecision.BF16)
                .visionPrecision(AmpPrecision.BF16)
                .visionEncoderPrecision(AmpPrecision.BF16)
                .visionProjectionPrecision(AmpPrecision.BF16)
                .crossAttentionPrecision(AmpPrecision.BF16)
                .fusionPrecision(AmpPrecision.FP32)
                .embeddingPrecision(AmpPrecision.BF16)
                .build();
    }

    /**
     * Create MultiModalAmp for mixed modality inference.
     */
    public static MultiModalAmp createForInference() {
        return builder()
                .textPrecision(AmpPrecision.FP16)
                .visionPrecision(AmpPrecision.FP16)
                .visionEncoderPrecision(AmpPrecision.FP16)
                .visionProjectionPrecision(AmpPrecision.FP16)
                .crossAttentionPrecision(AmpPrecision.FP16)
                .fusionPrecision(AmpPrecision.FP16)
                .embeddingPrecision(AmpPrecision.FP16)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private MultiModalAmp(Builder builder) {
        this.textPrecision = builder.textPrecision;
        this.visionPrecision = builder.visionPrecision;
        this.audioPrecision = builder.audioPrecision;
        this.fusionPrecision = builder.fusionPrecision;
        this.embeddingPrecision = builder.embeddingPrecision;
        this.visionEncoderPrecision = builder.visionEncoderPrecision;
        this.visionProjectionPrecision = builder.visionProjectionPrecision;
        this.crossAttentionPrecision = builder.crossAttentionPrecision;
    }

    /**
     * Cast tensor to text/LLM precision.
     */
    public Tensor castForText(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(textPrecision);
        textCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to vision precision.
     */
    public Tensor castForVision(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(visionPrecision);
        visionCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to vision encoder precision.
     */
    public Tensor castForVisionEncoder(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(visionEncoderPrecision);
        visionCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to vision projection precision.
     */
    public Tensor castForVisionProjection(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(visionProjectionPrecision);
        visionCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to audio precision.
     */
    public Tensor castForAudio(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(audioPrecision);
        audioCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to embedding precision.
     */
    public Tensor castForEmbedding(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(embeddingPrecision);
        crossModalCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor for cross-modal attention.
     */
    public Tensor castForCrossAttention(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(crossAttentionPrecision);
        crossModalCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor to fusion precision (usually FP32 for stability).
     */
    public Tensor castForFusion(Tensor input) {
        if (input == null || !input.defined()) return input;
        long start = System.currentTimeMillis();
        ScalarType targetType = AutocastContext.toScalarType(fusionPrecision);
        fusionCasts.incrementAndGet();
        totalTimeMs.addAndGet(System.currentTimeMillis() - start);
        return input.to(targetType);
    }

    /**
     * Cast tensor based on modal type.
     */
    public Tensor cast(Tensor input, Modal modal) {
        if (input == null || !input.defined()) return input;
        switch (modal) {
            case TEXT: return castForText(input);
            case VISION: return castForVision(input);
            case AUDIO: return castForAudio(input);
            case FUSION: return castForFusion(input);
            case EMBEDDING: return castForEmbedding(input);
            case CROSS_ATTENTION: return castForCrossAttention(input);
            default: return input;
        }
    }

    /**
     * Get precision for a modal type.
     */
    public AmpPrecision getPrecision(Modal modal) {
        switch (modal) {
            case TEXT: return textPrecision;
            case VISION: return visionPrecision;
            case AUDIO: return audioPrecision;
            case FUSION: return fusionPrecision;
            case EMBEDDING: return embeddingPrecision;
            case CROSS_ATTENTION: return crossAttentionPrecision;
            default: return AmpPrecision.FP32;
        }
    }

    /**
     * Get text precision.
     */
    public AmpPrecision getTextPrecision() { return textPrecision; }

    /**
     * Get vision precision.
     */
    public AmpPrecision getVisionPrecision() { return visionPrecision; }

    /**
     * Get audio precision.
     */
    public AmpPrecision getAudioPrecision() { return audioPrecision; }

    /**
     * Get fusion precision.
     */
    public AmpPrecision getFusionPrecision() { return fusionPrecision; }

    /**
     * Get embedding precision.
     */
    public AmpPrecision getEmbeddingPrecision() { return embeddingPrecision; }

    /**
     * Get vision encoder precision.
     */
    public AmpPrecision getVisionEncoderPrecision() { return visionEncoderPrecision; }

    /**
     * Get vision projection precision.
     */
    public AmpPrecision getVisionProjectionPrecision() { return visionProjectionPrecision; }

    /**
     * Get cross-attention precision.
     */
    public AmpPrecision getCrossAttentionPrecision() { return crossAttentionPrecision; }

    /**
     * Get multi-modal AMP statistics.
     */
    public MultiModalAmpStats getStats() {
        return new MultiModalAmpStats(
                textPrecision.name(),
                visionPrecision.name(),
                audioPrecision.name(),
                fusionPrecision.name(),
                embeddingPrecision.name(),
                visionEncoderPrecision.name(),
                visionProjectionPrecision.name(),
                crossAttentionPrecision.name(),
                textCasts.get(),
                visionCasts.get(),
                audioCasts.get(),
                fusionCasts.get(),
                crossModalCasts.get(),
                totalTimeMs.get()
        );
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[MultiModalAmp] Closed: text=%s, vision=%s, audio=%s, fusion=%s, " +
                "embed=%s, casts=%d/%d/%d/%d/%d, time=%.2fs%n",
                textPrecision, visionPrecision, audioPrecision, fusionPrecision,
                embeddingPrecision,
                textCasts.get(), visionCasts.get(), audioCasts.get(),
                fusionCasts.get(), crossModalCasts.get(),
                totalTimeMs.get() / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Multi-modal AMP statistics.
     */
    public static class MultiModalAmpStats {
        public final String textPrecision;
        public final String visionPrecision;
        public final String audioPrecision;
        public final String fusionPrecision;
        public final String embeddingPrecision;
        public final String visionEncoderPrecision;
        public final String visionProjectionPrecision;
        public final String crossAttentionPrecision;
        public final long textCasts;
        public final long visionCasts;
        public final long audioCasts;
        public final long fusionCasts;
        public final long crossModalCasts;
        public final long totalTimeMs;

        public MultiModalAmpStats(String textPrecision, String visionPrecision,
                              String audioPrecision, String fusionPrecision,
                              String embeddingPrecision, String visionEncoderPrecision,
                              String visionProjectionPrecision, String crossAttentionPrecision,
                              long textCasts, long visionCasts, long audioCasts,
                              long fusionCasts, long crossModalCasts, long totalTimeMs) {
            this.textPrecision = textPrecision;
            this.visionPrecision = visionPrecision;
            this.audioPrecision = audioPrecision;
            this.fusionPrecision = fusionPrecision;
            this.embeddingPrecision = embeddingPrecision;
            this.visionEncoderPrecision = visionEncoderPrecision;
            this.visionProjectionPrecision = visionProjectionPrecision;
            this.crossAttentionPrecision = crossAttentionPrecision;
            this.textCasts = textCasts;
            this.visionCasts = visionCasts;
            this.audioCasts = audioCasts;
            this.fusionCasts = fusionCasts;
            this.crossModalCasts = crossModalCasts;
            this.totalTimeMs = totalTimeMs;
        }

        public long totalCasts() {
            return textCasts + visionCasts + audioCasts + fusionCasts + crossModalCasts;
        }

        @Override
        public String toString() {
            return String.format(
                    "MultiModalAmpStats{text=%s, vision=%s, audio=%s, fusion=%s, " +
                    "precision=%s/%s/%s, casts=%d, time=%.2fs}",
                    textPrecision, visionPrecision, audioPrecision, fusionPrecision,
                    visionEncoderPrecision, visionProjectionPrecision, crossAttentionPrecision,
                    totalCasts(), totalTimeMs / 1000.0);
        }
    }

    /**
     * Builder for MultiModalAmp.
     */
    public static class Builder {
        private AmpPrecision textPrecision = AmpPrecision.FP16;
        private AmpPrecision visionPrecision = AmpPrecision.FP16;
        private AmpPrecision audioPrecision = AmpPrecision.FP16;
        private AmpPrecision fusionPrecision = AmpPrecision.FP32;
        private AmpPrecision embeddingPrecision = AmpPrecision.FP16;
        private AmpPrecision visionEncoderPrecision = AmpPrecision.FP16;
        private AmpPrecision visionProjectionPrecision = AmpPrecision.FP16;
        private AmpPrecision crossAttentionPrecision = AmpPrecision.FP16;

        public Builder textPrecision(AmpPrecision textPrecision) {
            this.textPrecision = textPrecision;
            return this;
        }

        public Builder visionPrecision(AmpPrecision visionPrecision) {
            this.visionPrecision = visionPrecision;
            return this;
        }

        public Builder audioPrecision(AmpPrecision audioPrecision) {
            this.audioPrecision = audioPrecision;
            return this;
        }

        public Builder fusionPrecision(AmpPrecision fusionPrecision) {
            this.fusionPrecision = fusionPrecision;
            return this;
        }

        public Builder embeddingPrecision(AmpPrecision embeddingPrecision) {
            this.embeddingPrecision = embeddingPrecision;
            return this;
        }

        public Builder visionEncoderPrecision(AmpPrecision visionEncoderPrecision) {
            this.visionEncoderPrecision = visionEncoderPrecision;
            return this;
        }

        public Builder visionProjectionPrecision(AmpPrecision visionProjectionPrecision) {
            this.visionProjectionPrecision = visionProjectionPrecision;
            return this;
        }

        public Builder crossAttentionPrecision(AmpPrecision crossAttentionPrecision) {
            this.crossAttentionPrecision = crossAttentionPrecision;
            return this;
        }

        /**
         * Configure for BF16 training.
         */
        public Builder bf16() {
            this.textPrecision = AmpPrecision.BF16;
            this.visionPrecision = AmpPrecision.BF16;
            this.audioPrecision = AmpPrecision.BF16;
            this.visionEncoderPrecision = AmpPrecision.BF16;
            this.visionProjectionPrecision = AmpPrecision.BF16;
            this.crossAttentionPrecision = AmpPrecision.BF16;
            this.embeddingPrecision = AmpPrecision.BF16;
            return this;
        }

        /**
         * Configure for FP16 training.
         */
        public Builder fp16() {
            this.textPrecision = AmpPrecision.FP16;
            this.visionPrecision = AmpPrecision.FP16;
            this.audioPrecision = AmpPrecision.FP16;
            this.visionEncoderPrecision = AmpPrecision.FP16;
            this.visionProjectionPrecision = AmpPrecision.FP16;
            this.crossAttentionPrecision = AmpPrecision.FP16;
            this.embeddingPrecision = AmpPrecision.FP16;
            return this;
        }

        public MultiModalAmp build() {
            return new MultiModalAmp(this);
        }
    }
}
