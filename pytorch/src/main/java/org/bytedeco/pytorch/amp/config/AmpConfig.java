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
package org.bytedeco.pytorch.amp.config;

import org.bytedeco.pytorch.amp.AmpPrecision;

/**
 * Configuration for Automatic Mixed Precision (AMP) training.
 *
 * <p>Provides a flexible configuration system for:
 * <ul>
 *   <li>Precision selection (FP16, BF16, FP8)</li>
 *   <li>GradScaler settings</li>
 *   <li>Distributed training settings</li>
 *   <li>Performance tuning options</li>
 * </ul>
 *
 * <p>Reference: PyTorch AMP documentation, NVIDIA AMP guidelines
 */
public class AmpConfig {

    // Version
    public static final String VERSION = "2.0";

    // Precision settings
    private final boolean enabled;
    private final AmpPrecision forwardPrecision;
    private final AmpPrecision backwardPrecision;
    private final AmpPrecision optimizerPrecision;

    // GradScaler settings
    private final float initScale;
    private final float minScale;
    private final float maxScale;
    private final float growthFactor;
    private final float backoffFactor;
    private final int growthInterval;

    // Gradient clipping
    private final double maxGradNorm;

    // Distributed settings
    private final boolean distributed;
    private final int worldSize;
    private final int rank;
    private final boolean syncBatchNorm;
    private final boolean averageGradients;

    // Performance settings
    private final boolean useFlashAttention;
    private final boolean useMemoryEfficientAttention;
    private final boolean enableKernelFallback;

    private AmpConfig(Builder builder) {
        this.enabled = builder.enabled;
        this.forwardPrecision = builder.forwardPrecision;
        this.backwardPrecision = builder.backwardPrecision;
        this.optimizerPrecision = builder.optimizerPrecision;
        this.initScale = builder.initScale;
        this.minScale = builder.minScale;
        this.maxScale = builder.maxScale;
        this.growthFactor = builder.growthFactor;
        this.backoffFactor = builder.backoffFactor;
        this.growthInterval = builder.growthInterval;
        this.maxGradNorm = builder.maxGradNorm;
        this.distributed = builder.distributed;
        this.worldSize = builder.worldSize;
        this.rank = builder.rank;
        this.syncBatchNorm = builder.syncBatchNorm;
        this.averageGradients = builder.averageGradients;
        this.useFlashAttention = builder.useFlashAttention;
        this.useMemoryEfficientAttention = builder.useMemoryEfficientAttention;
        this.enableKernelFallback = builder.enableKernelFallback;
    }

    /**
     * Create AmpConfig with default settings.
     */
    public static AmpConfig defaults() {
        return new Builder().build();
    }

    /**
     * Create AmpConfig for FP16 training.
     */
    public static AmpConfig fp16() {
        return new Builder().fp16().build();
    }

    /**
     * Create AmpConfig for BF16 training.
     */
    public static AmpConfig bf16() {
        return new Builder().bf16().build();
    }

    /**
     * Create AmpConfig for FP8 inference.
     */
    public static AmpConfig fp8() {
        return new Builder().fp8().build();
    }

    /**
     * Create AmpConfig for distributed training.
     */
    public static AmpConfig distributed(int worldSize, int rank) {
        return new Builder()
                .distributed(true)
                .worldSize(worldSize)
                .rank(rank)
                .bf16()
                .build();
    }

    /**
     * Create AmpConfig for LLaMA training.
     */
    public static AmpConfig forLLaMA() {
        return new Builder()
                .bf16()
                .initScale(65536.0f)
                .growthFactor(1.01f)
                .backoffFactor(0.5f)
                .growthInterval(2000)
                .maxGradNorm(1.0)
                .useFlashAttention(true)
                .build();
    }

    /**
     * Create AmpConfig for GPT training.
     */
    public static AmpConfig forGPT() {
        return new Builder()
                .fp16()
                .initScale(65536.0f)
                .growthFactor(2.0f)
                .backoffFactor(0.5f)
                .growthInterval(2000)
                .maxGradNorm(1.0)
                .useFlashAttention(true)
                .build();
    }

    /**
     * Create AmpConfig for training large models (Meta/LLaMA style).
     */
    public static AmpConfig forLargeModels() {
        return new Builder()
                .bf16()
                .initScale(65536.0f)
                .growthFactor(1.01f)
                .backoffFactor(0.5f)
                .growthInterval(2000)
                .maxGradNorm(1.0)
                .useFlashAttention(true)
                .useMemoryEfficientAttention(true)
                .build();
    }

    // Getters
    public boolean enabled() { return enabled; }
    public AmpPrecision forwardPrecision() { return forwardPrecision; }
    public AmpPrecision backwardPrecision() { return backwardPrecision; }
    public AmpPrecision optimizerPrecision() { return optimizerPrecision; }
    public float initScale() { return initScale; }
    public float minScale() { return minScale; }
    public float maxScale() { return maxScale; }
    public float growthFactor() { return growthFactor; }
    public float backoffFactor() { return backoffFactor; }
    public int growthInterval() { return growthInterval; }
    public double maxGradNorm() { return maxGradNorm; }
    public boolean distributed() { return distributed; }
    public int worldSize() { return worldSize; }
    public int rank() { return rank; }
    public boolean syncBatchNorm() { return syncBatchNorm; }
    public boolean averageGradients() { return averageGradients; }
    public boolean useFlashAttention() { return useFlashAttention; }
    public boolean useMemoryEfficientAttention() { return useMemoryEfficientAttention; }
    public boolean enableKernelFallback() { return enableKernelFallback; }

    @Override
    public String toString() {
        return String.format(
                "AmpConfig{enabled=%b, precision=%s/%s/%s, " +
                "scale=[%.1f, %.1f, %.1f], growth=%.2f/%.2f, interval=%d, " +
                "gradNorm=%.2f, distributed=%b, flashAttention=%b}",
                enabled, forwardPrecision, backwardPrecision, optimizerPrecision,
                initScale, minScale, maxScale, growthFactor, backoffFactor, growthInterval,
                maxGradNorm, distributed, useFlashAttention);
    }

    /**
     * Builder for AmpConfig.
     */
    public static class Builder {
        // Precision settings
        private boolean enabled = true;
        private AmpPrecision forwardPrecision = AmpPrecision.FP16;
        private AmpPrecision backwardPrecision = AmpPrecision.FP16;
        private AmpPrecision optimizerPrecision = AmpPrecision.FP32;

        // GradScaler settings
        private float initScale = 65536.0f;
        private float minScale = 1.0f;
        private float maxScale = 65536.0f;
        private float growthFactor = 1.01f;
        private float backoffFactor = 0.5f;
        private int growthInterval = 2000;

        // Gradient clipping
        private double maxGradNorm = 1.0;

        // Distributed settings
        private boolean distributed = false;
        private int worldSize = 1;
        private int rank = 0;
        private boolean syncBatchNorm = true;
        private boolean averageGradients = true;

        // Performance settings
        private boolean useFlashAttention = false;
        private boolean useMemoryEfficientAttention = false;
        private boolean enableKernelFallback = true;

        public Builder enabled(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public Builder forwardPrecision(AmpPrecision forwardPrecision) {
            this.forwardPrecision = forwardPrecision;
            return this;
        }

        public Builder backwardPrecision(AmpPrecision backwardPrecision) {
            this.backwardPrecision = backwardPrecision;
            return this;
        }

        public Builder optimizerPrecision(AmpPrecision optimizerPrecision) {
            this.optimizerPrecision = optimizerPrecision;
            return this;
        }

        public Builder initScale(float initScale) {
            this.initScale = initScale;
            return this;
        }

        public Builder minScale(float minScale) {
            this.minScale = minScale;
            return this;
        }

        public Builder maxScale(float maxScale) {
            this.maxScale = maxScale;
            return this;
        }

        public Builder growthFactor(float growthFactor) {
            this.growthFactor = growthFactor;
            return this;
        }

        public Builder backoffFactor(float backoffFactor) {
            this.backoffFactor = backoffFactor;
            return this;
        }

        public Builder growthInterval(int growthInterval) {
            this.growthInterval = growthInterval;
            return this;
        }

        public Builder maxGradNorm(double maxGradNorm) {
            this.maxGradNorm = maxGradNorm;
            return this;
        }

        public Builder distributed(boolean distributed) {
            this.distributed = distributed;
            return this;
        }

        public Builder worldSize(int worldSize) {
            this.worldSize = worldSize;
            return this;
        }

        public Builder rank(int rank) {
            this.rank = rank;
            return this;
        }

        public Builder syncBatchNorm(boolean syncBatchNorm) {
            this.syncBatchNorm = syncBatchNorm;
            return this;
        }

        public Builder averageGradients(boolean averageGradients) {
            this.averageGradients = averageGradients;
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

        public Builder enableKernelFallback(boolean enableKernelFallback) {
            this.enableKernelFallback = enableKernelFallback;
            return this;
        }

        /**
         * Configure for FP16 training (default PyTorch).
         */
        public Builder fp16() {
            this.forwardPrecision = AmpPrecision.FP16;
            this.backwardPrecision = AmpPrecision.FP16;
            this.optimizerPrecision = AmpPrecision.FP32;
            this.growthFactor = 2.0f;
            return this;
        }

        /**
         * Configure for BF16 training (recommended for stability).
         */
        public Builder bf16() {
            this.forwardPrecision = AmpPrecision.BF16;
            this.backwardPrecision = AmpPrecision.BF16;
            this.optimizerPrecision = AmpPrecision.FP32;
            this.growthFactor = 1.01f;
            return this;
        }

        /**
         * Configure for FP8 training (for inference and large models).
         */
        public Builder fp8() {
            this.forwardPrecision = AmpPrecision.FP8_E4M3;
            this.backwardPrecision = AmpPrecision.FP8_E5M2;
            this.optimizerPrecision = AmpPrecision.FP16;
            this.growthFactor = 1.01f;
            return this;
        }

        /**
         * Disable mixed precision (FP32 only).
         */
        public Builder fp32() {
            this.enabled = false;
            return this;
        }

        public AmpConfig build() {
            return new AmpConfig(this);
        }
    }
}
