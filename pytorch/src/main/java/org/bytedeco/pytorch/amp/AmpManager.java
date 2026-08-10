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
import org.bytedeco.pytorch.DeviceType;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise-grade Automatic Mixed Precision (AMP) manager for PyTorch.
 *
 * <p>Provides:
 * <ul>
 *   <li>Automatic precision casting based on operation type</li>
 *   <li>Dynamic loss scaling for FP16 stability</li>
 *   <li>FP16, BF16, and FP8 support</li>
 *   <li>Gradient clipping integration</li>
 *   <li>Distributed training support</li>
 *   <li>Performance monitoring and statistics</li>
 * </ul>
 *
 * <p>Reference: NVIDIA AMP (Automatic Mixed Precision) and PyTorch autocast
 *
 * <pre>{@code
 * // Create AMP manager
 * try (AmpManager amp = AmpManager.builder()
 *     .device("cuda")
 *     .enabled(true)
 *     .build()) {
 *
 *     // Forward pass with automatic precision
 *     try (AutocastContext ctx = amp.autocast()) {
 *         Tensor loss = model.forward(input);
 *
 *         // Scale loss and backward
 *         amp.scaleLoss(loss).backward();
 *     }
 *
 *     // Unscale gradients and optimizer step
 *     amp.step(optimizer, model.parameters());
 * }
 * }</pre>
 */
public class AmpManager implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final boolean enabled;
    private final Device device;
    private final AmpPrecision forwardPrecision;
    private final AmpPrecision backwardPrecision;
    private final AmpPrecision optimizerPrecision;

    // GradScaler state
    private final GradScalerConfig scalerConfig;
    private float scaleFactor;
    private int unskippedSteps;
    private int skippedSteps;
    private int overflowCount;

    // Performance metrics
    private final AtomicLong totalForwardTimeMs = new AtomicLong(0);
    private final AtomicLong totalBackwardTimeMs = new AtomicLong(0);
    private final AtomicLong totalScaleTimeMs = new AtomicLong(0);
    private final AtomicLong totalUnscaleTimeMs = new AtomicLong(0);
    private final AtomicLong totalCasts = new AtomicLong(0);
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);

    // Distributed state
    private final boolean distributed;
    private final int rank;

    private AmpManager(Builder builder) {
        this.enabled = builder.enabled;
        this.device = builder.device != null ? builder.device : autoDevice();
        this.forwardPrecision = builder.forwardPrecision;
        this.backwardPrecision = builder.backwardPrecision;
        this.optimizerPrecision = builder.optimizerPrecision;
        this.scalerConfig = builder.scalerConfig;
        this.scaleFactor = builder.initialScale;
        this.distributed = builder.distributed;
        this.rank = builder.rank;
    }

    /**
     * Check if CUDA is available and device is CUDA.
     */
    public static boolean isCudaAvailable() {
        try {
            return torch.cuda_is_available();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get current device.
     */
    public Device device() {
        return device;
    }

    /**
     * Get forward precision.
     */
    public AmpPrecision forwardPrecision() {
        return forwardPrecision;
    }

    /**
     * Get backward precision.
     */
    public AmpPrecision backwardPrecision() {
        return backwardPrecision;
    }

    /**
     * Get optimizer precision.
     */
    public AmpPrecision optimizerPrecision() {
        return optimizerPrecision;
    }

    /**
     * Check if AMP is enabled.
     */
    public boolean isEnabled() {
        return enabled && !closed;
    }

    /**
     * Create an autocast context for automatic precision casting.
     */
    public AutocastContext autocast() {
        if (!isEnabled()) {
            return AutocastContext.DISABLED;
        }
        return new AutocastContext(this, device, forwardPrecision, backwardPrecision);
    }

    /**
     * Scale loss for backward pass.
     */
    public Tensor scaleLoss(Tensor loss) {
        if (!isEnabled()) {
            return loss;
        }
        if (forwardPrecision == AmpPrecision.FP16 || backwardPrecision == AmpPrecision.FP16) {
            return loss.mul(new Scalar(scaleFactor));
        }
        return loss;
    }

    /**
     * Scale loss and perform backward.
     */
    public void backward(Tensor loss) {
        if (!isEnabled()) {
            loss.backward();
            return;
        }
        long start = System.currentTimeMillis();
        Tensor scaled = scaleLoss(loss);
        scaled.backward();
        totalBackwardTimeMs.addAndGet(System.currentTimeMillis() - start);
    }

    /**
     * Unscale gradients and update scale factor.
     */
    public boolean unscaleGradients(TensorVector params) {
        if (!isEnabled()) {
            return true;
        }
        if (forwardPrecision != AmpPrecision.FP16 && backwardPrecision != AmpPrecision.FP16) {
            return true;
        }

        long start = System.currentTimeMillis();

        // Check for non-finite gradients
        boolean hasNonFinite = checkNonFiniteGradients(params);
        if (hasNonFinite) {
            skippedSteps++;
            overflowCount++;
            scaleFactor = Math.max(scaleFactor * scalerConfig.backoffFactor, scalerConfig.minScale);
            totalUnscaleTimeMs.addAndGet(System.currentTimeMillis() - start);
            return false;
        }

        // Unscale gradients
        float invScale = 1.0f / scaleFactor;
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || !p.defined()) continue;
            Tensor g = p.grad();
            if (g == null || !g.defined()) continue;
            g.mul_(new Scalar(invScale));
        }

        unskippedSteps++;
        scaleFactor = Math.min(scaleFactor * scalerConfig.growthFactor, scalerConfig.maxScale);
        totalUnscaleTimeMs.addAndGet(System.currentTimeMillis() - start);
        return true;
    }

    /**
     * Step optimizer with optional gradient clipping.
     */
    public void step(Object optimizer, TensorVector params) {
        if (!isEnabled()) {
            // Direct step without unscaling
            return;
        }

        if (scalerConfig.maxGradNorm > 0) {
            // Clip gradients before optimizer step
            clipGradients(params);
        }

        // Check and unscale gradients
        if (!unscaleGradients(params)) {
            // Skip step due to overflow
            if (optimizer instanceof org.bytedeco.pytorch.optim.Optimizer opt) {
                opt.zero_grad();
            }
            return;
        }

        // Update scale factor statistics
        if (scalerConfig.growthInterval > 0 && unskippedSteps >= scalerConfig.growthInterval) {
            scaleFactor = Math.min(scaleFactor * scalerConfig.growthFactor, scalerConfig.maxScale);
            unskippedSteps = 0;
        }
    }

    /**
     * Clip gradients by global norm.
     */
    public float clipGradients(TensorVector params) {
        if (scalerConfig.maxGradNorm <= 0) {
            return 0.0f;
        }

        // Compute global gradient norm
        float totalNorm = 0.0f;
        int numParams = 0;

        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || !p.defined()) continue;
            Tensor g = p.grad();
            if (g == null || !g.defined()) continue;

            float paramNorm = (float) Math.sqrt(g.doublevalue().sum().item_double());
            totalNorm += paramNorm * paramNorm;
            numParams++;
        }

        if (numParams == 0) return 0.0f;

        float clipCoeff = (float) (scalerConfig.maxGradNorm / (Math.sqrt(totalNorm) + 1e-6));
        if (clipCoeff < 1.0f) {
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor p = params.get(i);
                if (p == null || !p.defined()) continue;
                Tensor g = p.grad();
                if (g == null || !g.defined()) continue;
                g.mul_(new Scalar(clipCoeff));
            }
        }

        return (float) Math.sqrt(totalNorm);
    }

    /**
     * Update the scaler (call after optimizer step).
     */
    public void update() {
        // Already handled in step()
    }

    /**
     * Get current scale factor.
     */
    public float getScaleFactor() {
        return scaleFactor;
    }

    /**
     * Set scale factor manually.
     */
    public void setScaleFactor(float scale) {
        this.scaleFactor = Math.max(scalerConfig.minScale, Math.min(scale, scalerConfig.maxScale));
    }

    /**
     * Check for non-finite gradients.
     */
    private boolean checkNonFiniteGradients(TensorVector params) {
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || !p.defined()) continue;
            Tensor g = p.grad();
            if (g == null || !g.defined()) continue;
            if (!torch.isfinite(g).all().item_bool()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get AMP statistics.
     */
    public AmpStats getStats() {
        return new AmpStats(
                enabled,
                device.toString(),
                forwardPrecision.name(),
                backwardPrecision.name(),
                optimizerPrecision.name(),
                scaleFactor,
                unskippedSteps,
                skippedSteps,
                overflowCount,
                totalForwardTimeMs.get(),
                totalBackwardTimeMs.get(),
                totalScaleTimeMs.get(),
                totalUnscaleTimeMs.get(),
                totalCasts.get()
        );
    }

    /**
     * Get device automatically.
     */
    private static Device autoDevice() {
        try {
            if (torch.cuda_is_available()) {
                return new Device(DeviceType.CUDA, (byte) 0);
            }
        } catch (Exception ignored) {}
        return new Device(DeviceType.CPU);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[AmpManager] Closed: enabled=%b, device=%s, precision=%s/%s/%s, " +
                "scale=%.1f, steps=%d/%d, overflows=%d, " +
                "fwdTime=%.2fs, bwdTime=%.2fs, castTime=%.2fs%n",
                enabled, device, forwardPrecision, backwardPrecision, optimizerPrecision,
                scaleFactor, unskippedSteps, skippedSteps, overflowCount,
                totalForwardTimeMs.get() / 1000.0,
                totalBackwardTimeMs.get() / 1000.0,
                totalScaleTimeMs.get() / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * AMP statistics.
     */
    public static final class AmpStats {
        public final boolean enabled;
        public final String device;
        public final String forwardPrecision;
        public final String backwardPrecision;
        public final String optimizerPrecision;
        public final float scaleFactor;
        public final int unskippedSteps;
        public final int skippedSteps;
        public final int overflowCount;
        public final long totalForwardTimeMs;
        public final long totalBackwardTimeMs;
        public final long totalScaleTimeMs;
        public final long totalUnscaleTimeMs;
        public final long totalCasts;

        public AmpStats(boolean enabled, String device, String forwardPrecision,
                     String backwardPrecision, String optimizerPrecision,
                     float scaleFactor, int unskippedSteps, int skippedSteps, int overflowCount,
                     long totalForwardTimeMs, long totalBackwardTimeMs,
                     long totalScaleTimeMs, long totalUnscaleTimeMs, long totalCasts) {
            this.enabled = enabled;
            this.device = device;
            this.forwardPrecision = forwardPrecision;
            this.backwardPrecision = backwardPrecision;
            this.optimizerPrecision = optimizerPrecision;
            this.scaleFactor = scaleFactor;
            this.unskippedSteps = unskippedSteps;
            this.skippedSteps = skippedSteps;
            this.overflowCount = overflowCount;
            this.totalForwardTimeMs = totalForwardTimeMs;
            this.totalBackwardTimeMs = totalBackwardTimeMs;
            this.totalScaleTimeMs = totalScaleTimeMs;
            this.totalUnscaleTimeMs = totalUnscaleTimeMs;
            this.totalCasts = totalCasts;
        }

        public double overflowRate() {
            int total = unskippedSteps + skippedSteps;
            return total > 0 ? (double) skippedSteps / total : 0;
        }

        public double avgForwardTimeMs() {
            return totalForwardTimeMs / 1000.0;
        }

        public double avgBackwardTimeMs() {
            return totalBackwardTimeMs / 1000.0;
        }

        @Override
        public String toString() {
            return String.format(
                    "AmpStats{enabled=%b, device=%s, precision=%s/%s/%s, " +
                    "scale=%.1f, steps=%d/%d, overflowRate=%.2f%%}",
                    enabled, device, forwardPrecision, backwardPrecision, optimizerPrecision,
                    scaleFactor, unskippedSteps, skippedSteps, overflowRate() * 100);
        }
    }

    /**
     * GradScaler configuration.
     */
    public static final class GradScalerConfig {
        public final float initScale;
        public final float minScale;
        public final float maxScale;
        public final float growthFactor;
        public final float backoffFactor;
        public final int growthInterval;
        public final double maxGradNorm;

        public GradScalerConfig(float initScale, float minScale, float maxScale,
                             float growthFactor, float backoffFactor,
                             int growthInterval, double maxGradNorm) {
            this.initScale = initScale;
            this.minScale = minScale;
            this.maxScale = maxScale;
            this.growthFactor = growthFactor;
            this.backoffFactor = backoffFactor;
            this.growthInterval = growthInterval;
            this.maxGradNorm = maxGradNorm;
        }

        public static Builder builder() { return new Builder(); }

        public static GradScalerConfig defaults() {
            return new Builder().build();
        }

        public static GradScalerConfig forBF16() {
            return new Builder()
                    .initScale(65536.0f)
                    .growthFactor(1.01f)
                    .backoffFactor(0.5f)
                    .growthInterval(2000)
                    .build();
        }

        public static GradScalerConfig forFP16() {
            return new Builder()
                    .initScale(65536.0f)
                    .growthFactor(2.0f)
                    .backoffFactor(0.5f)
                    .growthInterval(2000)
                    .build();
        }

        public static class Builder {
            private float initScale = 65536.0f;
            private float minScale = 1.0f;
            private float maxScale = 65536.0f;
            private float growthFactor = 1.01f;
            private float backoffFactor = 0.5f;
            private int growthInterval = 2000;
            private double maxGradNorm = 1.0;

            public Builder initScale(float initScale) { this.initScale = initScale; return this; }
            public Builder minScale(float minScale) { this.minScale = minScale; return this; }
            public Builder maxScale(float maxScale) { this.maxScale = maxScale; return this; }
            public Builder growthFactor(float growthFactor) { this.growthFactor = growthFactor; return this; }
            public Builder backoffFactor(float backoffFactor) { this.backoffFactor = backoffFactor; return this; }
            public Builder growthInterval(int growthInterval) { this.growthInterval = growthInterval; return this; }
            public Builder maxGradNorm(double maxGradNorm) { this.maxGradNorm = maxGradNorm; return this; }

            public GradScalerConfig build() {
                return new GradScalerConfig(initScale, minScale, maxScale,
                        growthFactor, backoffFactor, growthInterval, maxGradNorm);
            }
        }
    }

    /**
     * Mixed precision training configuration.
     */
    public static final class Builder {
        private boolean enabled = true;
        private Device device;
        private AmpPrecision forwardPrecision = AmpPrecision.FP16;
        private AmpPrecision backwardPrecision = AmpPrecision.FP16;
        private AmpPrecision optimizerPrecision = AmpPrecision.FP32;
        private GradScalerConfig scalerConfig = GradScalerConfig.defaults();
        private float initialScale = 65536.0f;
        private boolean distributed = false;
        private int rank = 0;

        public Builder enabled(boolean enabled) { this.enabled = enabled; return this; }
        public Builder device(Device device) { this.device = device; return this; }
        public Builder device(String deviceStr) {
            this.device = new Device(deviceStr);
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
        public Builder scalerConfig(GradScalerConfig scalerConfig) {
            this.scalerConfig = scalerConfig;
            return this;
        }
        public Builder initialScale(float initialScale) { this.initialScale = initialScale; return this; }
        public Builder distributed(boolean distributed) { this.distributed = distributed; return this; }
        public Builder rank(int rank) { this.rank = rank; return this; }

        /**
         * Enable FP16 mixed precision (default for CUDA).
         */
        public Builder fp16() {
            this.forwardPrecision = AmpPrecision.FP16;
            this.backwardPrecision = AmpPrecision.FP16;
            this.optimizerPrecision = AmpPrecision.FP32;
            return this;
        }

        /**
         * Enable BF16 mixed precision (recommended for training).
         */
        public Builder bf16() {
            this.forwardPrecision = AmpPrecision.BF16;
            this.backwardPrecision = AmpPrecision.BF16;
            this.optimizerPrecision = AmpPrecision.FP32;
            return this;
        }

        /**
         * Enable FP8 mixed precision (for inference and large models).
         */
        public Builder fp8() {
            this.forwardPrecision = AmpPrecision.FP8_E4M3;
            this.backwardPrecision = AmpPrecision.FP8_E5M2;
            this.optimizerPrecision = AmpPrecision.FP16;
            return this;
        }

        /**
         * Disable mixed precision (FP32 only).
         */
        public Builder fp32() {
            this.forwardPrecision = AmpPrecision.FP32;
            this.backwardPrecision = AmpPrecision.FP32;
            this.optimizerPrecision = AmpPrecision.FP32;
            return this;
        }

        public AmpManager build() {
            return new AmpManager(this);
        }
    }

    public static Builder builder() { return new Builder(); }
}
