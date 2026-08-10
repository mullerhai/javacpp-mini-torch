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
package org.bytedeco.pytorch.amp.trainer;

import org.bytedeco.pytorch.amp.*;
import org.bytedeco.pytorch.amp.config.AmpConfig;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.function.Consumer;

/**
 * Base class for AMP-enabled trainers.
 *
 * <p>Provides mixed precision training support for all trainers:
 * <ul>
 *   <li>Automatic precision management</li>
 *   <li>GradScaler integration</li>
 *   <li>Distributed training support</li>
 *   <li>Performance monitoring</li>
 * </ul>
 *
 * <p>Reference: PyTorch AMP, DeepSpeed, and Megatron-LM
 */
public abstract class BaseAmpTrainer implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // AMP configuration and components
    protected final AmpConfig ampConfig;
    protected AmpManager ampManager;
    protected GradScaler gradScaler;
    protected AutocastContext autocast;

    // Performance metrics
    protected long totalForwardTimeMs = 0;
    protected long totalBackwardTimeMs = 0;
    protected long totalStepTimeMs = 0;
    protected int totalSteps = 0;
    protected int totalOverflows = 0;

    /**
     * Create BaseAmpTrainer with configuration.
     */
    protected BaseAmpTrainer(AmpConfig ampConfig) {
        this.ampConfig = ampConfig != null ? ampConfig : AmpConfig.defaults();
        initializeAmp();
    }

    /**
     * Create BaseAmpTrainer with default configuration.
     */
    protected BaseAmpTrainer() {
        this(AmpConfig.defaults());
    }

    /**
     * Initialize AMP components.
     */
    private void initializeAmp() {
        // Initialize GradScaler
        GradScaler.Builder scalerBuilder = GradScaler.builder()
                .enabled(ampConfig.enabled())
                .initScale(ampConfig.initScale())
                .minScale(ampConfig.minScale())
                .maxScale(ampConfig.maxScale())
                .growthFactor(ampConfig.growthFactor())
                .backoffFactor(ampConfig.backoffFactor())
                .growthInterval(ampConfig.growthInterval());

        if (ampConfig.forwardPrecision() == AmpPrecision.BF16) {
            scalerBuilder.bf16();
        } else {
            scalerBuilder.fp16();
        }

        if (ampConfig.distributed()) {
            scalerBuilder.distributed(true).worldSize(ampConfig.worldSize()).rank(ampConfig.rank());
        }

        this.gradScaler = scalerBuilder.build();

        // Initialize AmpManager
        AmpManager.Builder managerBuilder = AmpManager.builder()
                .enabled(ampConfig.enabled())
                .forwardPrecision(ampConfig.forwardPrecision())
                .backwardPrecision(ampConfig.backwardPrecision())
                .optimizerPrecision(ampConfig.optimizerPrecision())
                .initialScale(ampConfig.initScale())
                .distributed(ampConfig.distributed())
                .rank(ampConfig.rank());

        this.ampManager = managerBuilder.build();

        // Initialize Autocast
        Device device = new Device("cuda");
        this.autocast = ampManager.autocast();
    }

    /**
     * Scale loss for mixed precision training.
     */
    protected Tensor scaleLoss(Tensor loss) {
        if (!ampConfig.enabled()) {
            return loss;
        }
        return gradScaler.scale(loss);
    }

    /**
     * Unscale gradients and check for overflow.
     */
    protected boolean unscaleAndCheck(TensorVector params) {
        if (!ampConfig.enabled()) {
            return true;
        }
        boolean valid = gradScaler.unscaleAndCheck(params);
        if (!valid) {
            totalOverflows++;
        }
        return valid;
    }

    /**
     * Perform a training step with AMP.
     */
    protected void ampStep(Tensor loss, Object optimizer, TensorVector params,
                          Consumer<TensorVector> optimizerStep) {
        long stepStart = System.currentTimeMillis();

        // Scale loss
        Tensor scaledLoss = scaleLoss(loss);

        // Backward pass
        long bwdStart = System.currentTimeMillis();
        scaledLoss.backward();
        totalBackwardTimeMs += System.currentTimeMillis() - bwdStart;

        // Unscale and check
        if (unscaleAndCheck(params)) {
            // Clip gradients if configured
            if (ampConfig.maxGradNorm() > 0) {
                ampManager.clipGradients(params);
            }

            // Optimizer step
            if (optimizerStep != null) {
                optimizerStep.accept(params);
            }
        } else {
            // Overflow detected - zero gradients
            if (optimizer instanceof org.bytedeco.pytorch.optim.Optimizer opt) {
                opt.zero_grad();
            }
        }

        // Update scaler
        gradScaler.update();

        totalStepTimeMs += System.currentTimeMillis() - stepStart;
        totalSteps++;
    }

    /**
     * Cast tensor to appropriate precision for forward pass.
     */
    protected Tensor castForward(Tensor tensor) {
        if (!ampConfig.enabled() || tensor == null || !tensor.defined()) {
            return tensor;
        }
        return autocast.cast(tensor);
    }

    /**
     * Cast tensor to FP32 for loss computation.
     */
    protected Tensor castLoss(Tensor tensor) {
        if (!ampConfig.enabled() || tensor == null || !tensor.defined()) {
            return tensor;
        }
        return autocast.castFp32(tensor);
    }

    /**
     * Get current scale factor.
     */
    public float getScaleFactor() {
        return gradScaler != null ? gradScaler.getScaleFactor() : 1.0f;
    }

    /**
     * Get GradScaler.
     */
    public GradScaler getGradScaler() {
        return gradScaler;
    }

    /**
     * Get AmpManager.
     */
    public AmpManager getAmpManager() {
        return ampManager;
    }

    /**
     * Get AutocastContext.
     */
    public AutocastContext getAutocast() {
        return autocast;
    }

    /**
     * Check if AMP is enabled.
     */
    public boolean isAmpEnabled() {
        return ampConfig.enabled() && !closed;
    }

    /**
     * Get AMP statistics.
     */
    public AmpTrainerStats getStats() {
        return new AmpTrainerStats(
                ampConfig.enabled(),
                ampConfig.forwardPrecision().name(),
                totalSteps,
                totalOverflows,
                totalForwardTimeMs,
                totalBackwardTimeMs,
                totalStepTimeMs,
                gradScaler != null ? gradScaler.getStats() : null
        );
    }

    /**
     * Reset statistics.
     */
    public void resetStats() {
        totalForwardTimeMs = 0;
        totalBackwardTimeMs = 0;
        totalStepTimeMs = 0;
        totalSteps = 0;
        totalOverflows = 0;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Close AMP components
        if (autocast != null) {
            try { autocast.close(); } catch (Exception ignored) {}
        }
        if (ampManager != null) {
            try { ampManager.close(); } catch (Exception ignored) {}
        }
        if (gradScaler != null) {
            try { gradScaler.close(); } catch (Exception ignored) {}
        }

        System.out.printf(
                "[BaseAmpTrainer] Closed: enabled=%b, precision=%s, steps=%d, " +
                "overflows=%d, fwdTime=%.2fs, bwdTime=%.2fs, stepTime=%.2fs%n",
                ampConfig.enabled(), ampConfig.forwardPrecision(),
                totalSteps, totalOverflows,
                totalForwardTimeMs / 1000.0,
                totalBackwardTimeMs / 1000.0,
                totalStepTimeMs / 1000.0);
    }

    public boolean isClosed() { return closed; }

    /**
     * AMP Trainer statistics.
     */
    public static class AmpTrainerStats {
        public final boolean ampEnabled;
        public final String precision;
        public final int totalSteps;
        public final int totalOverflows;
        public final long totalForwardTimeMs;
        public final long totalBackwardTimeMs;
        public final long totalStepTimeMs;
        public final GradScaler.GradScalerStats gradScalerStats;

        public AmpTrainerStats(boolean ampEnabled, String precision, int totalSteps,
                           int totalOverflows, long totalForwardTimeMs,
                           long totalBackwardTimeMs, long totalStepTimeMs,
                           GradScaler.GradScalerStats gradScalerStats) {
            this.ampEnabled = ampEnabled;
            this.precision = precision;
            this.totalSteps = totalSteps;
            this.totalOverflows = totalOverflows;
            this.totalForwardTimeMs = totalForwardTimeMs;
            this.totalBackwardTimeMs = totalBackwardTimeMs;
            this.totalStepTimeMs = totalStepTimeMs;
            this.gradScalerStats = gradScalerStats;
        }

        public double overflowRate() {
            return totalSteps > 0 ? (double) totalOverflows / totalSteps : 0;
        }

        public double avgStepTimeMs() {
            return totalSteps > 0 ? (double) totalStepTimeMs / totalSteps : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "AmpTrainerStats{enabled=%b, precision=%s, steps=%d, " +
                    "overflowRate=%.2f%%, avgStepTime=%.2fms}",
                    ampEnabled, precision, totalSteps,
                    overflowRate() * 100, avgStepTimeMs());
        }
    }
}
