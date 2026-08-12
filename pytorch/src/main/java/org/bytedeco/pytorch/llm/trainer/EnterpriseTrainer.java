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
package org.bytedeco.pytorch.llm.trainer;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Tensor;

import java.io.Closeable;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise-grade training interface for LLM models.
 *
 * <p>Provides:
 * <ul>
 *   <li>Unified training interface for all trainer types</li>
 *   <li>Comprehensive progress tracking and callbacks</li>
 *   <li>Performance monitoring and metrics</li>
 *   <li>Error handling and recovery</li>
 *   <li>Multi-modal support</li>
 * </ul>
 *
 * <p>Reference: HuggingFace Trainer, DeepSpeed Trainer
 *
 * <pre>{@code
 * try (EnterpriseTrainer trainer = EnterpriseTrainer.builder()
 *     .model(model)
 *     .optimizer(optimizer)
 *     .config(trainConfig)
 *     .build()) {
 *
 *     trainer.addCallback(new LoggingCallback());
 *     trainer.addCallback(new CheckpointCallback());
 *
 *     trainer.train();
 * }
 * }</pre>
 */
public interface EnterpriseTrainer extends AutoCloseable {

    // ============= Lifecycle =============

    /**
     * Initialize the trainer.
     */
    void initialize();

    /**
     * Run the training loop.
     */
    void train();

    /**
     * Run evaluation.
     */
    default void eval() {
        eval(null);
    }

    /**
     * Run evaluation with custom metrics.
     */
    void eval(List<String> metrics);

    /**
     * Predict on new data.
     */
    Tensor predict(Tensor input);

    /**
     * Predict batch on new data.
     */
    List<Tensor> predictBatch(List<Tensor> inputs);

    // ============= Training Control =============

    /**
     * Pause training.
     */
    void pause();

    /**
     * Resume training.
     */
    void resume();

    /**
     * Request training to stop gracefully.
     */
    void stop();

    /**
     * Force stop training immediately.
     */
    void kill();

    /**
     * Check if training is stopped.
     */
    boolean isStopped();

    /**
     * Check if training is paused.
     */
    boolean isPaused();

    // ============= State =============

    /**
     * Get current training state.
     */
    TrainingState getState();

    /**
     * Get current global step.
     */
    int getGlobalStep();

    /**
     * Get current epoch.
     */
    int getCurrentEpoch();

    /**
     * Get total training steps.
     */
    int getTotalSteps();

    /**
     * Get progress (0.0 to 1.0).
     */
    double getProgress();

    // ============= Model & Optimizer =============

    /**
     * Get the model being trained.
     */
    Module getModel();

    /**
     * Get the optimizer.
     */
    Optimizer getOptimizer();

    /**
     * Save model checkpoint.
     */
    void saveCheckpoint(String path);

    /**
     * Load model checkpoint.
     */
    void loadCheckpoint(String path);

    // ============= Metrics =============

    /**
     * Get current training metrics.
     */
    TrainingMetrics getMetrics();

    /**
     * Get training history.
     */
    List<TrainingMetrics> getHistory();

    /**
     * Get metrics for a specific step.
     */
    TrainingMetrics getMetricsAtStep(int step);

    // ============= Callbacks =============

    /**
     * Add a training callback.
     */
    void addCallback(TrainingCallback callback);

    /**
     * Remove a training callback.
     */
    void removeCallback(TrainingCallback callback);

    // ============= Inner types =============

    /**
     * Training state.
     */
    enum TrainingState {
        /** Initial state, not started. */
        IDLE,
        /** Training is running. */
        TRAINING,
        /** Training is paused. */
        PAUSED,
        /** Training is stopping (graceful). */
        STOPPING,
        /** Training completed. */
        COMPLETED,
        /** Training failed. */
        FAILED,
        /** Training killed (force stop). */
        KILLED
    }

    /**
     * Training metrics.
     */
    final class TrainingMetrics {
        private final int globalStep;
        private final int epoch;
        private final double progress;
        private final double loss;
        private final double learningRate;
        private final double gradNorm;
        private final double gpuMemoryMB;
        private final double throughputTokensPerSec;
        private final long stepTimeMs;
        private final long timestamp;

        public TrainingMetrics(int globalStep, int epoch, double progress,
                             double loss, double learningRate, double gradNorm,
                             double gpuMemoryMB, double throughputTokensPerSec,
                             long stepTimeMs, long timestamp) {
            this.globalStep = globalStep;
            this.epoch = epoch;
            this.progress = progress;
            this.loss = loss;
            this.learningRate = learningRate;
            this.gradNorm = gradNorm;
            this.gpuMemoryMB = gpuMemoryMB;
            this.throughputTokensPerSec = throughputTokensPerSec;
            this.stepTimeMs = stepTimeMs;
            this.timestamp = timestamp;
        }

        // Getters
        public int globalStep() { return globalStep; }
        public int epoch() { return epoch; }
        public double progress() { return progress; }
        public double loss() { return loss; }
        public double learningRate() { return learningRate; }
        public double gradNorm() { return gradNorm; }
        public double gpuMemoryMB() { return gpuMemoryMB; }
        public double throughputTokensPerSec() { return throughputTokensPerSec; }
        public long stepTimeMs() { return stepTimeMs; }
        public long timestamp() { return timestamp; }

        @Override
        public String toString() {
            return String.format(
                    "TrainingMetrics{step=%d, epoch=%d, progress=%.1f%%, " +
                    "loss=%.4f, lr=%.2e, gradNorm=%.4f, " +
                    "gpuMem=%.1fMB, tokens/s=%.0f, stepTime=%.1fms}",
                    globalStep, epoch, progress * 100, loss, learningRate,
                    gradNorm, gpuMemoryMB, throughputTokensPerSec, stepTimeMs);
        }
    }

    /**
     * Training callback interface.
     */
    interface TrainingCallback {

        /**
         * Called before training starts.
         */
        default void onTrainBegin(EnterpriseTrainer trainer) {}

        /**
         * Called after training ends.
         */
        default void onTrainEnd(EnterpriseTrainer trainer) {}

        /**
         * Called at the beginning of each step.
         */
        default void onStepBegin(EnterpriseTrainer trainer, int step) {}

        /**
         * Called at the end of each step.
         */
        default void onStepEnd(EnterpriseTrainer trainer, int step, TrainingMetrics metrics) {}

        /**
         * Called at the beginning of each epoch.
         */
        default void onEpochBegin(EnterpriseTrainer trainer, int epoch) {}

        /**
         * Called at the end of each epoch.
         */
        default void onEpochEnd(EnterpriseTrainer trainer, int epoch, TrainingMetrics metrics) {}

        /**
         * Called on training error.
         */
        default void onError(EnterpriseTrainer trainer, Throwable error) {}

        /**
         * Called when checkpoint is saved.
         */
        default void onCheckpointSave(EnterpriseTrainer trainer, String path) {}

        /**
         * Called when checkpoint is loaded.
         */
        default void onCheckpointLoad(EnterpriseTrainer trainer, String path) {}

        /**
         * Called on training pause.
         */
        default void onPause(EnterpriseTrainer trainer) {}

        /**
         * Called on training resume.
         */
        default void onResume(EnterpriseTrainer trainer) {}

        /**
         * Called on training stop.
         */
        default void onStop(EnterpriseTrainer trainer) {}
    }

    /**
     * Builder for creating trainers.
     */
    final class Builder {
        private Module model;
        private Optimizer optimizer;
        private TrainerConfig config;
        private List<TrainingCallback> callbacks = new CopyOnWriteArrayList<>();

        public Builder model(Module model) { this.model = model; return this; }
        public Builder optimizer(Optimizer optimizer) { this.optimizer = optimizer; return this; }
        public Builder config(TrainerConfig config) { this.config = config; return this; }
        public Builder addCallback(TrainingCallback callback) { this.callbacks.add(callback); return this; }

        public EnterpriseTrainer build() {
            return new EnterpriseTrainerImpl(model, optimizer, config, callbacks);
        }
    }

    static Builder builder() { return new Builder(); }
}
