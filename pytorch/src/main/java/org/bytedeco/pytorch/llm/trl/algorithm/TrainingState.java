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
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.algorithm;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Immutable snapshot of training state for checkpointing.
 *
 * <p>Contains all information needed to resume training from a checkpoint:
 * <ul>
 *   <li>Optimizer state (moments, variance for Adam-like optimizers)</li>
 *   <li>Learning rate scheduler state (if applicable)</li>
 *   <li>Step counters and epoch counters</li>
 *   <li>Custom algorithm-specific state</li>
 *   <li>Training metadata (timestamp, version, algorithm)</li>
 * </ul>
 *
 * <p>This class is immutable - create new instances with updated values
 * using the builder pattern or the {@link #toBuilder()} method.
 *
 * <pre>{@code
 * // Save checkpoint
 * TrainingState state = trainer.getState();
 * saveToFile(state);
 *
 * // Resume training
 * TrainingState state = loadFromFile();
 * trainer.loadState(state);
 * }</pre>
 */
public final class TrainingState {

    // ==================== Metadata ====================

    /** Algorithm identifier (e.g., "dpo", "grpo") */
    private final String algorithmId;

    /** Algorithm version */
    private final String version;

    /** Timestamp when checkpoint was created (milliseconds since epoch) */
    private final long timestamp;

    /** Global step counter */
    private final int globalStep;

    /** Current epoch number */
    private final int epoch;

    /** Training samples processed */
    private final long samplesProcessed;

    // ==================== Optimizer State ====================

    /** Optimizer state dict (parameter name -> state tensors) */
    private final Map<String, Map<String, Tensor>> optimizerState;

    /** Current learning rate */
    private final double currentLearningRate;

    // ==================== Scheduler State ====================

    /** Scheduler state (if applicable) */
    private final Map<String, Object> schedulerState;

    // ==================== Metrics ====================

    /** Running training metrics */
    private final Map<String, Double> metrics;

    /** Best achieved metric values (for early stopping) */
    private final Map<String, Double> bestMetrics;

    // ==================== Custom State ====================

    /** Algorithm-specific state dict */
    private final Map<String, Tensor> customState;

    /** Custom metadata */
    private final Map<String, String> metadata;

    // ==================== Builder ====================

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder()
            .algorithmId(this.algorithmId)
            .version(this.version)
            .timestamp(this.timestamp)
            .globalStep(this.globalStep)
            .epoch(this.epoch)
            .samplesProcessed(this.samplesProcessed)
            .optimizerState(this.optimizerState)
            .currentLearningRate(this.currentLearningRate)
            .schedulerState(this.schedulerState)
            .metrics(this.metrics)
            .bestMetrics(this.bestMetrics)
            .customState(this.customState)
            .metadata(this.metadata);
    }

    public static final class Builder {
        private String algorithmId = "unknown";
        private String version = "1.0";
        private long timestamp = System.currentTimeMillis();
        private int globalStep = 0;
        private int epoch = 0;
        private long samplesProcessed = 0;
        private Map<String, Map<String, Tensor>> optimizerState = Map.of();
        private double currentLearningRate = 0.0;
        private Map<String, Object> schedulerState = Map.of();
        private Map<String, Double> metrics = Map.of();
        private Map<String, Double> bestMetrics = Map.of();
        private Map<String, Tensor> customState = Map.of();
        private Map<String, String> metadata = Map.of();

        public Builder algorithmId(String v) { this.algorithmId = v; return this; }
        public Builder version(String v) { this.version = v; return this; }
        public Builder timestamp(long v) { this.timestamp = v; return this; }
        public Builder globalStep(int v) { this.globalStep = v; return this; }
        public Builder epoch(int v) { this.epoch = v; return this; }
        public Builder samplesProcessed(long v) { this.samplesProcessed = v; return this; }
        public Builder optimizerState(Map<String, Map<String, Tensor>> v) { this.optimizerState = v; return this; }
        public Builder currentLearningRate(double v) { this.currentLearningRate = v; return this; }
        public Builder schedulerState(Map<String, Object> v) { this.schedulerState = v; return this; }
        public Builder metrics(Map<String, Double> v) { this.metrics = v; return this; }
        public Builder bestMetrics(Map<String, Double> v) { this.bestMetrics = v; return this; }
        public Builder customState(Map<String, Tensor> v) { this.customState = v; return this; }
        public Builder metadata(Map<String, String> v) { this.metadata = v; return this; }

        public TrainingState build() {
            return new TrainingState(this);
        }
    }

    // ==================== Constructor ====================

    private TrainingState(Builder b) {
        this.algorithmId = Objects.requireNonNull(b.algorithmId, "algorithmId");
        this.version = b.version;
        this.timestamp = b.timestamp;
        this.globalStep = b.globalStep;
        this.epoch = b.epoch;
        this.samplesProcessed = b.samplesProcessed;
        this.optimizerState = b.optimizerState;
        this.currentLearningRate = b.currentLearningRate;
        this.schedulerState = b.schedulerState;
        this.metrics = b.metrics;
        this.bestMetrics = b.bestMetrics;
        this.customState = b.customState;
        this.metadata = b.metadata;
    }

    // ==================== Getters ====================

    public String algorithmId() { return algorithmId; }
    public String version() { return version; }
    public long timestamp() { return timestamp; }
    public int globalStep() { return globalStep; }
    public int epoch() { return epoch; }
    public long samplesProcessed() { return samplesProcessed; }
    public Map<String, Map<String, Tensor>> optimizerState() { return optimizerState; }
    public double currentLearningRate() { return currentLearningRate; }
    public Map<String, Object> schedulerState() { return schedulerState; }
    public Map<String, Double> metrics() { return metrics; }
    public Map<String, Double> bestMetrics() { return bestMetrics; }
    public Map<String, Tensor> customState() { return customState; }
    public Map<String, String> metadata() { return metadata; }

    // ==================== Utility Methods ====================

    /**
     * Get a specific metric value.
     */
    public double getMetric(String name, double defaultValue) {
        Double value = metrics.get(name);
        return value != null ? value : defaultValue;
    }

    /**
     * Get best metric value.
     */
    public double getBestMetric(String name, double defaultValue) {
        Double value = bestMetrics.get(name);
        return value != null ? value : defaultValue;
    }

    /**
     * Get custom state tensor.
     */
    public Tensor getCustomTensor(String key) {
        return customState.get(key);
    }

    /**
     * Get metadata value.
     */
    public String getMetadata(String key, String defaultValue) {
        String value = metadata.get(key);
        return value != null ? value : defaultValue;
    }

    /**
     * Create a copy with updated global step.
     */
    public TrainingState withGlobalStep(int step) {
        return toBuilder().globalStep(step).build();
    }

    /**
     * Create a copy with updated metrics.
     */
    public TrainingState withMetrics(Map<String, Double> newMetrics) {
        return toBuilder().metrics(newMetrics).build();
    }

    /**
     * Create a copy with updated best metrics.
     */
    public TrainingState withBestMetrics(Map<String, Double> newBestMetrics) {
        return toBuilder().bestMetrics(newBestMetrics).build();
    }

    // ==================== Object Methods ====================

    @Override
    public String toString() {
        return "TrainingState{" +
                "algorithm='" + algorithmId + '\'' +
                ", version='" + version + '\'' +
                ", step=" + globalStep +
                ", epoch=" + epoch +
                ", samples=" + samplesProcessed +
                ", lr=" + currentLearningRate +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TrainingState that = (TrainingState) o;
        return globalStep == that.globalStep &&
                epoch == that.epoch &&
                samplesProcessed == that.samplesProcessed &&
                Double.compare(that.currentLearningRate, currentLearningRate) == 0 &&
                algorithmId.equals(that.algorithmId) &&
                version.equals(that.version);
    }

    @Override
    public int hashCode() {
        return Objects.hash(algorithmId, version, globalStep, epoch,
                samplesProcessed, currentLearningRate);
    }
}
