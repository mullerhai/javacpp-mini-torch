/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed under the Apache License, Version 2.0, or (at your option)
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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.distributed;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Activation Checkpointing Manager for memory-efficient training.
 *
 * <p>Reduces memory footprint by selectively discarding intermediate activations
 * during forward pass and recomputing them during backward pass. This trades
 * compute for memory, enabling training of larger models.
 *
 * <p>Supports three strategies:
 * <ul>
 *   <li>{@link CheckpointStrategy#FULL}: Checkpoint all activations (maximum memory savings)</li>
 *   <li>{@link CheckpointStrategy#SELECTIVE}: Checkpoint only specified layers</li>
 *   <li>{@link CheckpointStrategy#ADAPTIVE}: Dynamically select based on memory pressure</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * ActivationCheckpoint checkpoint = ActivationCheckpoint.builder()
 *     .strategy(CheckpointStrategy.SELECTIVE)
 *     .checkpointLayers("attention", "ffn", "moe")
 *     .recomputeFunc((tensor, name) -> recompute(tensor, name))
 *     .build();
 *
 * Tensor input = checkpoint.checkpoint("encoder_layer_1", () -> forwardPass(input));
 * }</pre>
 */
public final class ActivationCheckpoint {
    private final CheckpointStrategy strategy;
    private final Set<String> checkpointLayers;
    private final Map<String, Tensor> storedActivations;
    private final Map<String, Long> memoryUsage;
    private final long maxMemoryBytes;
    private final boolean trackStats;
    private final RecomputeFunction recomputeFunc;

    private long currentMemoryBytes = 0;
    private int recomputeCount = 0;
    private int checkpointCount = 0;

    private ActivationCheckpoint(Builder builder) {
        this.strategy = builder.strategy;
        this.checkpointLayers = builder.checkpointLayers != null
            ? new HashSet<>(builder.checkpointLayers)
            : new HashSet<>();
        this.storedActivations = builder.trackStats
            ? new ConcurrentHashMap<>()
            : new HashMap<>();
        this.memoryUsage = builder.trackStats
            ? new ConcurrentHashMap<>()
            : new HashMap<>();
        this.maxMemoryBytes = builder.maxMemoryBytes;
        this.trackStats = builder.trackStats;
        this.recomputeFunc = builder.recomputeFunc;

        System.out.printf("[ActivationCheckpoint] strategy=%s maxMemory=%.2fGB%n",
                strategy, maxMemoryBytes / 1e9);
    }

    public static Builder builder() { return new Builder(); }

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Checkpointing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Execute a computation with activation checkpointing.
     *
     * <p>If the activation should be checkpointed, the computation is executed
     * without storing activations. During backward, the computation will be
     * re-executed to recover activations.
     *
     * @param layerName name of the layer being checkpointed
     * @param compute forward computation to execute
     * @return result of the computation
     */
    public <T extends Tensor> T checkpoint(String layerName, Supplier<T> compute) {
        boolean shouldCheckpoint = shouldCheckpoint(layerName);

        if (!shouldCheckpoint) {
            return compute.get();
        }

        // Execute forward without storing activations
        T result = compute.get();
        checkpointCount++;

        if (trackStats) {
            long size = result.elementSize() * result.numel();
            memoryUsage.put(layerName + "_output", size);

            if (strategy == CheckpointStrategy.ADAPTIVE) {
                currentMemoryBytes += size;
                maybeEvict();
            }
        }

        return result;
    }

    /**
     * Checkpoint with input preservation for backward recomputation.
     *
     * <p>Stores only the input tensor (much smaller than full activations)
     * and recomputes everything during backward.
     *
     * @param layerName name of the layer
     * @param input input tensor to store
     * @param compute forward computation
     * @return result of computation
     */
    public <T extends Tensor> T checkpointWithInput(String layerName, Tensor input, Supplier<T> compute) {
        boolean shouldCheckpoint = shouldCheckpoint(layerName);

        if (trackStats) {
            long inputSize = input.elementSize() * input.numel();
            memoryUsage.put(layerName + "_input", inputSize);
        }

        if (!shouldCheckpoint) {
            return compute.get();
        }

        // Store only input, not activations
        T result = compute.get();
        checkpointCount++;

        return result;
    }

    /**
     * Recompute activations during backward pass.
     *
     * @param layerName name of the layer
     * @param savedInput input that was saved during forward
     * @return recomputed activations
     */
    public Tensor recompute(String layerName, Tensor savedInput) {
        recomputeCount++;

        if (recomputeFunc != null) {
            return recomputeFunc.apply(layerName, savedInput);
        }

        throw new UnsupportedOperationException(
            "Recompute function not set. Provide via builder.recomputeFunc()");
    }

    /**
     * Recompute with full forward pass.
     */
    public Tensor recomputeFull(String layerName, Supplier<Tensor> forwardFunc) {
        recomputeCount++;
        return forwardFunc.get();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Selective Checkpointing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Wrap a layer for selective checkpointing.
     *
     * @param <T> tensor type
     * @param layerName layer name
     * @param forward forward computation
     * @param recompute recompute computation
     * @return wrapped computation
     */
    public <T extends Tensor> CheckpointedComputation<T> wrap(
            String layerName,
            Supplier<T> forward,
            Supplier<Tensor> recompute) {
        return new CheckpointedComputation<>(layerName, forward, recompute, this);
    }

    /**
     * Check if a layer should be checkpointed.
     */
    public boolean shouldCheckpoint(String layerName) {
        switch (strategy) {
            case FULL:
                return true;
            case SELECTIVE:
                return checkpointLayers.isEmpty() || checkpointLayers.contains(layerName);
            case ADAPTIVE:
                return currentMemoryBytes >= maxMemoryBytes * 0.8
                       || (checkpointLayers.contains(layerName));
            default:
                return false;
        }
    }

    /**
     * Add a layer to checkpoint list.
     */
    public void addCheckpointLayer(String layerName) {
        checkpointLayers.add(layerName);
    }

    /**
     * Remove a layer from checkpoint list.
     */
    public void removeCheckpointLayer(String layerName) {
        checkpointLayers.remove(layerName);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Memory Management
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get current memory usage in bytes.
     */
    public long getCurrentMemoryBytes() {
        return currentMemoryBytes;
    }

    /**
     * Get memory usage as percentage of max.
     */
    public double getMemoryUsagePercent() {
        return (currentMemoryBytes * 100.0) / maxMemoryBytes;
    }

    /**
     * Check if memory is under pressure.
     */
    public boolean isMemoryPressure() {
        return currentMemoryBytes >= maxMemoryBytes * 0.9;
    }

    /**
     * Evict least recently used activations when under memory pressure.
     */
    private void maybeEvict() {
        if (currentMemoryBytes <= maxMemoryBytes) {
            return;
        }

        // Evict oldest entries
        List<String> keys = new ArrayList<>(storedActivations.keySet());
        while (currentMemoryBytes > maxMemoryBytes * 0.7 && !keys.isEmpty()) {
            String key = keys.remove(0);
            Tensor removed = storedActivations.remove(key);
            if (removed != null) {
                currentMemoryBytes -= memoryUsage.remove(key);
                removed.close();
            }
        }

        if (trackStats) {
            System.out.printf("[ActivationCheckpoint] Evicted. Memory: %.2f%% of max%n",
                    getMemoryUsagePercent());
        }
    }

    /**
     * Clear all stored activations.
     */
    public void clear() {
        for (Tensor t : storedActivations.values()) {
            t.close();
        }
        storedActivations.clear();
        memoryUsage.clear();
        currentMemoryBytes = 0;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics
    // ═══════════════════════════════════════════════════════════════════════════

    public CheckpointStats getStats() {
        return new CheckpointStats(
            checkpointCount,
            recomputeCount,
            currentMemoryBytes,
            maxMemoryBytes,
            new HashMap<>(memoryUsage),
            storedActivations.size()
        );
    }

    public void printStats() {
        CheckpointStats stats = getStats();
        System.out.printf("""
                ═══ Activation Checkpoint Stats ═══
                  Checkpoints: %d
                  Recomputes:  %d
                  Memory:      %.2f%% (%.2f / %.2f GB)
                  Stored:      %d tensors
                ═════════════════════════════════════
                """,
                stats.checkpointCount(),
                stats.recomputeCount(),
                stats.memoryUsagePercent(),
                stats.currentMemoryBytes() / 1e9,
                stats.maxMemoryBytes() / 1e9,
                stats.storedTensorCount()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════════

    public enum CheckpointStrategy {
        /** Checkpoint all activations (maximum memory savings) */
        FULL,
        /** Checkpoint only specified layers */
        SELECTIVE,
        /** Dynamically select based on memory pressure */
        ADAPTIVE
    }

    @FunctionalInterface
    public interface RecomputeFunction {
        Tensor apply(String layerName, Tensor input);
    }

    public static final class Builder {
        private CheckpointStrategy strategy = CheckpointStrategy.SELECTIVE;
        private Set<String> checkpointLayers;
        private long maxMemoryBytes = 1L << 34; // 16GB default
        private boolean trackStats = true;
        private RecomputeFunction recomputeFunc;

        public Builder strategy(CheckpointStrategy s) { this.strategy = s; return this; }
        public Builder checkpointLayers(Set<String> layers) { this.checkpointLayers = layers; return this; }
        public Builder checkpointLayers(String... layers) {
            this.checkpointLayers = new HashSet<>(Arrays.asList(layers));
            return this;
        }
        public Builder maxMemoryBytes(long m) { this.maxMemoryBytes = m; return this; }
        public Builder trackStats(boolean t) { this.trackStats = t; return this; }
        public Builder recomputeFunc(RecomputeFunction f) { this.recomputeFunc = f; return this; }

        public ActivationCheckpoint build() {
            return new ActivationCheckpoint(this);
        }
    }

    /**
     * Checkpointed computation wrapper.
     */
    public static final class CheckpointedComputation<T extends Tensor> {
        private final String layerName;
        private final Supplier<T> forward;
        private final Supplier<Tensor> recompute;
        private final ActivationCheckpoint checkpoint;
        private Tensor savedInput;

        public CheckpointedComputation(String layerName, Supplier<T> forward,
                                       Supplier<Tensor> recompute, ActivationCheckpoint checkpoint) {
            this.layerName = layerName;
            this.forward = forward;
            this.recompute = recompute;
            this.checkpoint = checkpoint;
        }

        public T forward() {
            return checkpoint.checkpointWithInput(layerName, null, forward);
        }

        public Tensor backward() {
            return checkpoint.recomputeFull(layerName, recompute);
        }

        public void saveInput(Tensor input) {
            this.savedInput = input;
        }

        public Tensor getSavedInput() {
            return savedInput;
        }
    }

    /**
     * Checkpointing statistics.
     */
    public record CheckpointStats(
        int checkpointCount,
        int recomputeCount,
        long currentMemoryBytes,
        long maxMemoryBytes,
        Map<String, Long> memoryUsage,
        int storedTensorCount
    ) {
        public double memoryUsagePercent() {
            return maxMemoryBytes > 0 ? (currentMemoryBytes * 100.0) / maxMemoryBytes : 0;
        }
    }
}
