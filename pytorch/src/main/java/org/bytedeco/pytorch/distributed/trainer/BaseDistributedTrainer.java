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
package org.bytedeco.pytorch.distributed.trainer;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.TrainerStats;
import org.bytedeco.pytorch.distributed.config.MixedPrecisionConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Base interface for all distributed trainers in the JavaCPP PyTorch framework.
 *
 * <p>This interface defines the contract that all distributed training implementations
 * must follow, including DDP, FSDP, ZeRO, Pipeline Parallel, Tensor Parallel,
 * and Hybrid Parallel trainers.
 *
 * <p>Key operations:
 * <ul>
 *   <li>{@link #forward(Tensor)} - Run forward pass through the model</li>
 *   <li>{@link #step(Tensor, Tensor, Optimizer)} - Complete training step (forward + backward + optimize)</li>
 *   <li>{@link #trainingStep(Tensor, Tensor, Optimizer)} - Alias for step</li>
 *   <li>{@link #backward(Tensor)} - Backward pass with gradient synchronization</li>
 *   <li>{@link #synchronize()} - Explicit gradient synchronization</li>
 * </ul>
 *
 * <p>All trainers implement {@link AutoCloseable} for resource management.
 *
 * <pre>{@code
 * try (BaseDistributedTrainer trainer : createTrainer(config)) {
 *     for (int i = 0; i < steps; i++) {
 *         Tensor loss = trainer.step(input, target, optimizer);
 *     }
 * }
 * }</pre>
 */
public interface BaseDistributedTrainer extends AutoCloseable {

    /**
     * Get the underlying module being trained.
     */
    Module getModule();

    /**
     * Get the local module for this rank.
     * Default implementation returns {@link #getModule()}.
     */
    default Module getLocalModule() {
        return getModule();
    }

    /**
     * Run forward pass through the distributed model.
     *
     * @param input input tensor
     * @return output tensor
     */
    Tensor forward(Tensor input);

    /**
     * Complete training step: forward + loss computation + backward + optimizer step.
     *
     * @param input    input tensor
     * @param target   target tensor for loss computation
     * @param optimizer optimizer to use (may be null for custom training loops)
     * @return loss tensor
     */
    Tensor step(Tensor input, Tensor target, Optimizer optimizer);

    /**
     * Alias for {@link #step(Tensor, Tensor, Optimizer)}.
     * Provides a consistent API across different trainer implementations.
     */
    default Tensor trainingStep(Tensor input, Tensor target, Optimizer optimizer) {
        return step(input, target, optimizer);
    }

    /**
     * Backward pass with gradient synchronization.
     * When using gradient accumulation, call {@link #backward(Tensor)} for
     * each micro-step without sync, then call {@link #synchronize()} after.
     *
     * @param loss loss tensor from forward pass
     */
    default void backward(Tensor loss) {
        Objects.requireNonNull(loss, "loss");
        loss.backward();
    }

    /**
     * Explicitly synchronize gradients across all ranks.
     * Use this when doing gradient accumulation with disabled auto-sync.
     */
    default void synchronize() {
        // Default: no-op for trainers that handle sync automatically
    }

    /**
     * Disable gradient synchronization for the next backward pass.
     * Used for gradient accumulation without communication.
     */
    default void disableSync() {
        // Default: no-op for trainers that don't support this
    }

    /**
     * Re-enable gradient synchronization.
     */
    default void enableSync() {
        // Default: no-op for trainers that don't support this
    }

    /**
     * Check if gradient synchronization is currently enabled.
     */
    default boolean isSyncEnabled() {
        return true;
    }

    /**
     * Zero all parameter gradients.
     */
    default void zeroGrad() {
        Module module = getModule();
        if (module == null) return;
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            var p = params.get(i);
            if (p == null || p.isNull()) continue;
            try {
                var g = p.grad();
                if (g != null && !g.isNull() && g.defined()) g.zero_();
            } catch (Exception ignored) {
            }
        }
    }

    /**
     * Set model to training mode.
     */
    default void train() {
        Module module = getModule();
        if (module != null) module.train(true);
    }

    /**
     * Set model to evaluation mode.
     */
    default void eval() {
        Module module = getModule();
        if (module != null) module.eval();
    }

    /**
     * Check if model is in training mode.
     */
    default boolean isTraining() {
        Module module = getModule();
        return module != null && module.is_training();
    }

    /**
     * Collect all parameters from the model.
     */
    default List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        Module module = getModule();
        if (module == null) return params;
        TensorVector paramVec = module.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            var p = paramVec.get(i);
            if (p != null && !p.isNull()) {
                params.add(p);
            }
        }
        return params;
    }

    /**
     * Get the process group for distributed communication.
     */
    ProcessGroupWrapper getProcessGroup();

    /**
     * Get current rank in the distributed group.
     */
    default int getRank() {
        return getProcessGroup() != null ? getProcessGroup().getRank() : 0;
    }

    /**
     * Get total number of processes in the distributed group.
     */
    default int getWorldSize() {
        return getProcessGroup() != null ? getProcessGroup().getWorldSize() : 1;
    }

    /**
     * Check if this is the main process (rank 0).
     */
    default boolean isMainProcess() {
        return getRank() == 0;
    }

    /**
     * Get the device used for this rank.
     */
    default Device getDevice() {
        return getProcessGroup() != null ? getProcessGroup().getDevice() : null;
    }

    /**
     * Get mixed precision configuration.
     * Default implementation returns FP32 (full precision).
     */
    // Note: Removed default implementation to avoid conflicts with subclasses
    // that return different types. Subclasses must implement this.

    /**
     * Get training statistics.
     * Default implementation returns null.
     */
    default TrainerStats stats() {
        return null;
    }

    /**
     * Create a state dict for checkpointing.
     * Returns a map containing model state and training metadata.
     */
    default Map<String, Object> stateDict() {
        return Map.of();
    }

    /**
     * Load state dict from checkpoint.
     *
     * @param state state dict from {@link #stateDict()}
     */
    default void loadStateDict(Map<String, Object> state) {
        // Default: no-op
    }

    /**
     * Get number of forward calls.
     */
    default long getNumForwardCalls() {
        return 0;
    }

    /**
     * Get number of backward calls.
     */
    default long getNumBackwardCalls() {
        return 0;
    }

    /**
     * Get number of gradient synchronization calls.
     */
    default long getNumSyncCalls() {
        return 0;
    }

    /**
     * Reset training statistics.
     */
    default void resetStats() {
        // Default: no-op
    }

    /**
     * Builder interface for creating trainers.
     * All trainer implementations should provide a Builder that extends this.
     *
     * @param <T> trainer type
     * @param <B> builder type (for method chaining)
     */
    interface Builder<T extends BaseDistributedTrainer, B extends Builder<T, B>> {
        /**
         * Set the module to train.
         */
        B module(Module module);

        /**
         * Set the process group for distributed training.
         */
        B processGroup(ProcessGroupWrapper processGroup);

        /**
         * Build the trainer instance.
         */
        T build();
    }
}
