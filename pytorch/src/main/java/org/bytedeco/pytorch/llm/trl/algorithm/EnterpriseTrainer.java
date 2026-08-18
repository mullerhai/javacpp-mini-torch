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
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.llm.trl.config.TrainerConfig;

import java.util.Map;

/**
 * Enterprise unified trainer interface for all RLHF/alignment algorithms.
 *
 * <p>This interface provides a common contract for all training algorithms,
 * enabling polymorphic usage and consistent APIs across different methods.
 *
 * <p>Supported algorithms include:
 * <ul>
 *   <li><b>Preference Optimization</b>: DPO, SimPO, IPO, TDPO, ORPO, KTO, RLOO, GRPO</li>
 *   <li><b>Policy Gradient</b>: PPO, Nash-MD, SPO</li>
 *   <li><b>Rank-based</b>: RRHF, RAFT</li>
 *   <li><b>Supervised</b>: SFT</li>
 * </ul>
 *
 * <p>Reference papers:
 * <ul>
 *   <li>DPO: "Direct Preference Optimization" (Rafailov et al., 2023)</li>
 *   <li>SimPO: "SimPO: Simple Preference Optimization" (Meng et al., 2024)</li>
 *   <li>IPO: "Identity Preference Optimization" (Azar et al., 2024)</li>
 *   <li>KTO: "KTO: Kahneman-Tversky Optimization" (Ethayarajh et al., 2024)</li>
 *   <li>GRPO: "DeepSeek-R1" (DeepSeek AI, 2025)</li>
 *   <li>RRHF: "RRHF: Rank Responses to Rank Responses" (Yuan et al., 2023)</li>
 * </ul>
 */
public interface EnterpriseTrainer extends AutoCloseable {

    // ==================== Algorithm Identification ====================

    /**
     * Unique algorithm identifier.
     * Standard names: "dpo", "simpo", "ipo", "tdpo", "orpo", "kto", "rloo", "grpo",
     *                  "ppo", "nash-md", "spo", "rrhf", "sft", "cpo", "crai"
     */
    String algorithmId();

    /**
     * Human-readable algorithm name.
     */
    default String algorithmName() {
        return algorithmId().toUpperCase();
    }

    /**
     * Algorithm category for grouping and filtering.
     */
    enum AlgorithmCategory {
        /** Preference-based optimization using paired comparison data */
        PREFERENCE_OPTIMIZATION,
        /** Policy gradient methods with value functions */
        POLICY_GRADIENT,
        /** Ranking-based alignment methods */
        RANK_BASED,
        /** Standard supervised fine-tuning */
        SUPERVISED,
        /** Contrastive learning approaches */
        CONTRASTIVE,
        /** Constitutional AI methods */
        CONSTITUTIONAL,
        /** Multi-modal alignment */
        MULTIMODAL
    }

    /**
     * Returns the category of this algorithm.
     */
    AlgorithmCategory category();

    // ==================== Training Control ====================

    /**
     * Execute one training step on a batch.
     *
     * @param batch Map containing algorithm-specific tensors (e.g., "input_ids", "rewards", etc.)
     * @return Scalar loss value for this step
     */
    double trainingStep(Map<String, Tensor> batch);

    /**
     * Compute loss without performing an optimizer step.
     * Useful for evaluation and debugging.
     *
     * <p>Note: This method may not be available on all implementations.
     * If not supported, implementations should throw UnsupportedOperationException.
     *
     * @param batch Input batch
     * @return Scalar loss tensor (still attached to computation graph)
     */
    default Tensor computeLoss(Map<String, Tensor> batch) {
        throw new UnsupportedOperationException(
            "computeLoss is not publicly available in this implementation");
    }

    /**
     * Set training mode.
     */
    void train();

    /**
     * Set evaluation mode.
     */
    void eval();

    /**
     * Check if currently in training mode.
     */
    boolean isTraining();

    // ==================== Model Access ====================

    /**
     * Get the primary model being trained.
     */
    Module policy();

    /**
     * Get the reference model (may be null for reference-free algorithms).
     */
    Module reference();

    /**
     * Get the reward model (may be null if not used).
     */
    Module rewardModel();

    // ==================== Optimizer & Config ====================

    /**
     * Get the optimizer.
     */
    Optimizer optimizer();

    /**
     * Get the training configuration.
     */
    TrainerConfig config();

    // ==================== State & Metrics ====================

    /**
     * Get current global step counter.
     */
    int globalStep();


    /**
     * Reset accumulated metrics.
     */
    void resetMetrics();

    // ==================== Serialization ====================

    /**
     * Get a state snapshot for checkpointing.
     */
    TrainingState getState();

    /**
     * Restore from a state snapshot.
     *
     * @param state Previously saved TrainingState
     */
    void loadState(TrainingState state);

    // ==================== Lifecycle ====================

    /**
     * Check if trainer has been closed.
     */
    boolean isClosed();

    /**
     * Get version string.
     */
    default String version() {
        return "1.0";
    }

    @Override
    void close();

    // ==================== Default Implementations ====================

    /**
     * Default metrics implementation.
     */
    default Map<String, Double> getMetrics() {
        return Map.of(
            "global_step", (double) globalStep(),
            "training", isTraining() ? 1.0 : 0.0
        );
    }

    /**
     * Get accumulated training metrics.
     *
     * @return Map of metric name to current value
     */
//    Map<String, Double> getMetrics();


//    default void resetMetrics() {
//        // Default: no-op, override if needed
//    }

//    default void loadState(TrainingState state) {
//        // Default: no-op, override if needed
//    }
}
