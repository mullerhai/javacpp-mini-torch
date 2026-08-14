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

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for all alignment training algorithms.
 *
 * <p>Provides a unified factory interface for creating trainer instances
 * and maintains metadata about each algorithm including:
 * <ul>
 *   <li>Supported data formats (preference pairs, rankings, rewards)</li>
 *   <li>Reference model requirements</li>
 *   <li>Computational requirements (memory, GPU)</li>
 *   <li>Algorithm category and family</li>
 *   <li>Recommended use cases</li>
 * </ul>
 *
 * <p>Algorithm families:
 * <ul>
 *   <li><b>DPO Family</b>: DPO, IPO, SimPO, TDPO - Direct preference optimization</li>
 *   <li><b>Policy Gradient</b>: PPO, GRPO, RLOO - Advantage-based updates</li>
 *   <li><b>Contrastive</b>: KTO, ORPO - Asymmetric loss functions</li>
 *   <li><b>Rank-based</b>: RRHF, RAFT - Response ranking</li>
 * </ul>
 *
 * <pre>{@code
 * // List all available algorithms
 * AlgorithmRegistry.listAlgorithms();
 *
 * // Get algorithm metadata
 * AlgorithmInfo info = AlgorithmRegistry.getInfo("dpo");
 * System.out.println(info.description());
 *
 * // Create trainer (via factory)
 * EnterpriseTrainer trainer = AlgorithmRegistry.createTrainer("dpo", config);
 * }</pre>
 */
public final class AlgorithmRegistry {

    // ==================== Singleton ====================

    private static volatile AlgorithmRegistry INSTANCE = null;

    public static AlgorithmRegistry getInstance() {
        if (INSTANCE == null) {
            synchronized (AlgorithmRegistry.class) {
                if (INSTANCE == null) {
                    INSTANCE = new AlgorithmRegistry();
                    INSTANCE.registerBuiltins();
                }
            }
        }
        return INSTANCE;
    }

    /**
     * Reset registry (primarily for testing).
     */
    public static synchronized void reset() {
        if (INSTANCE != null) {
            INSTANCE.algorithms.clear();
            INSTANCE.registerBuiltins();
        }
    }

    // ==================== Data Structures ====================

    private final Map<String, AlgorithmInfo> algorithms = new ConcurrentHashMap<>();
    private final Map<String, TrainerFactory> factories = new ConcurrentHashMap<>();

    private AlgorithmRegistry() {}

    // ==================== Registration ====================

    /**
     * Register an algorithm with its metadata and factory.
     */
    public synchronized AlgorithmRegistry register(
            String id,
            String name,
            EnterpriseTrainer.AlgorithmCategory category,
            String description,
            String paper,
            Set<DataFormat> supportedFormats,
            Set<Requirement> requirements,
            String recommendedUseCase,
            TrainerFactory factory) {

        AlgorithmInfo info = new AlgorithmInfo(
                id, name, category, description, paper,
                supportedFormats, requirements, recommendedUseCase);
        algorithms.put(id.toLowerCase(), info);
        if (factory != null) {
            factories.put(id.toLowerCase(), factory);
        }
        return this;
    }

    /**
     * Register algorithm without factory (metadata only).
     */
    public synchronized AlgorithmRegistry register(AlgorithmInfo info) {
        algorithms.put(info.id().toLowerCase(), info);
        return this;
    }

    // ==================== Query ====================

    /**
     * Check if algorithm is registered.
     */
    public boolean isRegistered(String id) {
        return algorithms.containsKey(id.toLowerCase());
    }

    /**
     * Get algorithm info.
     */
    public AlgorithmInfo getInfo(String id) {
        return algorithms.get(id.toLowerCase());
    }

    /**
     * Get all registered algorithm IDs.
     */
    public Set<String> listAlgorithms() {
        return Collections.unmodifiableSet(algorithms.keySet());
    }

    /**
     * Get all algorithms in a category.
     */
    public List<AlgorithmInfo> getByCategory(EnterpriseTrainer.AlgorithmCategory category) {
        return algorithms.values().stream()
                .filter(info -> info.category() == category)
                .sorted(Comparator.comparing(AlgorithmInfo::name))
                .toList();
    }

    /**
     * Find algorithms by requirement.
     */
    public List<AlgorithmInfo> findByRequirement(Requirement requirement) {
        return algorithms.values().stream()
                .filter(info -> info.requirements().contains(requirement))
                .sorted(Comparator.comparing(AlgorithmInfo::name))
                .toList();
    }

    /**
     * Find algorithms supporting a data format.
     */
    public List<AlgorithmInfo> findByDataFormat(DataFormat format) {
        return algorithms.values().stream()
                .filter(info -> info.supportedFormats().contains(format))
                .sorted(Comparator.comparing(AlgorithmInfo::name))
                .toList();
    }

    // ==================== Factory Methods ====================

    /**
     * Create a trainer instance for the given algorithm.
     *
     * @throws IllegalArgumentException if algorithm is not registered or no factory available
     */
    public EnterpriseTrainer createTrainer(String algorithmId, TrainerFactoryArgs args) {
        TrainerFactory factory = factories.get(algorithmId.toLowerCase());
        if (factory == null) {
            throw new IllegalArgumentException(
                    "No factory registered for algorithm: " + algorithmId +
                    ". Available: " + listAlgorithms());
        }
        return factory.create(args);
    }

    // ==================== Builtin Registration ====================

    private void registerBuiltins() {
        // DPO
        register("dpo", "Direct Preference Optimization",
                EnterpriseTrainer.AlgorithmCategory.PREFERENCE_OPTIMIZATION,
                "Directly optimizes language models to align with human preferences using paired preference data. " +
                "Removes the need for a separate reward model by formulating alignment as a classification task.",
                "Rafailov et al. (2023) - https://arxiv.org/abs/2305.18290",
                Set.of(DataFormat.PREFERENCE_PAIRS, DataFormat.PRECOMPUTED_LOGPS),
                Set.of(Requirement.REFERENCE_MODEL, Requirement.PAIRED_DATA),
                "General preference alignment with high sample efficiency. Best when reference model is available.",
                null);

        // SimPO
        register("simpo", "Simple Preference Optimization",
                EnterpriseTrainer.AlgorithmCategory.PREFERENCE_OPTIMIZATION,
                "Simplified DPO variant that removes the reference model term, using target margin on log probabilities. " +
                "More robust to hyperparameters and doesn't require training a reference model.",
                "Meng et al. (2024) - https://arxiv.org/abs/2405.14734",
                Set.of(DataFormat.PREFERENCE_PAIRS, DataFormat.PRECOMPUTED_LOGPS),
                Set.of(Requirement.PAIRED_DATA),
                "When reference model is unavailable or memory is limited. Simpler hyperparameter tuning.",
                null);

        // IPO
        register("ipo", "Identity Preference Optimization",
                EnterpriseTrainer.AlgorithmCategory.PREFERENCE_OPTIMIZATION,
                "Theoretically grounded preference optimization that adds a regularization term to DPO. " +
                "Guarantees convergence to the optimal policy and has better finite-sample bounds.",
                "Azar et al. (2024) - https://arxiv.org/abs/2312.00079",
                Set.of(DataFormat.PREFERENCE_PAIRS, DataFormat.PRECOMPUTED_LOGPS),
                Set.of(Requirement.REFERENCE_MODEL, Requirement.PAIRED_DATA),
                "When theoretical guarantees are important or training stability is a concern.",
                null);

        // TDPO
        register("tdpo", "Token-level Direct Preference Optimization",
                EnterpriseTrainer.AlgorithmCategory.PREFERENCE_OPTIMIZATION,
                "Token-level extension of DPO that applies preference optimization at each token position. " +
                "Better for tasks requiring fine-grained token-level control.",
                "Dong et al. (2024) - Token-level DPO paper",
                Set.of(DataFormat.PREFERENCE_PAIRS, DataFormat.TOKEN_LEVEL_REWARDS),
                Set.of(Requirement.REFERENCE_MODEL, Requirement.PAIRED_DATA, Requirement.SEQUENTIAL),
                "Code generation, structured output, or tasks requiring precise token-level alignment.",
                null);

        // ORPO
        register("orpo", "Odds Ratio Preference Optimization",
                EnterpriseTrainer.AlgorithmCategory.CONTRASTIVE,
                "Reference-free preference optimization combining SFT loss with a odds-ratio based penalty. " +
                "Single-phase training without separate reference model.",
                "Ji et al. (2024) - https://arxiv.org/abs/2403.07691",
                Set.of(DataFormat.PREFERENCE_PAIRS, DataFormat.PRECOMPUTED_LOGPS),
                Set.of(Requirement.PAIRED_DATA),
                "Single-phase training without reference model. Good for resource-constrained scenarios.",
                null);

        // KTO
        register("kto", "Kahneman-Tversky Optimization",
                EnterpriseTrainer.AlgorithmCategory.CONTRASTIVE,
                "Alignment using prospect theory-inspired asymmetric loss. Uses binary acceptance signals " +
                "instead of paired preferences, achieving better helpfulness/diversity trade-off.",
                "Ethayarajh et al. (2024) - https://arxiv.org/abs/2402.01306",
                Set.of(DataFormat.BINARY_ACCEPTANCE, DataFormat.PAIRED_DATA),
                Set.of(Requirement.REFERENCE_MODEL_OPTIONAL),
                "When only binary feedback is available or diversity is important alongside helpfulness.",
                null);

        // GRPO
        register("grpo", "Group Relative Policy Optimization",
                EnterpriseTrainer.AlgorithmCategory.POLICY_GRADIENT,
                "DeepSeek's group-relative policy optimization that normalizes rewards within groups. " +
                "No value function required, uses self-critical baseline.",
                "DeepSeek AI (2025) - DeepSeek-R1 technical report",
                Set.of(DataFormat.REWARD_SCALARS, DataFormat.GROUP_REWARDS),
                Set.of(Requirement.REWARD_MODEL, Requirement.GENERATION_CAPABILITY),
                "Math, reasoning tasks, and any domain where self-verification rewards are available.",
                null);

        // PPO
        register("ppo", "Proximal Policy Optimization",
                EnterpriseTrainer.AlgorithmCategory.POLICY_GRADIENT,
                "Classic policy gradient algorithm with clipped objective for stable training. " +
                "Industry standard for RLHF with well-understood convergence properties.",
                "Schulman et al. (2017) - https://arxiv.org/abs/1707.06347",
                Set.of(DataFormat.REWARD_SCALARS, DataFormat.ADVANTAGES, DataFormat.VALUE_ESTIMATES),
                Set.of(Requirement.VALUE_NETWORK, Requirement.REWARD_MODEL, Requirement.GENERATION_CAPABILITY),
                "Production RLHF with reward model. Best when stability and theoretical guarantees matter.",
                null);

        // RLOO
        register("rloo", "REINFORCE Leave-One-Out",
                EnterpriseTrainer.AlgorithmCategory.POLICY_GRADIENT,
                "Simplified policy gradient using LOO baseline for advantage estimation. " +
                "No value network needed, batch-averaged rewards as baseline.",
                "Meta AI (2024) - A Minimalist Approach to LLM RL",
                Set.of(DataFormat.REWARD_SCALARS, DataFormat.GROUP_REWARDS),
                Set.of(Requirement.REWARD_MODEL, Requirement.GENERATION_CAPABILITY),
                "Memory-constrained scenarios where value network is infeasible. Simpler than PPO.",
                null);

        // RRHF
        register("rrhf", "Rank Responses to Rank Responses",
                EnterpriseTrainer.AlgorithmCategory.RANK_BASED,
                "Aligns language models using a ranking loss on generated responses. " +
                "Only requires a reward model for scoring, no reference model needed.",
                "Yuan et al. (2023) - https://arxiv.org/abs/2304.05302",
                Set.of(DataFormat.RANKINGS, DataFormat.GENERATED_RESPONSES),
                Set.of(Requirement.REWARD_MODEL, Requirement.GENERATION_CAPABILITY),
                "When you have ranking data or can score multiple responses with a reward model.",
                null);

        // CPO
        register("cpo", "Constrained Preference Optimization",
                EnterpriseTrainer.AlgorithmCategory.PREFERENCE_OPTIMIZATION,
                "DPO variant with explicit constraint handling for safety and other hard constraints. " +
                "Adds Lagrangian multiplier for constraint satisfaction.",
                "Balapour et al. (2024) - Constrained Preference Optimization",
                Set.of(DataFormat.CONSTRAINED_PREFERENCES, DataFormat.PREFERENCE_PAIRS),
                Set.of(Requirement.REFERENCE_MODEL, Requirement.PAIRED_DATA, Requirement.CONSTRAINTS),
                "Safety-critical applications or when explicit constraints must be satisfied.",
                null);

        // SFT
        register("sft", "Supervised Fine-Tuning",
                EnterpriseTrainer.AlgorithmCategory.SUPERVISED,
                "Standard supervised fine-tuning on demonstration data. Foundation for all other methods. " +
                "Often used as pre-training step before alignment algorithms.",
                "Standard practice in LLM training",
                Set.of(DataFormat.SFT_DEMONSTRATIONS),
                Set.of(Requirement.DEMONSTRATION_DATA),
                "Initial fine-tuning on curated demonstrations. Pre-requisite for RLHF.",
                null);
    }

    // ==================== Supporting Types ====================

    /**
     * Algorithm metadata.
     */
    public record AlgorithmInfo(
            String id,
            String name,
            EnterpriseTrainer.AlgorithmCategory category,
            String description,
            String paper,
            Set<DataFormat> supportedFormats,
            Set<Requirement> requirements,
            String recommendedUseCase
    ) {}

    /**
     * Supported data formats for different algorithms.
     */
    public enum DataFormat {
        /** Chosen and rejected response pairs */
        PREFERENCE_PAIRS,
        /** Precomputed log probabilities */
        PRECOMPUTED_LOGPS,
        /** Scalar rewards for each sample */
        REWARD_SCALARS,
        /** Grouped rewards for GRPO-style normalization */
        GROUP_REWARDS,
        /** Token-level advantage/reward signals */
        TOKEN_LEVEL_REWARDS,
        /** Binary accept/reject signals */
        BINARY_ACCEPTANCE,
        /** Preference data with constraints */
        CONSTRAINED_PREFERENCES,
        /** Ranked list of responses */
        RANKINGS,
        /** Generated responses to score */
        GENERATED_RESPONSES,
        /** SFT demonstration data */
        SFT_DEMONSTRATIONS,
        /** Advantages for policy gradient methods */
        ADVANTAGES,
        /** Value function estimates */
        VALUE_ESTIMATES,
        /** Sequential/multi-turn data */
        SEQUENTIAL,
        PAIRED_DATA
    }

    /**
     * Algorithm requirements/dependencies.
     */
    public enum Requirement {
        /** Requires a frozen reference model */
        REFERENCE_MODEL,
        /** Reference model optional but supported */
        REFERENCE_MODEL_OPTIONAL,
        /** Requires paired preference data */
        PAIRED_DATA,
        /** Requires a reward model for scoring */
        REWARD_MODEL,
        /** Requires value network for advantage estimation */
        VALUE_NETWORK,
        /** Requires text generation capability */
        GENERATION_CAPABILITY,
        /** Requires demonstration data */
        DEMONSTRATION_DATA,
        /** Requires constraint specification */
        CONSTRAINTS,
        /** Works best with sequential/multi-turn data */
        SEQUENTIAL
    }

    /**
     * Factory interface for creating trainer instances.
     */
    @FunctionalInterface
    public interface TrainerFactory {
        EnterpriseTrainer create(TrainerFactoryArgs args);
    }

    /**
     * Arguments for trainer factory.
     */
    public record TrainerFactoryArgs(
            org.bytedeco.pytorch.nn.Module policy,
            org.bytedeco.pytorch.nn.Module reference,
            org.bytedeco.pytorch.nn.Module rewardModel,
            org.bytedeco.pytorch.llm.trl.LlmForward policyForward,
            org.bytedeco.pytorch.llm.trl.LlmForward referenceForward,
            org.bytedeco.pytorch.optim.Optimizer optimizer,
            org.bytedeco.pytorch.llm.trl.config.TrainerConfig config
    ) {
        public static Builder builder() { return new Builder(); }

        public static final class Builder {
            private org.bytedeco.pytorch.nn.Module policy;
            private org.bytedeco.pytorch.nn.Module reference;
            private org.bytedeco.pytorch.nn.Module rewardModel;
            private org.bytedeco.pytorch.llm.trl.LlmForward policyForward;
            private org.bytedeco.pytorch.llm.trl.LlmForward referenceForward;
            private org.bytedeco.pytorch.optim.Optimizer optimizer;
            private org.bytedeco.pytorch.llm.trl.config.TrainerConfig config;

            public Builder policy(org.bytedeco.pytorch.nn.Module v) { this.policy = v; return this; }
            public Builder reference(org.bytedeco.pytorch.nn.Module v) { this.reference = v; return this; }
            public Builder rewardModel(org.bytedeco.pytorch.nn.Module v) { this.rewardModel = v; return this; }
            public Builder policyForward(org.bytedeco.pytorch.llm.trl.LlmForward v) { this.policyForward = v; return this; }
            public Builder referenceForward(org.bytedeco.pytorch.llm.trl.LlmForward v) { this.referenceForward = v; return this; }
            public Builder optimizer(org.bytedeco.pytorch.optim.Optimizer v) { this.optimizer = v; return this; }
            public Builder config(org.bytedeco.pytorch.llm.trl.config.TrainerConfig v) { this.config = v; return this; }

            public TrainerFactoryArgs build() {
                return new TrainerFactoryArgs(policy, reference, rewardModel,
                        policyForward, referenceForward, optimizer, config);
            }
        }
    }
}
