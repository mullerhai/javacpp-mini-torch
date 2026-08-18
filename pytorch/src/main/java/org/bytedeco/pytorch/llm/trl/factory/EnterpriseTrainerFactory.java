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
package org.bytedeco.pytorch.llm.trl.factory;

import org.bytedeco.pytorch.llm.trl.callback.TrainerCallback;
import org.bytedeco.pytorch.llm.trl.trainer.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.llm.trl.*;
import org.bytedeco.pytorch.llm.trl.algorithm.AlgorithmRegistry;
import org.bytedeco.pytorch.llm.trl.algorithm.AlgorithmRegistry.*;
import org.bytedeco.pytorch.llm.trl.algorithm.EnterpriseTrainer;
import org.bytedeco.pytorch.llm.trl.algorithm.TrainingState;
import org.bytedeco.pytorch.llm.trl.callback.CallbackManager;
import org.bytedeco.pytorch.llm.trl.checkpoint.CheckpointManager;
import org.bytedeco.pytorch.llm.trl.config.*;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import java.nio.file.Path;
import java.util.*;

/**
 * Enterprise factory for creating and managing RLHF trainers.
 *
 * <p>Provides a unified entry point for:
 * <ul>
 *   <li>Creating trainers for all supported algorithms</li>
 *   <li>Configuring optimizers and learning rate schedules</li>
 *   <li>Setting up callbacks and checkpointing</li>
 *   <li>Managing trainer lifecycle</li>
 * </ul>
 *
 * <p>Supported algorithms:
 * <ul>
 *   <li><b>Preference Optimization</b>: DPO, SimPO, IPO, TDPO, ORPO, KTO</li>
 *   <li><b>Policy Gradient</b>: PPO, GRPO, RLOO, Nash-MD</li>
 *   <li><b>Rank-based</b>: RRHF, RAFT</li>
 *   <li><b>Supervised</b>: SFT</li>
 * </ul>
 *
 * <pre>{@code
 * // Create a DPO trainer
 * EnterpriseTrainer trainer = EnterpriseTrainerFactory.builder("dpo")
 *     .policy(policyModel)
 *     .reference(referenceModel)
 *     .config(DPOConfig.builder().beta(0.1).build())
 *     .optimizer(new Adam(policy.parameters(), new AdamOptions().lr(1e-6)))
 *     .build();
 *
 * // Or create a GRPO trainer
 * EnterpriseTrainer grpoTrainer = EnterpriseTrainerFactory.builder("grpo")
 *     .policy(policyModel)
 *     .config(GRPOConfig.builder().numGenerations(8).build())
 *     .build();
 * }</pre>
 */
public final class EnterpriseTrainerFactory {

    // ==================== Static Factory Methods ====================

    /**
     * Create a trainer for the specified algorithm.
     *
     * @param algorithm Algorithm ID (e.g., "dpo", "simpo", "grpo")
     */
    public static TrainerBuilder builder(String algorithm) {
        return new TrainerBuilder(algorithm);
    }

    /**
     * Get list of supported algorithm IDs.
     */
    public static Set<String> supportedAlgorithms() {
        return AlgorithmRegistry.getInstance().listAlgorithms();
    }

    /**
     * Get algorithm info.
     */
    public static AlgorithmInfo getAlgorithmInfo(String algorithm) {
        return AlgorithmRegistry.getInstance().getInfo(algorithm);
    }

    // ==================== Builder ====================

    /**
     * Fluent builder for creating trainers.
     */
    public static final class TrainerBuilder {
        private final String algorithm;
        private Module policy;
        private Module reference;
        private Module rewardModel;
        private LlmForward policyForward;
        private LlmForward referenceForward;
        private Optimizer optimizer;
        private TrainerConfig config;

        // Callback and checkpointing
        private CallbackManager callbackManager;
        private CheckpointManager checkpointManager;

        // Training metadata
        private Path outputDir;
        private Map<String, String> metadata = new HashMap<>();

        TrainerBuilder(String algorithm) {
            this.algorithm = Objects.requireNonNull(algorithm, "algorithm");
        }

        // ==================== Model Configuration ====================

        /**
         * Set the policy model.
         */
        public TrainerBuilder policy(Module model) {
            this.policy = Objects.requireNonNull(model, "policy model");
            return this;
        }

        /**
         * Set the reference model (for DPO, PPO, etc.).
         */
        public TrainerBuilder reference(Module model) {
            this.reference = model;
            return this;
        }

        /**
         * Set the reward model (for GRPO, PPO, RRHF, etc.).
         */
        public TrainerBuilder rewardModel(Module model) {
            this.rewardModel = model;
            return this;
        }

        /**
         * Set the policy forward function for LLM forward pass.
         */
        public TrainerBuilder policyForward(LlmForward forward) {
            this.policyForward = forward;
            return this;
        }

        /**
         * Set the reference forward function.
         */
        public TrainerBuilder referenceForward(LlmForward forward) {
            this.referenceForward = forward;
            return this;
        }

        // ==================== Optimizer Configuration ====================

        /**
         * Set the optimizer.
         */
        public TrainerBuilder optimizer(Optimizer opt) {
            this.optimizer = opt;
            return this;
        }

        /**
         * Create and set an Adam optimizer with specified learning rate.
         */
        public TrainerBuilder adam(double lr) {
            if (policy == null) {
                throw new IllegalStateException("Policy model must be set before creating optimizer");
            }
            AdamOptions options = new AdamOptions();
            options.lr().put(lr);
            this.optimizer = new Adam(policy.parameters(), options);
            return this;
        }

        /**
         * Create and set an AdamW optimizer with specified learning rate.
         */
        public TrainerBuilder adamW(double lr) {
            if (policy == null) {
                throw new IllegalStateException("Policy model must be set before creating optimizer");
            }
            var options = new org.bytedeco.pytorch.optim.options.AdamWOptions();
            options.lr().put(lr);
            this.optimizer = new org.bytedeco.pytorch.optim.AdamW(policy.parameters(), options);
            return this;
        }

        // ==================== Configuration ====================

        /**
         * Set the trainer configuration.
         */
        public TrainerBuilder config(TrainerConfig cfg) {
            this.config = cfg;
            return this;
        }

        /**
         * Set DPO-specific config.
         */
        public TrainerBuilder config(DPOConfig cfg) { return config((TrainerConfig) cfg); }

        /**
         * Set SimPO-specific config.
         */
        public TrainerBuilder config(SimPOConfig cfg) { return config((TrainerConfig) cfg); }

        /**
         * Set PPO-specific config.
         */
        public TrainerBuilder config(PPOConfig cfg) { return config((TrainerConfig) cfg); }

        /**
         * Set GRPO-specific config.
         */
        public TrainerBuilder config(GRPOConfig cfg) { return config((TrainerConfig) cfg); }

        /**
         * Set KTO-specific config.
         */
        public TrainerBuilder config(KTOConfig cfg) { return config((TrainerConfig) cfg); }

        // ==================== Callback & Checkpointing ====================

        /**
         * Set callback manager.
         */
        public TrainerBuilder callbacks(CallbackManager manager) {
            this.callbackManager = manager;
            return this;
        }

        /**
         * Set checkpoint manager.
         */
        public TrainerBuilder checkpointing(CheckpointManager manager) {
            this.checkpointManager = manager;
            return this;
        }

        /**
         * Enable automatic checkpointing.
         */
        public TrainerBuilder checkpointing(Path outputDir, int saveInterval) {
            this.checkpointManager = CheckpointManager.builder(outputDir)
                    .saveInterval(saveInterval)
                    .build();
            return this;
        }

        // ==================== Metadata ====================

        /**
         * Add metadata for logging/tracking.
         */
        public TrainerBuilder metadata(String key, String value) {
            this.metadata.put(key, value);
            return this;
        }

        /**
         * Set output directory for logs/checkpoints.
         */
        public TrainerBuilder outputDir(Path dir) {
            this.outputDir = dir;
            return this;
        }

        // ==================== Build ====================

        /**
         * Build the trainer.
         */
        public BaseTrainer build() {
            // Validate
            validate();

            // Create trainer based on algorithm
            BaseTrainer trainer = createTrainer();

            // Setup callbacks
            if (callbackManager != null) {
                trainer.addCallback(new CallbackAdapter(callbackManager));
            }

            return trainer;
        }

        /**
         * Build and return as EnterpriseTrainer interface.
         */
        public EnterpriseTrainer buildEnterprise() {
            return new EnterpriseTrainerAdapter(build());
        }

        private void validate() {
            if (policy == null) {
                throw new IllegalStateException("Policy model is required");
            }
            if (optimizer == null) {
                // Use default Adam with reasonable LR
                adam(1e-5);
            }
            if (config == null) {
                // Use default config based on algorithm
                config = createDefaultConfig();
            }
        }

        private TrainerConfig createDefaultConfig() {
            return switch (algorithm.toLowerCase()) {
                case "dpo" -> DPOConfig.builder().build();
                case "simpo" -> SimPOConfig.builder().build();
                case "ipo" -> IPOConfig.builder().build();
                case "tdpo" -> TDPOConfig.builder().build();
                case "orpo" -> ORPOConfig.builder().build();
                case "kto" -> KTOConfig.builder().build();
                case "grpo" -> GRPOConfig.builder().build();
                case "rloo" -> RLOOConfig.builder().build();
                case "ppo" -> PPOConfig.builder().build();
                case "rrhf" -> RRHFConfig.builder().build();
                case "sft" -> SFTConfig.builder().build();
                default -> TrainerConfig.builder().build();
            };
        }

        private BaseTrainer createTrainer() {
            return switch (algorithm.toLowerCase()) {
                case "dpo" -> createDPOTrainer();
                case "simpo" -> createSimPOTrainer();
                case "ipo" -> createIPOTrainer();
                case "tdpo" -> createTDPOTrainer();
                case "orpo" -> createORPOTrainer();
                case "kto" -> createKTOTrainer();
                case "grpo" -> createGRPOTrainer();
                case "rloo" -> createRLOOTrainer();
                case "ppo" -> createPPOTrainer();
                case "rrhf" -> createRRHFTrainer();
                case "spo" -> createSPOTrainer();
                case "nash-md" -> createNashMDTrainer();
                case "sft" -> createSFTTrainer();
                default -> throw new IllegalArgumentException(
                        "Unsupported algorithm: " + algorithm +
                        ". Supported: " + supportedAlgorithms());
            };
        }

        private BaseTrainer createDPOTrainer() {
            return new DPOTrainer(policy, policyForward, reference, referenceForward, optimizer,
                    config instanceof DPOConfig ? (DPOConfig) config : DPOConfig.builder().build());
        }

        private BaseTrainer createSimPOTrainer() {
            return new SimPOTrainer(policy, policyForward, optimizer,
                    config instanceof SimPOConfig ? (SimPOConfig) config : SimPOConfig.builder().build());
        }

        private BaseTrainer createIPOTrainer() {
            return new IPOTrainer(policy, policyForward, reference, referenceForward, optimizer,
                    config instanceof IPOConfig ? (IPOConfig) config : IPOConfig.builder().build());
        }

        private BaseTrainer createTDPOTrainer() {
            return new TDPOTrainer(policy, policyForward, reference, referenceForward, optimizer,
                    config instanceof TDPOConfig ? (TDPOConfig) config : TDPOConfig.builder().build());
        }

        private BaseTrainer createORPOTrainer() {
            return new ORPOTrainer(policy, policyForward, optimizer,
                    config instanceof ORPOConfig ? (ORPOConfig) config : ORPOConfig.builder().build());
        }

        private BaseTrainer createKTOTrainer() {
            return new KTOTrainer(policy, policyForward, reference, referenceForward, optimizer,
                    config instanceof KTOConfig ? (KTOConfig) config : KTOConfig.builder().build());
        }

        private BaseTrainer createGRPOTrainer() {
            return new GRPOTrainer(policy, policyForward, optimizer,
                    config instanceof GRPOConfig ? (GRPOConfig) config : GRPOConfig.builder().build());
        }

        private BaseTrainer createRLOOTrainer() {
            return new RLOOTrainer(policy, policyForward, reference, referenceForward, optimizer,
                    config instanceof RLOOConfig ? (RLOOConfig) config : RLOOConfig.builder().build());
        }

        private BaseTrainer createPPOTrainer() {
            // PPOTrainer needs PolicyValueForward, not LlmForward
            // Use precomputed mode or adapt LlmForward to PolicyValueForward
            PPOTrainer.PolicyValueForward pvForward = null;
            if (policyForward != null) {
                final LlmForward fwd = policyForward;
                pvForward = (inputIds, attentionMask) -> {
                    var output = fwd.forward(inputIds, attentionMask);
                    return new PPOTrainer.PolicyValueOutput(output, null);
                };
            }
            return new PPOTrainer(
                    policy, pvForward, optimizer,
                    config instanceof PPOConfig ? (PPOConfig) config : PPOConfig.builder().build());
        }

        private BaseTrainer createRRHFTrainer() {
            return new RRHFTrainer(policy, policyForward, rewardModel, optimizer,
                    config instanceof RRHFConfig ? (RRHFConfig) config : RRHFConfig.builder().build());
        }

        private BaseTrainer createSPOTrainer() {
            return new SPOTrainer(policy, policyForward, optimizer,
                    config instanceof SPOConfig ? (SPOConfig) config : SPOConfig.builder().build());
        }

        private BaseTrainer createNashMDTrainer() {
            return new NashMDTrainer(policy, policyForward, reference, referenceForward, optimizer,
                    config instanceof NashMDConfig ? (NashMDConfig) config : NashMDConfig.builder().build());
        }

        private BaseTrainer createSFTTrainer() {
            return new SFTTrainer(policy, policyForward, optimizer,
                    config instanceof SFTConfig ? (SFTConfig) config : SFTConfig.builder().build());
        }
    }

    // ==================== Adapters ====================

    /**
     * Adapter to make BaseTrainer implement EnterpriseTrainer.
     */
    private static class EnterpriseTrainerAdapter implements EnterpriseTrainer {
        private final BaseTrainer delegate;
        private final String algoId;

        EnterpriseTrainerAdapter(BaseTrainer delegate) {
            this.delegate = Objects.requireNonNull(delegate);
            this.algoId = delegate.getClass().getSimpleName().replace("Trainer", "").toLowerCase();
        }

        @Override public String algorithmId() { return algoId; }
        @Override public AlgorithmCategory category() { return AlgorithmCategory.PREFERENCE_OPTIMIZATION; }
        @Override public Module policy() { return null; }
        @Override public Module reference() { return null; }
        @Override public Module rewardModel() { return null; }
        @Override public Optimizer optimizer() { return delegate.optimizer(); }
        @Override public TrainerConfig config() { return delegate.config(); }
        @Override public int globalStep() { return delegate.globalStep(); }
        @Override public Map<String, Double> getMetrics() { return Map.of("step", (double) globalStep()); }
        @Override public TrainingState getState() { return TrainingState.builder().build(); }
        @Override public boolean isClosed() { return delegate.isClosed(); }
        @Override public void train() { delegate.train(); }
        @Override public void eval() { delegate.eval(); }
        @Override public boolean isTraining() { return delegate.isTraining(); }
        @Override public double trainingStep(Map<String, org.bytedeco.pytorch.Tensor> batch) {
            return delegate.trainingStep(batch);
        }
        @Override public org.bytedeco.pytorch.Tensor computeLoss(Map<String, org.bytedeco.pytorch.Tensor> batch) {
            return delegate.computeLoss(batch);
        }
        @Override public void loadState(TrainingState state) {}
        @Override public void resetMetrics() {}
        @Override public void close() { delegate.close(); }
    }

    /**
     * Adapter to connect CallbackManager to TrainerCallback.
     */
    private static class CallbackAdapter implements TrainerCallback {
        private final CallbackManager manager;

        CallbackAdapter(CallbackManager manager) {
            this.manager = manager;
        }

        @Override
        public void onStepEnd(BaseTrainer trainer, int step, java.util.Map<String, Double> metrics) {
            manager.record("loss", metrics.getOrDefault("loss", 0.0), step);
        }
    }
}
