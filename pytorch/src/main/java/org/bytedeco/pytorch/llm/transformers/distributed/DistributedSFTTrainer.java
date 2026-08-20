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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.distributed;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.config.DistributedConfig;
import org.bytedeco.pytorch.llm.transformers.loading.SnapshotFiles;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.llm.transformers.mapping.WeightMap;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Distributed SFT training facade for causal LMs.
 *
 * <p>Glues {@link DistributedCausalLM} (load) with the project's
 * {@code org.bytedeco.pytorch.distributed} package (DDP / FSDP / hybrid /
 * ZeRO trainers) and the {@code trl.trainer} family (SFTTrainer etc.) into a
 * single, HF {@code Trainer}-style entry point.
 *
 * <p>This class is the JavaCPP-Mini analog of HF's
 * {@code transformers.Trainer} + {@code accelerate.prepare()} — except that
 * all model-parallel primitives are already implemented in
 * {@code org.bytedeco.pytorch.distributed}.
 *
 * <h3>Usage</h3>
 *
 * <pre>{@code
 * // 1) Distributed model load
 * DistributedConfig cfg = DistributedConfig.builder().rank(0).worldSize(8).backend("NCCL").build();
 * try (org.bytedeco.pytorch.distributed.DistributedStore store =
 *          org.bytedeco.pytorch.distributed.DistributedStore.create(0, 8);
 *      ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 8, store);
 *      DistributedCausalLM.Bundle loaded = DistributedCausalLM.fromPretrained(
 *          "Qwen/Qwen2-7B", hub, cfg, pg)) {
 *
 *     // 2) Pick a parallel strategy. Available: DDP, FSDP, HybridParallel,
 *     //    SequenceParallel, PipelineParallel, ExpertParallel, ZeRO, …
 *     Strategy strategy = Strategy.FSDP_FULL_SHARD;
 *
 *     // 3) Train
 *     try (DistributedSFTTrainer trainer = new DistributedSFTTrainer(
 *              loaded, pg, strategy,
 *              SFTConfig.builder().learningRate(2e-5).build())) {
 *         trainer.fit(inputIds, labels, /*optimizer* / new AdamW(...));
 *     }
 * }
 * }</pre>
 */
public final class DistributedSFTTrainer implements AutoCloseable {

    public enum Strategy {
        /** DDP — replicate model on each rank, all-reduce gradients. */
        DDP,
        /** FSDP full-shard — shard params, all-gather forward, reduce-scatter backward. */
        FSDP_FULL_SHARD,
        /** FSDP no-shard — replicate across ranks (memory-intensive but fast). */
        FSDP_NO_SHARD,
        /** Hybrid DP×TP (use {@code EnterpriseHybridTrainer}). */
        HYBRID_DP_TP,
        /** Pipeline parallelism — split layers across ranks. */
        PIPELINE,
        /** Sequence parallel — long-context splitting. */
        SEQUENCE_PARALLEL,
        /** Expert parallel — MoE router sharding. */
        EXPERT_PARALLEL,
        /** ZeRO-style optimizer state sharding. */
        ZERO
    }

    private final DistributedCausalLM.Bundle bundle;
    private final ProcessGroupWrapper processGroup;
    private final Strategy strategy;
    private final Object config;
    private final Module wrapped;
    private final Object trainer;
    private volatile boolean closed;

    public DistributedSFTTrainer(DistributedCausalLM.Bundle bundle,
                                 ProcessGroupWrapper processGroup,
                                 Strategy strategy,
                                 Object sftConfig) {
        this.bundle = Objects.requireNonNull(bundle);
        this.processGroup = Objects.requireNonNull(processGroup);
        this.strategy = strategy;
        this.config = sftConfig;
        this.wrapped = wrapModel(bundle, processGroup, strategy);
        this.trainer = buildTrainer(this.wrapped, processGroup, strategy);
    }

    private static Module wrapModel(DistributedCausalLM.Bundle bundle, ProcessGroupWrapper pg, Strategy strategy) {
        Module base = bundle.model();
        switch (strategy) {
            case DDP: {
                org.bytedeco.pytorch.distributed.trainer.DDPTrainer ddp =
                        new org.bytedeco.pytorch.distributed.trainer.DDPTrainer(base, pg);
                // DDP itself is a trainer, not a Module wrapper. We expose the
                // underlying module as wrapped (the trainer holds it).
                return base;
            }
            case FSDP_FULL_SHARD:
            case FSDP_NO_SHARD: {
                var strat = (strategy == Strategy.FSDP_NO_SHARD)
                        ? org.bytedeco.pytorch.distributed.enums.ShardingStrategy.NO_SHARD
                        : org.bytedeco.pytorch.distributed.enums.ShardingStrategy.FULL_SHARD;
                org.bytedeco.pytorch.distributed.trainer.FSDPTrainer fsdp =
                        new org.bytedeco.pytorch.distributed.trainer.FSDPTrainer(base, pg, strat, true, true);
                return base;
            }
            case HYBRID_DP_TP: {
                // Use the bundled EnterpriseHybridTrainer; user supplies the mesh.
                // Here we just record the user's intent; the trainer holds the model.
                return base;
            }
            case PIPELINE: {
                // PipelineParallelTrainer splits layers across ranks.
                return base;
            }
            case SEQUENCE_PARALLEL: {
                org.bytedeco.pytorch.distributed.trainer.SequenceParallelTrainer sp =
                        new org.bytedeco.pytorch.distributed.trainer.SequenceParallelTrainer(base, pg);
                return base;
            }
            case EXPERT_PARALLEL: {
                org.bytedeco.pytorch.distributed.trainer.ExpertParallelTrainer ep =
                        new org.bytedeco.pytorch.distributed.trainer.ExpertParallelTrainer(base, pg, 8, 2, base.parameters().size() > 0 ? 1L : 1L, 1L);
                return base;
            }
            case ZERO: {
                org.bytedeco.pytorch.distributed.trainer.ZeroRedundancyOptimizerTrainer zr =
                        new org.bytedeco.pytorch.distributed.trainer.ZeroRedundancyOptimizerTrainer(base, pg);
                return base;
            }
            default:
                throw new IllegalArgumentException("Unknown strategy: " + strategy);
        }
    }

    private static Object buildTrainer(Module wrapped, ProcessGroupWrapper pg, Strategy strategy) {
        return switch (strategy) {
            case DDP -> new org.bytedeco.pytorch.distributed.trainer.DDPTrainer(wrapped, pg);
            case FSDP_FULL_SHARD, FSDP_NO_SHARD -> new org.bytedeco.pytorch.distributed.trainer.FSDPTrainer(
                    wrapped, pg,
                    (strategy == Strategy.FSDP_NO_SHARD
                            ? org.bytedeco.pytorch.distributed.enums.ShardingStrategy.NO_SHARD
                            : org.bytedeco.pytorch.distributed.enums.ShardingStrategy.FULL_SHARD),
                    true, true);
            case SEQUENCE_PARALLEL -> new org.bytedeco.pytorch.distributed.trainer.SequenceParallelTrainer(wrapped, pg);
            case EXPERT_PARALLEL -> new org.bytedeco.pytorch.distributed.trainer.ExpertParallelTrainer(
                    wrapped, pg, 8, 2, 1L, 1L);
            case ZERO -> new org.bytedeco.pytorch.distributed.trainer.ZeroRedundancyOptimizerTrainer(wrapped, pg);
            case HYBRID_DP_TP -> null; // user wires it via EnterpriseHybridTrainer explicitly
            case PIPELINE -> null;     // user wires it via PipelineParallelTrainer explicitly
        };
    }

    /**
     * Run a single step: forward, loss, backward, optimizer step. Returns the loss.
     */
    public Tensor step(Tensor inputIds, Tensor labels, Optimizer optimizer) {
        switch (strategy) {
            case DDP:
                return ((org.bytedeco.pytorch.distributed.trainer.DDPTrainer) trainer)
                        .step(inputIds, labels, optimizer);
            case FSDP_FULL_SHARD:
            case FSDP_NO_SHARD:
                return ((org.bytedeco.pytorch.distributed.trainer.FSDPTrainer) trainer)
                        .step(inputIds, labels, optimizer);
            case SEQUENCE_PARALLEL:
                return ((org.bytedeco.pytorch.distributed.trainer.SequenceParallelTrainer) trainer)
                        .step(inputIds, labels, optimizer);
            case EXPERT_PARALLEL:
                return ((org.bytedeco.pytorch.distributed.trainer.ExpertParallelTrainer) trainer)
                        .step(inputIds, labels, optimizer);
            case ZERO:
                return ((org.bytedeco.pytorch.distributed.trainer.ZeroRedundancyOptimizerTrainer) trainer)
                        .step(inputIds, labels, optimizer);
            default:
                throw new UnsupportedOperationException("Strategy " + strategy
                        + " does not have a default step; please use the corresponding trainer directly.");
        }
    }

    /**
     * Run a single step with explicit loss tensor (already computed by caller).
     *
     * <p>Note: as of the current trainer API, only the {@code step(input, target, optimizer)}
     * entry point is supported by {@link org.bytedeco.pytorch.distributed.trainer.DDPTrainer}
     * and its peers. This method is provided as a convenience that backpropagates the
     * provided {@code loss} manually. If your model output already provides a scalar
     * loss, prefer calling {@link #step(Tensor, Tensor, Optimizer)} or invoking the
     * trainer's {@code step(...)} overload that takes an explicit target.
     */
    public void stepWithLoss(Tensor loss, Optimizer optimizer) {
        // Fallback path: since the underlying trainers don't expose stepWithLoss,
        // we perform the backward + optimizer step manually here.
        Objects.requireNonNull(loss, "loss");
        Objects.requireNonNull(optimizer, "optimizer");
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    /** Barrier synchronisation across the process group. */
    public void barrier() {
        if (processGroup.getWorldSize() > 1) processGroup.barrierWait();
    }

    public DistributedCausalLM.Bundle bundle() { return bundle; }
    public ProcessGroupWrapper processGroup() { return processGroup; }
    public Strategy strategy() { return strategy; }
    public Object trainer() { return trainer; }

    @Override public void close() {
        if (closed) return;
        closed = true;
        try { bundle.close(); } catch (Throwable ignored) {}
    }
}