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
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.*;

/**
 * Enterprise-Grade Hybrid Trainer with Full Production Features.
 *
 * <p>This trainer integrates all enterprise features for large-scale distributed training:
 * <ul>
 *   <li><b>Pipeline Parallelism</b>: Async 1F1B scheduling with micro-batches</li>
 *   <li><b>Tensor Parallelism</b>: TP for attention and FFN layers</li>
 *   <li><b>Expert Parallelism</b>: EP for MoE with load balancing</li>
 *   <li><b>Activation Checkpointing</b>: Memory-efficient recomputation</li>
 *   <li><b>Gradient Accumulation</b>: Large effective batch sizes</li>
 *   <li><b>Mixed Precision</b>: FP16 with loss scaling</li>
 *   <li><b>Checkpoint/Resume</b>: Fault tolerance</li>
 *   <li><b>Distributed Profiling</b>: Performance analysis</li>
 * </ul>
 *
 * <p>Architecture:
 * <pre>{@code
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    EnterpriseHybridTrainer                       │
 * ├─────────────────────────────────────────────────────────────────┤
 * │  AsyncPipeline      │  Pipeline parallelism with 1F1B           │
 * │  ParallelLayers     │  TP/DP/EP/SP parallelism                  │
 * │  ActivationCheckpoint│  Memory optimization                      │
 * │  GradientAccumulator │  Large batch support                      │
 * │  MixedPrecision      │  FP16 with loss scaling                   │
 * │  CheckpointManager   │  Fault tolerance                          │
 * │  ExpertLoadBalancer  │  MoE load balancing                       │
 * │  DistributedProfiler │  Performance analysis                     │
 * └─────────────────────────────────────────────────────────────────┘
 * }</pre>
 *
 * <p>Example usage:
 * <pre>{@code
 * EnterpriseHybridTrainer trainer = EnterpriseHybridTrainer.builder()
 *     .processGroup(pg)
 *     .numStages(4)
 *     .numMicroBatches(16)
 *     .accumulationSteps(8)
 *     .enableMixedPrecision(true)
 *     .enableCheckpointing(true)
 *     .checkpointDir("./checkpoints")
 *     .enableProfiling(true)
 *     .build();
 *
 * // Training
 * for (DataBatch batch : dataloader) {
 *     trainer.trainStep(batch);
 * }
 *
 * // Save checkpoint
 * trainer.saveCheckpoint();
 *
 * // Resume
 * trainer.loadCheckpoint();
 *
 * // Profiling report
 * trainer.printProfileReport();
 *
 * trainer.close();
 * }</pre>
 */
public final class EnterpriseHybridTrainer implements AutoCloseable {
    // Core components
    private final AsyncPipeline pipeline;
    private final ParallelLayers.HybridTrainer model;
    private final ProcessGroupWrapper pg;

    // Enterprise features
    private final ActivationCheckpoint checkpoint;
    private final GradientAccumulator gradientAccumulator;
    private final MixedPrecisionManager mixedPrecision;
    private final CheckpointManager checkpointManager;
    private final ExpertLoadBalancer loadBalancer;
    private final DistributedProfiler profiler;

    // Configuration
    private final Config config;
    private final int worldSize;
    private final int rank;

    // State
    private final AtomicLong globalStep;
    private final AtomicLong epochStep;
    private int currentEpoch = 0;
    private volatile boolean closed = false;

    // Training statistics
    private final List<Double> lossHistory = new ArrayList<>();
    private final List<Double> throughputHistory = new ArrayList<>();

    private EnterpriseHybridTrainer(Config config) throws Exception {
        this.config = config;
//        this.pg = config.pg;
        // Initialize process group (uses config.pg if already set, else creates default)
        this.pg = initProcessGroup();

        this.worldSize = pg != null ? pg.getWorldSize() : 1;
        this.rank = pg != null ? pg.getRank() : 0;

        this.globalStep = new AtomicLong(0);
        this.epochStep = new AtomicLong(0);

        System.out.printf("""
                ╔══════════════════════════════════════════════════════════════════╗
                ║         Enterprise Hybrid Trainer Initialization                  ║
                ╠══════════════════════════════════════════════════════════════════╣
                ║  World Size:    %3d                                             ║
                ║  Pipeline:      %s × %s stages                                   ║
                ║  Micro Batches: %d                                               ║
                ║  Accumulation:  %d                                               ║
                ║  Mixed Prec:    %s                                               ║
                ║  Checkpoint:    %s                                               ║
                ║  Profiling:     %s                                               ║
                ╚══════════════════════════════════════════════════════════════════╝
                """,
                worldSize,
                config.numPipelineStages, config.numMicroBatches,
                config.numMicroBatches,
                config.accumulationSteps,
                config.enableMixedPrecision,
                config.enableCheckpointing,
                config.enableProfiling
        );

        // Initialize profiler first
        this.profiler = createProfiler();


        // Initialize model
        this.model = initModel();

        // Initialize pipeline
        this.pipeline = createPipeline();

        // Initialize activation checkpointing
        this.checkpoint = createActivationCheckpoint();

        // Initialize gradient accumulator
        this.gradientAccumulator = createGradientAccumulator();

        // Initialize mixed precision
        this.mixedPrecision = createMixedPrecision();

        // Initialize checkpoint manager
        this.checkpointManager = createCheckpointManager();

        // Initialize load balancer (for MoE)
        this.loadBalancer = createLoadBalancer();

        System.out.printf("[Trainer] Initialization complete (rank %d)%n", rank);
    }

    public static Builder builder() { return new Builder(); }

    // ═══════════════════════════════════════════════════════════════════════════
    // Initialization
    // ═══════════════════════════════════════════════════════════════════════════

    private ProcessGroupWrapper initProcessGroup() {
        if (pg != null) return pg;

        // Create default local process group
        DistributedStore store = DistributedStore.createSingleProcess();
        ProcessGroupWrapper.Options opts = new ProcessGroupWrapper.Options()
                .forceCollective(config.enableProfiling)
                .syncCollectives(true);

        return ProcessGroupWrapper.create(opts, 0, 1, store);
    }

    private ParallelLayers.HybridTrainer initModel() {
        // Get or create mesh
        DeviceMesh mesh;
        if (config.tpSize > 1 || config.epSize > 1) {
            if (config.epSize > 1) {
                mesh = ParallelLayers.initDpTpEp(pg, config.tpSize, config.epSize);
            } else {
                mesh = ParallelLayers.initDpTp(pg, config.tpSize);
            }
        } else {
            mesh = DeviceMesh.init(pg, new int[]{1}, new String[]{"dp"});
        }

        return new ParallelLayers.HybridTrainer(
                mesh,
                config.vocabSize,
                config.hiddenDim,
                config.numHeads,
                config.intermediateDim,
                config.numExperts,
                config.topK,
                config.maxSeqLen
        );
    }

    private AsyncPipeline createPipeline() {
        if (config.numPipelineStages <= 1) {
            return null;
        }

        return AsyncPipeline.builder()
                .processGroup(pg)
                .numStages(config.numPipelineStages)
                .numMicroBatches(config.numMicroBatches)
                .enableOverlap(config.enableOverlap)
                .maxMemoryBytes(config.maxPipelineMemory)
                .build();
    }

    private ActivationCheckpoint createActivationCheckpoint() {
        if (!config.enableActivationCheckpointing) {
            return null;
        }

        ActivationCheckpoint.Builder builder = ActivationCheckpoint.builder()
                .strategy(ActivationCheckpoint.CheckpointStrategy.SELECTIVE)
                .maxMemoryBytes(config.maxCheckpointMemory)
                .trackStats(config.enableProfiling);

        // Selectively checkpoint heavy layers
        if (config.checkpointLayers != null) {
            builder.checkpointLayers(new HashSet<>(config.checkpointLayers));
        } else {
            builder.checkpointLayers("attention", "ffn", "moe");
        }

        return builder.build();
    }

    private GradientAccumulator createGradientAccumulator() {
        GradientAccumulator.Builder builder = GradientAccumulator.builder()
                .accumulationSteps(config.accumulationSteps)
                .maxGradNorm(config.maxGradNorm)
                .processGroup(pg)
                .syncBeforeStep(config.syncBeforeStep);

        if (config.enableProfiling) {
            builder.onStepComplete(acc -> {
                profiler.recordCounter("gradient_accumulation_steps", acc.getMicroBatchCount());
                profiler.recordCounter("optimizer_steps", acc.getOptimizerStepCount());
            });
        }

        return builder.build();
    }

    private MixedPrecisionManager createMixedPrecision() {
        if (!config.enableMixedPrecision) {
            return null;
        }

        return MixedPrecisionManager.builder()
                .initialScale(config.initialLossScale)
                .growthFactor(config.lossScaleGrowthFactor)
                .backoffFactor(config.lossScaleBackoffFactor)
                .growthInterval(config.lossScaleGrowthInterval)
                .processGroup(pg)
                .build();
    }

    private CheckpointManager createCheckpointManager() throws Exception {
        if (!config.enableCheckpointing) {
            return null;
        }

        CheckpointManager.Builder builder = CheckpointManager.builder()
                .checkpointDir(config.checkpointDir)
                .maxCheckpoints(config.maxCheckpoints)
                .saveIntervalSteps(config.checkpointInterval)
                .asyncSave(config.asyncCheckpointing)
                .compressCheckpoints(config.compressCheckpoints)
                .processGroup(pg);

        return builder.build();
    }

    private ExpertLoadBalancer createLoadBalancer() {
        if (config.numExperts <= 0) {
            return null;
        }

        return ExpertLoadBalancer.builder()
                .numExperts(config.numExperts)
                .numEPProcesses(config.epSize)
                .strategy(ExpertLoadBalancer.BalanceStrategy.HYBRID)
                .targetLoadFactor(config.targetLoadFactor)
                .adjustmentInterval(config.loadBalanceInterval)
                .processGroup(pg)
                .build();
    }

    private DistributedProfiler createProfiler() {
        if (!config.enableProfiling) {
            return DistributedProfiler.builder()
                    .enabled(false)
                    .build();
        }

        return DistributedProfiler.builder()
                .enabled(true)
                .traceOperations(true)
                .profileMemory(config.profileMemory)
                .recordShallowFlops(config.recordFlops)
                .outputDir(config.profileDir)
                .maxEvents(config.maxProfilerEvents)
                .processGroup(pg)
                .build();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Training Loop
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Single training step.
     *
     * @param input input tensor [batch, seq_len]
     * @param target target tensor [batch, seq_len]
     * @return loss tensor
     */
    public Tensor trainStep(Tensor input, Tensor target) {
        return trainStep(input, target, null);
    }

    /**
     * Training step with optimizer.
     */
    public Tensor trainStep(Tensor input, Tensor target, Optimizer optimizer) {
        if (closed) throw new IllegalStateException("Trainer is closed");

        long stepStart = System.nanoTime();

        try (DistributedProfiler.ProfilerRegion region = profiler.profile("train_step")) {
            // Forward pass
            Tensor loss = forward(input, target);

            // Backward pass
            backward(loss);

            // Optimizer step (if optimizer provided)
            if (optimizer != null) {
                step(optimizer);
            }

            // Update statistics
            updateStats(loss, stepStart);

            return loss;

        } catch (Exception e) {
            profiler.recordError("train_step", e);
            throw new RuntimeException("Training step failed", e);
        }
    }

    /**
     * Forward pass with all enterprise features.
     */
    public Tensor forward(Tensor input, Tensor target) {
        // Compute everything inside the profiler lambda so we still
        // measure the full forward pass. We use a one-element array
        // holder (`xRef`) to satisfy the "effectively final" capture
        // rule for the nested lambdas passed to
        // {@link ActivationCheckpoint#checkpoint(String, Supplier)}.
        return profiler.profile("forward", () -> {
            final Tensor[] xRef = new Tensor[]{input};

            // Mixed precision casting
            if (mixedPrecision != null && input.scalar_type() != org.bytedeco.pytorch.global.torch.ScalarType.Half) {
                xRef[0] = xRef[0].to(org.bytedeco.pytorch.global.torch.ScalarType.Half);
            }

            // Apply activation checkpointing
            if (checkpoint != null) {
                xRef[0] = checkpoint.checkpoint("embedding", () -> model.embedding.forward(xRef[0]));
                xRef[0] = checkpoint.checkpoint("prefill", () -> model.prefill.forward(xRef[0]));
                xRef[0] = checkpoint.checkpoint("ffn", () -> model.ffn.forward(xRef[0]));
                xRef[0] = checkpoint.checkpoint("moe", () -> model.moe.forward(xRef[0]));
            } else {
                xRef[0] = model.embedding.forward(xRef[0]);
                xRef[0] = model.prefill.forward(xRef[0]);
                xRef[0] = model.ffn.forward(xRef[0]);
                xRef[0] = model.moe.forward(xRef[0]);
            }

            // MoE load-balancing hook (reserved for future use)
            if (loadBalancer != null && config.enableLoadBalancing) {
                // Routing would happen in moe.forward(), apply balancer there
            }

            // Snapshot the final tensor and target into locals so they can
            // be captured by the loss-supplier lambda below.
            final Tensor x = xRef[0];
            final Tensor tgt = target;
            final java.util.function.Supplier<Tensor> lossComputation =
                    () -> DistributedLoss.crossEntropy(x, tgt);

            // Compute loss
            Tensor loss;
            if (mixedPrecision != null) {
                loss = mixedPrecision.scaleLoss(lossComputation);
            } else {
                loss = lossComputation.get();
            }

            // Record FLOPs
            if (config.recordFlops) {
                long flops = estimateFlops(input);
                profiler.recordFlops("forward", flops);
            }

            return loss;
        });
    }

    /**
     * Backward pass.
     */
    public void backward(Tensor loss) {
        profiler.profile("backward", () -> {
            loss.backward();

            // Record backward FLOPs
            if (config.recordFlops) {
                profiler.recordFlops("backward", profiler.getOrCreateHistogram("forward")
                        .total() > 0 ? 2 : 0);
            }
        });
    }

    /**
     * Optimizer step with gradient accumulation.
     */
    public void step(Optimizer optimizer) {
        profiler.profile("optimizer_step", () -> {
            Module modelModule = model.getModule();

            // Accumulate gradients
            gradientAccumulator.accumulate(modelModule);

            // Step if accumulated enough
            if (gradientAccumulator.step(modelModule, optimizer)) {
                // Record throughput
                if (config.enableProfiling) {
                    profiler.recordCounter("optimizer_steps_total", 1);
                }
            }
        });
    }

    /**
     * Pipeline training step.
     */
    public void pipelineTrainStep(List<Tensor> microBatches, Optimizer optimizer) {
        if (pipeline == null) {
            // Fall back to standard training
            for (Tensor batch : microBatches) {
                trainStep(batch, null, optimizer);
            }
            return;
        }

        profiler.profile("pipeline_train", () -> {
            // 1F1B schedule
            pipeline.train1F1B(microBatches);

            // Sync and step
            if (optimizer != null) {
                gradientAccumulator.syncGradients();
                gradientAccumulator.forceStep(model.getModule(), optimizer);
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Checkpointing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Save checkpoint.
     */
    public String saveCheckpoint() {
        return saveCheckpoint(globalStep.get());
    }

    /**
     * Save checkpoint at specific step.
     */
    public String saveCheckpoint(long step) {
        if (checkpointManager == null) {
            System.out.println("[Trainer] Checkpointing disabled");
            return null;
        }

        Map<String, Object> extraState = new HashMap<>();
        extraState.put("epoch", currentEpoch);
        extraState.put("global_step", step);
        extraState.put("loss_history", lossHistory);

        try {
            return profiler.profile("save_checkpoint", () -> {
                try {
                    return checkpointManager.save(model.getModule(), null, (int) step, extraState);
                } catch (java.io.IOException ioe) {
                    throw new RuntimeException(ioe);
                }
            });
        } catch (RuntimeException e) {
            Throwable cause = e.getCause() != null ? e.getCause() : e;
            System.err.println("[EnterpriseHybridTrainer] Failed to save checkpoint: " + cause.getMessage());
            return null;
        }
    }

    /**
     * Load latest checkpoint.
     */
    public int loadCheckpoint() {
        if (checkpointManager == null) {
            System.out.println("[Trainer] Checkpointing disabled");
            return -1;
        }

        return profiler.profile("load_checkpoint", () -> {
            int step = checkpointManager.loadLatest(model.getModule(), null);
            if (step >= 0) {
                globalStep.set(step);
                System.out.printf("[Trainer] Resumed from step %d%n", step);
            }
            return step;
        });
    }

    /**
     * Check if should save checkpoint.
     */
    public boolean shouldSaveCheckpoint() {
        return checkpointManager != null && checkpointManager.shouldSave((int) globalStep.get());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics & Monitoring
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Update training statistics.
     */
    private void updateStats(Tensor loss, long stepStart) {
        long stepTime = System.nanoTime() - stepStart;

        // Record loss
        double lossValue = loss.item().toDouble();
        lossHistory.add(lossValue);
        if (lossHistory.size() > 1000) {
            lossHistory.remove(0);
        }

        // Record throughput
        double throughput = config.microBatchSize * config.maxSeqLen / (stepTime / 1e9);
        throughputHistory.add(throughput);

        // Update profiler counters
        if (config.enableProfiling) {
            profiler.recordCounter("global_step", globalStep.incrementAndGet());
            profiler.recordCounter("loss", (long) (lossValue * 1000));
            profiler.recordCounter("throughput_tokens_per_sec", (long) throughput);
        }

        // Print periodic report
        if (globalStep.get() % config.logInterval == 0 && rank == 0) {
            printProgress();
        }

        // Auto checkpoint
        if (shouldSaveCheckpoint()) {
            saveCheckpoint();
        }

        // Auto profiling export
        if (config.enableProfiling && globalStep.get() % config.profileInterval == 0) {
            profiler.exportChromeTrace(String.format("step_%d_trace.json", globalStep.get()));
        }
    }

    /**
     * Print training progress.
     */
    public void printProgress() {
        double avgLoss = lossHistory.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);

        double avgThroughput = throughputHistory.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);

        String lossScaleStr = mixedPrecision != null
                ? String.format(" scale=%.0f", mixedPrecision.getScale())
                : "";

        System.out.printf("""
                ╔══════════════════════════════════════════════════════════════════╗
                ║  Step %7d  │  Loss: %.4f  │  Throughput: %8.0f tokens/s     ║%s
                ╚══════════════════════════════════════════════════════════════════╝
                """,
                globalStep.get(),
                avgLoss,
                avgThroughput,
                lossScaleStr
        );
    }

    /**
     * Get current loss.
     */
    public double getCurrentLoss() {
        return lossHistory.isEmpty() ? 0 : lossHistory.get(lossHistory.size() - 1);
    }

    /**
     * Get average loss over recent steps.
     */
    public double getAverageLoss(int window) {
        int size = Math.min(window, lossHistory.size());
        if (size == 0) return 0;

        double sum = 0;
        for (int i = lossHistory.size() - size; i < lossHistory.size(); i++) {
            sum += lossHistory.get(i);
        }
        return sum / size;
    }

    /**
     * Get throughput.
     */
    public double getThroughput() {
        return throughputHistory.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FLOP Estimation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Estimate FLOPs for a forward pass.
     */
    private long estimateFlops(Tensor input) {
        long batch = input.sizes().get(0);
        long seqLen = input.sizes().get(1);

        // Embedding FLOPs
        long embeddingFlops = batch * seqLen * config.vocabSize;

        // Attention FLOPs
        long attnFlops = 4L * batch * seqLen * config.hiddenDim * config.hiddenDim;

        // FFN FLOPs
        long ffnFlops = 3L * batch * seqLen * config.hiddenDim * config.intermediateDim;

        // MoE FLOPs (if enabled)
        long moeFlops = config.numExperts > 0
                ? 3L * batch * seqLen * config.hiddenDim * config.hiddenDim * config.numExperts * config.topK
                : 0;

        return embeddingFlops + attnFlops + ffnFlops + moeFlops;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get trainer configuration.
     */
    public Config getConfig() {
        return config;
    }

    /**
     * Get current global step.
     */
    public long getGlobalStep() {
        return globalStep.get();
    }

    /**
     * Get current epoch.
     */
    public int getCurrentEpoch() {
        return currentEpoch;
    }

    /**
     * Set current epoch.
     */
    public void setEpoch(int epoch) {
        this.currentEpoch = epoch;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Sub-component Access
    // ═══════════════════════════════════════════════════════════════════════════

    public ParallelLayers.HybridTrainer getModel() { return model; }
    public ProcessGroupWrapper getProcessGroup() { return pg; }
    public MixedPrecisionManager getMixedPrecision() { return mixedPrecision; }
    public GradientAccumulator getGradientAccumulator() { return gradientAccumulator; }
    public ActivationCheckpoint getActivationCheckpoint() { return checkpoint; }
    public CheckpointManager getCheckpointManager() { return checkpointManager; }
    public ExpertLoadBalancer getLoadBalancer() { return loadBalancer; }
    public DistributedProfiler getProfiler() { return profiler; }

    // ═══════════════════════════════════════════════════════════════════════════
    // Profiling & Reporting
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Print detailed profiling report.
     */
    public void printProfileReport() {
        System.out.println("\n" + "═".repeat(70));
        System.out.println("                     PROFILING REPORT");
        System.out.println("═".repeat(70));

        profiler.printReport();

        if (gradientAccumulator != null) {
            gradientAccumulator.printStats();
        }

        if (checkpoint != null) {
            checkpoint.printStats();
        }

        if (mixedPrecision != null) {
            mixedPrecision.printStats();
        }

        if (checkpointManager != null) {
            checkpointManager.printStats();
        }

        if (loadBalancer != null) {
            loadBalancer.printStats();
        }

        System.out.println("═".repeat(70));
    }

    /**
     * Export profiling data.
     */
    public void exportProfile(String suffix) {
        profiler.exportChromeTrace("train_" + suffix + "_trace.json");
        profiler.exportReport("train_" + suffix + "_report.txt");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Final checkpoint
        if (checkpointManager != null && config.saveOnClose) {
            saveCheckpoint();
        }

        // Print final report
        if (rank == 0) {
            printProgress();
            if (config.enableProfiling) {
                printProfileReport();
            }
        }

        // Close all components
        if (profiler != null) profiler.close();
        if (checkpointManager != null) checkpointManager.close();
        if (mixedPrecision != null) mixedPrecision.close();
        if (gradientAccumulator != null) gradientAccumulator.close();
        if (checkpoint != null) checkpoint.clear();
        if (loadBalancer != null) loadBalancer.close();
        if (pipeline != null) pipeline.close();

        System.out.printf("[Trainer] Shutdown complete (total steps: %d)%n", globalStep.get());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════════

    public static final class Builder {
        // Process group
        private ProcessGroupWrapper pg;

        // Model configuration
        private long vocabSize = 32000;
        private long hiddenDim = 4096;
        private int numHeads = 32;
        private long intermediateDim = 16384;
        private int numExperts = 8;
        private int topK = 2;
        private long maxSeqLen = 2048;

        // Parallelism configuration
        private int tpSize = 1;
        private int epSize = 1;
        private int numPipelineStages = 1;
        private int numMicroBatches = 1;
        private boolean enableOverlap = true;
        private long maxPipelineMemory = 1L << 36; // 64GB

        // Training configuration
        private int accumulationSteps = 1;
        private float maxGradNorm = 1.0f;
        private float learningRate = 1e-4f;
        private int microBatchSize = 1;
        private int logInterval = 100;

        // Mixed precision
        private boolean enableMixedPrecision = true;
        private float initialLossScale = 65536.0f;
        private float lossScaleGrowthFactor = 2.0f;
        private float lossScaleBackoffFactor = 0.5f;
        private int lossScaleGrowthInterval = 2000;

        // Activation checkpointing
        private boolean enableActivationCheckpointing = true;
        private long maxCheckpointMemory = 1L << 34; // 16GB
        private List<String> checkpointLayers;

        // Gradient accumulation
        private boolean syncBeforeStep = true;

        // Checkpointing
        private boolean enableCheckpointing = true;
        private String checkpointDir = "./checkpoints";
        private int maxCheckpoints = 5;
        private int checkpointInterval = 1000;
        private boolean asyncCheckpointing = true;
        private boolean compressCheckpoints = true;
        private boolean saveOnClose = true;

        // Load balancing
        private boolean enableLoadBalancing = true;
        private float targetLoadFactor = 0.8f;
        private int loadBalanceInterval = 100;

        // Profiling
        private boolean enableProfiling = true;
        private boolean profileMemory = true;
        private boolean recordFlops = true;
        private String profileDir = "./profiler_output";
        private int maxProfilerEvents = 100000;
        private int profileInterval = 1000;

        // Builder methods
        public Builder processGroup(ProcessGroupWrapper pg) { this.pg = pg; return this; }
        public Builder vocabSize(long v) { this.vocabSize = v; return this; }
        public Builder hiddenDim(long d) { this.hiddenDim = d; return this; }
        public Builder numHeads(int h) { this.numHeads = h; return this; }
        public Builder intermediateDim(long i) { this.intermediateDim = i; return this; }
        public Builder numExperts(int e) { this.numExperts = e; return this; }
        public Builder topK(int k) { this.topK = k; return this; }
        public Builder maxSeqLen(long s) { this.maxSeqLen = s; return this; }
        public Builder tpSize(int t) { this.tpSize = t; return this; }
        public Builder epSize(int e) { this.epSize = e; return this; }
        public Builder numPipelineStages(int s) { this.numPipelineStages = s; return this; }
        public Builder numMicroBatches(int m) { this.numMicroBatches = m; return this; }
        public Builder enableOverlap(boolean e) { this.enableOverlap = e; return this; }
        public Builder maxPipelineMemory(long m) { this.maxPipelineMemory = m; return this; }
        public Builder accumulationSteps(int a) { this.accumulationSteps = a; return this; }
        public Builder maxGradNorm(float n) { this.maxGradNorm = n; return this; }
        public Builder learningRate(float lr) { this.learningRate = lr; return this; }
        public Builder microBatchSize(int m) { this.microBatchSize = m; return this; }
        public Builder logInterval(int l) { this.logInterval = l; return this; }
        public Builder enableMixedPrecision(boolean e) { this.enableMixedPrecision = e; return this; }
        public Builder initialLossScale(float s) { this.initialLossScale = s; return this; }
        public Builder lossScaleGrowthFactor(float f) { this.lossScaleGrowthFactor = f; return this; }
        public Builder lossScaleBackoffFactor(float f) { this.lossScaleBackoffFactor = f; return this; }
        public Builder lossScaleGrowthInterval(int i) { this.lossScaleGrowthInterval = i; return this; }
        public Builder enableActivationCheckpointing(boolean e) { this.enableActivationCheckpointing = e; return this; }
        public Builder maxCheckpointMemory(long m) { this.maxCheckpointMemory = m; return this; }
        public Builder checkpointLayers(List<String> l) { this.checkpointLayers = l; return this; }
        public Builder syncBeforeStep(boolean s) { this.syncBeforeStep = s; return this; }
        public Builder enableCheckpointing(boolean e) { this.enableCheckpointing = e; return this; }
        public Builder checkpointDir(String d) { this.checkpointDir = d; return this; }
        public Builder maxCheckpoints(int m) { this.maxCheckpoints = m; return this; }
        public Builder checkpointInterval(int i) { this.checkpointInterval = i; return this; }
        public Builder asyncCheckpointing(boolean a) { this.asyncCheckpointing = a; return this; }
        public Builder compressCheckpoints(boolean c) { this.compressCheckpoints = c; return this; }
        public Builder saveOnClose(boolean s) { this.saveOnClose = s; return this; }
        public Builder enableLoadBalancing(boolean e) { this.enableLoadBalancing = e; return this; }
        public Builder targetLoadFactor(float f) { this.targetLoadFactor = f; return this; }
        public Builder loadBalanceInterval(int i) { this.loadBalanceInterval = i; return this; }
        public Builder enableProfiling(boolean e) { this.enableProfiling = e; return this; }
        public Builder profileMemory(boolean m) { this.profileMemory = m; return this; }
        public Builder recordFlops(boolean r) { this.recordFlops = r; return this; }
        public Builder profileDir(String d) { this.profileDir = d; return this; }
        public Builder maxProfilerEvents(int m) { this.maxProfilerEvents = m; return this; }
        public Builder profileInterval(int i) { this.profileInterval = i; return this; }

        public EnterpriseHybridTrainer build() throws Exception {
            return new EnterpriseHybridTrainer(new Config(this));
        }
    }

    /**
     * Configuration record.
     */
    private static class Config {
        ProcessGroupWrapper pg;
        long vocabSize;
        long hiddenDim;
        int numHeads;
        long intermediateDim;
        int numExperts;
        int topK;
        long maxSeqLen;
        int tpSize;
        int epSize;
        int numPipelineStages;
        int numMicroBatches;
        boolean enableOverlap;
        long maxPipelineMemory;
        int accumulationSteps;
        float maxGradNorm;
        float learningRate;
        int microBatchSize;
        int logInterval;
        boolean enableMixedPrecision;
        float initialLossScale;
        float lossScaleGrowthFactor;
        float lossScaleBackoffFactor;
        int lossScaleGrowthInterval;
        boolean enableActivationCheckpointing;
        long maxCheckpointMemory;
        List<String> checkpointLayers;
        boolean syncBeforeStep;
        boolean enableCheckpointing;
        String checkpointDir;
        int maxCheckpoints;
        int checkpointInterval;
        boolean asyncCheckpointing;
        boolean compressCheckpoints;
        boolean saveOnClose;
        boolean enableLoadBalancing;
        float targetLoadFactor;
        int loadBalanceInterval;
        boolean enableProfiling;
        boolean profileMemory;
        boolean recordFlops;
        String profileDir;
        int maxProfilerEvents;
        int profileInterval;

        Config(Builder b) {
            this.pg = b.pg;
            this.vocabSize = b.vocabSize;
            this.hiddenDim = b.hiddenDim;
            this.numHeads = b.numHeads;
            this.intermediateDim = b.intermediateDim;
            this.numExperts = b.numExperts;
            this.topK = b.topK;
            this.maxSeqLen = b.maxSeqLen;
            this.tpSize = b.tpSize;
            this.epSize = b.epSize;
            this.numPipelineStages = b.numPipelineStages;
            this.numMicroBatches = b.numMicroBatches;
            this.enableOverlap = b.enableOverlap;
            this.maxPipelineMemory = b.maxPipelineMemory;
            this.accumulationSteps = b.accumulationSteps;
            this.maxGradNorm = b.maxGradNorm;
            this.learningRate = b.learningRate;
            this.microBatchSize = b.microBatchSize;
            this.logInterval = b.logInterval;
            this.enableMixedPrecision = b.enableMixedPrecision;
            this.initialLossScale = b.initialLossScale;
            this.lossScaleGrowthFactor = b.lossScaleGrowthFactor;
            this.lossScaleBackoffFactor = b.lossScaleBackoffFactor;
            this.lossScaleGrowthInterval = b.lossScaleGrowthInterval;
            this.enableActivationCheckpointing = b.enableActivationCheckpointing;
            this.maxCheckpointMemory = b.maxCheckpointMemory;
            this.checkpointLayers = b.checkpointLayers;
            this.syncBeforeStep = b.syncBeforeStep;
            this.enableCheckpointing = b.enableCheckpointing;
            this.checkpointDir = b.checkpointDir;
            this.maxCheckpoints = b.maxCheckpoints;
            this.checkpointInterval = b.checkpointInterval;
            this.asyncCheckpointing = b.asyncCheckpointing;
            this.compressCheckpoints = b.compressCheckpoints;
            this.saveOnClose = b.saveOnClose;
            this.enableLoadBalancing = b.enableLoadBalancing;
            this.targetLoadFactor = b.targetLoadFactor;
            this.loadBalanceInterval = b.loadBalanceInterval;
            this.enableProfiling = b.enableProfiling;
            this.profileMemory = b.profileMemory;
            this.recordFlops = b.recordFlops;
            this.profileDir = b.profileDir;
            this.maxProfilerEvents = b.maxProfilerEvents;
            this.profileInterval = b.profileInterval;
        }
    }

    /**
     * Profiler region for try-with-resources.
     */
    public final class ProfilerRegion implements AutoCloseable {
        private final String name;
        private final long startTime;
        private boolean closed = false;

        ProfilerRegion(String name) {
            this.name = name;
            this.startTime = System.nanoTime();
        }

        public void close() {
            if (!closed) {
                long elapsed = System.nanoTime() - startTime;
                profiler.recordTiming(name, elapsed, "region");
                closed = true;
            }
        }
    }
}
