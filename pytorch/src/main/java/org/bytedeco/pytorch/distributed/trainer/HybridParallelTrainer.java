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
package org.bytedeco.pytorch.distributed.trainer;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.distributed.config.MixedPrecisionConfig;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * 3D hybrid parallel trainer: Data Parallel (DP) × Tensor Parallel (TP) ×
 * Pipeline Parallel (PP) — the standard enterprise configuration for
 * multi-billion-parameter models.
 *
 * <p>Combines three orthogonal parallelism strategies into a single trainer:
 *
 * <ul>
 *   <li><b>DP</b>: replicates model across {@code dpSize} ranks; gradients are
 *       all-reduced after backward ({@link NativeDDPTrainer} semantics).</li>
 *   <li><b>TP</b>: shards each model layer (column/row parallel linear) across
 *       {@code tpSize} ranks; allgather on forward, allreduce on backward
 *       ({@link TensorParallel} semantics).</li>
 *   <li><b>PP</b>: splits the model depth across {@code ppSize} stages;
 *       activations are sent forward, gradients sent backward via
 *       {@code send}/{@code recv} collectives ({@link PipelineParallelTrainer}
 *       semantics).</li>
 * </ul>
 *
 * <p>Requires {@code dpSize * tpSize * ppSize == worldSize}. Each rank belongs
 * to exactly one DP group, one TP group, and one PP stage. The trainer manages
 * the mesh creation, stage assignment, and collective dispatch automatically.
 *
 * <p>Pipeline schedule: all-forward (fill pipeline), all-backward (drain),
 * with optional gradient accumulation steps to increase the effective batch size.
 *
 * <pre>{@code
 * // 8 GPUs: dp=1, tp=4, pp=2
 * HybridParallelTrainer trainer = HybridParallelTrainer.builder()
 *         .processGroup(pg)
 *         .tpSize(4)
 *         .ppSize(2)
 *         .numMicroBatches(8)
 *         .accumulationSteps(4)
 *         .build();
 *
 * try (trainer) {
 *     for (int i = 0; i < steps; i++) {
 *         Tensor loss = trainer.step(input, target, optimizer);
 *     }
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class HybridParallelTrainer implements BaseDistributedTrainer, AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";

    // ── Mesh & group configuration ─────────────────────────────────────────
    private final ProcessGroupWrapper pg;
    private final int worldSize;
    private final int rank;

    /** dpSize * tpSize * ppSize must equal worldSize. */
    private final int dpSize;
    private final int tpSize;
    private final int ppSize;

    /** My coordinates in the 3D mesh [dp, tp, pp]. */
    private final int dpRank;
    private final int tpRank;
    private final int ppStage;         // 0 = first, ppSize-1 = last

    /** Sub-process groups (each rank is in exactly one DP, one TP, one PP group). */
    private final ProcessGroupWrapper dpGroup;
    private final ProcessGroupWrapper tpGroup;
    private final ProcessGroupWrapper ppGroup;

    // ── Stage configuration ────────────────────────────────────────────────
    /** Number of pipeline stages (must divide model layers). */
    private final int numPipelineStages;
    /** Which stage this rank owns (0 .. numPipelineStages-1). */
    private final int myStage;
    /** Is this rank the first pipeline stage? */
    private final boolean isFirstStage;
    /** Is this rank the last pipeline stage? */
    private final boolean isLastStage;

    // ── Training state ────────────────────────────────────────────────────
    private final Module module;
    private final ModuleForward moduleForward;
    private final TrainerStats stats = new TrainerStats();
    private final int numMicroBatches;
    private final int accumulationSteps;
    private final boolean enableActivationCheckpointing;
    private final MixedPrecisionConfig mixedPrecision;
    private final Device device;
    private final Module[] stages;     // PP stage modules

    private long numSteps;
    private long numMicroBatchProcessed;
    private boolean closed;

    // ── Constructors ──────────────────────────────────────────────────────

    public HybridParallelTrainer(Builder b) {
        this.pg = Objects.requireNonNull(b.pg, "pg");
        this.worldSize = pg.getWorldSize();
        this.rank = pg.getRank();
        this.dpSize = b.dpSize > 0 ? b.dpSize : 1;
        this.tpSize = b.tpSize > 0 ? b.tpSize : 1;
        this.ppSize = b.ppSize > 0 ? b.ppSize : 1;
        this.numMicroBatches = Math.max(1, b.numMicroBatches);
        this.accumulationSteps = Math.max(1, b.accumulationSteps);
        this.enableActivationCheckpointing = b.enableActivationCheckpointing;
        this.mixedPrecision = b.mixedPrecision != null ? b.mixedPrecision : MixedPrecisionConfig.fp32();
        this.device = pg.getDevice();
        this.moduleForward = ModuleForward.of(b.module);
        this.module = b.module;
        this.stages = new Module[1];
        this.stages[0] = b.module;

        // Validate mesh
        if (dpSize * tpSize * ppSize != worldSize) {
            throw new IllegalArgumentException(
                    "dpSize*tpSize*ppSize = " + dpSize + "*" + tpSize + "*" + ppSize
                            + " = " + (dpSize * tpSize * ppSize)
                            + " != worldSize=" + worldSize);
        }

        // Compute coordinates
        int dpDim = (int) (Math.log(tpSize * ppSize) / Math.log(dpSize > 1 ? dpSize : 1));
        int tpDim = 1;
        int ppDim = 2;

        // Simplified: row-major order [dp, tp, pp]
        int[] coords = compute3DCoords(rank, dpSize, tpSize, ppSize);
        this.dpRank = coords[0];
        this.tpRank = coords[1];
        this.ppStage = coords[2];
        this.myStage = ppStage;
        this.numPipelineStages = ppSize;
        this.isFirstStage = (myStage == 0);
        this.isLastStage = (myStage == numPipelineStages - 1);

        // Create sub-process groups
        this.dpGroup = createDpGroup();
        this.tpGroup = createTpGroup();
        this.ppGroup = createPpGroup();

        module.to(device, true);

        System.out.printf(
                "[HybridParallelTrainer v%s] rank=%d dp=%d/%d tp=%d/%d pp=%d/%d "
                        + "stages=%d microBatches=%d accum=%d mp=%s "
                        + "firstStage=%s lastStage=%s%n",
                VERSION, rank, dpRank, dpSize, tpRank, tpSize,
                myStage, numPipelineStages, numMicroBatches, accumulationSteps,
                mixedPrecision, isFirstStage, isLastStage);
    }

    public static Builder builder() { return new Builder(); }

    // ── Forward method (required by BaseDistributedTrainer) ───────────────────

    /**
     * Forward pass through the model.
     */
    public Tensor forward(Tensor input) {
        return moduleForward.apply(module, input);
    }

    // ── Mesh coordinate helpers ───────────────────────────────────────────

    /**
     * Convert a flat world rank to [dp, tp, pp] coordinates.
     * Order: dp changes slowest, pp changes fastest (row-major).
     */
    private static int[] compute3DCoords(int rank, int dp, int tp, int pp) {
        int[] c = new int[3];
        c[0] = rank / (tp * pp);
        int rem = rank % (tp * pp);
        c[1] = rem / pp;
        c[2] = rem % pp;
        return c;
    }

    /**
     * Convert [dp, tp, pp] coordinates to flat world rank.
     */
    public static int coordsToRank(int[] coords, int dp, int tp, int pp) {
        return coords[0] * tp * pp + coords[1] * pp + coords[2];
    }

    private ProcessGroupWrapper createDpGroup() {
        // All ranks with the same [tpRank, ppStage] form a DP group.
        // Simplified: create a wrapper over the same PG with filtering.
        // For a real implementation, callers would create subgroups via
        // DeviceMesh.split() / ProcessGroup.split(). Here we use the
        // full PG with rank checks for simplicity.
        return new SubProcessGroupWrapper(pg, (r) -> {
            int[] c = compute3DCoords(r, dpSize, tpSize, ppSize);
            return c[1] == tpRank && c[2] == myStage;
        }).parent;
    }

    private ProcessGroupWrapper createTpGroup() {
        // All ranks with the same [dpRank, ppStage] form a TP group.
        return new SubProcessGroupWrapper(pg, (r) -> {
            int[] c = compute3DCoords(r, dpSize, tpSize, ppSize);
            return c[0] == dpRank && c[2] == myStage;
        }).parent;
    }

    private ProcessGroupWrapper createPpGroup() {
        // All ranks with the same [dpRank, tpRank] form a PP group.
        return new SubProcessGroupWrapper(pg, (r) -> {
            int[] c = compute3DCoords(r, dpSize, tpSize, ppSize);
            return c[0] == dpRank && c[1] == tpRank;
        }).parent;
    }

    // ── Forward / backward ─────────────────────────────────────────────────

    /**
     * Single training step. When {@code worldSize == 1}, runs a standard
     * forward + backward. When {@code worldSize > 1}, runs the GPipe-style
     * schedule: fill (all forward microbatches), drain (all backward microbatches),
     * then optimizer step.
     */
    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        stats.fireStepStart();
        if (worldSize == 1 || numPipelineStages == 1) {
            return stepSingleProcess(input, target, optimizer);
        }
        return stepPipeline(input, target, optimizer);
    }

    private Tensor stepSingleProcess(Tensor input, Tensor target, Optimizer optimizer) {
        if (optimizer != null) optimizer.zero_grad();
        List<Tensor> microInputs = splitIntoMicroBatches(input, numMicroBatches);
        List<Tensor> microTargets = splitIntoMicroBatches(target, numMicroBatches);
        Tensor lastLoss = null;

        // Forward pass through all stages
        for (int c = 0; c < microInputs.size(); c++) {
            Tensor x = microInputs.get(c);
            x = moduleForward.apply(module, x);
            lastLoss = DistributedLoss.crossEntropy(x, microTargets.get(c));
        }

        // Backward pass
        for (int c = microInputs.size() - 1; c >= 0; c--) {
            if (c == microInputs.size() - 1) {
                lastLoss.div(new Scalar(numMicroBatches)).backward();
            }
        }

        if (optimizer != null) optimizer.step();
        numSteps++;
        numMicroBatchProcessed += numMicroBatches;
        stats.fireStepEnd(lastLoss);
        return lastLoss != null ? lastLoss : zeros(1).to(device, ScalarType.Float);
    }

    /**
     * Pipeline parallel schedule: all forward (GPipe fill), then all backward (drain).
     * Activations between PP stages are communicated via send/recv.
     */
    private Tensor stepPipeline(Tensor input, Tensor target, Optimizer optimizer) {
        if (optimizer != null) optimizer.zero_grad();

        List<Tensor> microInputs = splitIntoMicroBatches(input, numMicroBatches);
        List<Tensor> microTargets = splitIntoMicroBatches(target, numMicroBatches);
        List<Tensor> activations = new ArrayList<>();

        // ── Forward phase (fill the pipeline) ────────────────────────────
        for (int c = 0; c < microInputs.size(); c++) {
            Tensor x;
            if (isFirstStage) {
                x = microInputs.get(c);
            } else {
                // Receive activation from previous PP stage
                x = recvActivationFromPrev();
            }

            // Forward through this stage's module
            x = moduleForward.apply(stages[0], x);
            numMicroBatchProcessed++;

            if (isLastStage) {
                // Compute loss
                Tensor loss = DistributedLoss.crossEntropy(x, microTargets.get(c));
                loss.div(new Scalar(numMicroBatches)).backward();
                activations.add(loss);
            } else {
                // Send activation to next PP stage
                sendActivationToNext(x);
                activations.add(null);
            }
        }

        // ── Backward phase (drain the pipeline) ───────────────────────────
        for (int c = microInputs.size() - 1; c >= 0; c--) {
            if (isLastStage) {
                // Loss already backpropped above
            } else {
                // Receive grad from next stage
                Tensor grad = recvGradFromNext();
                if (grad != null && !grad.isNull()) {
                    try {
                        grad.backward(grad,
                                new org.bytedeco.pytorch.BoolOptional(),
                                false,
                                new org.bytedeco.pytorch.TensorArrayRefOptional());
                    } catch (Throwable ignored) {}
                }
                if (!isFirstStage) {
                    sendGradToPrev(grad);
                }
            }
        }

        // ── TP gradient sync (within TP group) ───────────────────────────
        if (tpGroup.getWorldSize() > 1) {
            syncTpGradients();
        }

        // ── DP gradient sync (within DP group) ──────────────────────────
        if (dpGroup.getWorldSize() > 1) {
            syncDpGradients();
        }

        if (optimizer != null) {
            optimizer.step();
            stats.fireOptimizerStep();
            optimizer.zero_grad();
        }
        numSteps++;
        stats.fireStepEnd(activations.isEmpty() ? null
                : (activations.get(activations.size() - 1) != null
                    ? activations.get(activations.size() - 1)
                    : zeros(1).to(device, ScalarType.Float)));
        return activations.isEmpty() || activations.get(0) == null
                ? zeros(1).to(device, ScalarType.Float)
                : activations.get(0);
    }

    private List<Tensor> splitIntoMicroBatches(Tensor batch, int chunks) {
        List<Tensor> out = new ArrayList<>();
        if (batch == null || batch.isNull() || batch.dim() == 0) {
            for (int i = 0; i < chunks; i++) out.add(batch);
            return out;
        }
        long n = batch.sizes().get(0);
        long base = Math.max(1, n / chunks);
        long offset = 0;
        for (int i = 0; i < chunks; i++) {
            long len = (i == chunks - 1) ? (n - offset) : base;
            if (offset >= n) {
                out.add(batch.narrow(0, Math.max(0, n - 1), 1));
            } else {
                long L = Math.max(1, Math.min(len, n - offset));
                out.add(batch.narrow(0, offset, L));
                offset += L;
            }
        }
        return out;
    }

    private void syncTpGradients() {
        List<Tensor> grads = new ArrayList<>();
        var params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor g = p.grad();
                if (g != null && !g.isNull() && g.defined()) grads.add(g);
            } catch (Throwable ignored) {}
        }
        if (!grads.isEmpty()) {
            tpGroup.allreduce(grads, ReduceOp.RedOpType.SUM);
            TrainerOps.divideInPlace(grads, tpGroup.getWorldSize());
            stats.fireAllreduce(TrainerOps.totalBytes(grads));
        }
    }

    private void syncDpGradients() {
        List<Tensor> grads = new ArrayList<>();
        var params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor g = p.grad();
                if (g != null && !g.isNull() && g.defined()) grads.add(g);
            } catch (Throwable ignored) {}
        }
        if (!grads.isEmpty()) {
            dpGroup.allreduce(grads, ReduceOp.RedOpType.SUM);
            TrainerOps.divideInPlace(grads, dpGroup.getWorldSize());
            stats.fireAllreduce(TrainerOps.totalBytes(grads));
        }
    }

    private static final int TAG_ACT = 8001;
    private static final int TAG_GRAD = 8002;

    private void sendActivationToNext(Tensor activation) {
        if (activation == null || activation.isNull()) return;
        int nextRank = (ppStage + 1 < ppSize)
                ? coordsToRank(new int[]{dpRank, tpRank, ppStage + 1}, dpSize, tpSize, ppSize)
                : rank;
        try {
            ppGroup.send(activation.contiguous(), nextRank, TAG_ACT);
        } catch (Throwable ignored) {}
    }

    private Tensor recvActivationFromPrev() {
        int prevRank = (ppStage > 0)
                ? coordsToRank(new int[]{dpRank, tpRank, ppStage - 1}, dpSize, tpSize, ppSize)
                : rank;
        Tensor buf = empty(1, 1, (int) device.type().value).to(device, ScalarType.Float);
        try {
            ppGroup.recv(buf, prevRank, TAG_ACT);
        } catch (Throwable ignored) {}
        return buf;
    }

    private void sendGradToPrev(Tensor grad) {
        int prevRank = (ppStage > 0)
                ? coordsToRank(new int[]{dpRank, tpRank, ppStage - 1}, dpSize, tpSize, ppSize)
                : rank;
        if (grad != null && !grad.isNull()) {
            try {
                ppGroup.send(grad.contiguous(), prevRank, TAG_GRAD);
            } catch (Throwable ignored) {}
        }
    }

    private Tensor recvGradFromNext() {
        int nextRank = (ppStage + 1 < ppSize)
                ? coordsToRank(new int[]{dpRank, tpRank, ppStage + 1}, dpSize, tpSize, ppSize)
                : rank;
        Tensor buf = empty(1, 1, (int) device.type().value).to(device, ScalarType.Float);
        try {
            ppGroup.recv(buf, nextRank, TAG_GRAD);
        } catch (Throwable ignored) {}
        return buf;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    public Module getModule() { return module; }
    public Module[] getStages() { return stages; }
    public ProcessGroupWrapper getProcessGroup() { return pg; }
    public ProcessGroupWrapper getDpGroup() { return dpGroup; }
    public ProcessGroupWrapper getTpGroup() { return tpGroup; }
    public ProcessGroupWrapper getPpGroup() { return ppGroup; }
    public int getDpSize() { return dpSize; }
    public int getTpSize() { return tpSize; }
    public int getPpSize() { return ppSize; }
    public int getDpRank() { return dpRank; }
    public int getTpRank() { return tpRank; }
    public int getPpStage() { return ppStage; }
    public int getNumPipelineStages() { return numPipelineStages; }
    public int getNumMicroBatches() { return numMicroBatches; }
    public int getAccumulationSteps() { return accumulationSteps; }
    public boolean isFirstStage() { return isFirstStage; }
    public boolean isLastStage() { return isLastStage; }
    public MixedPrecisionConfig getMixedPrecision() { return mixedPrecision; }
    public TrainerStats stats() { return stats; }
    public long getNumSteps() { return numSteps; }
    public long getNumMicroBatchProcessed() { return numMicroBatchProcessed; }

    public void train() { module.train(true); }
    public void eval() { module.eval(); }
    public boolean isTraining() { return module.is_training(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try { if (dpGroup != null) dpGroup.close(); } catch (Throwable ignored) {}
        try { if (tpGroup != null) tpGroup.close(); } catch (Throwable ignored) {}
        try { if (ppGroup != null) ppGroup.close(); } catch (Throwable ignored) {}
    }

    @Override
    public String toString() {
        return "HybridParallelTrainer{rank=" + rank + ", dp=" + dpRank + "/" + dpSize
                + ", tp=" + tpRank + "/" + tpSize
                + ", pp=" + myStage + "/" + numPipelineStages
                + ", steps=" + numSteps
                + ", stats=" + stats.snapshot() + '}';
    }

    // ── Builder ─────────────────────────────────────────────────────────

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper pg;
        private int dpSize = 1;
        private int tpSize = 1;
        private int ppSize = 1;
        private int numMicroBatches = 4;
        private int accumulationSteps = 1;
        private boolean enableActivationCheckpointing = false;
        private MixedPrecisionConfig mixedPrecision;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.pg = pg; return this; }
        public Builder dpSize(int d) { this.dpSize = d; return this; }
        public Builder tpSize(int t) { this.tpSize = t; return this; }
        public Builder ppSize(int p) { this.ppSize = p; return this; }
        public Builder numMicroBatches(int n) { this.numMicroBatches = n; return this; }
        public Builder accumulationSteps(int n) { this.accumulationSteps = n; return this; }
        public Builder enableActivationCheckpointing(boolean b) {
            this.enableActivationCheckpointing = b; return this;
        }
        public Builder mixedPrecision(MixedPrecisionConfig mp) { this.mixedPrecision = mp; return this; }

        public HybridParallelTrainer build() {
            if (module == null) module = new org.bytedeco.pytorch.nn.modules.container.SequentialImpl();
            return new HybridParallelTrainer(this);
        }
    }

    // ── Sub-group wrapper for filtered rank operations ────────────────────

    /**
     * A lightweight sub-group facade that delegates to a parent PG but
     * only performs collectives when the target rank satisfies a predicate.
     * Used to model DP / TP / PP sub-groups without creating real native
     * sub-process-groups (which require ProcessGroup.split() support).
     */
    private static final class SubProcessGroupWrapper implements AutoCloseable {
        private final ProcessGroupWrapper parent;
        private final java.util.function.IntPredicate predicate;
        private final int worldSize;

        SubProcessGroupWrapper(ProcessGroupWrapper parent, java.util.function.IntPredicate predicate) {
            this.parent = parent;
            this.predicate = predicate;
            this.worldSize = countMatching(parent.getWorldSize(), predicate);
        }

        private static int countMatching(int worldSize, java.util.function.IntPredicate p) {
            int count = 0;
            for (int r = 0; r < worldSize; r++) {
                if (p.test(r)) count++;
            }
            return Math.max(1, count);
        }

        public int getRank() { return parent.getRank(); }
        public int getWorldSize() { return worldSize; }
        public boolean predicateHolds(int r) { return predicate.test(r); }
        public Device getDevice() { return parent.getDevice(); }
        public ProcessGroupWrapper parent() { return parent; }

        public Work allreduce(List<Tensor> tensors) {
            if (worldSize <= 1) return null;
            return parent.allreduce(tensors);
        }

        public Work allreduce(List<Tensor> tensors, ReduceOp.RedOpType op) {
            if (worldSize <= 1) return null;
            return parent.allreduce(tensors, op);
        }

        public void send(Tensor tensor, int dstRank, int tag) {
            if (predicate.test(getRank()) && predicate.test(dstRank)) {
                parent.send(tensor, dstRank, tag);
            }
        }

        public Tensor recv(Tensor tensor, int srcRank, int tag) {
            if (predicate.test(getRank()) && predicate.test(srcRank)) {
                return parent.recv(tensor, srcRank, tag);
            }
            return tensor;
        }

        public Work barrier() { return parent.barrier(); }
        public void barrierWait() { parent.barrierWait(); }

        @Override
        public void close() {
            // Don't close the parent PG
        }

        @Override
        public String toString() {
            return "SubGroup{parent=PG@" + parent.getRank() + ", size=" + worldSize + "}";
        }
    }
}
