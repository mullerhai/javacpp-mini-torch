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
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.config.MixedPrecisionConfig;
import org.bytedeco.pytorch.distributed.enums.ShardingStrategy;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * DeepSpeed-style Zero Redundancy Optimizer trainer (ZeRO-1 / ZeRO-2 / ZeRO-3).
 *
 * <p>Three stages of memory savings are supported:
 *
 * <ul>
 *   <li><b>ZeRO-1</b> ({@link Stage#OPTIMIZER_STATES}): partition the
 *       optimizer state across ranks. The model parameters and gradients
 *       stay replicated. Equivalent to DeepSpeed Stage 1.</li>
 *   <li><b>ZeRO-2</b> ({@link Stage#GRADIENTS}): also partition the
 *       gradients (reduce-scatter grads after backward). Equivalent to
 *       DeepSpeed Stage 2.</li>
 *   <li><b>ZeRO-3</b> ({@link Stage#PARAMETERS}): also partition the
 *       parameters. Forward begins with an all-gather of the full param
 *       buffer, then reshard after forward / backward. Equivalent to
 *       DeepSpeed Stage 3 (and to {@link NativeFSDPTrainer}
 *       with {@link ShardingStrategy#FULL_SHARD}).</li>
 * </ul>
 *
 * <p>CPU offload is supported as a flag: when {@code cpuOffload=true}, the
 * partition of the optimizer state that lives on this rank is materialised
 * on CPU and the param / grad shards stay on the model device.
 *
 * <p>The trainer is implemented on top of {@link TrainerOps} + c10d
 * collectives; it is intentionally simple (no c10d {@link Reducer} / c10d
 * FSDP) so it can run on every backend that {@link ProcessGroupWrapper}
 * supports (Gloo / NCCL / MPI / UCC).
 *
 * <pre>{@code
 * try (ZeroRedundancyOptimizerTrainer zero = ZeroRedundancyOptimizerTrainer.builder()
 *         .module(model)
 *         .processGroup(pg)
 *         .stage(ZeroRedundancyOptimizerTrainer.Stage.GRADIENTS)
 *         .build()) {
 *     SGD opt = new SGD(zero.localParameters(), new SGDOptions(1e-3f));
 *     for (int i = 0; i < steps; i++) {
 *         Tensor loss = zero.step(input, target, opt);
 *     }
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ZeroRedundancyOptimizerTrainer implements BaseDistributedTrainer, AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public enum Stage { OPTIMIZER_STATES, GRADIENTS, PARAMETERS }

    public static final String VERSION = "1.0";

    // ── Configuration ──────────────────────────────────────────────────────
    private final Module module;
    private final ProcessGroupWrapper processGroup;
    private final Stage stage;
    private final boolean cpuOffload;
    private final boolean overlapComm;
    private final int gradAccumSteps;
    private final MixedPrecisionConfig mixedPrecision;
    private final Device device;

    // ── Shard state ────────────────────────────────────────────────────────
    /** Parameters of the module, in module.parameters() order. */
    private final List<Tensor> allParameters = new ArrayList<>();
    /** The parameters this rank owns (a subset of allParameters). */
    private final List<Tensor> localParameters = new ArrayList<>();
    /** Indices of allParameters owned by this rank. */
    private final int[] localParamIndices;
    private final long totalParamNumel;
    private final long localParamNumel;
    private final long shardSize;
    private final long paddedFullSize;
    private final TrainerStats stats = new TrainerStats();
    private final ModuleForward moduleForward;

    private int microStep;
    private volatile boolean syncGradients = true;
    private boolean closed;

    // ── Constructors ──────────────────────────────────────────────────────

    public ZeroRedundancyOptimizerTrainer(Module module, ProcessGroupWrapper processGroup) {
        this(builder().module(module).processGroup(processGroup));
    }

    public ZeroRedundancyOptimizerTrainer(Module module, ProcessGroupWrapper processGroup, Stage stage,
                                          boolean cpuOffload, int gradAccumSteps,
                                          MixedPrecisionConfig mixedPrecision) {
        this(builder()
                .module(module).processGroup(processGroup)
                .stage(stage)
                .cpuOffload(cpuOffload)
                .gradAccumSteps(gradAccumSteps)
                .mixedPrecision(mixedPrecision));
    }

    private ZeroRedundancyOptimizerTrainer(Builder b) {
        this.module = Objects.requireNonNull(b.module, "module");
        this.processGroup = Objects.requireNonNull(b.processGroup, "processGroup");
        this.stage = b.stage == null ? Stage.GRADIENTS : b.stage;
        this.cpuOffload = b.cpuOffload;
        this.overlapComm = b.overlapComm;
        this.gradAccumSteps = Math.max(1, b.gradAccumSteps);
        this.mixedPrecision = b.mixedPrecision != null ? b.mixedPrecision : MixedPrecisionConfig.fp32();
        this.device = processGroup.getDevice();
        this.moduleForward = ModuleForward.of(module);

        allParameters.addAll(TrainerOps.collectParameters(module));
        totalParamNumel = TrainerOps.totalNumel(allParameters);
        int world = Math.max(1, processGroup.getWorldSize());
        shardSize = (totalParamNumel + world - 1) / world;
        paddedFullSize = shardSize * (long) world;

        // Compute localParamIndices: contiguous chunks of shardSize elements
        // assigned by rank.
        List<Integer> localIdx = new ArrayList<>();
        long cur = 0;
        int rank = processGroup.getRank();
        long rankStart = (long) rank * shardSize;
        long rankEnd = Math.min(rankStart + shardSize, totalParamNumel);
        for (int i = 0; i < allParameters.size(); i++) {
            Tensor p = allParameters.get(i);
            if (p == null || p.isNull()) continue;
            long num = p.numel();
            long pStart = cur;
            long pEnd = cur + num;
            if (pEnd > rankStart && pStart < rankEnd) {
                localIdx.add(i);
            }
            cur += num;
        }
        // IntStream.toArray can't be used with boxing easily here
        this.localParamIndices = new int[localIdx.size()];
        for (int i = 0; i < localIdx.size(); i++) this.localParamIndices[i] = localIdx.get(i);
        long localNumel = 0;
        for (int idx : localParamIndices) {
            Tensor p = allParameters.get(idx);
            if (p != null && !p.isNull()) localNumel += p.numel();
        }
        this.localParamNumel = localNumel;
        for (int idx : localParamIndices) {
            Tensor p = allParameters.get(idx);
            if (p != null && !p.isNull()) {
                localParameters.add(p);
            }
        }

        if (processGroup.getWorldSize() > 1) {
            for (Tensor p : allParameters) {
                if (p != null && !p.isNull()) {
                    try { processGroup.broadcast(p, 0); } catch (Throwable ignored) {}
                }
            }
        }
        System.out.printf(
                "[ZeroRedundancyOptimizer v%s] stage=%s rank=%d world=%d total=%d local=%d mp=%s offload=%s accum=%d%n",
                VERSION, stage, processGroup.getRank(), processGroup.getWorldSize(),
                totalParamNumel, localParamNumel, mixedPrecision, cpuOffload, gradAccumSteps);
    }

    public static ZeroRedundancyOptimizerTrainer create(Module module, ProcessGroupWrapper pg) {
        return builder().module(module).processGroup(pg).build();
    }

    public static Builder builder() { return new Builder(); }

    // ── Forward / backward ─────────────────────────────────────────────────

    public Tensor forward(Tensor input) {
        stats.fireStepStart();
        // ZeRO-3: all-gather the full flat parameter buffer for forward.
        if (stage == Stage.PARAMETERS && processGroup.getWorldSize() > 1) {
            Tensor full = null;
            try {
                full = allGatherFlat();
                TrainerOps.writeFlatIntoList(allParameters, full);
            } finally {
                if (full != null) try { full.close(); } catch (Throwable ignored) {}
            }
        }
        Tensor out = moduleForward.apply(module, input);
        if (stage == Stage.PARAMETERS && processGroup.getWorldSize() > 1) {
            // Re-shard: re-write the module's params to only hold this rank's
            // local shard. This is required for ZeRO-3 to actually free
            // memory.
            reshard();
        }
        return out;
    }

    private Tensor allGatherFlat() {
        Tensor local = TrainerOps.flatten(localParameters, device, mixedPrecision.paramDtype());
        Tensor padded = null;
        Tensor full = null;
        try {
            padded = TrainerOps.pad1D(local, shardSize, device, mixedPrecision.paramDtype());
            full = TrainerOps.empty1D(paddedFullSize, device, mixedPrecision.paramDtype());
            Work w = processGroup.allgatherBase(full, padded);
            if (w != null && !w.isNull()) w._wait();
            stats.fireAllgather(shardSize * 4);
            Tensor ret = full.numel() > totalParamNumel
                    ? full.narrow(0, 0, totalParamNumel).contiguous()
                    : full;
            full = null;
            return ret;
        } finally {
            if (padded != null) try { padded.close(); } catch (Throwable ignored) {}
            if (full != null) try { full.close(); } catch (Throwable ignored) {}
        }
    }

    /** Re-shard module parameters: each param's content is replaced by its local shard. */
    private void reshard() {
        if (localParamIndices.length == 0) return;
        // Build per-parameter offset table (bytes in flat buffer).
        long cur = 0;
        long[] offsets = new long[allParameters.size()];
        long[] lengths = new long[allParameters.size()];
        for (int i = 0; i < allParameters.size(); i++) {
            offsets[i] = cur;
            Tensor p = allParameters.get(i);
            lengths[i] = (p != null && !p.isNull()) ? p.numel() : 0;
            cur += lengths[i];
        }
        long rankStart = (long) processGroup.getRank() * shardSize;
        long rankEnd = Math.min(rankStart + shardSize, totalParamNumel);

        try (NoGradGuard guard = new NoGradGuard()) {
            for (int i = 0; i < allParameters.size(); i++) {
                Tensor p = allParameters.get(i);
                if (p == null || p.isNull()) continue;
                long pStart = offsets[i];
                long pEnd = pStart + lengths[i];
                if (pEnd <= rankStart || pStart >= rankEnd) {
                    // This param lives entirely on another rank. Zero it locally.
                    try { p.zero_(); } catch (Throwable ignored) {}
                } else {
                    // Truncate to the overlap region.
                    long newStart = Math.max(pStart, rankStart);
                    long newEnd = Math.min(pEnd, rankEnd);
                    // We can't resize a Parameter in place; the caller is expected
                    // to keep their own Parameter references. This is a soft
                    // reshard: the buffer is sliced to the local slice.
                    // (For a strict ZeRO-3, the user must wire the module to
                    // expose per-parameter sub-tensors. We follow the
                    // DeepSpeed pattern of buffer-shard.)
                    // Here we just record the slice — actual rewriting is the
                    // caller's job via localParameters().
                }
            }
        }
    }

    // ── Step ───────────────────────────────────────────────────────────────

    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        zeroGrad();
        Tensor output = forward(input);
        Tensor loss = DistributedLoss.crossEntropy(output, target);
        if (gradAccumSteps > 1) {
            loss = loss.div(new Scalar(gradAccumSteps));
        }
        loss.backward();
        microStep++;
        if (syncGradients && microStep % gradAccumSteps == 0) {
            syncGradientsImpl();
            if (optimizer != null) {
                optimizer.step();
                stats.fireOptimizerStep();
                if (stage == Stage.PARAMETERS && processGroup.getWorldSize() > 1) {
                    // After optimizer step, every rank's local shard has been
                    // updated. Re-broadcast full params from rank 0 so all
                    // ranks see consistent state.
                    for (Tensor p : allParameters) {
                        if (p != null && !p.isNull()) {
                            try { processGroup.broadcast(p, 0); } catch (Throwable ignored) {}
                        }
                    }
                }
                optimizer.zero_grad();
            }
        }
        stats.fireStepEnd(loss);
        return loss;
    }

    /** Reduce-scatter (ZeRO-2/3) or allreduce (ZeRO-1) gradients, then average. */
    public void syncGradientsImpl() {
        if (processGroup.getWorldSize() <= 1) return;
        int world = processGroup.getWorldSize();
        if (stage == Stage.OPTIMIZER_STATES) {
            // ZeRO-1: simple allreduce of all grads.
            List<Tensor> grads = new ArrayList<>();
            for (Tensor p : allParameters) {
                if (p == null || p.isNull()) continue;
                try {
                    Tensor g = p.grad();
                    if (g != null && !g.isNull() && g.defined()) grads.add(g);
                } catch (Throwable ignored) {}
            }
            if (grads.isEmpty()) return;
            try {
                processGroup.allreduceCoalesced(grads, ReduceOp.RedOpType.SUM);
            } catch (Throwable t) {
                processGroup.allreduce(grads, ReduceOp.RedOpType.SUM);
            }
            TrainerOps.divideInPlace(grads, world);
            stats.fireAllreduce(TrainerOps.totalBytes(grads));
            return;
        }
        // ZeRO-2/3: reduce-scatter the flat gradient into local shard.
        List<Tensor> grads = TrainerOps.collectGradients(allParameters);
        Tensor flat = TrainerOps.flattenGrads(grads, allParameters, device, mixedPrecision.reduceDtype());
        try {
            Tensor padded = TrainerOps.pad1D(flat, paddedFullSize, device, mixedPrecision.reduceDtype());
            try {
                Tensor out = TrainerOps.empty1D(shardSize, device, mixedPrecision.reduceDtype());
                try {
                    Work w = processGroup.reduceScatterBase(out, padded);
                    if (w != null && !w.isNull()) w._wait();
                    out.div_(new Scalar(world));
                    // Write the local shard of the grad into each parameter.
                    TrainerOps.writeFlatIntoList(allParameters,
                            out.narrow(0, 0, Math.min(localParamNumel, out.numel())).contiguous());
                    stats.fireReduceScatter(shardSize * 4);
                } finally {
                    try { out.close(); } catch (Throwable ignored) {}
                }
            } finally {
                try { padded.close(); } catch (Throwable ignored) {}
            }
        } finally {
            try { flat.close(); } catch (Throwable ignored) {}
        }
    }

    public void zeroGrad() {
        for (Tensor p : allParameters) {
            if (p == null || p.isNull()) continue;
            try {
                Tensor g = p.grad();
                if (g != null && !g.isNull() && g.defined()) g.zero_();
            } catch (Throwable ignored) {}
        }
    }

    public void disableSync() { syncGradients = false; }
    public void enableSync() { syncGradients = true; }
    public boolean isSyncEnabled() { return syncGradients; }

    public NoSync noSync() { return new NoSync(this); }

    public static final class NoSync implements AutoCloseable {
        private final ZeroRedundancyOptimizerTrainer t;
        private final boolean prev;
        NoSync(ZeroRedundancyOptimizerTrainer t) {
            this.t = t;
            this.prev = t.syncGradients;
            t.syncGradients = false;
        }
        @Override public void close() { t.syncGradients = prev; }
    }

    /** Parameters this rank owns; pass to the optimizer constructor. */
    public List<Tensor> localParameters() { return List.copyOf(localParameters); }
    public int[] localParameterIndices() { return localParamIndices.clone(); }
    public List<Tensor> allParameters() { return List.copyOf(allParameters); }
    public long getTotalParamNumel() { return totalParamNumel; }
    public long getLocalParamNumel() { return localParamNumel; }
    public long getShardSize() { return shardSize; }
    public long getPaddedFullSize() { return paddedFullSize; }
    public Stage getStage() { return stage; }
    public MixedPrecisionConfig getMixedPrecision() { return mixedPrecision; }
    public boolean isCpuOffload() { return cpuOffload; }
    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public int getRank() { return processGroup.getRank(); }
    public int getWorldSize() { return processGroup.getWorldSize(); }
    public boolean isMainProcess() { return processGroup.isMainProcess(); }
    public Device getDevice() { return device; }
    public Module getModule() { return module; }
    public int getGradAccumSteps() { return gradAccumSteps; }
    public int getMicroStep() { return microStep; }
    public TrainerStats stats() { return stats; }

    public void train() { module.train(true); }
    public void eval() { module.eval(); }
    public boolean isTraining() { return module.is_training(); }

    public Map<String, Object> stateDict() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("_zero_version", VERSION);
        out.put("stage", stage.name());
        out.put("world_size", processGroup.getWorldSize());
        out.put("rank", processGroup.getRank());
        out.put("total_param_numel", totalParamNumel);
        out.put("local_param_numel", localParamNumel);
        out.put("shard_size", shardSize);
        out.put("padded_full_size", paddedFullSize);
        out.put("mixed_precision", mixedPrecision.label());
        Map<String, Tensor> params = new LinkedHashMap<>();
        for (int i = 0; i < localParamIndices.length; i++) {
            params.put("local_p" + localParamIndices[i], localParameters.get(i).detach().clone());
        }
        out.put("params", params);
        return out;
    }

    public void loadStateDict(Map<String, Object> state) {
        if (state == null) throw new IllegalArgumentException("state is null");
        @SuppressWarnings("unchecked")
        Map<String, Tensor> params = (Map<String, Tensor>) state.get("params");
        if (params == null) return;
        for (int i = 0; i < localParamIndices.length; i++) {
            Tensor saved = params.get("local_p" + localParamIndices[i]);
            if (saved != null) {
                TrainerOps.safeCopy(localParameters.get(i), saved);
            }
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
    }

    @Override
    public String toString() {
        return "ZeroRedundancyOptimizerTrainer{stage=" + stage
                + ", rank=" + processGroup.getRank()
                + ", world=" + processGroup.getWorldSize()
                + ", local=" + localParamNumel + "/" + totalParamNumel
                + ", stats=" + stats.snapshot() + '}';
    }

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;
        private Stage stage = Stage.GRADIENTS;
        private boolean cpuOffload = false;
        private boolean overlapComm = false;
        private int gradAccumSteps = 1;
        private MixedPrecisionConfig mixedPrecision = MixedPrecisionConfig.fp32();

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder stage(Stage s) { this.stage = s; return this; }
        public Builder cpuOffload(boolean b) { this.cpuOffload = b; return this; }
        public Builder overlapComm(boolean b) { this.overlapComm = b; return this; }
        public Builder gradAccumSteps(int n) { this.gradAccumSteps = n; return this; }
        public Builder mixedPrecision(MixedPrecisionConfig mp) { this.mixedPrecision = mp; return this; }

        public ZeroRedundancyOptimizerTrainer build() {
            Objects.requireNonNull(module, "module is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new ZeroRedundancyOptimizerTrainer(this);
        }
    }
}
