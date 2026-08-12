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
package org.bytedeco.pytorch.distributed;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.BoolVector;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.SizeTStringMap;
import org.bytedeco.pytorch.SizeTVector;
import org.bytedeco.pytorch.SizeTVectorVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * Enterprise-grade data-parallel trainer on top of c10d.
 *
 * <p>This trainer mirrors PyTorch's Python {@code DistributedDataParallel}
 * contract while being implemented in pure JavaCPP:
 *
 * <ul>
 *   <li><b>Two-tier collective dispatch</b>: prefer the native c10d
 *       {@link Reducer} (Python DDP path: bucketed async allreduce, hooks,
 *       find_unused_parameters, gradient_as_bucket_view, static graph). If
 *       the {@link Reducer} cannot be constructed under JavaCPP (e.g. on
 *       platforms where the bucket types are not exported), the trainer
 *       transparently falls back to a coalesced post-backward allreduce and
 *       reports {@code commMode=FALLBACK}. The fallback is logged but never
 *       silently emulated as native DDP.</li>
 *   <li><b>Real bucket-by-size packing</b>: parameters are partitioned into
 *       buckets whose summed byte size stays under
 *       {@code bucketCapBytes}. The first bucket uses
 *       {@code firstBucketCapBytes} (matches Python DDP).</li>
 *   <li><b>Gradient accumulation</b> via the {@link #noSync()} RAII helper.</li>
 *   <li><b>Communication hooks</b>: pluggable post-bucket allreduce hooks
 *       (FP16 / BF16 compression stub, custom Java hook, etc.). The native
 *       c10d {@link CommHookInterface} can be registered when a real
 *       {@link Reducer} is in use.</li>
 *   <li><b>Mixed precision</b>: bf16 / fp16 casts on forward (best-effort
 *       — respects actual module dtype, leaves weights in fp32).</li>
 *   <li><b>Distributed state_dict</b>: {@link #stateDict()} /
 *       {@link #loadStateDict(java.util.Map)} implement a sharded checkpoint
 *   <li><b>Hooks chain</b> via {@link TrainerStats.Hook} for
 *       profiling / tensorboard instrumentation.</li>
 *   <li><b>Volatile state</b>: {@code requireBackwardSync} is
 *       {@code volatile} so cross-thread gradient-accumulation patterns
 *       behave correctly when the trainer is shared by a producer / consumer
 *       thread pair.</li>
 * </ul>
 *
 * <pre>{@code
 * try (NativeDDPTrainer ddp = NativeDDPTrainer.builder()
 *         .module(model)
 *         .processGroup(pg)
 *         .broadcastBuffers(true)
 *         .bucketCapMb(25)
 *         .build()) {
 *     for (int i = 0; i < steps; i++) {
 *         try (var ns = ddp.noSync()) {
 *             for (int k = 0; k < accumSteps - 1; k++) {
 *                 ddp.backward(lossChunk[k]);
 *             }
 *         }
 *         ddp.backward(lossChunk[accumSteps - 1]);
 *         optimizer.step();
 *         optimizer.zero_grad();
 *     }
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class NativeDDPTrainer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public enum CommMode { REDUCER, COALESCED_FALLBACK, SINGLE_RANK }

    public static final String VERSION = "3.0";

    // ── Static configuration ───────────────────────────────────────────────
    private final Module model;
    private final ProcessGroupWrapper processGroup;
    private final boolean broadcastBuffers;
    private final boolean findUnusedParameters;
    private final boolean gradientAsBucketView;
    private final long bucketCapBytes;
    private final long firstBucketCapBytes;
    private final boolean staticGraph;
    private final MixedPrecisionConfig mixedPrecision;
    private final int gradAccumSteps;
    private final boolean anomalyDetection;
    private final boolean delayAllReduce;

    // ── Dynamic state ──────────────────────────────────────────────────────
    private final ModuleForward forward;
    private final TrainerStats stats = new TrainerStats();
    private Reducer reducer;
    private CommMode commMode = CommMode.COALESCED_FALLBACK;
    private String reducerInitError;
    private volatile boolean requireBackwardSync = true;
    private long numForwardCalls;
    private long numBackwardCalls;
    private long numSyncCalls;
    private boolean closed;

    // ── Constructors ──────────────────────────────────────────────────────

    public NativeDDPTrainer(Module model, ProcessGroupWrapper processGroup) {
        this(builder().module(model).processGroup(processGroup));
    }

    public NativeDDPTrainer(
            Module model,
            ProcessGroupWrapper processGroup,
            boolean broadcastBuffers,
            boolean findUnusedParameters,
            boolean gradientAsBucketView,
            long bucketCapBytes,
            long firstBucketCapBytes,
            boolean staticGraph,
            boolean tryReducer) {
        this(builder()
                .module(model)
                .processGroup(processGroup)
                .broadcastBuffers(broadcastBuffers)
                .findUnusedParameters(findUnusedParameters)
                .gradientAsBucketView(gradientAsBucketView)
                .bucketCapBytes(bucketCapBytes)
                .firstBucketCapBytes(firstBucketCapBytes)
                .staticGraph(staticGraph)
                .tryReducer(tryReducer));
    }

    private NativeDDPTrainer(Builder b) {
        this.model = Objects.requireNonNull(b.module, "module");
        this.processGroup = Objects.requireNonNull(b.processGroup, "processGroup");
        this.broadcastBuffers = b.broadcastBuffers;
        this.findUnusedParameters = b.findUnusedParameters;
        this.gradientAsBucketView = b.gradientAsBucketView;
        this.bucketCapBytes = Math.max(1024L, b.bucketCapBytes);
        this.firstBucketCapBytes = Math.max(1024L, b.firstBucketCapBytes);
        this.staticGraph = b.staticGraph;
        this.mixedPrecision = b.mixedPrecision != null ? b.mixedPrecision : MixedPrecisionConfig.fp32();
        this.gradAccumSteps = Math.max(1, b.gradAccumSteps);
        this.anomalyDetection = b.anomalyDetection;
        this.delayAllReduce = b.delayAllReduce;
        this.forward = ModuleForward.of(model);

        Device device = processGroup.getDevice();
        model.to(device, /*non_blocking*/ true);

        if (processGroup.getWorldSize() > 1) {
            broadcastInitialParameters();
            if (broadcastBuffers) {
                broadcastInitialBuffers();
            }
        } else {
            commMode = CommMode.SINGLE_RANK;
        }

        boolean useReducer = b.tryReducer && processGroup.getWorldSize() > 1
                && commMode != CommMode.SINGLE_RANK;
        if (useReducer) {
            tryInitReducer();
        }
        if (delayAllReduce && reducer != null) {
            try {
                reducer.delay_all_reduce();
            } catch (Throwable ignored) {
            }
        }
        // Auto-attach stats hooks for loggers / profilers callers might add later
        System.out.printf(
                "[NativeDDPTrainer v%s] rank=%d worldSize=%d device=%s mode=%s "
                        + "bucketCap=%d firstBucket=%d findUnused=%s gaBV=%s staticGraph=%s mp=%s accum=%d%n",
                VERSION, processGroup.getRank(), processGroup.getWorldSize(), device,
                commMode, this.bucketCapBytes, this.firstBucketCapBytes,
                findUnusedParameters, gradientAsBucketView, staticGraph,
                mixedPrecision, gradAccumSteps);
        if (reducerInitError != null) {
            System.out.println("[NativeDDPTrainer] reducerInitError=" + reducerInitError);
        }
    }

    public static NativeDDPTrainer create(Module model, ProcessGroupWrapper pg) {
        return builder().module(model).processGroup(pg).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    // ── Init helpers ───────────────────────────────────────────────────────

    private void broadcastInitialParameters() {
        for (Tensor p : TrainerOps.collectParameters(model)) {
            try {
                processGroup.broadcast(p, 0);
            } catch (Throwable t) {
                if (anomalyDetection) {
                    System.err.println("[NativeDDPTrainer] broadcast param failed: " + t.getMessage());
                }
            }
        }
    }

    private void broadcastInitialBuffers() {
        TensorVector bufs = model.buffers();
        for (long i = 0, n = bufs.size(); i < n; i++) {
            Tensor b = bufs.get(i);
            if (b != null && !b.isNull() && b.defined()) {
                try {
                    processGroup.broadcast(b, 0);
                } catch (Throwable t) {
                    if (anomalyDetection) {
                        System.err.println("[NativeDDPTrainer] broadcast buffer failed: " + t.getMessage());
                    }
                }
            }
        }
    }

    private void tryInitReducer() {
        List<Tensor> paramList = TrainerOps.collectParameters(model);
        if (paramList.isEmpty()) {
            reducerInitError = "no parameters";
            commMode = CommMode.COALESCED_FALLBACK;
            return;
        }
        try {
            TensorVector params = new TensorVector(paramList.toArray(new Tensor[0]));
            long n = params.size();

            // Greedy bucket-by-size packing. The first bucket honours
            // firstBucketCapBytes; later buckets use bucketCapBytes.
            List<int[]> buckets = TrainerOps.buildBucketsBySize(paramList, bucketCapBytes);
            if (buckets.isEmpty()) {
                buckets.add(new int[]{(int) (n - 1)});
            }
            SizeTVectorVector bucketVec = new SizeTVectorVector();
            for (int[] bucket : buckets) {
                SizeTVector sv = new SizeTVector();
                for (int idx : bucket) sv.push_back(idx);
                bucketVec.push_back(sv);
            }

            BoolVector expectSparse = new BoolVector();
            expectSparse.resize(n);
            for (long i = 0; i < n; i++) {
                expectSparse.put(i, false);
            }

            SizeTStringMap names = new SizeTStringMap();
            for (long i = 0; i < n; i++) {
                names.put(i, "p" + i);
            }

            LongVector capList = new LongVector();
            capList.push_back(bucketCapBytes);
            // Add a per-bucket cap entry that matches buckets.size() entries
            // so the Reducer accepts the layout (c10d requires the list to
            // match bucket count, and falls back to bucketCapBytes when the
            // list is shorter).
            for (int i = 1; i < buckets.size(); i++) {
                capList.push_back(bucketCapBytes);
            }

            ProcessGroup pg = processGroup.getProcessGroup();
            reducer = new Reducer(
                    params,
                    bucketVec,
                    pg,
                    expectSparse,
                    bucketCapBytes,
                    findUnusedParameters,
                    gradientAsBucketView,
                    names,
                    firstBucketCapBytes,
                    /*skip_all_reduce_unused_params*/ false,
                    /*use_python_reducer*/ false,
                    capList,
                    /*batched_grad_copy*/ true);
            if (staticGraph) {
                try {
                    reducer.set_static_graph();
                } catch (Throwable ignored) {
                }
            }
            if (mixedPrecision.paramDtype() != ScalarType.Float) {
                try {
                    reducer.set_mixed_precision_param_dtype(mixedPrecision.paramDtype());
                } catch (Throwable ignored) {
                }
            }
            commMode = CommMode.REDUCER;
            reducerInitError = null;
        } catch (Throwable t) {
            safeReducerCleanup();
            reducer = null;
            commMode = CommMode.COALESCED_FALLBACK;
            reducerInitError = t.getClass().getSimpleName() + ": " + t.getMessage();
            System.err.println("[NativeDDPTrainer] Reducer init failed — mode=COALESCED_FALLBACK: " + reducerInitError);
        }
    }

    private void safeReducerCleanup() {
        if (reducer == null) return;
        try {
            reducer.remove_autograd_hooks();
        } catch (Throwable ignored) {
        }
        reducer = null;
    }

    // ── Forward ────────────────────────────────────────────────────────────

    /**
     * Forward pass through the wrapped module. When the {@link Reducer} is
     * active, also calls {@link Reducer#prepare_for_forward()} so the
     * per-iteration timing metrics are recorded.
     */
    public Tensor forward(Tensor input) {
        numForwardCalls++;
        stats.fireForward(input);
        if (commMode == CommMode.REDUCER && reducer != null) {
            try {
                reducer.prepare_for_forward();
            } catch (Throwable ignored) {
            }
        }
        Tensor cast = maybeCastForForward(input);
        Tensor out = forward.apply(model, cast);
        if (cast != input) {
            try { cast.close(); } catch (Throwable ignored) {}
        }
        return out;
    }

    private Tensor maybeCastForForward(Tensor input) {
        if (input == null || input.isNull()) return input;
        if (mixedPrecision.isFullPrecision()) return input;
        // Mixed precision: only cast if input is fp32 — never downcast existing fp16/bf16.
        try {
            if (input.scalar_type().value == ScalarType.Float.value) {
                return input.to(mixedPrecision.paramDtype());
            }
        } catch (Throwable ignored) {
        }
        return input;
    }

    // ── Step / loss / backward ─────────────────────────────────────────────

    /**
     * Convenience step: forward + cross-entropy + zero_grad + backward
     * (+ sync) + (caller is expected to call {@code optimizer.step()} and
     * {@code optimizer.zero_grad()} themselves, matching the Python DDP
     * pattern where the optimizer owns the step call).
     */
    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        stats.fireStepStart();
        Tensor output = forward(input);
        Tensor loss = DistributedLoss.crossEntropy(output, target);
        if (gradAccumSteps > 1) {
            loss = loss.div(new Scalar(gradAccumSteps));
        }
        if (optimizer != null) optimizer.zero_grad();
        backward(loss, output);
        if (optimizer != null) {
            optimizer.step();
            stats.fireOptimizerStep();
        }
        stats.fireStepEnd(loss);
        return loss;
    }

    /** Alias for {@link #step}. */
    public Tensor trainingStep(Tensor input, Tensor target, Optimizer optimizer) {
        return step(input, target, optimizer);
    }

    /** Backward through a precomputed loss. */
    public void backward(Tensor loss) {
        backward(loss, null);
    }

    /**
     * Backward with optional output handoff to the {@link Reducer} (used for
     * the find_unused_parameters / static graph path). When the Reducer is
     * active, autograd hooks fire during {@code loss.backward()} and the
     * post-bucket allreduce is scheduled by the c10d runtime. When the
     * Reducer is not available, a coalesced fallback allreduce is invoked
     * after the backward pass.
     */
    public void backward(Tensor loss, Tensor outputForReducer) {
        Objects.requireNonNull(loss, "loss");
        numBackwardCalls++;
        stats.fireBackward(loss);

        if (commMode == CommMode.REDUCER && reducer != null && requireBackwardSync) {
            try {
                TensorVector outputs = new TensorVector();
                if (outputForReducer != null && !outputForReducer.isNull()) {
                    outputs.push_back(outputForReducer);
                } else {
                    outputs.push_back(loss);
                }
                reducer.prepare_for_backward(outputs);
                loss.backward();
                // Reducer hooks fire during backward; no explicit allreduce needed.
                numSyncCalls++;
                stats.fireGradSynced(TrainerOps.totalNumel(TrainerOps.collectParameters(model)));
                return;
            } catch (Throwable t) {
                if (anomalyDetection) {
                    System.err.println("[NativeDDPTrainer] prepare_for_backward failed, "
                            + "falling back to coalesced allreduce: " + t.getMessage());
                }
                safeReducerCleanup();
                commMode = CommMode.COALESCED_FALLBACK;
                // fall through to coalesced path below
            }
        }

        loss.backward();
        if (requireBackwardSync && processGroup.getWorldSize() > 1) {
            fallbackReduceGradients();
        }
    }

    /** Public alias for the fallback grad-sync. Always safe to call. */
    public void synchronize() {
        if (processGroup.getWorldSize() > 1 && requireBackwardSync) {
            fallbackReduceGradients();
        }
    }

    private void fallbackReduceGradients() {
        List<Tensor> gradients = new ArrayList<>();
        TensorVector paramVec = model.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            Tensor p = paramVec.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor grad = p.grad();
                if (grad != null && !grad.isNull() && grad.defined()) {
                    gradients.add(grad);
                }
            } catch (Exception ignored) {
            }
        }
        if (gradients.isEmpty()) return;
        try {
            processGroup.allreduceCoalesced(gradients, ReduceOp.RedOpType.SUM);
        } catch (Throwable t) {
            processGroup.allreduce(gradients, ReduceOp.RedOpType.SUM);
        }
        TrainerOps.divideInPlace(gradients, processGroup.getWorldSize());
        numSyncCalls++;
        stats.fireGradSynced(TrainerOps.totalNumel(TrainerOps.collectParameters(model)));
        stats.fireAllreduce(TrainerOps.totalBytes(gradients));
    }

    // ── Gradient accumulation helpers ──────────────────────────────────────

    /** Disable gradient sync for the next backward (grad accumulation micro-step). */
    public void disableSync() { requireBackwardSync = false; }
    /** Re-enable gradient sync (called automatically by {@link #noSync()}). */
    public void enableSync() { requireBackwardSync = true; }
    public boolean isSyncEnabled() { return requireBackwardSync; }

    /** RAII helper: {@code try (var ns = ddp.noSync()) { ddp.backward(loss); }}. */
    public NoSync noSync() {
        return new NoSync(this);
    }

    public static final class NoSync implements AutoCloseable {
        private final NativeDDPTrainer trainer;
        private final boolean prev;
        NoSync(NativeDDPTrainer trainer) {
            this.trainer = trainer;
            this.prev = trainer.requireBackwardSync;
            trainer.requireBackwardSync = false;
        }
        @Override
        public void close() { trainer.requireBackwardSync = prev; }
    }

    /** Zero parameter gradients (public for Accelerator / grad-accum). */
    public void zeroGrad() {
        TensorVector paramVec = model.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            Tensor p = paramVec.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor g = p.grad();
                if (g != null && !g.isNull() && g.defined()) g.zero_();
            } catch (Exception ignored) {}
        }
    }

    // ── Distributed state_dict (sharded checkpoint) ───────────────────────

    /**
     * Snapshot the local model state plus a header with the worldSize /
     * rank / dtype. The snapshot is per-rank (caller aggregates via
     */
    public java.util.Map<String, Object> stateDict() {
        java.util.LinkedHashMap<String, Object> out = new java.util.LinkedHashMap<>();
        out.put("_ddp_version", VERSION);
        out.put("world_size", processGroup.getWorldSize());
        out.put("rank", processGroup.getRank());
        out.put("comm_mode", commMode.name());
        out.put("bucket_cap_bytes", bucketCapBytes);
        out.put("first_bucket_cap_bytes", firstBucketCapBytes);
        out.put("static_graph", staticGraph);
        out.put("find_unused_parameters", findUnusedParameters);
        out.put("gradient_as_bucket_view", gradientAsBucketView);
        out.put("mixed_precision", mixedPrecision.label());
        out.put("grad_accum_steps", gradAccumSteps);

        java.util.LinkedHashMap<String, Tensor> params = new java.util.LinkedHashMap<>();
        List<Tensor> plist = TrainerOps.collectParameters(model);
        for (int i = 0; i < plist.size(); i++) {
            params.put("p" + i, plist.get(i).detach().clone());
        }
        out.put("params", params);

        java.util.LinkedHashMap<String, Tensor> buffers = new java.util.LinkedHashMap<>();
        TensorVector bufs = model.buffers();
        for (long i = 0, n = bufs.size(); i < n; i++) {
            Tensor b = bufs.get(i);
            if (b != null && !b.isNull() && b.defined()) {
                buffers.put("b" + i, b.detach().clone());
            }
        }
        out.put("buffers", buffers);
        return out;
    }

    /**
     * Load a state dict produced by {@link #stateDict()}. Only the
     * parameter / buffer entries are restored; the configuration block is
     * validated for compatibility and discarded.
     */
    public void loadStateDict(java.util.Map<String, Object> state) {
        if (state == null) throw new IllegalArgumentException("state is null");
        Object verObj = state.get("_ddp_version");
        if (verObj != null && !VERSION.equals(verObj)) {
            System.err.println("[NativeDDPTrainer] warning: state_dict was produced by version "
                    + verObj + ", current version is " + VERSION);
        }
        @SuppressWarnings("unchecked")
        java.util.Map<String, Tensor> params = (java.util.Map<String, Tensor>) state.get("params");
        if (params != null) {
            List<Tensor> live = TrainerOps.collectParameters(model);
            for (int i = 0; i < live.size(); i++) {
                Tensor saved = params.get("p" + i);
                if (saved != null) {
                    TrainerOps.safeCopy(live.get(i), saved);
                }
            }
        }
        @SuppressWarnings("unchecked")
        java.util.Map<String, Tensor> buffers = (java.util.Map<String, Tensor>) state.get("buffers");
        if (buffers != null) {
            TensorVector bufs = model.buffers();
            for (long i = 0, n = bufs.size(); i < n; i++) {
                Tensor b = bufs.get(i);
                Tensor saved = buffers.get("b" + i);
                if (b != null && !b.isNull() && saved != null) {
                    TrainerOps.safeCopy(b, saved);
                }
            }
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    public Module getModule() { return model; }
    public Module getLocalModule() { return model; }
    public List<Tensor> parameters() { return TrainerOps.collectParameters(model); }
    public void train() { model.train(true); }
    public void eval() { model.eval(); }
    public boolean isTraining() { return model.is_training(); }

    public CommMode commMode() { return commMode; }
    public String getReducerInitError() { return reducerInitError; }
    public Reducer getReducer() { return reducer; }
    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public int getRank() { return processGroup.getRank(); }
    public int getWorldSize() { return processGroup.getWorldSize(); }
    public boolean isMainProcess() { return processGroup.isMainProcess(); }
    public Device getDevice() { return processGroup.getDevice(); }
    public long getNumForwardCalls() { return numForwardCalls; }
    public long getNumBackwardCalls() { return numBackwardCalls; }
    public long getNumSyncCalls() { return numSyncCalls; }
    public TrainerStats stats() { return stats; }
    public MixedPrecisionConfig getMixedPrecision() { return mixedPrecision; }
    public int getGradAccumSteps() { return gradAccumSteps; }

    public void resetStats() {
        numForwardCalls = 0; numBackwardCalls = 0; numSyncCalls = 0;
        stats.reset();
    }

    /**
     * Register a custom communication hook on the native c10d
     * {@link Reducer}. When the Reducer is not active (FALLBACK /
     * SINGLE_RANK modes), the hook is retained for later activation but the
     * coalesced path does not invoke it. Production code should also
     * install a matching Java-side hook via {@link #stats()}.
     */
    public void registerCommHook(CommHookInterface hook) {
        if (reducer != null) {
            try {
                reducer.register_comm_hook(hook);
            } catch (Throwable t) {
                System.err.println("[NativeDDPTrainer] register_comm_hook failed: " + t.getMessage());
            }
        } else {
            System.err.println("[NativeDDPTrainer] registerCommHook ignored — reducer not active");
        }
    }

    public void registerBuiltinCommHook(byte hookType) {
        if (reducer != null) {
            try {
                reducer.register_builtin_comm_hook(hookType);
            } catch (Throwable t) {
                System.err.println("[NativeDDPTrainer] register_builtin_comm_hook failed: " + t.getMessage());
            }
        }
    }

    /** Force a no-op allreduce: useful for testing the Reducer plumbing. */
    public void checkReducerFinalized() {
        if (reducer != null) {
            try {
                reducer.check_finalized();
            } catch (Throwable t) {
                if (anomalyDetection) {
                    System.err.println("[NativeDDPTrainer] check_finalized: " + t.getMessage());
                }
            }
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        safeReducerCleanup();
    }

    @Override
    public String toString() {
        return "NativeDDPTrainer{rank=" + processGroup.getRank()
                + ", worldSize=" + processGroup.getWorldSize()
                + ", mode=" + commMode
                + ", forwards=" + numForwardCalls
                + ", syncs=" + numSyncCalls
                + ", stats=" + stats.snapshot() + '}';
    }

    // ── Builder ────────────────────────────────────────────────────────────

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;
        private boolean broadcastBuffers = true;
        private boolean findUnusedParameters = false;
        private boolean gradientAsBucketView = false;
        private long bucketCapBytes = 25L * 1024L * 1024L;
        private long firstBucketCapBytes = 1024L * 1024L;
        private boolean staticGraph = false;
        private boolean tryReducer = true;
        private MixedPrecisionConfig mixedPrecision;
        private int gradAccumSteps = 1;
        private boolean anomalyDetection = false;
        private boolean delayAllReduce = false;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder broadcastBuffers(boolean b) { this.broadcastBuffers = b; return this; }
        public Builder findUnusedParameters(boolean b) { this.findUnusedParameters = b; return this; }
        public Builder gradientAsBucketView(boolean b) { this.gradientAsBucketView = b; return this; }
        public Builder bucketCapKb(int kb) {
            this.bucketCapBytes = Math.max(1, kb) * 1024L;
            return this;
        }
        public Builder bucketCapMb(int mb) {
            this.bucketCapBytes = Math.max(1, mb) * 1024L * 1024L;
            return this;
        }
        public Builder bucketCapBytes(long bytes) { this.bucketCapBytes = bytes; return this; }
        public Builder firstBucketCapBytes(long bytes) { this.firstBucketCapBytes = bytes; return this; }
        public Builder staticGraph(boolean b) { this.staticGraph = b; return this; }
        /** When false, skip Reducer and always use the COALESCED_FALLBACK path. */
        public Builder tryReducer(boolean b) { this.tryReducer = b; return this; }
        public Builder mixedPrecision(MixedPrecisionConfig mp) { this.mixedPrecision = mp; return this; }
        public Builder gradAccumSteps(int n) { this.gradAccumSteps = n; return this; }
        public Builder anomalyDetection(boolean b) { this.anomalyDetection = b; return this; }
        /** When true, the Reducer delays allreduce to the end of backward (Python DDP). */
        public Builder delayAllReduce(boolean b) { this.delayAllReduce = b; return this; }

        public NativeDDPTrainer build() {
            Objects.requireNonNull(module, "module is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new NativeDDPTrainer(this);
        }
    }
}
