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

import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-trainer training-step metrics + lifecycle hook support used by the
 * distributed trainers.
 *
 * <p>Tracks a few counters that the Python trainers expose via
 * {@code DistributedDataParallel.__init__} and a small hook chain that lets
 * embedders observe {@code on_step_start}, {@code on_forward},
 * {@code on_backward}, {@code on_grad_synced}, {@code on_step_end}.
 * The native c10d Reducer / FSDP callbacks fire on the C++ side; the Java
 * side exposes a parallel hook chain that the user can attach profiling or
 * tensorboard logging to.
 */
public final class TrainerStats {

    private final AtomicLong forwardCalls = new AtomicLong();
    private final AtomicLong backwardCalls = new AtomicLong();
    private final AtomicLong gradSyncCalls = new AtomicLong();
    private final AtomicLong allreduceCalls = new AtomicLong();
    private final AtomicLong allgatherCalls = new AtomicLong();
    private final AtomicLong reduceScatterCalls = new AtomicLong();
    private final AtomicLong optimizerSteps = new AtomicLong();
    private final AtomicLong bytesAllreduced = new AtomicLong();
    private final AtomicLong bytesAllgathered = new AtomicLong();
    private final AtomicLong bytesReduceScattered = new AtomicLong();
    private final AtomicLong stepCalls = new AtomicLong();
    private final AtomicLong lossSum = new AtomicLong();       // as fixed-point x1e6
    private final AtomicLong lossCount = new AtomicLong();

    private final List<Hook> hooks = new CopyOnWriteArrayList<>();

    /** Add a hook. Returns the hook for removal. */
    public Hook addHook(Hook h) {
        Objects.requireNonNull(h, "hook");
        hooks.add(h);
        return h;
    }

    public void removeHook(Hook h) { hooks.remove(h); }
    public void clearHooks() { hooks.clear(); }

    public void fireStepStart() {
        stepCalls.incrementAndGet();
        for (Hook h : hooks) h.onStepStart();
    }

    public void fireForward(Tensor input) {
        forwardCalls.incrementAndGet();
        for (Hook h : hooks) h.onForward(input);
    }

    public void fireBackward(Tensor loss) {
        backwardCalls.incrementAndGet();
        for (Hook h : hooks) h.onBackward(loss);
    }

    public void fireGradSynced(long numParams) {
        gradSyncCalls.incrementAndGet();
        for (Hook h : hooks) h.onGradSynced(numParams);
    }

    public void fireAllreduce(long bytes) {
        allreduceCalls.incrementAndGet();
        if (bytes > 0) bytesAllreduced.addAndGet(bytes);
        for (Hook h : hooks) h.onAllreduce(bytes);
    }

    public void fireAllgather(long bytes) {
        allgatherCalls.incrementAndGet();
        if (bytes > 0) bytesAllgathered.addAndGet(bytes);
        for (Hook h : hooks) h.onAllgather(bytes);
    }

    public void fireReduceScatter(long bytes) {
        reduceScatterCalls.incrementAndGet();
        if (bytes > 0) bytesReduceScattered.addAndGet(bytes);
        for (Hook h : hooks) h.onReduceScatter(bytes);
    }

    public void fireOptimizerStep() {
        optimizerSteps.incrementAndGet();
        for (Hook h : hooks) h.onOptimizerStep();
    }

    public void fireStepEnd(Tensor loss) {
        if (loss != null && !loss.isNull() && loss.defined() && loss.numel() > 0) {
            try {
                double lv = loss.item().toDouble();
                if (Double.isFinite(lv)) {
                    lossSum.addAndGet((long) (lv * 1_000_000.0));
                    lossCount.incrementAndGet();
                }
            } catch (Throwable ignored) {
            }
        }
        for (Hook h : hooks) h.onStepEnd(loss);
    }

    public void reset() {
        forwardCalls.set(0);
        backwardCalls.set(0);
        gradSyncCalls.set(0);
        allreduceCalls.set(0);
        allgatherCalls.set(0);
        reduceScatterCalls.set(0);
        optimizerSteps.set(0);
        bytesAllreduced.set(0);
        bytesAllgathered.set(0);
        bytesReduceScattered.set(0);
        stepCalls.set(0);
        lossSum.set(0);
        lossCount.set(0);
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    public long forwardCalls() { return forwardCalls.get(); }
    public long backwardCalls() { return backwardCalls.get(); }
    public long gradSyncCalls() { return gradSyncCalls.get(); }
    public long allreduceCalls() { return allreduceCalls.get(); }
    public long allgatherCalls() { return allgatherCalls.get(); }
    public long reduceScatterCalls() { return reduceScatterCalls.get(); }
    public long optimizerSteps() { return optimizerSteps.get(); }
    public long stepCalls() { return stepCalls.get(); }
    public long bytesAllreduced() { return bytesAllreduced.get(); }
    public long bytesAllgathered() { return bytesAllgathered.get(); }
    public long bytesReduceScattered() { return bytesReduceScattered.get(); }
    public double meanLoss() {
        long c = lossCount.get();
        return c == 0 ? Double.NaN : (lossSum.get() / 1_000_000.0) / c;
    }

    public Snapshot snapshot() {
        return new Snapshot(
                forwardCalls.get(), backwardCalls.get(), gradSyncCalls.get(),
                allreduceCalls.get(), allgatherCalls.get(), reduceScatterCalls.get(),
                optimizerSteps.get(), stepCalls.get(),
                bytesAllreduced.get(), bytesAllgathered.get(), bytesReduceScattered.get(),
                meanLoss());
    }

    @Override
    public String toString() {
        return snapshot().toString();
    }

    /** Immutable copy of a stats snapshot — safe to log / send. */
    public static final class Snapshot {
        public final long forwardCalls, backwardCalls, gradSyncCalls;
        public final long allreduceCalls, allgatherCalls, reduceScatterCalls;
        public final long optimizerSteps, stepCalls;
        public final long bytesAllreduced, bytesAllgathered, bytesReduceScattered;
        public final double meanLoss;

        public Snapshot(long fwd, long bwd, long sync, long ar, long ag, long rs,
                        long opt, long step, long arB, long agB, long rsB, double ml) {
            this.forwardCalls = fwd;
            this.backwardCalls = bwd;
            this.gradSyncCalls = sync;
            this.allreduceCalls = ar;
            this.allgatherCalls = ag;
            this.reduceScatterCalls = rs;
            this.optimizerSteps = opt;
            this.stepCalls = step;
            this.bytesAllreduced = arB;
            this.bytesAllgathered = agB;
            this.bytesReduceScattered = rsB;
            this.meanLoss = ml;
        }

        @Override
        public String toString() {
            return String.format(
                    "steps=%d fwd=%d bwd=%d sync=%d allreduce=%d allgather=%d reduceScatter=%d " +
                    "bytes(ar=%d ag=%d rs=%d) meanLoss=%.6f",
                    stepCalls, forwardCalls, backwardCalls, gradSyncCalls,
                    allreduceCalls, allgatherCalls, reduceScatterCalls,
                    bytesAllreduced, bytesAllgathered, bytesReduceScattered,
                    Double.isNaN(meanLoss) ? 0.0 : meanLoss);
        }
    }

    /**
     * Observability hook. Default methods are no-ops so an embedder can
     * override only the events it cares about.
     */
    public interface Hook {
        default void onStepStart() {}
        default void onForward(Tensor input) {}
        default void onBackward(Tensor loss) {}
        default void onGradSynced(long numParams) {}
        default void onAllreduce(long bytes) {}
        default void onAllgather(long bytes) {}
        default void onReduceScatter(long bytes) {}
        default void onOptimizerStep() {}
        default void onStepEnd(Tensor loss) {}
    }

    /**
     * Convenience: a list of hooks composed in declaration order.
     */
    public static final class CompositeHook implements Hook {
        private final List<Hook> list = new ArrayList<>();
        public CompositeHook add(Hook h) { list.add(h); return this; }
        public CompositeHook remove(Hook h) { list.remove(h); return this; }
        public List<Hook> hooks() { return list; }
        @Override public void onStepStart() { for (Hook h : list) h.onStepStart(); }
        @Override public void onForward(Tensor input) { for (Hook h : list) h.onForward(input); }
        @Override public void onBackward(Tensor loss) { for (Hook h : list) h.onBackward(loss); }
        @Override public void onGradSynced(long numParams) { for (Hook h : list) h.onGradSynced(numParams); }
        @Override public void onAllreduce(long bytes) { for (Hook h : list) h.onAllreduce(bytes); }
        @Override public void onAllgather(long bytes) { for (Hook h : list) h.onAllgather(bytes); }
        @Override public void onReduceScatter(long bytes) { for (Hook h : list) h.onReduceScatter(bytes); }
        @Override public void onOptimizerStep() { for (Hook h : list) h.onOptimizerStep(); }
        @Override public void onStepEnd(Tensor loss) { for (Hook h : list) h.onStepEnd(loss); }
    }
}
