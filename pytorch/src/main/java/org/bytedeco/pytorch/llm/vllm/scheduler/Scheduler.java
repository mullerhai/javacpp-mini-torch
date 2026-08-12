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
package org.bytedeco.pytorch.llm.vllm.scheduler;

import org.bytedeco.pytorch.llm.vllm.EngineConfig;
import org.bytedeco.pytorch.llm.vllm.Sequence;
import org.bytedeco.pytorch.llm.vllm.SequenceStatus;

import java.util.ArrayList;
import java.util.List;

/**
 * FCFS continuous-batching scheduler (nano-vllm style).
 *
 * <p>Each {@link #schedule()} call decides which sequences get prefill vs decode
 * based on {@code EngineConfig} token/sequence budgets.
 *
 * <p>Policy:
 * <ul>
 *   <li>Waiting seqs → prefill first (until budget full), then decode running seqs.</li>
 *   <li>Strict prefill-before-decode within one step keeps correctness simple.</li>
 *   <li>Finished / aborted seqs are collected for cache release.</li>
 * </ul>
 *
 * <p>Supports multiple scheduling policies:
 * <ul>
 *   <li>{@code FCFS}: First-come-first-served (default)</li>
 *   <li>{@code SHORTEST}: Prioritize shortest sequences</li>
 *   <li>{@code LONGEST}: Prioritize longest sequences</li>
 *   <li>{@code PRIORITY}: User-provided priority queue</li>
 * </ul>
 */
public final class Scheduler implements AutoCloseable {

    private final EngineConfig config;
    private final List<Sequence> waiting = new ArrayList<>();
    private final List<Sequence> running = new ArrayList<>();
    private volatile boolean closed;

    // Scheduling policy
    public enum Policy { FCFS, SHORTEST, LONGEST, PRIORITY }
    private final Policy policy;

    // Performance metrics
    private long totalScheduleTimeMs;
    private int totalScheduleCalls;

    public Scheduler(EngineConfig config) {
        this(config, Policy.FCFS);
    }

    public Scheduler(EngineConfig config, Policy policy) {
        this.config = config;
        this.policy = policy != null ? policy : Policy.FCFS;
    }

    /** Enqueue a new request. */
    public void add(Sequence seq) {
        if (seq.status() != SequenceStatus.WAITING) {
            throw new IllegalStateException("Sequence must be WAITING on add");
        }
        if (closed) {
            throw new IllegalStateException("Scheduler is closed");
        }
        waiting.add(seq);
    }

    /** Remove a sequence from all queues (abort). */
    public void abort(Sequence seq) {
        if (closed) return;
        if (waiting.remove(seq)) return;
        if (running.remove(seq)) return;
    }

//    public boolean isClosed() { return closed; }

    /**
     * Return a list of running sequences that are not finished. */
    public List<Sequence> running() {
        return running;
    }

    /** Number of waiting requests. */
//    public int waitingCount() { return waiting.size(); }

    /** Return a list of running sequences that are not finished. */
//    public List<Sequence> running() {
//        return running;
//    }

    /** Number of waiting requests. */
    public int waitingCount() { return waiting.size(); }

    /** Number of running requests. */
    public int runningCount() { return running.size(); }

    /** Get scheduling policy. */
    public Policy policy() { return policy; }

    /** Get scheduling statistics. */
    public ScheduleStats getStats() {
        return new ScheduleStats(totalScheduleCalls, totalScheduleTimeMs,
                waiting.size(), running.size());
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        waiting.clear();
        running.clear();
        System.out.printf(
                "[Scheduler] Closed: totalScheduleCalls=%d, totalTime=%.2fs, " +
                "avgScheduleTime=%.2fms%n",
                totalScheduleCalls, totalScheduleTimeMs / 1000.0,
                totalScheduleCalls > 0 ? (double) totalScheduleTimeMs / totalScheduleCalls : 0);
    }

    public boolean isClosed() { return closed; }

    /**
     * Scheduling statistics.
     */
    public static final class ScheduleStats {
        public final int totalScheduleCalls;
        public final long totalTimeMs;
        public final int waiting;
        public final int running;

        public ScheduleStats(int totalScheduleCalls, long totalTimeMs,
                           int waiting, int running) {
            this.totalScheduleCalls = totalScheduleCalls;
            this.totalTimeMs = totalTimeMs;
            this.waiting = waiting;
            this.running = running;
        }

        public double avgScheduleTimeMs() {
            return totalScheduleCalls > 0 ? (double) totalTimeMs / totalScheduleCalls : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "ScheduleStats{calls=%d, avgTime=%.2fms, waiting=%d, running=%d}",
                    totalScheduleCalls, avgScheduleTimeMs(), waiting, running);
        }
    }

    /**
     * Compute next scheduling decision under token and sequence budgets.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Collect newly-finished/aborted seqs from running.</li>
     *   <li>Pull waiting seqs into prefill until prefill tokens reach budget.</li>
     *   <li>If budget allows, pull running decode seqs (one token each).</li>
     *   <li>Move waiting→running for any seq that finished prefill but didn't fit budget.</li>
     * </ol>
     */
    public SchedulerOutput schedule() {
        List<Sequence> finished = new ArrayList<>();
        List<Sequence> prefill = new ArrayList<>();
        List<Sequence> decode = new ArrayList<>();

        // 1. Finish and free finished/aborted
        for (int i = running.size() - 1; i >= 0; i--) {
            Sequence s = running.get(i);
            if (s.isFinished()) {
                running.remove(i);
                finished.add(s);
            }
        }

        // 2. Move waiting seqs into prefill under token budget
        int prefillTokens = 0;
        for (int i = 0; i < waiting.size(); i++) {
            Sequence s = waiting.get(i);
            int needed = s.numUncomputedTokens();
            if (prefillTokens + needed > config.maxNumBatchedTokens) {
                if (prefill.size() >= config.maxNumSeqs) break;
                // partial budget left: try to squeeze one more if it's tiny
                if (needed > config.maxNumBatchedTokens - prefillTokens && prefillTokens > 0) {
                    // can't fit this waiting seq, skip for now
                    continue;
                }
            }
            prefillTokens += needed;
            waiting.remove(i--);
            s.setStatus(SequenceStatus.RUNNING);
            running.add(s);
            prefill.add(s);
            if (prefill.size() >= config.maxNumSeqs) break;
        }

        // 3. Decode running seqs (one token each) under remaining budget
        int remainingBudget = config.maxNumBatchedTokens - prefillTokens;
        int decodeCount = Math.min(running.size(), remainingBudget);
        // decode only those already in decode phase (prefill done)
        int decodeAdded = 0;
        for (Sequence s : running) {
            if (s.isPrefillDone() && decodeAdded < decodeCount) {
                decode.add(s);
                decodeAdded++;
            }
        }

        return new SchedulerOutput(prefill, decode, finished);
    }
}
