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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.vllm;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.vllm.cache.CacheEngine;
import org.bytedeco.pytorch.llm.vllm.metrics.EngineMetrics;
import org.bytedeco.pytorch.llm.vllm.runner.EmbeddingRunner;
import org.bytedeco.pytorch.llm.vllm.runner.ModelRunner;
import org.bytedeco.pytorch.llm.vllm.sampling.Sampler;
import org.bytedeco.pytorch.llm.vllm.scheduler.Scheduler;
import org.bytedeco.pytorch.llm.vllm.scheduler.SchedulerOutput;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Core continuous-batching inference engine (nano-vLLM style).
 *
 * <p>Lifecycle:
 * <ol>
 *   <li>{@link #addRequest} enqueues a new sequence.</li>
 *   <li>Client drives {@link #step()} until {@link #hasPending()} is false.</li>
 *   <li>Finished seqs are returned via {@link #finishPendingRequests()}.</li>
 * </ol>
 *
 * <p>The engine owns the KV cache, scheduler, model runner, and sampler.
 */
public final class LLMEngine implements AutoCloseable {

    private final EngineConfig config;
    private final Scheduler scheduler;
    private final CacheEngine cache;
    private final ModelRunner runner;
    private final EngineMetrics metrics;
    private final FastTokenizer tokenizer;   // for decoding output token ids → text

    private final Map<Long, Sequence> activeSeqs = new HashMap<>();
    private final List<RequestOutput> finishedOutputs = new ArrayList<>();
    private volatile boolean closed;

    // Performance metrics
    private long totalStepTimeMs;
    private long totalPrefillTimeMs;
    private long totalDecodeTimeMs;
    private long totalSchedulingTimeMs;
    private int totalSteps;

    public LLMEngine(EngineConfig config, ModelRunner runner, CacheEngine cache,
                   FastTokenizer tokenizer) {
        this.config = Objects.requireNonNull(config, "config");
        this.scheduler = new Scheduler(config);
        this.runner = Objects.requireNonNull(runner, "runner");
        this.cache = Objects.requireNonNull(cache, "cache");
        this.tokenizer = tokenizer;
        this.metrics = new EngineMetrics();
    }

    public EngineConfig config() { return config; }
    public EngineMetrics metrics() { return metrics; }
    public CacheEngine cache() { return cache; }

    /** Add a new generation request. */
    public long addRequest(int[] promptTokenIds, SamplingParams samplingParams,
                           String promptText, String promptSystem) {
        // Apply chat template if system prompt present
        int[] finalIds = promptTokenIds;
        if (promptSystem != null && !promptSystem.isBlank()) {
            // Simple concatenation; caller should ideally use ChatTemplate
            // For now just note it — the LLM facade will handle this
        }
        Sequence seq = new Sequence(finalIds, samplingParams, null, promptText);
        long id = seq.requestId();
        activeSeqs.put(id, seq);
        scheduler.add(seq);
        cache.createSequence(seq);
        metrics.numRequests.increment();
        return id;
    }

    /** Abort a running request. */
    public void abortRequest(long requestId) {
        Sequence seq = activeSeqs.remove(requestId);
        if (seq == null) return;
        scheduler.abort(seq);
        cache.releaseSequence(seq);
        seq.markAborted();
        finishedOutputs.add(RequestOutput.fromSequence(seq, ""));
    }

    /** Execute one scheduling + forward + sample step. */
    public void step() {
        if (closed) throw new IllegalStateException("Engine closed");
        if (!hasPending()) return;

        long stepStart = System.currentTimeMillis();
        long schedulingStart = stepStart;

        // 1. Schedule (also harvests seqs finished on a prior step)
        SchedulerOutput sched = scheduler.schedule();
        long schedulingTime = System.currentTimeMillis() - schedulingStart;

        // Always free previously-finished seqs, even when this step has no new work.
        // Otherwise generateAll() can hang: last sample marks FINISHED, next schedule
        // collects them into finishedSeqs, hasWork() is false, and they never land in
        // finishedOutputs / leave the running set's bookkeeping.
        for (Sequence fin : sched.finishedSeqs) {
            cache.releaseSequence(fin);
            activeSeqs.remove(fin.requestId());
            String decoded = tokenizer != null
                ? tokenizer.decode(fin.outputTokenIdsArray(), true)
                : "";
        finishedOutputs.add(RequestOutput.fromSequence(fin, decoded));
            metrics.numFinished.increment();
        }
        if (!sched.hasWork()) {
            long stepMs = System.currentTimeMillis() - stepStart;
            totalStepTimeMs += stepMs;
            totalSchedulingTimeMs += schedulingTime;
            totalSteps++;
            metrics.recordStep(stepMs);
            return;
        }

        // 2. Build cache id list for scheduled seqs
        List<Sequence> allSeqs = new ArrayList<>(sched.prefillSeqs);
        allSeqs.addAll(sched.decodeSeqs);
        long[] cacheIds = new long[allSeqs.size()];
        for (int i = 0; i < allSeqs.size(); i++) cacheIds[i] = allSeqs.get(i).cacheSeqId();

        // 3. Forward (prefill + decode sequentially per seq)
        long prefillStart = System.currentTimeMillis();
        List<Tensor> logitsList = runner.forwardBatch(sched.prefillSeqs, sched.decodeSeqs, cacheIds);
        long prefillTime = System.currentTimeMillis() - prefillStart;

        // 4. Sample
        long decodeStart = System.currentTimeMillis();
        int li = 0;
        for (Sequence seq : sched.prefillSeqs) {
            if (!seq.isFinished()) {
                // Prefill tokens accounted once, on first sample after prompt forward.
                if (seq.numOutputTokens() == 0) {
                    metrics.numPrefillTokens.add(seq.promptLen());
                }
                Sampler.sample(logitsList.get(li++), seq);
                metrics.numTokens.increment();
            } else {
                li++;
            }
        }
        for (Sequence seq : sched.decodeSeqs) {
            if (!seq.isFinished()) {
                Sampler.sample(logitsList.get(li++), seq);
                metrics.numTokens.increment();
                metrics.numDecodeTokens.increment();
            } else {
                li++;
            }
        }
        long decodeTime = System.currentTimeMillis() - decodeStart;

        long stepMs = System.currentTimeMillis() - stepStart;
        totalStepTimeMs += stepMs;
        totalPrefillTimeMs += prefillTime;
        totalDecodeTimeMs += decodeTime;
        totalSchedulingTimeMs += schedulingTime;
        totalSteps++;
        metrics.recordStep(stepMs);
    }

    /** Run steps until all requests finish. */
    public List<RequestOutput> generateAll() {
        while (hasPending()) step();
        List<RequestOutput> out = new ArrayList<>(finishedOutputs);
        finishedOutputs.clear();
        return out;
    }

    /** True if there are sequences waiting or running. */
    public boolean hasPending() {
        return scheduler.waitingCount() > 0 || scheduler.runningCount() > 0;
    }

    /** Return and clear newly-finished request outputs. */
    public List<RequestOutput> finishPendingRequests() {
        List<RequestOutput> out = new ArrayList<>(finishedOutputs);
        finishedOutputs.clear();
        return out;
    }

    /** Embedding batch (separate from generation path). */
    public float[][] embedTexts(List<String> texts, EmbeddingRunner embedRunner) {
        return embedRunner.encodeBatch(texts);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Abort all pending requests
        for (Long requestId : new ArrayList<>(activeSeqs.keySet())) {
            abortRequest(requestId);
        }

        // Close sub-components
        try { scheduler.close(); } catch (Throwable ignored) {}
        try { runner.close(); } catch (Throwable ignored) {}
        try { cache.close(); } catch (Throwable ignored) {}

        // Print final stats
        System.out.printf(
                "[LLMEngine] Closed: totalSteps=%d, totalTime=%.2fs, " +
                "prefillTime=%.2fs, decodeTime=%.2fs, schedulingTime=%.2fs%n",
                totalSteps, totalStepTimeMs / 1000.0,
                totalPrefillTimeMs / 1000.0, totalDecodeTimeMs / 1000.0,
                totalSchedulingTimeMs / 1000.0);
    }

    /**
     * Check if engine is closed.
     */
    public boolean isClosed() { return closed; }

    /**
     * Get engine statistics.
     */
    public EngineStats getStats() {
        return new EngineStats(
                totalSteps, totalStepTimeMs, totalPrefillTimeMs,
                totalDecodeTimeMs, totalSchedulingTimeMs,
                metrics.numRequests.longValue(),
                metrics.numFinished.longValue(),
                metrics.numTokens.longValue(),
                metrics.numPrefillTokens.longValue(),
                metrics.numDecodeTokens.longValue()
        );
    }

    /**
     * Engine performance statistics.
     */
    public static final class EngineStats {
        public final int totalSteps;
        public final long totalStepTimeMs;
        public final long totalPrefillTimeMs;
        public final long totalDecodeTimeMs;
        public final long totalSchedulingTimeMs;
        public final long totalRequests;
        public final long totalFinished;
        public final long totalTokens;
        public final long totalPrefillTokens;
        public final long totalDecodeTokens;

        public EngineStats(int totalSteps, long totalStepTimeMs, long totalPrefillTimeMs,
                         long totalDecodeTimeMs, long totalSchedulingTimeMs,
                         long totalRequests, long totalFinished,
                         long totalTokens, long totalPrefillTokens, long totalDecodeTokens) {
            this.totalSteps = totalSteps;
            this.totalStepTimeMs = totalStepTimeMs;
            this.totalPrefillTimeMs = totalPrefillTimeMs;
            this.totalDecodeTimeMs = totalDecodeTimeMs;
            this.totalSchedulingTimeMs = totalSchedulingTimeMs;
            this.totalRequests = totalRequests;
            this.totalFinished = totalFinished;
            this.totalTokens = totalTokens;
            this.totalPrefillTokens = totalPrefillTokens;
            this.totalDecodeTokens = totalDecodeTokens;
        }

        public double avgStepTimeMs() {
            return totalSteps > 0 ? (double) totalStepTimeMs / totalSteps : 0;
        }

        public double avgThroughputTokensPerSec() {
            return totalStepTimeMs > 0 ? totalTokens * 1000.0 / totalStepTimeMs : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "EngineStats{steps=%d, requests=%d, finished=%d, " +
                    "tokens=%d (prefill=%d, decode=%d), " +
                    "avgStepTime=%.2fms, throughput=%.1f tok/s",
                    totalSteps, totalRequests, totalFinished,
                    totalTokens, totalPrefillTokens, totalDecodeTokens,
                    avgStepTimeMs(), avgThroughputTokensPerSec()
            );
        }
    }
}
