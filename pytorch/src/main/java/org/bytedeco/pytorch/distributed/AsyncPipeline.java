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

import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.Scalar;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Asynchronous pipeline parallel implementation with 1F1B scheduling.
 *
 * <p>Implements micro-batch pipelining where computation and communication
 * overlap for improved GPU utilization. Uses 1F1B (one-forward-one-backward)
 * schedule for optimal throughput.
 *
 * <p>Example:
 * <pre>{@code
 * AsyncPipeline pipeline = AsyncPipeline.builder()
 *     .processGroup(pg)
 *     .numStages(4)
 *     .numMicroBatches(16)
 *     .forwardFunc((input, stage) -> forwardStage(input, stage))
 *     .backwardFunc((grad, stage) -> backwardStage(grad, stage))
 *     .build();
 * pipeline.train(microBatches);
 * }</pre>
 */
public final class AsyncPipeline implements AutoCloseable {
    private final ProcessGroupWrapper pg;
    private final int numStages;
    private final int numMicroBatches;
    private final int stageId;
    private final int worldSize;
    private final int rank;
    private final ExecutorService executor;
    private final BlockingQueue<Tensor> sendQueue;
    private final BlockingQueue<Tensor> recvQueue;
    private final boolean enableOverlap;
    private final long maxMemoryBytes;

    // Statistics
    private final AtomicLong totalForwardTime = new AtomicLong(0);
    private final AtomicLong totalBackwardTime = new AtomicLong(0);
    private final AtomicLong totalCommTime = new AtomicLong(0);
    private final AtomicInteger forwardCount = new AtomicInteger(0);
    private final AtomicInteger backwardCount = new AtomicInteger(0);

    private volatile boolean closed = false;

    private AsyncPipeline(Builder builder) {
        this.pg = builder.pg;
        this.numStages = builder.numStages;
        this.numMicroBatches = builder.numMicroBatches;
        this.stageId = builder.stageId;
        this.worldSize = builder.worldSize;
        this.rank = builder.rank;
        this.enableOverlap = builder.enableOverlap;
        this.maxMemoryBytes = builder.maxMemoryBytes;

        int queueCapacity = builder.queueCapacity > 0 ? builder.queueCapacity : numMicroBatches * 2;
        this.sendQueue = new LinkedBlockingQueue<>(queueCapacity);
        this.recvQueue = new LinkedBlockingQueue<>(queueCapacity);

        int numThreads = enableOverlap ? Runtime.getRuntime().availableProcessors() : 1;
        this.executor = Executors.newFixedThreadPool(numThreads);

        System.out.printf("[AsyncPipeline] stage=%d/%d threads=%d overlap=%b queue=%d%n",
                stageId, numStages, numThreads, enableOverlap, queueCapacity);
    }

    public static Builder builder() { return new Builder(); }

    // ═══════════════════════════════════════════════════════════════════════════
    // Pipeline Stages
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Forward pass for a micro-batch.
     *
     * @param input input tensor
     * @param microBatchId micro-batch index
     * @return output activations
     */
    public Tensor forward(Tensor input, int microBatchId) {
        long start = System.nanoTime();
        try {
            if (stageId > 0) {
                // Receive from previous stage
                input = receiveFromPrevious();
            }

            // Compute local forward
            Tensor output = computeForward(input, microBatchId);

            if (stageId < numStages - 1) {
                // Send to next stage asynchronously if overlap enabled
                if (enableOverlap) {
                    executor.submit(() -> {
                        try {
                            sendToNext(output);
                        } catch (Exception e) {
                            System.err.println("[AsyncPipeline] Send failed: " + e.getMessage());
                        }
                    });
                } else {
                    sendToNext(output);
                }
            }

            long elapsed = System.nanoTime() - start;
            totalForwardTime.addAndGet(elapsed);
            forwardCount.incrementAndGet();

            return output;
        } catch (Exception e) {
            throw new RuntimeException("Forward failed at stage " + stageId, e);
        }
    }

    /**
     * Backward pass for a micro-batch.
     *
     * @param gradOutput gradient of output
     * @param microBatchId micro-batch index
     * @return gradient of input
     */
    public Tensor backward(Tensor gradOutput, int microBatchId) {
        long start = System.nanoTime();
        try {
            if (stageId < numStages - 1) {
                // Receive grad from next stage
                gradOutput = receiveGradFromNext();
            }

            // Compute local backward
            Tensor gradInput = computeBackward(gradOutput, microBatchId);

            if (stageId > 0) {
                // Send grad to previous stage
                if (enableOverlap) {
                    executor.submit(() -> {
                        try {
                            sendGradToPrevious(gradInput);
                        } catch (Exception e) {
                            System.err.println("[AsyncPipeline] GradSend failed: " + e.getMessage());
                        }
                    });
                } else {
                    sendGradToPrevious(gradInput);
                }
            }

            long elapsed = System.nanoTime() - start;
            totalBackwardTime.addAndGet(elapsed);
            backwardCount.incrementAndGet();

            return gradInput;
        } catch (Exception e) {
            throw new RuntimeException("Backward failed at stage " + stageId, e);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1F1B Schedule
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Run 1F1B schedule: interleave forwards and backwards.
     *
     * <p>Schedule:
     * - Warmup: run k forwards
     * - Steady state: 1 forward, 1 backward
     * - Cooldown: run remaining backwards
     */
    public void train1F1B(List<Tensor> microBatches) {
        int warmup = Math.min(numStages - stageId, numMicroBatches);
        int cooldown = 0;

        System.out.printf("[AsyncPipeline] 1F1B: warmup=%d total=%d%n", warmup, numMicroBatches);

        // Warmup phase: only forwards
        List<Tensor> forwardOutputs = new ArrayList<>(warmup);
        for (int i = 0; i < warmup; i++) {
            forwardOutputs.add(forward(microBatches.get(i), i));
        }

        // Steady state: alternate forward and backward
        int forwardIdx = warmup;
        int backwardIdx = 0;
        while (forwardIdx < numMicroBatches || backwardIdx < warmup) {
            // Launch forward
            if (forwardIdx < numMicroBatches) {
                final int mbId = forwardIdx;
                executor.submit(() -> {
                    try {
                        forward(microBatches.get(mbId), mbId);
                    } catch (Exception e) {
                        System.err.println("[AsyncPipeline] Forward " + mbId + " failed: " + e.getMessage());
                    }
                });
                forwardIdx++;
            }

            // Launch backward
            if (backwardIdx < warmup) {
                final int bId = backwardIdx++;
                executor.submit(() -> {
                    try {
                        backward(forwardOutputs.get(bId).contiguous(), bId);
                    } catch (Exception e) {
                        System.err.println("[AsyncPipeline] Backward " + bId + " failed: " + e.getMessage());
                    }
                });
            }
        }

        // Cooldown: remaining backwards
        while (backwardIdx < numMicroBatches) {
            final int bId = backwardIdx++;
            backward(forwardOutputs.get(Math.min(bId, forwardOutputs.size() - 1)).contiguous(), bId);
        }

        // Wait for all pending operations
        executor.shutdown();
        try {
            executor.awaitTermination(1, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Communication Primitives
    // ═══════════════════════════════════════════════════════════════════════════

    private Tensor receiveFromPrevious() {
        long start = System.nanoTime();
        try {
            Tensor received = recvQueue.take();
            totalCommTime.addAndGet(System.nanoTime() - start);
            return received;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Receive interrupted", e);
        }
    }

    private void sendToNext(Tensor tensor) {
        long start = System.nanoTime();
        int nextRank = (rank + 1) % worldSize;

        // Use point-to-point send
        pg.send(tensor, nextRank);
        totalCommTime.addAndGet(System.nanoTime() - start);
    }

    private Tensor receiveGradFromNext() {
        long start = System.nanoTime();
        try {
            int prevRank = (rank - 1 + worldSize) % worldSize;
            // For gradients, we use a pre-allocated buffer
            long elementSize = 8; // Assume float64 max
            long numElements = maxMemoryBytes / elementSize;
            Tensor gradBuffer = torch.empty(numElements);

            pg.recv(gradBuffer, prevRank);
            totalCommTime.addAndGet(System.nanoTime() - start);
            return gradBuffer;
        } catch (Exception e) {
            throw new RuntimeException("Grad receive failed", e);
        }
    }

    private void sendGradToPrevious(Tensor grad) {
        long start = System.nanoTime();
        int prevRank = (rank - 1 + worldSize) % worldSize;
        pg.send(grad, prevRank);
        totalCommTime.addAndGet(System.nanoTime() - start);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Compute Methods (override these or use builder functions)
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Compute local forward pass. Override or set via builder.
     */
    public Tensor computeForward(Tensor input, int microBatchId) {
        // Default: identity (override in subclass or use builder)
        return input;
    }

    /**
     * Compute local backward pass. Override or set via builder.
     */
    public Tensor computeBackward(Tensor gradOutput, int microBatchId) {
        // Default: identity (override in subclass or use builder)
        return gradOutput;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics & Profiling
    // ═══════════════════════════════════════════════════════════════════════════

    public PipelineStats getStats() {
        return new PipelineStats(
            totalForwardTime.get(),
            totalBackwardTime.get(),
            totalCommTime.get(),
            forwardCount.get(),
            backwardCount.get()
        );
    }

    public void printStats() {
        PipelineStats stats = getStats();
        System.out.printf("""
                ═══ Pipeline Stats (Stage %d) ═══
                  Forward:  %,.2f ms (%d runs)
                  Backward: %,.2f ms (%d runs)
                  Comm:     %,.2f ms
                  Efficiency: %.1f%%
                ═══════════════════════════════════
                """,
                stageId,
                stats.forwardTimeMs(),
                stats.forwardCount(),
                stats.backwardTimeMs(),
                stats.backwardCount(),
                stats.commTimeMs(),
                stats.efficiency()
        );
    }

    @Override
    public void close() {
        closed = true;
        executor.shutdownNow();
        try {
            executor.awaitTermination(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════════

    public static final class Builder {
        private ProcessGroupWrapper pg;
        private int numStages = 1;
        private int numMicroBatches = 1;
        private int stageId = 0;
        private int worldSize = 1;
        private int rank = 0;
        private boolean enableOverlap = true;
        private int queueCapacity = 0;
        private long maxMemoryBytes = 1L << 30; // 1GB default
        private java.util.function.Function<Tensor, Tensor> forwardFunc;
        private java.util.function.Function<Tensor, Tensor> backwardFunc;

        public Builder processGroup(ProcessGroupWrapper pg) {
            this.pg = pg;
            this.worldSize = pg.getWorldSize();
            this.rank = pg.getRank();
            return this;
        }

        public Builder numStages(int n) { this.numStages = n; return this; }
        public Builder numMicroBatches(int n) { this.numMicroBatches = n; return this; }
        public Builder stageId(int id) { this.stageId = id; return this; }
        public Builder worldSize(int ws) { this.worldSize = ws; return this; }
        public Builder rank(int r) { this.rank = r; return this; }
        public Builder enableOverlap(boolean e) { this.enableOverlap = e; return this; }
        public Builder queueCapacity(int q) { this.queueCapacity = q; return this; }
        public Builder maxMemoryBytes(long m) { this.maxMemoryBytes = m; return this; }
        public Builder forwardFunc(java.util.function.Function<Tensor, Tensor> f) { this.forwardFunc = f; return this; }
        public Builder backwardFunc(java.util.function.Function<Tensor, Tensor> f) { this.backwardFunc = f; return this; }

        public AsyncPipeline build() {
            return new AsyncPipeline(this);
        }
    }

    /**
     * Pipeline statistics.
     */
    public record PipelineStats(
        long forwardTimeNs,
        long backwardTimeNs,
        long commTimeNs,
        int forwardCount,
        int backwardCount
    ) {
        public double forwardTimeMs() { return forwardTimeNs / 1e6; }
        public double backwardTimeMs() { return backwardTimeNs / 1e6; }
        public double commTimeMs() { return commTimeNs / 1e6; }
        public double totalTimeMs() { return forwardTimeMs() + backwardTimeMs() + commTimeMs(); }
        public double efficiency() {
            double compute = forwardTimeMs() + backwardTimeMs();
            return compute > 0 ? (compute / (compute + commTimeMs())) * 100 : 0;
        }
    }
}
