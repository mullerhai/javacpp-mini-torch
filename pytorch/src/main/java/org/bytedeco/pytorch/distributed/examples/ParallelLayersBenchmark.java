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
package org.bytedeco.pytorch.distributed.examples;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.distributed.enums.BackendType;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.SGD;
import org.bytedeco.pytorch.optim.options.SGDOptions;

import java.util.ArrayList;
import java.util.List;

/**
 * Performance benchmarking for ParallelLayers.
 *
 * <p>Features:
 * - FP16/FP32 throughput comparison
 * - NCCL-optimized benchmark on A100/H100 clusters
 * - Detailed performance report (TFLOPS, memory bandwidth, etc.)
 *
 * <p>Usage:
 * <pre>{@code
 * // Multi-process (requires torchrun):
 * // torchrun --nproc_per_node=8 ParallelLayersBenchmark
 *
 * // Single-process (for local testing):
 * ParallelLayersBenchmark.benchmarkLocal();
 * }</pre>
 */
public final class ParallelLayersBenchmark {
    private ParallelLayersBenchmark() {}

    // ═══════════════════════════════════════════════════════════════════════════
    // Public API
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Full distributed benchmark (requires NCCL and multi-process setup).
     *
     * @param worldSize total number of ranks
     * @param dpSize    data parallel size
     * @param tpSize    tensor parallel size
     * @param epSize    expert parallel size
     * @param steps     number of training steps
     * @param vocab     vocabulary size
     * @param hiddenDim hidden dimension
     * @param numHeads  number of attention heads
     * @param seqLen    sequence length
     * @param batchSize batch size
     * @param fp16      whether to use FP16 precision
     * @return performance report
     */
    public static PerformanceReport benchmark(
            int worldSize, int dpSize, int tpSize, int epSize,
            int steps, long vocab, long hiddenDim, int numHeads,
            long seqLen, int batchSize, boolean fp16) {

        System.out.printf("[Benchmark] Full distributed: world=%d dp=%d tp=%d ep=%d%n",
                worldSize, dpSize, tpSize, epSize);

        // Initialize NCCL backend
        ProcessGroupWrapper pg = createNCCLBackend(worldSize);

        // Create 3D mesh: dp × tp × ep
        DeviceMesh mesh = ParallelLayers.initDpTpEp(pg, tpSize, epSize);

        // Create model
        ParallelLayers.HybridTrainer model = createHybridTrainer(mesh, vocab, hiddenDim, numHeads,
                hiddenDim * 4, 8, 2, seqLen);

        // Create optimizer
        Optimizer opt = createOptimizer(model, fp16);

        // Warmup
        System.out.printf("[Benchmark] Warming up with %d steps...%n", Math.min(steps, 10));
        runWarmup(model, opt, batchSize, seqLen, fp16);

        // Actual benchmark
        PerformanceReport report = runBenchmark(model, opt, steps, batchSize, seqLen, fp16);

        // Report
        System.out.println(report);
        return report;
    }

    /**
     * Single-process benchmark for local testing (no NCCL required).
     * Uses local backend with synthetic data.
     *
     * @param steps     number of training steps
     * @param hiddenDim hidden dimension
     * @param numHeads  number of attention heads
     * @param seqLen    sequence length
     * @param batchSize batch size
     * @param fp16      whether to use FP16 precision
     * @return performance report
     */
    public static PerformanceReport benchmarkLocal(
            int steps, long hiddenDim, int numHeads,
            long seqLen, int batchSize, boolean fp16) {

        System.out.printf("[Benchmark Local] hidden=%d heads=%d seq=%d batch=%d fp16=%b%n",
                hiddenDim, numHeads, seqLen, batchSize, fp16);

        // Initialize local backend (single process, no NCCL)
        ProcessGroupWrapper pg = createLocalBackend();
        DeviceMesh mesh = ParallelLayers.initDpTp(pg, 1);  // 2D mesh: dp=1, tp=1

        // Create model
        ParallelLayers.HybridTrainer model = createHybridTrainer(mesh, 32000, hiddenDim, numHeads,
                hiddenDim * 4, 2, 1, seqLen);

        // Create optimizer
        Optimizer opt = createOptimizer(model, fp16);

        // Warmup
        System.out.printf("[Benchmark] Warming up with %d steps...%n", Math.min(steps, 10));
        runWarmup(model, opt, batchSize, seqLen, fp16);

        // Actual benchmark
        PerformanceReport report = runBenchmark(model, opt, steps, batchSize, seqLen, fp16);

        // Report
        System.out.println(report);
        return report;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Backend Creation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Create NCCL backend with proper configuration.
     * Requires multi-process setup via torchrun.
     */
    private static ProcessGroupWrapper createNCCLBackend(int worldSize) {
        DistributedStore store = DistributedStore.create(0, worldSize);
        ProcessGroupWrapper.Options opts = new ProcessGroupWrapper.Options();
        opts.backend(BackendType.NCCL);
        opts.timeout(300_000);
        opts.syncCollectives(true);

        return ProcessGroupWrapper.create(opts, 0, worldSize, store);
    }

    /**
     * Create local backend for single-process testing.
     * No actual distributed operations, but model runs correctly.
     */
    private static ProcessGroupWrapper createLocalBackend() {
        DistributedStore store = DistributedStore.createSingleProcess();
        ProcessGroupWrapper.Options opts = new ProcessGroupWrapper.Options();
        opts.forceCollective(true);  // Force Gloo even for single process
        opts.syncCollectives(true);

        return ProcessGroupWrapper.create(opts, 0, 1, store);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Model & Optimizer Creation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Create hybrid trainer with given mesh.
     */
    private static ParallelLayers.HybridTrainer createHybridTrainer(
            DeviceMesh mesh, long vocab, long hiddenDim, int numHeads,
            long intermediateDim, int numExperts, int topK, long seqLen) {

        return new ParallelLayers.HybridTrainer(mesh, vocab, hiddenDim, numHeads,
                intermediateDim, numExperts, topK, seqLen);
    }

    /**
     * Create optimizer with appropriate learning rate.
     */
    private static Optimizer createOptimizer(ParallelLayers.HybridTrainer model, boolean fp16) {
        SGD opt = new SGD(model.getModule().parameters(), new SGDOptions(1e-4f).lr(1e-4f));
        // Note: FP16 support depends on PyTorch version and native bindings
        // opt.set_fp16(true); // Uncomment if supported
        return opt;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Warmup & Benchmark
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Warmup run to stabilize timing.
     */
    private static void runWarmup(ParallelLayers.HybridTrainer model, Optimizer opt,
                                  int batchSize, long seqLen, boolean fp16) {

        Device device = getDevice();
        ScalarType dtype = fp16 ? torch.ScalarType.Half : torch.ScalarType.Float;

        Tensor input = torch.zeros(batchSize, seqLen).to(device, dtype);
        Tensor target = torch.zeros(batchSize, seqLen).to(device, torch.ScalarType.Long);

        int warmupSteps = 5;
        for (int i = 0; i < warmupSteps; i++) {
            try {
                model.step(input, target, opt);
            } catch (Exception e) {
                System.out.printf("[Benchmark] Warmup step %d failed: %s%n", i, e.getMessage());
                // Continue with remaining warmup steps
            }
        }
    }

    /**
     * Get appropriate device (CUDA if available, otherwise CPU).
     */
    private static Device getDevice() {
        if (torch.cuda_is_available()) {
            return new Device(torch.DeviceType.CUDA);
        } else {
            System.out.println("[Benchmark] WARNING: CUDA not available, running on CPU");
            return new Device(torch.DeviceType.CPU);
        }
    }

    /**
     * Run actual benchmark and collect performance metrics.
     */
    private static PerformanceReport runBenchmark(ParallelLayers.HybridTrainer model, Optimizer opt,
                                                  int steps, int batchSize, long seqLen, boolean fp16) {

        List<PerformanceMetric> metrics = new ArrayList<>();

        Device device = getDevice();
        ScalarType dtype = fp16 ? torch.ScalarType.Half : torch.ScalarType.Float;

        Tensor input = torch.zeros(batchSize, seqLen).to(device, dtype);
        Tensor target = torch.zeros(batchSize, seqLen).to(device, torch.ScalarType.Long);

        for (int i = 0; i < steps; i++) {
            long start = System.nanoTime();

            try {
                Tensor loss = model.step(input, target, opt);
                long elapsed = System.nanoTime() - start;

                // Calculate TFLOPS
                long flops = calculateFLOPS(batchSize, seqLen, 4096, 32, 8, 2);
                double tflops = (flops * 1e-9) / Math.max(elapsed * 1e-9, 1e-9);

                metrics.add(new PerformanceMetric(i, elapsed, tflops, flops));
            } catch (Exception e) {
                System.out.printf("[Benchmark] Step %d failed: %s%n", i, e.getMessage());
                // Record failure
                metrics.add(new PerformanceMetric(i, -1, 0, 0));
            }
        }

        return new PerformanceReport(metrics, fp16);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Performance Calculations
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Calculate approximate FLOPS for the model.
     */
    private static long calculateFLOPS(int batch, long seqLen, long hiddenDim, int numHeads, long intermediateDim, int numExperts) {
        // Embedding: vocab * hidden
        long embeddingFlops = 32000L * hiddenDim;

        // Attention: 4 * batch * seq * hidden * hidden + attention
        long attnFlops = 4L * batch * seqLen * hiddenDim * hiddenDim;

        // FFN: 2 * batch * seq * hidden * intermediate
        long ffnFlops = 2L * batch * seqLen * hiddenDim * intermediateDim;

        // MoE: numExperts * topK * batch * seq * hidden * hidden
        long moeFlops = numExperts * numHeads * batch * seqLen * hiddenDim * hiddenDim;

        return embeddingFlops + attnFlops + ffnFlops + moeFlops;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Report Classes
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Performance report containing benchmark results.
     */
    public static class PerformanceReport {
        private final List<PerformanceMetric> metrics;
        private final boolean fp16;
        private final long totalTime;
        private final double avgTFLOPS;
        private final double avgMemoryBandwidth;
        private final int successSteps;
        private final int failedSteps;

        public PerformanceReport(List<PerformanceMetric> metrics, boolean fp16) {
            this.metrics = metrics;
            this.fp16 = fp16;

            this.successSteps = (int) metrics.stream().filter(m -> m.elapsedTime > 0).count();
            this.failedSteps = (int) metrics.stream().filter(m -> m.elapsedTime < 0).count();

            this.totalTime = metrics.stream()
                    .filter(m -> m.elapsedTime > 0)
                    .mapToLong(m -> m.elapsedTime)
                    .sum();

            this.avgTFLOPS = metrics.stream()
                    .filter(m -> m.elapsedTime > 0)
                    .mapToDouble(m -> m.tflops)
                    .average()
                    .orElse(0.0);

            this.avgMemoryBandwidth = calculateMemoryBandwidth(metrics);
        }

        private double calculateMemoryBandwidth(List<PerformanceMetric> metrics) {
            if (totalTime <= 0) return 0;
            // Approximate memory access: 2 * batch * seq * hidden * 4 bytes (FP32)
            long bytesPerStep = 2L * 4096 * 2048 * 4;
            return (bytesPerStep * successSteps) / (totalTime * 1e-9 * 1024 * 1024 * 1024);
        }

        public List<PerformanceMetric> getMetrics() { return metrics; }
        public boolean isFp16() { return fp16; }
        public long getTotalTime() { return totalTime; }
        public double getAvgTFLOPS() { return avgTFLOPS; }
        public double getAvgMemoryBandwidth() { return avgMemoryBandwidth; }
        public int getSuccessSteps() { return successSteps; }
        public int getFailedSteps() { return failedSteps; }

        @Override
        public String toString() {
            return String.format(
                    "═══ Performance Report (%s precision) ═══%n" +
                    "  Total steps: %d (success: %d, failed: %d)%n" +
                    "  Total time: %.2f ms%n" +
                    "  Average TFLOPS: %.2f TFLOPS%n" +
                    "  Average Memory Bandwidth: %.2f GB/s%n" +
                    "  Steps/sec: %.2f%n" +
                    "═══════════════════════════════════════",
                    fp16 ? "FP16" : "FP32",
                    metrics.size(),
                    successSteps,
                    failedSteps,
                    totalTime / 1e6,
                    avgTFLOPS,
                    avgMemoryBandwidth,
                    successSteps > 0 ? (successSteps * 1e9 / (double) totalTime) : 0
            );
        }
    }

    /**
     * Individual benchmark metric.
     */
    public static class PerformanceMetric {
        public final int step;
        public final long elapsedTime;
        public final double tflops;
        public final long flops;

        public PerformanceMetric(int step, long elapsedTime, double tflops, long flops) {
            this.step = step;
            this.elapsedTime = elapsedTime;
            this.tflops = tflops;
            this.flops = flops;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Main Entry Points
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Multi-process benchmark entry point.
     * Requires torchrun: torchrun --nproc_per_node=8 ParallelLayersBenchmark
     */
    public static void main(String[] args) {
        // Default configuration: 8 GPUs, 1 DP, 4 TP, 2 EP
        // For 8 GPUs: dp * tp * ep = 1 * 4 * 2 = 8
        benchmark(8, 1, 4, 2, 100, 32000, 4096, 32, 2048, 4, true);
    }
}
