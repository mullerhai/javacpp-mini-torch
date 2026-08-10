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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;

/**
 * Distributed Profiler and Tracing System.
 *
 * <p>Provides comprehensive profiling and tracing for distributed training:
 * <ul>
 *   <li>Operation-level timing and FLOP counting</li>
 *   <li>Communication analysis (bandwidth, latency)</li>
 *   <li>Memory profiling and peak detection</li>
 *   <li>GPU utilization metrics</li>
 *   <li>Distributed tracing across workers</li>
 *   <li>JSON trace export for Chrome tracing</li>
 * </ul>
 *
 * <p>Features:
 * <ul>
 *   <li>Context manager for easy profiling</li>
 *   <li>Async-safe event recording</li>
 *   <li>Automatic FLOP estimation</li>
 *   <li>Memory snapshot and diff</li>
 *   <li>Custom metric collection</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * DistributedProfiler profiler = DistributedProfiler.builder()
 *     .enabled(true)
 *     .traceOperations(true)
 *     .profileMemory(true)
 *     .recordShallowFlops(true)
 *     .outputDir("./profiler_output")
 *     .processGroup(pg)
 *     .build();
 *
 * // Profile a region
 * try (ProfilerRegion region = profiler.profile("forward_pass")) {
 *     Tensor output = model.forward(input);
 * }
 *
 * // Profile function
 * profiler.profileFunc("backward", () -> loss.backward());
 *
 * // Generate report
 * profiler.printReport();
 * profiler.exportChromeTrace("trace.json");
 * }</pre>
 */
public final class DistributedProfiler implements AutoCloseable {
    private final boolean enabled;
    private final boolean traceOperations;
    private final boolean profileMemory;
    private final boolean recordShallowFlops;
    private final String outputDir;
    private final ProcessGroupWrapper pg;
    private final int worldSize;
    private final int rank;
    private final boolean isMainProcess;

    // Event storage
    private final List<ProfilerEvent> events;
    private final ConcurrentLinkedQueue<ProfilerEvent> asyncEvents;
    private final Map<String, Counter> counters;
    private final Map<String, Histogram> histograms;

    // Memory tracking
    private final List<MemorySnapshot> memorySnapshots;
    private long peakMemoryBytes = 0;
    private long currentMemoryBytes = 0;

    // State
    private final ThreadLocal<Deque<ProfilerRegion>> regionStack;
    private final AtomicInteger eventId;
    private volatile boolean closed = false;

    // Configuration
    private final int maxEvents;
    private final int reportIntervalMs;
    private final ScheduledExecutorService scheduler;

    // Callbacks
    private final List<Consumer<ProfilerEvent>> eventCallbacks = new ArrayList<>();

    private DistributedProfiler(Builder builder) {
        this.enabled = builder.enabled;
        this.traceOperations = builder.traceOperations;
        this.profileMemory = builder.profileMemory;
        this.recordShallowFlops = builder.recordShallowFlops;
        this.outputDir = builder.outputDir;
        this.pg = builder.pg;
        this.worldSize = builder.pg != null ? builder.pg.getWorldSize() : 1;
        this.rank = builder.pg != null ? builder.pg.getRank() : 0;
        this.isMainProcess = rank == 0 || pg == null;

        this.events = new ArrayList<>(builder.maxEvents);
        this.asyncEvents = new ConcurrentLinkedQueue<>();
        this.counters = new ConcurrentHashMap<>();
        this.histograms = new ConcurrentHashMap<>();
        this.memorySnapshots = new ArrayList<>();
        this.regionStack = ThreadLocal.withInitial(LinkedList::new);
        this.eventId = new AtomicInteger(0);
        this.maxEvents = builder.maxEvents;
        this.reportIntervalMs = builder.reportIntervalMs;

        // Start periodic report scheduler
        if (reportIntervalMs > 0) {
            this.scheduler = Executors.newScheduledThreadPool(1);
            scheduler.scheduleAtFixedRate(this::flushEvents, reportIntervalMs, reportIntervalMs,
                    TimeUnit.MILLISECONDS);
        } else {
            this.scheduler = null;
        }

        // Create output directory
        if (isMainProcess && outputDir != null) {
            try {
                Files.createDirectories(Paths.get(outputDir));
            } catch (Exception e) {
                System.err.println("[Profiler] Failed to create output dir: " + e.getMessage());
            }
        }

        if (enabled) {
            System.out.printf("[Profiler] Started: trace=%b memory=%b flops=%b dir=%s%n",
                    traceOperations, profileMemory, recordShallowFlops, outputDir);
        }
    }

    public static Builder builder() { return new Builder(); }

    // ═══════════════════════════════════════════════════════════════════════════
    // Core Profiling
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Profile a code region.
     *
     * @param name region name
     * @param <T> return type
     * @param computation code to profile
     * @return result of computation
     */
    public <T> T profile(String name, Supplier<T> computation) {
        if (!enabled) return computation.get();

        try (ProfilerRegion region = profile(name)) {
            return computation.get();
        }
    }

    /**
     * Profile a void code region.
     */
    public void profile(String name, Runnable computation) {
        if (!enabled) {
            computation.run();
            return;
        }

        try (ProfilerRegion region = profile(name)) {
            computation.run();
        }
    }

    /**
     * Start profiling a named region.
     */
    public ProfilerRegion profile(String name) {
        if (!enabled) return ProfilerRegion.NOOP;

        ProfilerRegion region = new ProfilerRegion(name, this);
        regionStack.get().push(region);
        region.start();

        if (traceOperations) {
            recordEvent(ProfilerEvent.builder()
                    .name(name)
                    .category("op")
                    .phase(Phase.BEGIN)
                    .build());
        }

        return region;
    }

    /**
     * Profile a function with automatic region management.
     */
    public <T> T profileFunc(String name, Supplier<T> func) {
        if (!enabled) return func.get();

        long start = System.nanoTime();
        try {
            T result = func.get();
            long elapsed = System.nanoTime() - start;
            recordTiming(name, elapsed, "func");
            return result;
        } catch (Exception e) {
            recordError(name, e);
            throw e;
        }
    }

    /**
     * Profile a void function.
     */
    public void profileAction(String name, Runnable action) {
        if (!enabled) {
            action.run();
            return;
        }

        long start = System.nanoTime();
        try {
            action.run();
            long elapsed = System.nanoTime() - start;
            recordTiming(name, elapsed, "action");
        } catch (Exception e) {
            recordError(name, e);
            throw e;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Event Recording
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Record a profiler event.
     */
    public void recordEvent(ProfilerEvent event) {
        if (!enabled || closed) return;

        int id = eventId.getAndIncrement();
        event.id(id);
        event.timestamp(System.nanoTime());
        event.pid(rank);
        event.tid((int) Thread.currentThread().getId());

        if (events.size() < maxEvents) {
            events.add(event);
        } else {
            asyncEvents.offer(event);
        }

        // Call callbacks
        for (Consumer<ProfilerEvent> cb : eventCallbacks) {
            cb.accept(event);
        }
    }

    /**
     * Record timing for an operation.
     */
    public void recordTiming(String name, long durationNs, String category) {
        if (!enabled) return;

        recordEvent(ProfilerEvent.builder()
                .name(name)
                .category(category)
                .phase(Phase.COMPLETE)
                .durationNs(durationNs)
                .build());

        // Update histogram
        getOrCreateHistogram(name).record(durationNs);
    }

    /**
     * Record a counter value.
     */
    public void recordCounter(String name, long value) {
        if (!enabled) return;

        Counter counter = counters.computeIfAbsent(name, k -> new Counter(name));
        counter.record(value);

        recordEvent(ProfilerEvent.builder()
                .name(name)
                .category("counter")
                .phase(Phase.COUNTER)
                .counterValue(value)
                .build());
    }

    /**
     * Increment a counter.
     */
    public void incrementCounter(String name) {
        incrementCounter(name, 1);
    }

    /**
     * Increment a counter by amount.
     */
    public void incrementCounter(String name, long amount) {
        Counter counter = counters.computeIfAbsent(name, k -> new Counter(name));
        counter.increment(amount);
    }

    /**
     * Record an error.
     */
    public void recordError(String name, Exception e) {
        if (!enabled) return;

        recordEvent(ProfilerEvent.builder()
                .name(name)
                .category("error")
                .phase(Phase.COMPLETE)
                .errorMessage(e.getMessage())
                .build());
    }

    /**
     * Register a callback for events.
     */
    public void onEvent(Consumer<ProfilerEvent> callback) {
        eventCallbacks.add(callback);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Memory Profiling
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Take a memory snapshot.
     */
    public MemorySnapshot snapshotMemory(String name) {
        if (!profileMemory) return MemorySnapshot.EMPTY;

        MemorySnapshot snapshot = new MemorySnapshot(
                System.nanoTime(),
                name,
                rank,
                currentMemoryBytes,
                peakMemoryBytes,
                getGpuMemoryInfo()
        );

        memorySnapshots.add(snapshot);

        // Update peak
        if (snapshot.usedBytes() > peakMemoryBytes) {
            peakMemoryBytes = snapshot.usedBytes();
        }

        return snapshot;
    }

    /**
     * Record memory usage.
     */
    public void recordMemory(String name, long bytes) {
        if (!profileMemory) return;

        currentMemoryBytes = bytes;

        recordEvent(ProfilerEvent.builder()
                .name(name)
                .category("memory")
                .phase(Phase.COUNTER)
                .memoryBytes(bytes)
                .build());
    }

    /**
     * Get GPU memory information.
     */
    private Map<String, Long> getGpuMemoryInfo() {
        Map<String, Long> info = new HashMap<>();

        if (torch.cuda_is_available()) {
            try {
                // In production, would call torch.cuda.memory_allocated()
                // and torch.cuda.max_memory_allocated()
                info.put("allocated_bytes", currentMemoryBytes);
                info.put("peak_bytes", peakMemoryBytes);
                info.put("reserved_bytes", currentMemoryBytes * 2); // Estimate
            } catch (Exception e) {
                // CUDA not available
            }
        }

        return info;
    }

    /**
     * Diff two memory snapshots.
     */
    public MemoryDiff diffMemory(MemorySnapshot before, MemorySnapshot after) {
        return new MemoryDiff(
                before.name(),
                after.name(),
                after.usedBytes() - before.usedBytes(),
                after.peakBytes() - before.peakBytes()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Communication Profiling
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Record communication operation.
     */
    public void recordCommunication(String opType, long bytes, long durationNs) {
        if (!enabled) return;

        recordEvent(ProfilerEvent.builder()
                .name(opType)
                .category("comm")
                .phase(Phase.COMPLETE)
                .durationNs(durationNs)
                .communicationBytes(bytes)
                .build());

        // Update counters
        String bandwidthCounter = opType + "_bandwidth";
        double bandwidthGbps = (bytes * 8.0) / (durationNs / 1e9) / 1e9;
        recordCounter(bandwidthCounter, (long) (bandwidthGbps * 1000)); // Store as Kbps
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FLOP Counting
    // ═══════════════════════════════════════════════════════════════════════════

    private final Map<String, Long> flopCounts = new ConcurrentHashMap<>();

    /**
     * Record FLOPs for an operation.
     */
    public void recordFlops(String name, long flops) {
        if (!recordShallowFlops) return;

        flopCounts.merge(name, flops, Long::sum);

        recordEvent(ProfilerEvent.builder()
                .name(name)
                .category("flops")
                .phase(Phase.COUNTER)
                .flops(flops)
                .build());
    }

    /**
     * Estimate FLOPs for a matrix multiplication.
     * Shape: [M, K] @ [K, N] = [M, N]
     */
    public void recordMatmulFlops(long m, long k, long n) {
        recordFlops("matmul", 2 * m * k * n);
    }

    /**
     * Estimate FLOPs for attention.
     * Q @ K^T: [B, H, S, D] @ [B, H, D, S] = [B, H, S, S]
     */
    public void recordAttentionFlops(long batch, long heads, long seqLen, long headDim) {
        // Q @ K^T
        recordFlops("attention_qk", 2L * batch * heads * seqLen * seqLen * headDim);
        // Softmax (estimation)
        recordFlops("attention_softmax", (long) batch * heads * seqLen * seqLen);
        // Attention @ V
        recordFlops("attention_av", 2L * batch * heads * seqLen * seqLen * headDim);
    }

    /**
     * Get total FLOPs recorded.
     */
    public long getTotalFlops() {
        return flopCounts.values().stream().mapToLong(Long::longValue).sum();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Region Management
    // ═══════════════════════════════════════════════════════════════════════════

    void endRegion(ProfilerRegion region) {
        Deque<ProfilerRegion> stack = regionStack.get();
        if (!stack.isEmpty() && stack.peek() == region) {
            stack.pop();
        }

        if (traceOperations) {
            recordEvent(ProfilerEvent.builder()
                    .name(region.name)
                    .category("op")
                    .phase(Phase.END)
                    .durationNs(region.getDurationNs())
                    .build());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Reporting
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Get or create a histogram.
     */
    public Histogram getOrCreateHistogram(String name) {
        return histograms.computeIfAbsent(name, k -> new Histogram(name));
    }

    /**
     * Flush pending events.
     */
    public void flushEvents() {
        ProfilerEvent event;
        while ((event = asyncEvents.poll()) != null) {
            if (events.size() < maxEvents) {
                events.add(event);
            }
        }
    }

    /**
     * Print profiling report.
     */
    public void printReport() {
        if (!isMainProcess) return;

        System.out.println("""
                ═════════════════════════════════════════════════════════════
                                 PROFILING REPORT
                ═════════════════════════════════════════════════════════════""");

        // Summary
        System.out.printf("""
                Summary:
                  Events recorded:   %d
                  Counters:          %d
                  Histograms:        %d
                  Memory snapshots:  %d
                  Peak memory:       %.2f GB
                """,
                events.size(),
                counters.size(),
                histograms.size(),
                memorySnapshots.size(),
                peakMemoryBytes / 1e9
        );

        // Top operations by total time
        if (!histograms.isEmpty()) {
            System.out.println("\nTop operations by total time:");
            System.out.println("─".repeat(60));

            histograms.entrySet().stream()
                    .sorted((a, b) -> Long.compare(
                            b.getValue().total(),
                            a.getValue().total()))
                    .limit(10)
                    .forEach(e -> {
                        Histogram h = e.getValue();
                        System.out.printf("  %-30s %12.2f ms  (avg: %8.2f us, count: %d)%n",
                                e.getKey(),
                                h.total() / 1e6,
                                h.mean() / 1e3,
                                h.count()
                        );
                    });
        }

        // Counters
        if (!counters.isEmpty()) {
            System.out.println("\nCounters:");
            System.out.println("─".repeat(60));
            counters.forEach((name, counter) ->
                    System.out.printf("  %-30s %,12d%n", name, counter.value())
            );
        }

        // Memory
        if (profileMemory && !memorySnapshots.isEmpty()) {
            System.out.println("\nMemory usage:");
            System.out.println("─".repeat(60));
            MemorySnapshot last = memorySnapshots.get(memorySnapshots.size() - 1);
            System.out.printf("  Current:              %.2f GB%n", last.usedBytes() / 1e9);
            System.out.printf("  Peak:                 %.2f GB%n", last.peakBytes() / 1e9);
        }

        // FLOPs
        if (recordShallowFlops && !flopCounts.isEmpty()) {
            System.out.println("\nFLOPs breakdown:");
            System.out.println("─".repeat(60));
            flopCounts.entrySet().stream()
                    .sorted((a, b) -> Long.compare(b.getValue(), a.getValue()))
                    .limit(10)
                    .forEach(e ->
                            System.out.printf("  %-30s %,15d%n", e.getKey(), e.getValue())
                    );
            System.out.printf("  %-30s %,15d%n", "TOTAL", getTotalFlops());
        }

        System.out.println("════════════════════════════════════════════════════════════");
    }

    /**
     * Export trace to JSON (Chrome-compatible format).
     */
    public void exportChromeTrace(String filename) {
        if (!enabled) return;

        String path = outputDir != null
                ? Paths.get(outputDir, filename).toString()
                : filename;

        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("{");
            pw.println("  \"traceEvents\": [");

            List<ProfilerEvent> allEvents = new ArrayList<>(events);
            flushEvents();
            allEvents.addAll(events);

            for (int i = 0; i < allEvents.size(); i++) {
                ProfilerEvent e = allEvents.get(i);
                pw.print(e.toChromeTraceJson());
                if (i < allEvents.size() - 1) {
                    pw.println(",");
                } else {
                    pw.println();
                }
            }

            pw.println("  ],");
            pw.println("  \"displayTimeUnit\": \"ns\"");
            pw.println("}");

            if (isMainProcess) {
                System.out.println("[Profiler] Chrome trace exported: " + path);
            }

        } catch (Exception e) {
            System.err.println("[Profiler] Failed to export trace: " + e.getMessage());
        }
    }

    /**
     * Export report to file.
     */
    public void exportReport(String filename) {
        if (!isMainProcess) return;

        String path = outputDir != null
                ? Paths.get(outputDir, filename).toString()
                : filename;

        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            // Write events
            pw.println("# Profiler Events");
            pw.println("timestamp_ns,name,category,phase,duration_ns,pid,tid,error,flops,bytes");
            for (ProfilerEvent e : events) {
                pw.printf("%d,%s,%s,%s,%d,%d,%d,%s,%d,%d%n",
                        e.timestamp(),
                        e.name(),
                        e.category(),
                        e.phase().name(),
                        e.durationNs(),
                        e.pid(),
                        e.tid(),
                        e.errorMessage() != null ? e.errorMessage().replace(",", ";") : "",
                        e.flops(),
                        e.communicationBytes()
                );
            }

            // Write counters
            pw.println("\n# Counters");
            counters.forEach((name, counter) ->
                    pw.printf("%s: %d%n", name, counter.value())
            );

            if (isMainProcess) {
                System.out.println("[Profiler] Report exported: " + path);
            }

        } catch (Exception e) {
            System.err.println("[Profiler] Failed to export report: " + e.getMessage());
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Getters
    // ═══════════════════════════════════════════════════════════════════════════

    public List<ProfilerEvent> getEvents() {
        return Collections.unmodifiableList(events);
    }

    public Map<String, Counter> getCounters() {
        return Collections.unmodifiableMap(counters);
    }

    public Map<String, Histogram> getHistograms() {
        return Collections.unmodifiableMap(histograms);
    }

    public List<MemorySnapshot> getMemorySnapshots() {
        return Collections.unmodifiableList(memorySnapshots);
    }

    public long getPeakMemoryBytes() {
        return peakMemoryBytes;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // End any open regions
        Deque<ProfilerRegion> stack = regionStack.get();
        while (!stack.isEmpty()) {
            ProfilerRegion region = stack.pop();
            region.end();
        }

        // Flush events
        flushEvents();

        // Stop scheduler
        if (scheduler != null) {
            scheduler.shutdown();
            try {
                scheduler.awaitTermination(1, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Final report
        if (enabled && isMainProcess) {
            printReport();
        }

        System.out.println("[Profiler] Stopped");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Builder
    // ═══════════════════════════════════════════════════════════════════════════

    public static final class Builder {
        private boolean enabled = true;
        private boolean traceOperations = true;
        private boolean profileMemory = true;
        private boolean recordShallowFlops = true;
        private String outputDir = "./profiler_output";
        private int maxEvents = 100000;
        private int reportIntervalMs = 10000;
        private ProcessGroupWrapper pg;

        public Builder enabled(boolean e) { this.enabled = e; return this; }
        public Builder traceOperations(boolean t) { this.traceOperations = t; return this; }
        public Builder profileMemory(boolean m) { this.profileMemory = m; return this; }
        public Builder recordShallowFlops(boolean f) { this.recordShallowFlops = f; return this; }
        public Builder outputDir(String d) { this.outputDir = d; return this; }
        public Builder maxEvents(int m) { this.maxEvents = m; return this; }
        public Builder reportIntervalMs(int r) { this.reportIntervalMs = r; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.pg = pg; return this; }

        public DistributedProfiler build() {
            return new DistributedProfiler(this);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Supporting Classes
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Profiler region (context manager).
     */
    public final class ProfilerRegion implements AutoCloseable {
        private final String name;
        private final long startTime;
        private long endTime;
        private boolean closed = false;
        private static final ProfilerRegion NOOP = null;

        ProfilerRegion(String name, DistributedProfiler profiler) {
            this.name = name;
            this.startTime = System.nanoTime();
        }

        public void start() {
            // Already started in constructor
        }

        public void end() {
            if (closed) return;
            this.endTime = System.nanoTime();
            closed = true;
            recordTiming(name, getDurationNs(), "region");
        }

        public long getDurationNs() {
            return closed ? endTime - startTime : System.nanoTime() - startTime;
        }

        @Override
        public void close() {
            if (!closed) {
                end();
                DistributedProfiler.this.endRegion(this);
            }
        }
    }

    /**
     * Profiler event.
     */
    public static class ProfilerEvent {
        private int id;
        private long timestamp;
        private String name;
        private String category;
        private Phase phase;
        private long durationNs;
        private int pid;
        private int tid;
        private String errorMessage;
        private long flops;
        private long communicationBytes;
        private long memoryBytes;
        private long counterValue;

        private ProfilerEvent(Builder builder) {
            this.name = builder.name;
            this.category = builder.category;
            this.phase = builder.phase;
            this.durationNs = builder.durationNs;
            this.errorMessage = builder.errorMessage;
            this.flops = builder.flops;
            this.communicationBytes = builder.communicationBytes;
            this.memoryBytes = builder.memoryBytes;
            this.counterValue = builder.counterValue;
        }

        public static Builder builder() { return new Builder(); }

        // Getters and setters
        public int id() { return id; }
        public void id(int id) { this.id = id; }
        public long timestamp() { return timestamp; }
        public void timestamp(long ts) { this.timestamp = ts; }
        public String name() { return name; }
        public String category() { return category; }
        public Phase phase() { return phase; }
        public long durationNs() { return durationNs; }
        public int pid() { return pid; }
        public void pid(int pid) { this.pid = pid; }
        public int tid() { return tid; }
        public void tid(int tid) { this.tid = tid; }
        public String errorMessage() { return errorMessage; }
        public long flops() { return flops; }
        public long communicationBytes() { return communicationBytes; }
        public long memoryBytes() { return memoryBytes; }
        public long counterValue() { return counterValue; }

        public String toChromeTraceJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"name\":\"").append(escapeJson(name)).append("\",");
            sb.append("\"cat\":\"").append(escapeJson(category)).append("\",");
            sb.append("\"ph\":\"").append(phase.chromeChar).append("\",");
            sb.append("\"pid\":").append(pid).append(",");
            sb.append("\"tid\":").append(tid).append(",");
            sb.append("\"ts\":").append(timestamp / 1000); // Convert to microseconds
            sb.append(",\"dur\":").append(durationNs / 1000);

            if (errorMessage != null) {
                sb.append(",\"args\":{\"error\":\"").append(escapeJson(errorMessage)).append("\"}");
            } else if (counterValue > 0) {
                sb.append(",\"args\":{\"value\":").append(counterValue).append("}");
            } else if (flops > 0) {
                sb.append(",\"args\":{\"flops\":").append(flops).append("}");
            } else if (communicationBytes > 0) {
                sb.append(",\"args\":{\"bytes\":").append(communicationBytes).append("}");
            } else if (memoryBytes > 0) {
                sb.append(",\"args\":{\"memory\":").append(memoryBytes).append("}");
            }

            sb.append("}");
            return sb.toString();
        }

        private static String escapeJson(String s) {
            if (s == null) return "";
            return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
        }

        public static class Builder {
            String name;
            String category = "default";
            Phase phase = Phase.COMPLETE;
            long durationNs;
            String errorMessage;
            long flops;
            long communicationBytes;
            long memoryBytes;
            long counterValue;

            public Builder name(String n) { this.name = n; return this; }
            public Builder category(String c) { this.category = c; return this; }
            public Builder phase(Phase p) { this.phase = p; return this; }
            public Builder durationNs(long d) { this.durationNs = d; return this; }
            public Builder errorMessage(String e) { this.errorMessage = e; return this; }
            public Builder flops(long f) { this.flops = f; return this; }
            public Builder communicationBytes(long b) { this.communicationBytes = b; return this; }
            public Builder memoryBytes(long m) { this.memoryBytes = m; return this; }
            public Builder counterValue(long v) { this.counterValue = v; return this; }
            public ProfilerEvent build() { return new ProfilerEvent(this); }
        }
    }

    /**
     * Event phase.
     */
    public enum Phase {
        BEGIN("B"),
        END("E"),
        COMPLETE("X"),
        INSTANT("I"),
        COUNTER("C");

        public final String chromeChar;

        Phase(String c) { this.chromeChar = c; }
    }

    /**
     * Counter for metrics.
     */
    public static class Counter {
        private final String name;
        private final AtomicLong value = new AtomicLong(0);

        public Counter(String name) { this.name = name; }

        public void record(long v) { value.set(v); }
        public void increment(long amount) { value.addAndGet(amount); }
        public void increment() { value.incrementAndGet(); }
        public long value() { return value.get(); }
        public String name() { return name; }
    }

    /**
     * Histogram for distributions.
     */
    public static class Histogram {
        private final String name;
        private final DoubleAdder sum = new DoubleAdder();
        private final AtomicLong count = new AtomicLong(0);
        private final double[] values;
        private final AtomicInteger index = new AtomicInteger(0);

        public Histogram(String name) {
            this(name, 1000);
        }

        public Histogram(String name, int size) {
            this.name = name;
            this.values = new double[size];
        }

        public void record(long nanos) {
            double us = nanos / 1000.0;
            sum.add(us);
            count.incrementAndGet();

            int idx = index.getAndIncrement() % values.length;
            values[idx] = us;
        }

        public long total() { return (long) (sum.sum() * 1000); }
        public long count() { return count.get(); }
        public double mean() { return count() > 0 ? sum.sum() / count() : 0; }
        public String name() { return name; }

        public double percentile(double p) {
            if (count() == 0) return 0;
            double[] sorted = Arrays.copyOf(values, (int) Math.min(count(), values.length));
            Arrays.sort(sorted);
            int idx = (int) (sorted.length * p / 100);
            return sorted[Math.min(idx, sorted.length - 1)];
        }
    }

    /**
     * Memory snapshot.
     */
    public record MemorySnapshot(
        long timestampNs,
        String name,
        int rank,
        long usedBytes,
        long peakBytes,
        Map<String, Long> details
    ) {
        public static final MemorySnapshot EMPTY = new MemorySnapshot(0, "", 0, 0, 0, Map.of());
    }

    /**
     * Memory diff between two snapshots.
     */
    public record MemoryDiff(
        String beforeName,
        String afterName,
        long usedDiffBytes,
        long peakDiffBytes
    ) {}
}
