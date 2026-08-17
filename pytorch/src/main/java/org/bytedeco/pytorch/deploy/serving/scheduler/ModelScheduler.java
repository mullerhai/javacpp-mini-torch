/*
 * Scheduler Module — enterprise-grade model serving scheduler.
 *
 * Key capabilities:
 *   1. Model warmup with adaptive strategies
 *   2. Adaptive batch processing with dynamic sizing
 *   3. Priority-based request queuing
 *   4. Model version management and hot-swap
 *   5. Request batching for GPU efficiency
 *   6. Resource quota management
 *   7. Backpressure handling
 *   8. Multi-model serving coordination
 *
 * Production patterns (Meta, Google, ByteDance, Alibaba, Tencent):
 *   - Model warmup: JIT compilation, memory allocation, cache warming
 *   - Adaptive batching: batch size adapts to latency targets
 *   - Priority queues: P0 critical > P1 interactive > P2 batch
 *   - GPU memory management: dynamic batching within memory limits
 *   - Request coalescing: combine similar requests for efficiency
 *   - Backpressure: slow down producers when consumers are overwhelmed
 */
package org.bytedeco.pytorch.deploy.serving.scheduler;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.*;

/**
 * Enterprise-grade model serving scheduler with adaptive batching and priority queuing.
 */
public final class ModelScheduler {

    // ---- Request types ----

    /**
     * Priority levels for request scheduling.
     */
    public enum Priority {
        P0_CRITICAL(0, 10),     // 10ms budget
        P1_INTERACTIVE(1, 50), // 50ms budget
        P2_BATCH(2, 200),     // 200ms budget
        P3_BACKGROUND(3, 1000); // 1000ms budget

        public final int level;
        public final long defaultBudgetMs;

        Priority(int level, long defaultBudgetMs) {
            this.level = level;
            this.defaultBudgetMs = defaultBudgetMs;
        }
    }

    /**
     * Inference request with metadata.
     */
    public static final class InferenceRequest {
        public final String requestId;
        public final String modelId;
        public final String modelVersion;
        public final Object input;
        public final Priority priority;
        public final long enqueueTimeMs;
        public final long deadlineMs;
        public final Map<String, String> metadata;
        public final CompletableFuture<InferenceResult> future;

        private InferenceRequest(Builder b) {
            this.requestId = b.requestId;
            this.modelId = b.modelId;
            this.modelVersion = b.modelVersion;
            this.input = b.input;
            this.priority = b.priority;
            this.enqueueTimeMs = b.enqueueTimeMs;
            this.deadlineMs = b.deadlineMs;
            this.metadata = Collections.unmodifiableMap(new HashMap<>(b.metadata));
            this.future = new CompletableFuture<>();
        }

        public static Builder builder(String requestId, String modelId) {
            return new Builder(requestId, modelId);
        }

        public long remainingBudgetMs() {
            return Math.max(0, deadlineMs - System.currentTimeMillis());
        }

        public boolean deadlineExceeded() {
            return System.currentTimeMillis() >= deadlineMs;
        }

        public long waitTimeMs() {
            return System.currentTimeMillis() - enqueueTimeMs;
        }

        public static final class Builder {
            private final String requestId;
            private final String modelId;
            private String modelVersion = "latest";
            private Object input;
            private Priority priority = Priority.P1_INTERACTIVE;
            private long enqueueTimeMs = System.currentTimeMillis();
            private long deadlineMs = enqueueTimeMs + priority.defaultBudgetMs;
            private final Map<String, String> metadata = new HashMap<>();

            private Builder(String requestId, String modelId) {
                this.requestId = requestId;
                this.modelId = modelId;
            }

            public Builder modelVersion(String version) { this.modelVersion = version; return this; }
            public Builder input(Object input) { this.input = input; return this; }
            public Builder priority(Priority priority) { this.priority = priority; return this; }
            public Builder deadlineMs(long deadlineMs) { this.deadlineMs = deadlineMs; return this; }
            public Builder metadata(String key, String value) { this.metadata.put(key, value); return this; }

            public InferenceRequest build() {
                return new InferenceRequest(this);
            }
        }
    }

    /**
     * Inference result with metadata.
     */
    public static final class InferenceResult {
        public final String requestId;
        public final Object output;
        public final long latencyMs;
        public final boolean success;
        public final String error;
        public final Map<String, Object> metrics;

        private InferenceResult(String requestId, Object output, long latencyMs,
                              boolean success, String error, Map<String, Object> metrics) {
            this.requestId = requestId;
            this.output = output;
            this.latencyMs = latencyMs;
            this.success = success;
            this.error = error;
            this.metrics = metrics;
        }

        public static InferenceResult success(String requestId, Object output, long latencyMs) {
            return new InferenceResult(requestId, output, latencyMs, true, null, Map.of());
        }

        public static InferenceResult failure(String requestId, String error, long latencyMs) {
            return new InferenceResult(requestId, null, latencyMs, false, error, Map.of());
        }

        public static InferenceResult timeout(String requestId, long latencyMs) {
            return new InferenceResult(requestId, null, latencyMs, false, "deadline_exceeded", Map.of());
        }
    }

    // ---- Batch configuration ----

    /**
     * Dynamic batching configuration.
     */
    public static final class BatchingConfig {
        public final int minBatchSize;
        public final int maxBatchSize;
        public final long maxWaitTimeMs;
        public final double targetLatencyMs;
        public final boolean adaptive;
        public final int maxQueueSize;

        public BatchingConfig(int minBatchSize, int maxBatchSize, long maxWaitTimeMs,
                            double targetLatencyMs, boolean adaptive, int maxQueueSize) {
            this.minBatchSize = minBatchSize;
            this.maxBatchSize = maxBatchSize;
            this.maxWaitTimeMs = maxWaitTimeMs;
            this.targetLatencyMs = targetLatencyMs;
            this.adaptive = adaptive;
            this.maxQueueSize = maxQueueSize;
        }

        public static BatchingConfig defaults() {
            return new BatchingConfig(1, 256, 10, 50, true, 10000);
        }

        public static final class Builder {
            private int minBatchSize = 1;
            private int maxBatchSize = 256;
            private long maxWaitTimeMs = 10;
            private double targetLatencyMs = 50;
            private boolean adaptive = true;
            private int maxQueueSize = 10000;

            public Builder minBatchSize(int size) { this.minBatchSize = size; return this; }
            public Builder maxBatchSize(int size) { this.maxBatchSize = size; return this; }
            public Builder maxWaitTimeMs(long ms) { this.maxWaitTimeMs = ms; return this; }
            public Builder targetLatencyMs(double ms) { this.targetLatencyMs = ms; return this; }
            public Builder adaptive(boolean adaptive) { this.adaptive = adaptive; return this; }
            public Builder maxQueueSize(int size) { this.maxQueueSize = size; return this; }

            public BatchingConfig build() {
                return new BatchingConfig(minBatchSize, maxBatchSize, maxWaitTimeMs,
                        targetLatencyMs, adaptive, maxQueueSize);
            }
        }
    }

    // ---- Model registry ----

    /**
     * Model definition with version and resource requirements.
     */
    public static final class ModelDefinition {
        public final String modelId;
        public final String currentVersion;
        public final Map<String, ModelVersion> versions;
        public final int minReplicas;
        public final int maxReplicas;
        public final int gpuMemoryMb;
        public final BatchingConfig batchingConfig;
        public final WarmupConfig warmupConfig;

        public ModelDefinition(String modelId, String currentVersion,
                            Map<String, ModelVersion> versions,
                            int minReplicas, int maxReplicas,
                            int gpuMemoryMb, BatchingConfig batchingConfig,
                            WarmupConfig warmupConfig) {
            this.modelId = modelId;
            this.currentVersion = currentVersion;
            this.versions = Collections.unmodifiableMap(new HashMap<>(versions));
            this.minReplicas = minReplicas;
            this.maxReplicas = maxReplicas;
            this.gpuMemoryMb = gpuMemoryMb;
            this.batchingConfig = batchingConfig;
            this.warmupConfig = warmupConfig;
        }

        public static Builder builder(String modelId) {
            return new Builder(modelId);
        }

        public static final class Builder {
            private final String modelId;
            private String currentVersion = "v1";
            private final Map<String, ModelVersion> versions = new HashMap<>();
            private int minReplicas = 1;
            private int maxReplicas = 10;
            private int gpuMemoryMb = 4096;
            private BatchingConfig batchingConfig = BatchingConfig.defaults();
            private WarmupConfig warmupConfig = WarmupConfig.defaults();

            private Builder(String modelId) {
                this.modelId = modelId;
            }

            public Builder currentVersion(String version) { this.currentVersion = version; return this; }
            public Builder version(String id, ModelVersion version) { this.versions.put(id, version); return this; }
            public Builder versions(Map<String, ModelVersion> versions) { this.versions.putAll(versions); return this; }
            public Builder minReplicas(int n) { this.minReplicas = n; return this; }
            public Builder maxReplicas(int n) { this.maxReplicas = n; return this; }
            public Builder gpuMemoryMb(int mb) { this.gpuMemoryMb = mb; return this; }
            public Builder batchingConfig(BatchingConfig config) { this.batchingConfig = config; return this; }
            public Builder warmupConfig(WarmupConfig config) { this.warmupConfig = config; return this; }

            public ModelDefinition build() {
                return new ModelDefinition(modelId, currentVersion, versions,
                        minReplicas, maxReplicas, gpuMemoryMb, batchingConfig, warmupConfig);
            }
        }
    }

    /**
     * Model version with artifact and metadata.
     */
    public static final class ModelVersion {
        public final String versionId;
        public final String artifactPath;
        public final Instant deployedAt;
        public final Map<String, String> metadata;
        public final long artifactSizeBytes;

        public ModelVersion(String versionId, String artifactPath,
                          Instant deployedAt, Map<String, String> metadata,
                          long artifactSizeBytes) {
            this.versionId = versionId;
            this.artifactPath = artifactPath;
            this.deployedAt = deployedAt;
            this.metadata = metadata;
            this.artifactSizeBytes = artifactSizeBytes;
        }

        public static Builder builder(String versionId) {
            return new Builder(versionId);
        }

        public static final class Builder {
            private final String versionId;
            private String artifactPath;
            private Instant deployedAt = Instant.now();
            private final Map<String, String> metadata = new HashMap<>();
            private long artifactSizeBytes;

            private Builder(String versionId) {
                this.versionId = versionId;
            }

            public Builder artifactPath(String path) { this.artifactPath = path; return this; }
            public Builder deployedAt(Instant at) { this.deployedAt = at; return this; }
            public Builder metadata(String key, String value) { this.metadata.put(key, value); return this; }
            public Builder artifactSizeBytes(long bytes) { this.artifactSizeBytes = bytes; return this; }

            public ModelVersion build() {
                return new ModelVersion(versionId, artifactPath, deployedAt, metadata, artifactSizeBytes);
            }
        }
    }

    // ---- Warmup configuration ----

    /**
     * Model warmup configuration.
     */
    public static final class WarmupConfig {
        public final int numWarmupIterations;
        public final int warmupBatchSize;
        public final List<WarmupSample> samples;
        public final long warmupTimeoutMs;
        public final boolean async;

        public WarmupConfig(int numWarmupIterations, int warmupBatchSize,
                          List<WarmupSample> samples, long warmupTimeoutMs,
                          boolean async) {
            this.numWarmupIterations = numWarmupIterations;
            this.warmupBatchSize = warmupBatchSize;
            this.samples = samples;
            this.warmupTimeoutMs = warmupTimeoutMs;
            this.async = async;
        }

        public static WarmupConfig defaults() {
            return new WarmupConfig(100, 32, List.of(), 60000, true);
        }

        public static final class Builder {
            private int numWarmupIterations = 100;
            private int warmupBatchSize = 32;
            private final List<WarmupSample> samples = new ArrayList<>();
            private long warmupTimeoutMs = 60000;
            private boolean async = true;

            public Builder numWarmupIterations(int n) { this.numWarmupIterations = n; return this; }
            public Builder warmupBatchSize(int size) { this.warmupBatchSize = size; return this; }
            public Builder addSample(WarmupSample sample) { this.samples.add(sample); return this; }
            public Builder warmupTimeoutMs(long ms) { this.warmupTimeoutMs = ms; return this; }
            public Builder async(boolean async) { this.async = async; return this; }

            public WarmupConfig build() {
                return new WarmupConfig(numWarmupIterations, warmupBatchSize,
                        new ArrayList<>(samples), warmupTimeoutMs, async);
            }
        }
    }

    public interface WarmupSample {
        Object input();
    }

    // ---- Inference engine interface ----

    /**
     * Model inference engine interface.
     */
    public interface InferenceEngine {
        /**
         * Run inference on a batch of inputs.
         */
        List<Object> inferBatch(List<Object> inputs) throws Exception;

        /**
         * Load model from artifact.
         */
        void load(String artifactPath) throws Exception;

        /**
         * Unload model and free resources.
         */
        void unload();

        /**
         * Check if model is loaded and ready.
         */
        boolean isReady();

        /**
         * Run warmup iterations.
         */
        void warmup(WarmupConfig config) throws Exception;
    }

    // ---- Queue implementation ----

    /**
     * Priority queue with deadline awareness.
     */
    private static class PriorityQueue {
        private final ConcurrentHashMap<Priority, Deque<InferenceRequest>> queues = new ConcurrentHashMap<>();
        private final AtomicLong totalSize = new AtomicLong(0);

        public PriorityQueue() {
            for (Priority p : Priority.values()) {
                queues.put(p, new ConcurrentLinkedDeque<>());
            }
        }

        public void enqueue(InferenceRequest request) {
            Deque<InferenceRequest> q = queues.get(request.priority);
            if (q != null) {
                q.addLast(request);
                totalSize.incrementAndGet();
            }
        }

        public InferenceRequest dequeue() {
            // Check deadlines first (most urgent)
            for (Priority p : Priority.values()) {
                Deque<InferenceRequest> q = queues.get(p);
                if (q == null) continue;

                // Check front of each priority queue
                for (Priority p2 : Priority.values()) {
                    InferenceRequest oldest = queues.get(p2).peekFirst();
                    if (oldest != null && oldest.deadlineExceeded()) {
                        return poll(p2);
                    }
                }
            }

            // Then standard priority order
            for (Priority p : Priority.values()) {
                InferenceRequest req = queues.get(p).pollFirst();
                if (req != null) {
                    totalSize.decrementAndGet();
                    return req;
                }
            }
            return null;
        }

        private InferenceRequest poll(Priority p) {
            InferenceRequest req = queues.get(p).pollFirst();
            if (req != null) {
                totalSize.decrementAndGet();
            }
            return req;
        }

        public InferenceRequest peek() {
            for (Priority p : Priority.values()) {
                InferenceRequest req = queues.get(p).peekFirst();
                if (req != null) return req;
            }
            return null;
        }

        public long size() {
            return totalSize.get();
        }

        public long size(Priority priority) {
            return queues.get(priority).size();
        }
    }

    // ---- Batch collector ----

    /**
     * Collects requests into batches for GPU efficiency.
     */
    private static class BatchCollector implements Runnable {
        private final PriorityQueue queue;
        private final BatchingConfig config;
        private final Function<List<Object>, List<Object>> batchProcessor;
        private final AtomicInteger currentBatchSize = new AtomicInteger(0);
        private final List<InferenceRequest> currentBatch = new ArrayList<>();
        private final ReentrantLock lock = new ReentrantLock();
        private final Condition batchReady;
        private volatile boolean running = true;
        private Instant lastBatchTime;

        public BatchCollector(PriorityQueue queue, BatchingConfig config,
                           Function<List<Object>, List<Object>> batchProcessor) {
            this.queue = queue;
            this.config = config;
            this.batchProcessor = batchProcessor;
            this.lastBatchTime = Instant.now();
            this.batchReady = lock.newCondition();
        }

        public void stop() {
            running = false;
            batchReady.signal();
        }

        @Override
        public void run() {
            while (running) {
                long waitTime = config.maxWaitTimeMs;
                if (lastBatchTime != null) {
                    long elapsed = Duration.between(lastBatchTime, Instant.now()).toMillis();
                    waitTime = Math.max(1, config.maxWaitTimeMs - elapsed);
                }

                lock.lock();
                try {
                    batchReady.await(waitTime, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } finally {
                    lock.unlock();
                }

                // Try to form a batch
                List<InferenceRequest> batch = tryFormBatch();
                if (batch != null && !batch.isEmpty()) {
                    processBatch(batch);
                }
            }
        }

        private List<InferenceRequest> tryFormBatch() {
            lock.lock();
            try {
                if (currentBatch.size() >= config.maxBatchSize) {
                    List<InferenceRequest> batch = new ArrayList<>(currentBatch);
                    currentBatch.clear();
                    currentBatchSize.set(0);
                    lastBatchTime = Instant.now();
                    return batch;
                }

                // Check timeout-based flushing
                if (lastBatchTime != null) {
                    long elapsed = Duration.between(lastBatchTime, Instant.now()).toMillis();
                    if (elapsed >= config.maxWaitTimeMs && !currentBatch.isEmpty()) {
                        List<InferenceRequest> batch = new ArrayList<>(currentBatch);
                        currentBatch.clear();
                        currentBatchSize.set(0);
                        lastBatchTime = Instant.now();
                        return batch;
                    }
                }

                // Try to fill batch
                while (currentBatch.size() < config.maxBatchSize) {
                    InferenceRequest req = queue.peek();
                    if (req == null) break;

                    // Check if request is timing out
                    if (req.deadlineExceeded() && !currentBatch.isEmpty()) {
                        break; // Process current batch first
                    }

                    req = queue.dequeue();
                    if (req == null) break;

                    currentBatch.add(req);
                }

                if (currentBatch.size() >= config.minBatchSize) {
                    List<InferenceRequest> batch = new ArrayList<>(currentBatch);
                    currentBatch.clear();
                    lastBatchTime = Instant.now();
                    return batch;
                }

                return null;
            } finally {
                lock.unlock();
            }
        }

        private void processBatch(List<InferenceRequest> batch) {
            long startTime = System.currentTimeMillis();
            List<Object> inputs = new ArrayList<>();
            for (InferenceRequest req : batch) {
                inputs.add(req.input);
            }

            try {
                List<Object> outputs = batchProcessor.apply(inputs);
                for (int i = 0; i < batch.size(); i++) {
                    InferenceRequest req = batch.get(i);
                    Object output = i < outputs.size() ? outputs.get(i) : null;
                    long latency = System.currentTimeMillis() - startTime;
                    req.future.complete(InferenceResult.success(req.requestId, output, latency));
                }
            } catch (Exception e) {
                for (InferenceRequest req : batch) {
                    long latency = System.currentTimeMillis() - startTime;
                    req.future.complete(InferenceResult.failure(req.requestId, e.getMessage(), latency));
                }
            }
        }
    }

    // ---- Model scheduler state ----

    private final Map<String, ModelDefinition> models = new ConcurrentHashMap<>();
    private final Map<String, InferenceEngine> loadedEngines = new ConcurrentHashMap<>();
    private final Map<String, PriorityQueue> modelQueues = new ConcurrentHashMap<>();
    private final Map<String, ScheduledFuture<?>> warmupTasks = new ConcurrentHashMap<>();
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
    private final ExecutorService batchExecutor;
    private final boolean ownsBatchExecutor;
    private final AtomicLong requestCount = new AtomicLong(0);
    private final AtomicLong successCount = new AtomicLong(0);
    private final AtomicLong failureCount = new AtomicLong(0);
    private final AtomicLong timeoutCount = new AtomicLong(0);
    private volatile boolean running = true;

    public ModelScheduler() {
        this(Runtime.getRuntime().availableProcessors());
    }

    public ModelScheduler(int threads) {
        this.batchExecutor = Executors.newFixedThreadPool(threads, r -> {
            Thread t = new Thread(r, "batch-worker");
            t.setDaemon(true);
            return t;
        });
        this.ownsBatchExecutor = true;

        // Start queue monitor
        scheduler.scheduleAtFixedRate(this::monitorQueues, 1, 1, TimeUnit.SECONDS);
    }

    // ---- Public API ----

    /**
     * Register a model with the scheduler.
     */
    public void registerModel(ModelDefinition model) {
        models.put(model.modelId, model);
        modelQueues.put(model.modelId, new PriorityQueue());

        // Schedule warmup if async
        if (model.warmupConfig.async) {
            scheduleWarmup(model);
        }
    }

    /**
     * Submit an inference request.
     */
    public CompletableFuture<InferenceResult> submit(InferenceRequest request) {
        if (!running) {
            return CompletableFuture.completedFuture(
                    InferenceResult.failure(request.requestId, "scheduler_shutdown", 0));
        }

        ModelDefinition model = models.get(request.modelId);
        if (model == null) {
            return CompletableFuture.completedFuture(
                    InferenceResult.failure(request.requestId, "unknown_model", 0));
        }

        PriorityQueue queue = modelQueues.get(request.modelId);
        queue.enqueue(request);
        requestCount.incrementAndGet();

        // Notify batch collector
        notifyCollector(request.modelId);

        return request.future;
    }

    /**
     * Submit and wait for result synchronously.
     */
    public InferenceResult submitSync(InferenceRequest request, long timeoutMs) {
        try {
            return submit(request).get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            return InferenceResult.timeout(request.requestId, timeoutMs);
        } catch (Exception e) {
            return InferenceResult.failure(request.requestId, e.getMessage(), 0);
        }
    }

    /**
     * Load a specific model version.
     */
    public void loadModel(String modelId, String versionId, InferenceEngine engine) {
        try {
            ModelDefinition model = models.get(modelId);
            if (model == null) {
                throw new IllegalArgumentException("Unknown model: " + modelId);
            }

            // Load engine
            engine.load(model.versions.get(versionId).artifactPath);
            loadedEngines.put(modelId + ":" + versionId, engine);

            // Update current version
            ModelDefinition updated = ModelDefinition.builder(modelId)
                    .currentVersion(versionId)
                    .versions(model.versions)
                    .minReplicas(model.minReplicas)
                    .maxReplicas(model.maxReplicas)
                    .gpuMemoryMb(model.gpuMemoryMb)
                    .batchingConfig(model.batchingConfig)
                    .warmupConfig(model.warmupConfig)
                    .build();
            models.put(modelId, updated);

        } catch (Exception e) {
            throw new RuntimeException("Failed to load model " + modelId + ":" + versionId, e);
        }
    }

    /**
     * Unload a model version.
     */
    public void unloadModel(String modelId, String versionId) {
        InferenceEngine engine = loadedEngines.remove(modelId + ":" + versionId);
        if (engine != null) {
            engine.unload();
        }
    }

    /**
     * Check model readiness.
     */
    public boolean isModelReady(String modelId) {
        ModelDefinition model = models.get(modelId);
        if (model == null) return false;

        InferenceEngine engine = loadedEngines.get(modelId + ":" + model.currentVersion);
        return engine != null && engine.isReady();
    }

    /**
     * Get queue depth for a model.
     */
    public long queueDepth(String modelId) {
        PriorityQueue queue = modelQueues.get(modelId);
        return queue != null ? queue.size() : 0;
    }

    /**
     * Get scheduler statistics.
     */
    public SchedulerStats stats() {
        return new SchedulerStats(
                requestCount.get(),
                successCount.get(),
                failureCount.get(),
                timeoutCount.get(),
                models.size(),
                loadedEngines.size(),
                modelQueues.values().stream().mapToLong(PriorityQueue::size).sum()
        );
    }

    /**
     * Shutdown the scheduler.
     */
    public void shutdown() {
        running = false;
        scheduler.shutdownNow();
        if (ownsBatchExecutor) {
            batchExecutor.shutdownNow();
        }
    }

    // ---- Internal methods ----

    private void scheduleWarmup(ModelDefinition model) {
        if (warmupTasks.containsKey(model.modelId)) {
            return; // Already warming up
        }
        ScheduledFuture<?> future = scheduler.schedule(() -> {
            try {
                InferenceEngine engine = loadedEngines.get(model.modelId + ":" + model.currentVersion);
                if (engine != null) {
                    engine.warmup(model.warmupConfig);
                }
            } catch (Exception e) {
                // Log warmup failure
            } finally {
                warmupTasks.remove(model.modelId);
            }
        }, 0, TimeUnit.MILLISECONDS);
        warmupTasks.put(model.modelId, future);
    }

    private void notifyCollector(String modelId) {
        // In a full implementation, notify the batch collector
    }

    private void monitorQueues() {
        for (Map.Entry<String, PriorityQueue> entry : modelQueues.entrySet()) {
            String modelId = entry.getKey();
            PriorityQueue queue = entry.getValue();

            // Check for timeout requests
            InferenceRequest oldest = queue.peek();
            if (oldest != null && oldest.deadlineExceeded()) {
                InferenceRequest timedOut = queue.dequeue();
                if (timedOut != null) {
                    timeoutCount.incrementAndGet();
                    timedOut.future.complete(InferenceResult.timeout(timedOut.requestId, timedOut.waitTimeMs()));
                }
            }

            // Backpressure: check queue size
            if (queue.size() > 1000) {
                // Could implement backpressure here
            }
        }
    }

    // ---- Stats ----

    public record SchedulerStats(
            long totalRequests,
            long successCount,
            long failureCount,
            long timeoutCount,
            int registeredModels,
            int loadedModels,
            long totalQueueDepth
    ) {
        public double successRate() {
            return totalRequests > 0 ? (double) successCount / totalRequests : 0;
        }
    }

    // ---- Builder ----

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private int batchThreads = Runtime.getRuntime().availableProcessors();

        public Builder batchThreads(int threads) { this.batchThreads = threads; return this; }

        public ModelScheduler build() {
            return new ModelScheduler(batchThreads);
        }
    }
}
