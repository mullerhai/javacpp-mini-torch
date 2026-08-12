/*
 * ExecutionConfig — runtime parameters for Daft execution (workers / memory / IO).
 *
 * Mirrors Python Daft's ExecutionConfig dataclass fields:
 *   num_workers, batch_size, shuffle_partitions, memory_bytes, spill_threshold,
 *   io_thread_pool_size, scan_tasks_min, scan_tasks_max, shuffle_algorithm,
 *   enable_actor_pool, etc.
 */
package org.bytedeco.pytorch.utils.daft.engine;

/**
 * Daft execution configuration (immutable).
 *
 * <p>Build with {@link Builder}. Pass to {@code DaftSession.setExecutionConfig}.
 */
public final class ExecutionConfig {

    public enum ShuffleAlgorithm {
        MAP_REDUCE,
        RADIX_SHUFFLE,
        BROADCAST_HASH_JOIN
    }

    public final int numWorkers;
    public final int ioThreads;
    public final int batchSize;
    public final int shufflePartitions;
    public final long memoryBytes;
    public final double spillThreshold; // 0..1
    public final int scanTasksMin;
    public final int scanTasksMax;
    public final ShuffleAlgorithm shuffleAlgorithm;
    public final boolean enableActorPool;
    public final String rayEndpoint;
    public final boolean useNativeRunner;

    private ExecutionConfig(Builder b) {
        this.numWorkers = b.numWorkers;
        this.ioThreads = b.ioThreads;
        this.batchSize = b.batchSize;
        this.shufflePartitions = b.shufflePartitions;
        this.memoryBytes = b.memoryBytes;
        this.spillThreshold = b.spillThreshold;
        this.scanTasksMin = b.scanTasksMin;
        this.scanTasksMax = b.scanTasksMax;
        this.shuffleAlgorithm = b.shuffleAlgorithm;
        this.enableActorPool = b.enableActorPool;
        this.rayEndpoint = b.rayEndpoint;
        this.useNativeRunner = b.useNativeRunner;
    }

    public static Builder builder() { return new Builder(); }

    public static ExecutionConfig defaults() {
        return new Builder().build();
    }

    public static final class Builder {
        private int numWorkers = Math.max(2, Runtime.getRuntime().availableProcessors());
        private int ioThreads = 16;
        private int batchSize = 4096;
        private int shufflePartitions = Math.max(8, numWorkers * 2);
        private long memoryBytes = (long) Math.min(8L * 1024 * 1024 * 1024,
                Runtime.getRuntime().maxMemory() / 4);
        private double spillThreshold = 0.7;
        private int scanTasksMin = 32;
        private int scanTasksMax = 4096;
        private ShuffleAlgorithm shuffleAlgorithm = ShuffleAlgorithm.MAP_REDUCE;
        private boolean enableActorPool = false;
        private String rayEndpoint = null;
        private boolean useNativeRunner = true;

        public Builder numWorkers(int n) { this.numWorkers = n; return this; }
        public Builder ioThreads(int n) { this.ioThreads = n; return this; }
        public Builder batchSize(int n) { this.batchSize = n; return this; }
        public Builder shufflePartitions(int n) { this.shufflePartitions = n; return this; }
        public Builder memoryBytes(long b) { this.memoryBytes = b; return this; }
        public Builder spillThreshold(double t) { this.spillThreshold = t; return this; }
        public Builder scanTasksMin(int n) { this.scanTasksMin = n; return this; }
        public Builder scanTasksMax(int n) { this.scanTasksMax = n; return this; }
        public Builder shuffleAlgorithm(ShuffleAlgorithm a) { this.shuffleAlgorithm = a; return this; }
        public Builder enableActorPool(boolean b) { this.enableActorPool = b; return this; }
        public Builder rayEndpoint(String e) { this.rayEndpoint = e; return this; }
        public Builder useNativeRunner(boolean b) { this.useNativeRunner = b; return this; }

        public ExecutionConfig build() {
            if (scanTasksMax < scanTasksMin) {
                throw new IllegalArgumentException("scanTasksMax < scanTasksMin");
            }
            if (spillThreshold <= 0.0 || spillThreshold >= 1.0) {
                throw new IllegalArgumentException("spillThreshold must be in (0,1)");
            }
            return new ExecutionConfig(this);
        }
    }
}
