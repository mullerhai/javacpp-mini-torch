/*
 * DaftSession — global runtime singleton (analog of daft.context + engine).
 *
 * Holds the execution configuration (parallelism, memory, IO thread pool)
 * and an io factory reference. Created lazily on first access; can be
 * replaced via {@link #setExecutionConfig} at runtime for tests.
 */
package org.bytedeco.pytorch.utils.daft;

import org.bytedeco.pytorch.utils.daft.engine.ExecutionConfig;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Global singleton controlling Daft execution.
 *
 * <p>Mirrors Python Daft's {@code daft.context} + {@code daft.set_runner_*}
 * surface but stored as a Java atomic reference so tests can swap it
 * without polluting the production default.
 */
public final class DaftSession {

    private static final AtomicReference<DaftSession> INSTANCE = new AtomicReference<>();

    private final ExecutionConfig config;
    private final ExecutorService ioPool;
    private final ForkJoinPool workerPool;

    private DaftSession(ExecutionConfig config) {
        this.config = config;
        this.ioPool = Executors.newFixedThreadPool(
                Math.max(2, config.ioThreads),
                r -> {
                    Thread t = new Thread(r, "daft-io");
                    t.setDaemon(true);
                    return t;
                });
        this.workerPool = new ForkJoinPool(
                Math.max(1, config.numWorkers),
                ForkJoinPool.defaultForkJoinWorkerThreadFactory,
                null, true);
    }

    /** Lazy access to the session; creates a default on first call. */
    public static DaftSession get() {
        return INSTANCE.updateAndGet(prev -> prev != null ? prev : new DaftSession(ExecutionConfig.defaults()));
    }

    /** Replace the global session (used in tests / multi-tenant deployments). */
    public static void setExecutionConfig(ExecutionConfig config) {
        DaftSession old = INSTANCE.getAndSet(new DaftSession(config));
        if (old != null) {
            old.ioPool.shutdown();
            old.workerPool.shutdown();
        }
    }

    public ExecutionConfig config() { return config; }

    public ExecutorService ioPool() { return ioPool; }

    public ForkJoinPool workerPool() { return workerPool; }

    /** Shutdown all pools (call on JVM exit / unit tests). */
    public void shutdown() {
        ioPool.shutdown();
        workerPool.shutdown();
        INSTANCE.compareAndSet(this, null);
    }
}
