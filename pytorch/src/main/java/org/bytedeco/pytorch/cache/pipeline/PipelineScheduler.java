/*
 * PipelineScheduler -- runs a pipeline DAG with retry / backoff.
 *
 * <p>Two scheduling modes:
 * <ul>
 *   <li>Sequential -- one stage at a time, retries on RETRY result with
 *       exponential backoff.</li>
 *   <li>Submits batches via {@link #runAsync} on a shared executor.</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.pipeline;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public final class PipelineScheduler implements AutoCloseable {

    private final ExecutorService pool;
    private final int maxRetries;
    private final long baseBackoffMs;

    public PipelineScheduler() {
        this(2, 3, 100);
    }

    public PipelineScheduler(int parallelism, int maxRetries, long baseBackoffMs) {
        this.maxRetries = Math.max(0, maxRetries);
        this.baseBackoffMs = Math.max(1, baseBackoffMs);
        AtomicInteger seq = new AtomicInteger();
        this.pool = Executors.newFixedThreadPool(Math.max(1, parallelism), new ThreadFactory() {
            @Override public Thread newThread(Runnable r) {
                Thread t = new Thread(r, "cache-pipeline-" + seq.incrementAndGet());
                t.setDaemon(true);
                return t;
            }
        });
    }

    public PipelineReport run(List<PipelineStage> stages) {
        return run(stages, new PipelineContext("default"));
    }

    public PipelineReport run(List<PipelineStage> stages, PipelineContext ctx) {
        long t0 = System.currentTimeMillis();
        boolean ok = true;
        boolean skipped = false;
        PipelineReport report = new PipelineReport(ctx.name(), t0, t0, true, false);

        for (PipelineStage stage : stages) {
            long s0 = System.nanoTime();
            PipelineStage.StageResult result = PipelineStage.StageResult.CONTINUE;
            String error = null;
            try {
                int attempts = 0;
                while (true) {
                    try {
                        result = stage.apply(ctx);
                        break;
                    } catch (Exception e) {
                        attempts++;
                        if (attempts > maxRetries) throw e;
                        try {
                            Thread.sleep(baseBackoffMs * (1L << (attempts - 1)));
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw e;
                        }
                    }
                }
                if (result == PipelineStage.StageResult.ABORT_FATAL) {
                    ok = false;
                    error = "abort-fatal";
                } else if (result == PipelineStage.StageResult.SKIP) {
                    skipped = true;
                }
            } catch (Exception e) {
                ok = false;
                error = e.getClass().getSimpleName() + ":" + e.getMessage();
                ctx.recordError(stage.name(), error);
            }
            long dur = System.nanoTime() - s0;
            report.recordStage(stage.name(), dur, result, error);
            if (!ok) break;
        }

        long t1 = System.currentTimeMillis();
        PipelineReport finalReport = new PipelineReport(ctx.name(), t0, t1, ok, skipped);
        for (PipelineReport.StageStat s : report.stages()) {
            finalReport.recordStage(s.name, s.durationNanos, s.result, s.error);
        }
        finalReport.setCounters(ctx.counters());
        finalReport.setErrors(ctx.errors());
        return finalReport;
    }

    public Future<PipelineReport> runAsync(List<PipelineStage> stages, PipelineContext ctx) {
        return pool.submit(() -> run(stages, ctx));
    }

    @Override
    public void close() {
        pool.shutdown();
        try { pool.awaitTermination(5, TimeUnit.SECONDS); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
