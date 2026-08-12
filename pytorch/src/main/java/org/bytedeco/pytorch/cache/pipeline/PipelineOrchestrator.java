/*
 * PipelineOrchestrator -- cron-driven pipeline runner.
 *
 * <p>Accepts a set of named pipelines (each with a cron schedule) and runs
 * them on a single daemon thread pool. Pipeline runs are independent; failed
 * runs are recorded in a {@link PipelineReport} and never propagated as
 * exceptions (the scheduler is fire-and-forget).
 *
 * <p>Design:
 * <ul>
 *   <li>One tick thread per pipeline so a long-running pipeline does not
 *       block quick-cadence ones</li>
 *   <li>{@link #triggerNow(String)} provides a synchronous test hook</li>
 *   <li>{@link CronSchedule} is loosely coupled -- callers can swap in any
 *       Schedule-like object via the {@link Runnable} overload</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.pipeline;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public final class PipelineOrchestrator implements AutoCloseable {

    public interface PipelineFactory {
        String name();
        CronSchedule schedule();
        List<PipelineStage> stages();
        PipelineContext freshContext();
    }

    private final Map<String, PipelineFactory> factories = new LinkedHashMap<>();
    private final Map<String, PipelineReport> latestReports = new LinkedHashMap<>();
    private final ScheduledExecutorService scheduler;
    private final Map<String, ScheduledFuture<?>> futures = new LinkedHashMap<>();
    private final ZoneId zone;
    private final PipelineScheduler pipelineScheduler;

    public PipelineOrchestrator(PipelineScheduler pipelineScheduler) {
        this(pipelineScheduler, ZoneId.systemDefault());
    }

    public PipelineOrchestrator(PipelineScheduler pipelineScheduler, ZoneId zone) {
        this.pipelineScheduler = pipelineScheduler;
        this.zone = zone == null ? ZoneId.systemDefault() : zone;
        AtomicInteger seq = new AtomicInteger();
        this.scheduler = Executors.newScheduledThreadPool(2, new ThreadFactory() {
            @Override public Thread newThread(Runnable r) {
                Thread t = new Thread(r, "cache-pipeline-orch-" + seq.incrementAndGet());
                t.setDaemon(true);
                return t;
            }
        });
    }

    public PipelineOrchestrator register(PipelineFactory f) {
        factories.put(f.name(), f);
        return this;
    }

    public void start() {
        for (PipelineFactory f : factories.values()) {
            ScheduledFuture<?> existing = futures.get(f.name());
            if (existing != null) existing.cancel(false);
            long initialDelaySec = computeInitialDelay(f.schedule());
            ScheduledFuture<?> sf = scheduler.scheduleAtFixedRate(
                    () -> runOnce(f.name()),
                    initialDelaySec, 60, TimeUnit.SECONDS);
            futures.put(f.name(), sf);
        }
    }

    public PipelineReport triggerNow(String name) {
        return runOnce(name);
    }

    public PipelineReport latestReport(String name) { return latestReports.get(name); }

    public Map<String, PipelineReport> snapshot() { return new LinkedHashMap<>(latestReports); }

    private PipelineReport runOnce(String name) {
        PipelineFactory f = factories.get(name);
        if (f == null) return null;
        PipelineContext ctx = f.freshContext();
        if (ctx == null) ctx = new PipelineContext(name);
        try {
            PipelineReport r = pipelineScheduler.run(f.stages(), ctx);
            latestReports.put(name, r);
            return r;
        } catch (Exception e) {
            PipelineReport r = new PipelineReport(name, System.currentTimeMillis(),
                    System.currentTimeMillis(), false, false);
            r.recordError("orchestrator", e.getClass().getSimpleName() + ":" + e.getMessage());
            latestReports.put(name, r);
            return r;
        }
    }

    private long computeInitialDelay(CronSchedule cron) {
        LocalDateTime next = cron.nextAfter(LocalDateTime.now(zone));
        if (next == null) return 60;
        long now = System.currentTimeMillis();
        long nextMs = next.atZone(zone).toInstant().toEpochMilli();
        return Math.max(1, (nextMs - now) / 1000);
    }

    @Override
    public void close() {
        for (ScheduledFuture<?> sf : futures.values()) sf.cancel(false);
        scheduler.shutdown();
        try { scheduler.awaitTermination(5, TimeUnit.SECONDS); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
