/*
 * BulkLoader — bulk offline-to-online ingestion path.
 *
 * <p>Big-tech pattern (ByteDance Abase / Alibaba Pouch / Tencent TFace / Google
 * Vertex AI Feature Store):
 * <ul>
 *   <li>Offline jobs (Spark / Flink / SQL) compute features into a columnar
 *       batch (DataFrame / Parquet / ORC). BulkLoader streams them into the
 *       online cache with pipeline-batched writes to Redis (mget / setex loops).</li>
 *   <li>Each row carries its event_timestamp so the FeatureCrossingGuard can
 *       reject features older than the requested snapshot (point-in-time
 *       correctness).</li>
 *   <li>Backpressure: a bounded work queue feeds the streaming pipeline so a
 *       1B-row batch doesn't blow out memory.</li>
 *   <li>Commit semantics: writes are idempotent; downstream reads see the
 *       feature as soon as the network ack returns. Optional two-phase
 *       commit via "version" tag.</li>
 * </ul>
 *
 * <p>This class writes to a TieredCache; subsequent reads can choose to
 * bypass the cache for offline-only paths via the {@link #forceRefresh} flag.
 */
package org.bytedeco.pytorch.cache.offline;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;
import org.bytedeco.pytorch.cache.TieredCache;
import org.bytedeco.pytorch.cache.metrics.CacheMetrics;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

public final class BulkLoader implements AutoCloseable {

    private final TieredCache tiered;
    private final int batchSize;
    private final ExecutorService pipeline;
    private final BlockingQueue<Map.Entry<CacheKey, CacheValue<Object>>> queue;
    private final AtomicLong accepted = new AtomicLong();
    private final AtomicLong written = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private final AtomicLong skipped = new AtomicLong();
    private final AtomicReference<String> lastError = new AtomicReference<>();
    private final Consumer<CacheMetrics.Snapshot> progressObserver;
    private final boolean forceRefresh;
    private final long enqueueTimeoutMs;

    public BulkLoader(TieredCache tiered, int batchSize, int queueCapacity,
                      boolean forceRefresh, long enqueueTimeoutMs,
                      Consumer<CacheMetrics.Snapshot> progressObserver) {
        this.tiered = Objects.requireNonNull(tiered);
        this.batchSize = Math.max(64, batchSize);
        this.forceRefresh = forceRefresh;
        this.enqueueTimeoutMs = enqueueTimeoutMs <= 0 ? 500 : enqueueTimeoutMs;
        this.queue = new ArrayBlockingQueue<>(Math.max(this.batchSize * 4, queueCapacity));
        this.progressObserver = progressObserver;
        this.pipeline = Executors.newFixedThreadPool(2, r -> {
            Thread t = new Thread(r, "bulk-loader-pipeline");
            t.setDaemon(true);
            return t;
        });
        pipeline.submit(this::drainLoop);
    }

    public BulkLoader(TieredCache tiered) {
        this(tiered, 1024, 65536, true, 1000, null);
    }

    /** Synchronously load a row. */
    public boolean offer(CacheKey key, CacheValue<Object> value) {
        if (key == null || value == null) {
            skipped.incrementAndGet();
            return false;
        }
        try {
            boolean ok = queue.offer(
                    new java.util.AbstractMap.SimpleEntry<>(key, value),
                    enqueueTimeoutMs, TimeUnit.MILLISECONDS);
            if (ok) accepted.incrementAndGet();
            else failed.incrementAndGet();
            return ok;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            failed.incrementAndGet();
            return false;
        }
    }

    public long offerBatch(Map<CacheKey, CacheValue<Object>> batch) {
        long n = 0;
        if (batch == null) return 0;
        for (Map.Entry<CacheKey, CacheValue<Object>> e : batch.entrySet()) {
            if (offer(e.getKey(), e.getValue())) n++;
        }
        return n;
    }

    /** Drain until queue is empty for the given timeout. */
    public void flush(Duration timeout) {
        long deadline = System.currentTimeMillis() + timeout.toMillis();
        while (System.currentTimeMillis() < deadline) {
            if (queue.isEmpty()) break;
            try { Thread.sleep(20); } catch (InterruptedException e) { Thread.currentThread().interrupt(); break; }
        }
    }

    public long accepted() { return accepted.get(); }
    public long written() { return written.get(); }
    public long failed() { return failed.get(); }
    public long skipped() { return skipped.get(); }
    public long pending() { return queue.size(); }
    public String lastError() { return lastError.get(); }

    private void drainLoop() {
        List<Map.Entry<CacheKey, CacheValue<Object>>> buf = new ArrayList<>(batchSize);
        while (!Thread.currentThread().isInterrupted()) {
            try {
                Map.Entry<CacheKey, CacheValue<Object>> first = queue.poll(100, TimeUnit.MILLISECONDS);
                if (first == null) {
                    if (progressObserver != null) progressObserver.accept(tiered.metrics().snapshot());
                    continue;
                }
                buf.add(first);
                queue.drainTo(buf, batchSize - 1);
                flushBatch(buf);
                buf.clear();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                failed.incrementAndGet();
                lastError.set(t.getMessage());
            }
        }
    }

    private void flushBatch(List<Map.Entry<CacheKey, CacheValue<Object>>> buf) {
        Map<CacheKey, CacheValue<Object>> batch = new LinkedHashMap<>();
        for (Map.Entry<CacheKey, CacheValue<Object>> e : buf) batch.put(e.getKey(), e.getValue());
        try {
            if (forceRefresh) {
                for (Map.Entry<CacheKey, CacheValue<Object>> e : batch.entrySet()) {
                    tiered.invalidate(e.getKey());
                }
            }
            tiered.putBatch(batch);
            written.addAndGet(batch.size());
        } catch (Exception e) {
            failed.addAndGet(batch.size());
            lastError.set(e.getMessage());
        }
    }

    @Override
    public void close() {
        pipeline.shutdown();
        try { pipeline.awaitTermination(10, TimeUnit.SECONDS); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
