/*
 * MetricsPipeline -- export {@link org.bytedeco.pytorch.cache.metrics.CacheMetrics}
 * snapshots to an external sink on a configurable cadence.
 *
 * <p>Pipeline shape:
 * <pre>
 *   1. snapshot        -> ctx.set("snapshot", CacheMetrics.Snapshot)
 *   2. transformProm   -> ctx.set("prom", Map&lt;String,Object&gt;)
 *   3. pushToGateway   -> ctx.increment("pushed", 1)
 *   4. healthCheck     -> ctx.increment("health", 1)
 * </pre>
 */
package org.bytedeco.pytorch.cache.pipeline;

import org.bytedeco.pytorch.cache.metrics.CacheMetrics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;

public final class MetricsPipeline {

    private final CacheMetrics metrics;
    private final Consumer<Map<String, Object>> prometheusSink;

    public MetricsPipeline(CacheMetrics metrics, Consumer<Map<String, Object>> prometheusSink) {
        this.metrics = metrics;
        this.prometheusSink = prometheusSink;
    }

    public java.util.List<PipelineStage> stages() {
        java.util.List<PipelineStage> out = new java.util.ArrayList<>();
        out.add(new TakeSnapshot(metrics));
        out.add(new TransformProm());
        out.add(new PushToGateway(prometheusSink));
        out.add(new HealthCheck());
        return out;
    }

    public PipelineReport run(PipelineContext ctx, PipelineScheduler scheduler) {
        return scheduler.run(stages(), ctx);
    }

    public static final class TakeSnapshot implements PipelineStage {
        private final CacheMetrics metrics;
        public TakeSnapshot(CacheMetrics metrics) { this.metrics = metrics; }
        @Override public String name() { return "snapshot"; }
        @Override public StageResult apply(PipelineContext ctx) {
            ctx.set("snapshot", metrics.snapshot());
            return StageResult.CONTINUE;
        }
    }

    public static final class TransformProm implements PipelineStage {
        @Override public String name() { return "transform-prom"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object o = ctx.get("snapshot");
            if (!(o instanceof CacheMetrics.Snapshot)) return StageResult.SKIP;
            CacheMetrics.Snapshot s = (CacheMetrics.Snapshot) o;
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("cache_l1_hits_total", s.l1Hits);
            out.put("cache_l2_hits_total", s.l2Hits);
            out.put("cache_misses_total", s.misses);
            out.put("cache_negative_hits_total", s.negativeHits);
            out.put("cache_penetration_total", s.penetrations);
            out.put("cache_stampede_total", s.stampedes);
            out.put("cache_avalanche_total", s.avalanches);
            out.put("cache_invalidated_total", s.invalidated);
            out.put("cache_errors_total", s.errors);
            out.put("cache_hit_ratio", s.hitRatio);
            out.put("cache_p50_ms", s.p50Ms);
            out.put("cache_p95_ms", s.p95Ms);
            out.put("cache_p99_ms", s.p99Ms);
            ctx.set("prom", out);
            return StageResult.CONTINUE;
        }
    }

    public static final class PushToGateway implements PipelineStage {
        private final Consumer<Map<String, Object>> sink;
        public PushToGateway(Consumer<Map<String, Object>> sink) { this.sink = sink; }
        @Override public String name() { return "push-gateway"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object o = ctx.get("prom");
            if (!(o instanceof Map)) return StageResult.SKIP;
            if (sink != null) sink.accept((Map<String, Object>) o);
            ctx.increment("pushed", 1);
            return StageResult.CONTINUE;
        }
    }

    public static final class HealthCheck implements PipelineStage {
        @Override public String name() { return "health-check"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object o = ctx.get("snapshot");
            if (!(o instanceof CacheMetrics.Snapshot)) {
                return StageResult.ABORT_FATAL;
            }
            CacheMetrics.Snapshot s = (CacheMetrics.Snapshot) o;
            if (s.errors > 0 && s.hitRatio < 0.5) {
                ctx.recordError(name(), "low hit ratio with errors");
                return StageResult.ABORT_FATAL;
            }
            ctx.increment("health", 1);
            return StageResult.CONTINUE;
        }
    }
}
