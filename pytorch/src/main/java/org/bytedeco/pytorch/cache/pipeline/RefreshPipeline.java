/*
 * RefreshPipeline -- orchestrates stale-while-revalidate refresh at scale.
 *
 * <p>Pipeline shape:
 * <pre>
 *   1. enumerateHotKeys     -> ctx.set("hotKeys", List&lt;CacheKey&gt;)
 *   2. classifyByTier       -> ctx.set("tiers", Map&lt;Tier,List&lt;CacheKey&gt;&gt;)
 *   3. loadFromSourcePerTier -> ctx.increment("loaded", N)
 *   4. writeBack            -> ctx.increment("written", N)
 *   5. invalidateStale      -> ctx.increment("invalidated", N)
 *   6. emitMetrics          -> ctx.increment("emitted", 1)
 * </pre>
 *
 * <p>Production refresh pipelines typically run on a cron schedule; this
 * class is the per-run orchestrator.
 */
package org.bytedeco.pytorch.cache.pipeline;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;
import org.bytedeco.pytorch.cache.LoadFunction;
import org.bytedeco.pytorch.cache.TieredCache;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class RefreshPipeline {

    private final TieredCache cache;
    private final LoadFunction loader;
    private final CacheBackend source;
    private final long maxBatchSize;

    public RefreshPipeline(TieredCache cache, LoadFunction loader, CacheBackend source, long maxBatchSize) {
        this.cache = cache;
        this.loader = loader;
        this.source = source;
        this.maxBatchSize = Math.max(1, maxBatchSize);
    }

    public List<PipelineStage> stages() {
        List<PipelineStage> out = new ArrayList<>();
        out.add(new EnumerateHotKeys(cache));
        out.add(new ClassifyByTier());
        out.add(new LoadFromSource(loader, source));
        out.add(new WriteBack(cache));
        out.add(new InvalidateStale(cache));
        out.add(new EmitMetrics());
        return out;
    }

    public PipelineReport run(PipelineContext ctx, PipelineScheduler scheduler) {
        return scheduler.run(stages(), ctx);
    }

    // ----- stages -----

    public static final class EnumerateHotKeys implements PipelineStage {
        private final TieredCache cache;
        public EnumerateHotKeys(TieredCache cache) { this.cache = cache; }
        @Override public String name() { return "enumerate-hot-keys"; }
        @Override public StageResult apply(PipelineContext ctx) {
            // Iteration over the cached tier is usually impossible (no index);
            // production variants read the hot-keys list from a separate
            // analytics stream. Here we accept a list attached to the context.
            Object provided = ctx.get("hotKeys");
            if (provided == null) {
                ctx.recordError(name(), "no hotKeys provided");
                return StageResult.ABORT_FATAL;
            }
            ctx.increment("hotKeysCount", ((Collection<?>) provided).size());
            return StageResult.CONTINUE;
        }
    }

    public static final class ClassifyByTier implements PipelineStage {
        @Override public String name() { return "classify-by-tier"; }
        @Override public StageResult apply(PipelineContext ctx) throws Exception {
            Object provided = ctx.get("hotKeys");
            if (!(provided instanceof Collection)) {
                return StageResult.SKIP;
            }
            Map<String, List<CacheKey>> byView = new LinkedHashMap<>();
            for (Object o : (Collection<?>) provided) {
                if (!(o instanceof CacheKey)) continue;
                CacheKey k = (CacheKey) o;
                byView.computeIfAbsent(k.view(), v -> new ArrayList<>()).add(k);
            }
            ctx.set("byView", byView);
            ctx.increment("views", byView.size());
            return StageResult.CONTINUE;
        }
    }

    public static final class LoadFromSource implements PipelineStage {
        private final LoadFunction loader;
        private final CacheBackend source;
        public LoadFromSource(LoadFunction loader, CacheBackend source) {
            this.loader = loader;
            this.source = source;
        }
        @Override public String name() { return "load-from-source"; }
        @Override public StageResult apply(PipelineContext ctx) throws Exception {
            @SuppressWarnings("unchecked")
            Map<String, List<CacheKey>> byView = (Map<String, List<CacheKey>>) ctx.get("byView");
            if (byView == null) return StageResult.SKIP;
            long loaded = 0, failed = 0;
            for (Map.Entry<String, List<CacheKey>> e : byView.entrySet()) {
                for (CacheKey k : e.getValue()) {
                    try {
                        CacheValue<Object> v = loader == null ? null : loader.load(k);
                        if (v == null && source != null) {
                            v = source.get(k).orElse(null);
                        }
                        if (v != null) {
                            ctx.set("v:" + k.toStorageKey(), v);
                            loaded++;
                        }
                    } catch (Exception ex) {
                        failed++;
                    }
                }
            }
            ctx.increment("loaded", loaded);
            ctx.increment("loadFailed", failed);
            return StageResult.CONTINUE;
        }
    }

    public static final class WriteBack implements PipelineStage {
        private final TieredCache cache;
        public WriteBack(TieredCache cache) { this.cache = cache; }
        @Override public String name() { return "write-back"; }
        @Override public StageResult apply(PipelineContext ctx) {
            long written = 0;
            for (Map.Entry<String, Object> e : ctx.attributes().entrySet()) {
                if (!e.getKey().startsWith("v:")) continue;
                if (!(e.getValue() instanceof CacheValue)) continue;
                CacheKey k = CacheKey.fromStorageKey(e.getKey().substring(2));
                cache.put(k, (CacheValue<Object>) e.getValue());
                written++;
            }
            ctx.increment("written", written);
            return StageResult.CONTINUE;
        }
    }

    public static final class InvalidateStale implements PipelineStage {
        private final TieredCache cache;
        public InvalidateStale(TieredCache cache) { this.cache = cache; }
        @Override public String name() { return "invalidate-stale"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object provided = ctx.get("hotKeys");
            if (!(provided instanceof Collection)) return StageResult.SKIP;
            long n = 0;
            for (Object o : (Collection<?>) provided) {
                if (o instanceof CacheKey) {
                    cache.invalidate((CacheKey) o);
                    n++;
                }
            }
            ctx.increment("invalidated", n);
            return StageResult.CONTINUE;
        }
    }

    public static final class EmitMetrics implements PipelineStage {
        @Override public String name() { return "emit-metrics"; }
        @Override public StageResult apply(PipelineContext ctx) {
            // In production this would push to Prometheus / OTel.
            ctx.increment("emitted", 1);
            return StageResult.CONTINUE;
        }
    }
}
