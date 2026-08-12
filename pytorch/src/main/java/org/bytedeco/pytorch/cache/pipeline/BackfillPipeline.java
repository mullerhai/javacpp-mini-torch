/*
 * BackfillPipeline -- rebuild a cache from a primary source.
 *
 * <p>Used for:
 * <ul>
 *   <li>cold-start when a new region/cluster is brought online</li>
 *   <li>migration from one backend to another</li>
 *   <li>disaster recovery (rebuild after a hot-zone failure)</li>
 * </ul>
 *
 * <p>Pipeline shape:
 * <pre>
 *   1. listEntities   -> ctx.set("entities", List&lt;CacheKey&gt;)
 *   2. fetchBatches   -> ctx.increment("fetched", N)
 *   3. writeToTarget  -> ctx.increment("written", N)
 *   4. verifyCounts   -> ctx.increment("verified", N)
 * </pre>
 */
package org.bytedeco.pytorch.cache.pipeline;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public final class BackfillPipeline {

    private final CacheBackend source;
    private final CacheBackend target;

    public BackfillPipeline(CacheBackend source, CacheBackend target) {
        this.source = source;
        this.target = target;
    }

    public List<PipelineStage> stages() {
        List<PipelineStage> out = new ArrayList<>();
        out.add(new ListEntities());
        out.add(new FetchBatches(source));
        out.add(new WriteToTarget(target));
        out.add(new VerifyCounts());
        return out;
    }

    public PipelineReport run(PipelineContext ctx, PipelineScheduler scheduler) {
        return scheduler.run(stages(), ctx);
    }

    public static final class ListEntities implements PipelineStage {
        @Override public String name() { return "list-entities"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object entities = ctx.get("entities");
            if (!(entities instanceof Collection)) {
                ctx.recordError(name(), "no entities provided");
                return StageResult.ABORT_FATAL;
            }
            ctx.increment("entitiesCount", ((Collection<?>) entities).size());
            return StageResult.CONTINUE;
        }
    }

    public static final class FetchBatches implements PipelineStage {
        private final CacheBackend source;
        public FetchBatches(CacheBackend source) { this.source = source; }
        @Override public String name() { return "fetch-batches"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object entities = ctx.get("entities");
            if (!(entities instanceof Collection)) return StageResult.SKIP;
            AtomicLong fetched = new AtomicLong();
            long batchSize = 256;
            for (Object o : (Collection<?>) entities) {
                if (!(o instanceof CacheKey)) continue;
                CacheKey k = (CacheKey) o;
                source.get(k).ifPresent(v -> {
                    ctx.set("v:" + k.toStorageKey(), v);
                    fetched.incrementAndGet();
                });
            }
            ctx.increment("fetched", fetched.get());
            return StageResult.CONTINUE;
        }
    }

    public static final class WriteToTarget implements PipelineStage {
        private final CacheBackend target;
        public WriteToTarget(CacheBackend target) { this.target = target; }
        @Override public String name() { return "write-target"; }
        @Override public StageResult apply(PipelineContext ctx) {
            long written = 0;
            for (java.util.Map.Entry<String, Object> e : ctx.attributes().entrySet()) {
                if (!e.getKey().startsWith("v:")) continue;
                if (!(e.getValue() instanceof CacheValue)) continue;
                CacheKey k = CacheKey.fromStorageKey(e.getKey().substring(2));
                target.put(k, (CacheValue<Object>) e.getValue());
                written++;
            }
            ctx.increment("written", written);
            return StageResult.CONTINUE;
        }
    }

    public static final class VerifyCounts implements PipelineStage {
        @Override public String name() { return "verify-counts"; }
        @Override public StageResult apply(PipelineContext ctx) {
            long fetched = ctx.counters().getOrDefault("fetched", 0L);
            long written = ctx.counters().getOrDefault("written", 0L);
            if (fetched != written) {
                ctx.recordError(name(), "fetched=" + fetched + " != written=" + written);
                return StageResult.ABORT_FATAL;
            }
            ctx.increment("verified", fetched);
            return StageResult.CONTINUE;
        }
    }
}
