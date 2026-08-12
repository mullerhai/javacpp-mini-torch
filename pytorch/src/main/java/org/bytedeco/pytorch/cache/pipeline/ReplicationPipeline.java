/*
 * ReplicationPipeline -- master → replica replication across regions.
 *
 * <p>Pipeline shape:
 * <pre>
 *   1. detectMaster   -> ctx.set("master", regionId)
 *   2. diffKeys       -> ctx.set("diff", List&lt;CacheKey&gt;)
 *   3. transfer       -> ctx.increment("transferred", N)
 *   4. verify         -> ctx.increment("verified", N)
 *   5. promote        -> ctx.increment("promoted", N)
 * </pre>
 *
 * <p>Production replication traffic is incident-driven (primary failover,
 * region resync) and is fed by a CDC stream. The pipeline accepts the diff
 * via {@link PipelineContext#set(String, Object)}.
 */
package org.bytedeco.pytorch.cache.pipeline;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ReplicationPipeline {

    private final CacheBackend master;
    private final CacheBackend replica;
    private final String region;

    public ReplicationPipeline(CacheBackend master, CacheBackend replica, String region) {
        this.master = master;
        this.replica = replica;
        this.region = region == null ? "default" : region;
    }

    public List<PipelineStage> stages() {
        List<PipelineStage> out = new ArrayList<>();
        out.add(new DetectMaster());
        out.add(new DiffKeys());
        out.add(new Transfer(master, replica));
        out.add(new Verify(replica));
        out.add(new Promote());
        return out;
    }

    public PipelineReport run(PipelineContext ctx, PipelineScheduler scheduler) {
        return scheduler.run(stages(), ctx);
    }

    public static final class DetectMaster implements PipelineStage {
        @Override public String name() { return "detect-master"; }
        @Override public StageResult apply(PipelineContext ctx) {
            String master = (String) ctx.get("master");
            if (master == null) {
                ctx.recordError(name(), "no master region set");
                return StageResult.ABORT_FATAL;
            }
            return StageResult.CONTINUE;
        }
    }

    public static final class DiffKeys implements PipelineStage {
        @Override public String name() { return "diff-keys"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object diff = ctx.get("diff");
            if (!(diff instanceof Collection)) {
                ctx.recordError(name(), "diff is not a collection");
                return StageResult.ABORT_FATAL;
            }
            ctx.increment("diffCount", ((Collection<?>) diff).size());
            return StageResult.CONTINUE;
        }
    }

    public static final class Transfer implements PipelineStage {
        private final CacheBackend master;
        private final CacheBackend replica;
        public Transfer(CacheBackend master, CacheBackend replica) {
            this.master = master;
            this.replica = replica;
        }
        @Override public String name() { return "transfer"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object diff = ctx.get("diff");
            if (!(diff instanceof Collection)) return StageResult.SKIP;
            long n = 0, failed = 0;
            List<CacheKey> batch = new ArrayList<>(512);
            for (Object o : (Collection<?>) diff) {
                if (!(o instanceof CacheKey)) continue;
                CacheKey k = (CacheKey) o;
                master.get(k).ifPresent(v -> {
                    replica.put(k, v);
                    batch.add(k);
                });
                n++;
            }
            ctx.increment("transferred", n);
            ctx.increment("failed", failed);
            ctx.set("transferredKeys", batch);
            return StageResult.CONTINUE;
        }
    }

    public static final class Verify implements PipelineStage {
        private final CacheBackend replica;
        public Verify(CacheBackend replica) { this.replica = replica; }
        @Override public String name() { return "verify"; }
        @Override public StageResult apply(PipelineContext ctx) {
            Object obj = ctx.get("transferredKeys");
            if (!(obj instanceof Collection)) return StageResult.SKIP;
            long verified = 0;
            for (Object o : (Collection<?>) obj) {
                if (o instanceof CacheKey) {
                    if (replica.get((CacheKey) o).isPresent()) verified++;
                }
            }
            ctx.increment("verified", verified);
            return StageResult.CONTINUE;
        }
    }

    public static final class Promote implements PipelineStage {
        @Override public String name() { return "promote"; }
        @Override public StageResult apply(PipelineContext ctx) {
            long verified = ctx.counters().getOrDefault("verified", 0L);
            long transferred = ctx.counters().getOrDefault("transferred", 0L);
            if (verified == transferred && transferred > 0) {
                ctx.increment("promoted", 1);
                return StageResult.CONTINUE;
            }
            // partial replication -- refuse to promote
            ctx.recordError(name(), "verified=" + verified + " < transferred=" + transferred);
            return StageResult.ABORT_FATAL;
        }
    }
}
