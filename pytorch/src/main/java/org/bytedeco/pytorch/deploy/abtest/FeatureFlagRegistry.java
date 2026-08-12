/*
 * Feature flag registry — manages all flags and provides a typed lookup
 * facade used by service code (the "client SDK" in LaunchDarkly parlance).
 *
 * This is the runtime analogue of the experiment LayeredExperimentManager
 * but for parameter delivery rather than statistical experiment analysis.
 *
 * Industry practice:
 *   - LaunchDarkly / Split / Optimizely: client-side SDK with local cache
 *   - Meta / ByteDance: dynamic-config (DC) service with local in-memory
 *     fallback and snapshot-based read path
 *   - Google: configuration service pinned per request
 *
 * Cache invalidation strategy: the registry is itself an immutable snapshot
 * in production. We expose {@link #swapSnapshot} so the config service can
 * atomically swap versions without impacting in-flight reads.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Thread-safe in-memory feature flag registry.
 */
public final class FeatureFlagRegistry {

    private final AtomicReference<Snapshot> snapshot = new AtomicReference<>();
    private final long version;

    public FeatureFlagRegistry() {
        this(0L);
    }

    public FeatureFlagRegistry(long version) {
        this.version = version;
        this.snapshot.set(new Snapshot(version, Collections.emptyMap(), Instant.now()));
    }

    /**
     * Atomic snapshot replacement (use this from a config-push listener).
     */
    public void swapSnapshot(Map<String, FeatureFlag> flags, Instant now) {
        snapshot.set(new Snapshot(version + 1, new LinkedHashMap<>(flags), now));
    }

    public void put(FeatureFlag flag) {
        Snapshot current = snapshot.get();
        Map<String, FeatureFlag> next = new LinkedHashMap<>(current.flags);
        next.put(flag.key, flag);
        snapshot.set(new Snapshot(current.version + 1, next, Instant.now()));
    }

    public FeatureFlag get(String key) {
        return snapshot.get().flags.get(key);
    }

    public List<FeatureFlag> list() {
        return List.copyOf(snapshot.get().flags.values());
    }

    public long version() {
        return snapshot.get().version;
    }

    public boolean boolValue(String key, DiversionContext ctx, boolean fallback) {
        FeatureFlag f = get(key);
        return f != null ? f.boolValue(ctx, fallback) : fallback;
    }

    public long intValue(String key, DiversionContext ctx, long fallback) {
        FeatureFlag f = get(key);
        return f != null ? f.intValue(ctx, fallback) : fallback;
    }

    public double doubleValue(String key, DiversionContext ctx, double fallback) {
        FeatureFlag f = get(key);
        return f != null ? f.doubleValue(ctx, fallback) : fallback;
    }

    public String stringValue(String key, DiversionContext ctx, String fallback) {
        FeatureFlag f = get(key);
        return f != null ? f.stringValue(ctx, fallback) : fallback;
    }

    /** Immutable snapshot of the registry. */
    public static final class Snapshot {
        public final long version;
        public final Map<String, FeatureFlag> flags;
        public final Instant createdAt;

        public Snapshot(long version, Map<String, FeatureFlag> flags, Instant createdAt) {
            this.version = version;
            this.flags = Collections.unmodifiableMap(new LinkedHashMap<>(flags));
            this.createdAt = createdAt;
        }

        @Override
        public String toString() {
            return "Snapshot{version=" + version + ", flags=" + flags.size() + "}";
        }
    }
}
