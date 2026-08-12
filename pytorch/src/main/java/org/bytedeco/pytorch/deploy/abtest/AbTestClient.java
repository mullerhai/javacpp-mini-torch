/*
 * AbTestClient — the runtime facade used by service code.
 *
 * Industry reference: LaunchDarkly / Split.io SDK. Service code calls
 * one entry point and gets:
 *   - variant assignment (with override / force / targeting / whitelist)
 *   - feature flag evaluation
 *   - exposure logging
 *   - metric observation
 *
 * This class is intentionally thin: it composes the existing modules
 * ({@link LayeredExperimentManager}, {@link ForceAssignmentRegistry},
 * {@link FeatureFlagRegistry}, {@link OnlineMetricsCollector}) and
 * exposes a single API for the calling service.
 *
 * Thread-safety: all underlying registries are thread-safe; this class
 * adds no mutable state of its own beyond composition.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * High-level runtime facade for online A/B + feature flag delivery.
 */
public final class AbTestClient {

    private final LayeredExperimentManager experimentManager;
    private final FeatureFlagRegistry flagRegistry;
    private final ForceAssignmentRegistry forceRegistry;
    private final OnlineMetricsCollector metrics;
    private final CopyOnWriteArrayList<Consumer<ExposureRecord>> exposureListeners =
            new CopyOnWriteArrayList<>();

    public AbTestClient(LayeredExperimentManager experimentManager,
                        FeatureFlagRegistry flagRegistry,
                        ForceAssignmentRegistry forceRegistry,
                        OnlineMetricsCollector metrics) {
        this.experimentManager = Objects.requireNonNull(experimentManager, "experimentManager");
        this.flagRegistry = flagRegistry != null ? flagRegistry : new FeatureFlagRegistry();
        this.forceRegistry = forceRegistry != null ? forceRegistry : new ForceAssignmentRegistry();
        this.metrics = metrics != null ? metrics : new OnlineMetricsCollector();
    }

    public static Builder builder() { return new Builder(); }

    public LayeredExperimentManager experimentManager() { return experimentManager; }
    public FeatureFlagRegistry flagRegistry() { return flagRegistry; }
    public ForceAssignmentRegistry forceRegistry() { return forceRegistry; }
    public OnlineMetricsCollector metrics() { return metrics; }

    /** Subscribe to exposure events for downstream logging (Kafka / ClickHouse). */
    public void addExposureListener(Consumer<ExposureRecord> listener) {
        exposureListeners.add(Objects.requireNonNull(listener));
    }

    /**
     * One-shot variant resolution for one experiment + context.
     * Honors:
     *   1. Force rules (override)
     *   2. Targeting rules (eligibility filter)
     *   3. Standard bucket assignment
     * Emits an exposure record regardless of whether the unit matched.
     */
    public AssignmentResult resolve(String experimentId, DiversionContext ctx) {
        Objects.requireNonNull(experimentId, "experimentId");
        Objects.requireNonNull(ctx, "ctx");
        Experiment exp = experimentManager.getExperiment(experimentId);
        if (exp == null) {
            return AssignmentResult.unmatched(experimentId, ctx.unitId(), "unknown_experiment");
        }
        ForceAssignmentRegistry.ForceRule force = forceRegistry.match(experimentId, ctx);
        if (force != null) {
            BucketAssigner.Assignment a = new BucketAssigner.Assignment(
                    experimentId, exp.layerId(), force.variantId, false,
                    -1L, ctx.unitId(), exp.diversionUnit(), ctx.timestamp().toEpochMilli());
            recordExposure(a, true, force.id);
            return new AssignmentResult(a, true, "force:" + force.id);
        }
        BucketAssigner.Assignment a = BucketAssigner.assign(exp, ctx.unitId(), ctx.timestamp().toEpochMilli());
        boolean matched = a != null;
        recordExposure(a, matched, null);
        return new AssignmentResult(a, matched, matched ? "bucket" : "no_match");
    }

    /** Resolve all active experiments for the unit. */
    public List<BucketAssigner.Assignment> resolveAll(DiversionContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        List<BucketAssigner.Assignment> base = experimentManager.resolve(
                ctx.unitId(), ctx.timestamp().toEpochMilli());
        for (BucketAssigner.Assignment a : base) {
            recordExposure(a, true, null);
        }
        return base;
    }

    /** Resolve and merge variant parameter overlays across all matched experiments. */
    public Map<String, String> resolveParameters(DiversionContext ctx) {
        Map<String, String> params = experimentManager.resolveParameters(ctx.unitId());
        Map<String, String> out = new LinkedHashMap<>(params);
        // Merge feature flag STRING values into the param map for callers that
        // expect a single bag (matches the LayeredExperimentManager convention).
        for (FeatureFlag f : flagRegistry.list()) {
            FeatureFlag.Value v = f.evaluate(ctx);
            if (v.type == FeatureFlag.Type.STRING) {
                out.put("flag." + f.key, v.stringValue);
            }
        }
        return Collections.unmodifiableMap(out);
    }

    public boolean boolFlag(String key, DiversionContext ctx, boolean fallback) {
        return flagRegistry.boolValue(key, ctx, fallback);
    }

    public long intFlag(String key, DiversionContext ctx, long fallback) {
        return flagRegistry.intValue(key, ctx, fallback);
    }

    public double doubleFlag(String key, DiversionContext ctx, double fallback) {
        return flagRegistry.doubleValue(key, ctx, fallback);
    }

    public String stringFlag(String key, DiversionContext ctx, String fallback) {
        return flagRegistry.stringValue(key, ctx, fallback);
    }

    /** Record a metric observation for the current exposure. */
    public void observe(String experimentId, String variantId, String metric, double value) {
        metrics.observe(experimentId, variantId, metric, value);
    }

    public void observeBinary(String experimentId, String variantId, String metric, boolean success) {
        metrics.observeBinary(experimentId, variantId, metric, success);
    }

    private void recordExposure(BucketAssigner.Assignment a, boolean matched, String forceId) {
        if (a == null) return;
        metrics.recordExposure(a.experimentId(), a.variantId());
        ExposureRecord rec = new ExposureRecord(
                a.experimentId(), a.layerId(), a.variantId(), a.unitId(),
                a.assignedAtEpochMs(), a.bucket(), matched, forceId);
        for (Consumer<ExposureRecord> l : exposureListeners) {
            try {
                l.accept(rec);
            } catch (RuntimeException ignored) {
                // listeners must not break hot path
            }
        }
    }

    /** Exposure record broadcast to listeners. */
    public static final class ExposureRecord {
        public final String experimentId;
        public final String layerId;
        public final String variantId;
        public final String unitId;
        public final long timestampMs;
        public final long bucket;
        public final boolean matched;
        public final String forceRuleId;

        public ExposureRecord(String experimentId, String layerId, String variantId,
                              String unitId, long timestampMs, long bucket,
                              boolean matched, String forceRuleId) {
            this.experimentId = experimentId;
            this.layerId = layerId;
            this.variantId = variantId;
            this.unitId = unitId;
            this.timestampMs = timestampMs;
            this.bucket = bucket;
            this.matched = matched;
            this.forceRuleId = forceRuleId;
        }

        @Override
        public String toString() {
            return "Exposure{exp=" + experimentId + " variant=" + variantId
                    + " unit=" + unitId + " matched=" + matched
                    + (forceRuleId != null ? " force=" + forceRuleId : "") + "}";
        }
    }

    /** Resolution outcome for one experiment. */
    public static final class AssignmentResult {
        public final BucketAssigner.Assignment assignment; // null when not matched
        public final boolean matched;
        public final String reason;

        private AssignmentResult(BucketAssigner.Assignment assignment, boolean matched, String reason) {
            this.assignment = assignment;
            this.matched = matched;
            this.reason = reason;
        }

        static AssignmentResult unmatched(String expId, String unitId, String reason) {
            return new AssignmentResult(null, false, reason);
        }

        public String variantId() {
            return assignment == null ? null : assignment.variantId();
        }
    }

    public static final class Builder {
        private LayeredExperimentManager em;
        private FeatureFlagRegistry fr;
        private ForceAssignmentRegistry fcr;
        private OnlineMetricsCollector mc;

        public Builder experimentManager(LayeredExperimentManager em) { this.em = em; return this; }
        public Builder flagRegistry(FeatureFlagRegistry fr) { this.fr = fr; return this; }
        public Builder forceRegistry(ForceAssignmentRegistry fcr) { this.fcr = fcr; return this; }
        public Builder metrics(OnlineMetricsCollector mc) { this.mc = mc; return this; }

        public AbTestClient build() {
            if (em == null) em = new LayeredExperimentManager();
            if (fr == null) fr = new FeatureFlagRegistry();
            if (fcr == null) fcr = new ForceAssignmentRegistry();
            if (mc == null) mc = new OnlineMetricsCollector();
            return new AbTestClient(em, fr, fcr, mc);
        }
    }
}