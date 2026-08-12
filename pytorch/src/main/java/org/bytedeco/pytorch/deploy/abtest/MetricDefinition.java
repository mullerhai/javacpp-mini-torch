/*
 * Metric definition framework — typed metric specs with computation strategy.
 *
 * Industry:
 *   - Meta XP / Google Ads: metric definitions carry aggregation type
 *     (count, mean, ratio, quantile, distinct_count, funnel).
 *   - ByteDance Libra: "指标定义" + "事件计算口径" — declaration is separate
 *     from computation (the latter is pluggable per data source).
 *   - Microsoft ExP: stored metric specs with parameter spec.
 *
 * Why this exists:
 *   The existing {@link OnlineMetricsCollector} accepts arbitrary double
 *   observations, which is good for fast iteration but lacks the contract
 *   required for production: a metric has a type, an aggregation strategy,
 *   a default window, and a meaning to both client and analyst.
 *
 * Three concrete aggregation strategies:
 *   - SUM     : metric value is sum of event weights (CTR numerator, GMV)
 *   - MEAN    : metric value is per-event mean (latency, dwell time)
 *   - RATIO   : metric is sum(success) / sum(trial) — Bernoulli mean
 *   - QUANTILE: pre-aggregated via P² / t-digest sketch (lazy; we expose
 *               Welford sketch as a placeholder and quantile estimator hook)
 *   - FUNNEL  : ordered sequence of binary events; the metric is conversion
 *               rate at each step (registered as ordered metrics).
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Metric type taxonomy.
 */
public final class MetricDefinition {

    public enum Aggregation {
        /** Sum of event weights. */
        SUM,
        /** Average of event weights. */
        MEAN,
        /** Sum(success) / Sum(trial) — Bernoulli rate. */
        RATIO,
        /** Quantile sketch (p50/p90/p99) — populated lazily via QuantileEstimator. */
        QUANTILE,
        /** Ordered funnel steps. Each step is a binary success event. */
        FUNNEL
    }

    public enum Direction {
        /** Higher is better (CTR, conversion, GMV). */
        HIGHER_IS_BETTER,
        /** Lower is better (latency, error rate). */
        LOWER_IS_BETTER
    }

    public final String key;
    public final Aggregation aggregation;
    public final Direction direction;
    public final String description;
    public final String unit;
    public final boolean primary;
    public final boolean guardrail;
    public final double minSample;
    public final Double minMde; // minimum detectable effect (optional)

    private MetricDefinition(Builder b) {
        this.key = Objects.requireNonNull(b.key, "key");
        this.aggregation = Objects.requireNonNull(b.aggregation, "aggregation");
        this.direction = b.direction != null ? b.direction : Direction.HIGHER_IS_BETTER;
        this.description = b.description != null ? b.description : "";
        this.unit = b.unit != null ? b.unit : "";
        this.primary = b.primary;
        this.guardrail = b.guardrail;
        this.minSample = b.minSample;
        this.minMde = b.minMde;
    }

    public boolean isImprovement(double delta) {
        // delta > 0 means treatment higher than control.
        return direction == Direction.HIGHER_IS_BETTER ? delta > 0.0 : delta < 0.0;
    }

    @Override
    public String toString() {
        return "Metric[" + key + ":" + aggregation + ":" + direction + "]";
    }

    public static Builder builder(String key) { return new Builder(key); }

    public static final class Builder {
        private final String key;
        private Aggregation aggregation;
        private Direction direction;
        private String description;
        private String unit;
        private boolean primary;
        private boolean guardrail;
        private double minSample = 100.0;
        private Double minMde;

        private Builder(String key) { this.key = key; }

        public Builder aggregation(Aggregation a) { this.aggregation = a; return this; }
        public Builder direction(Direction d) { this.direction = d; return this; }
        public Builder description(String d) { this.description = d; return this; }
        public Builder unit(String u) { this.unit = u; return this; }
        public Builder primary() { this.primary = true; return this; }
        public Builder guardrail() { this.guardrail = true; return this; }
        public Builder minSample(double n) { this.minSample = n; return this; }
        public Builder minMde(double m) { this.minMde = m; return this; }

        public MetricDefinition build() { return new MetricDefinition(this); }
    }
}
