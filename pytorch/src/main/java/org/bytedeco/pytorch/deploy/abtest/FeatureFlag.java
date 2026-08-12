/*
 * Feature flags / dynamic config — runtime parameter overrides keyed by
 * experiment assignment or by a separate flag id.
 *
 * Industry practice:
 *   - LaunchDarkly / Split.io / Optimizely: typed flag with rules
 *   - Meta XP: parameter bag attached to variant, dynamic config (DC) service
 *   - ByteDance: "动态配置" dynamic config + "实验参数" experiment parameters
 *   - Google: configuration service pinned per arm
 *   - Uber / DoorDash / Meituan: feature-flag-as-a-service with typed values
 *
 * Differences vs experiment:
 *   - FeatureFlag focuses on parameter / config delivery (no primary metric
 *     analysis). Experiment focuses on statistical comparison.
 *   - Flags can be layered on top of experiments (e.g. "experiment only
 *     matters when flag new_recsys_v2 is on").
 *
 * Flag types: boolean, integer, double, string, json (string-encoded).
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

/**
 * Typed feature flag with targeting + percentage rollout.
 */
public final class FeatureFlag {

    public enum Type { BOOLEAN, INTEGER, DOUBLE, STRING }

    public static final class Value {
        public final Type type;
        public final boolean boolValue;
        public final long intValue;
        public final double doubleValue;
        public final String stringValue;

        private Value(Type type, boolean b, long i, double d, String s) {
            this.type = type;
            this.boolValue = b;
            this.intValue = i;
            this.doubleValue = d;
            this.stringValue = s;
        }

        public static Value ofBoolean(boolean v) { return new Value(Type.BOOLEAN, v, 0L, 0.0, ""); }
        public static Value ofInt(long v) { return new Value(Type.INTEGER, false, v, 0.0, ""); }
        public static Value ofDouble(double v) { return new Value(Type.DOUBLE, false, 0L, v, ""); }
        public static Value ofString(String v) { return new Value(Type.STRING, false, 0L, 0.0, v); }

        public static Value of(String raw, Type type) {
            switch (type) {
                case BOOLEAN: return ofBoolean(Boolean.parseBoolean(raw));
                case INTEGER: return ofInt(Long.parseLong(raw.trim()));
                case DOUBLE: return ofDouble(Double.parseDouble(raw.trim()));
                case STRING: return ofString(raw);
                default: throw new IllegalArgumentException("unknown type");
            }
        }

        @Override
        public String toString() {
            switch (type) {
                case BOOLEAN: return "bool(" + boolValue + ")";
                case INTEGER: return "int(" + intValue + ")";
                case DOUBLE: return "double(" + doubleValue + ")";
                case STRING: return "string(" + stringValue + ")";
                default: return "?";
            }
        }
    }

    public static final class Rule {
        public final String id;
        public final TargetingRule.Rule predicate;
        public final Value value;
        public final double rolloutPercent;
        public final String salt;

        private Rule(Builder b) {
            this.id = Objects.requireNonNull(b.id, "id");
            this.predicate = b.predicate != null ? b.predicate : TargetingRule.ALWAYS;
            this.value = Objects.requireNonNull(b.value, "value");
            this.rolloutPercent = b.rolloutPercent;
            this.salt = b.salt != null ? b.salt : b.id;
        }

        public boolean applies(DiversionContext ctx) {
            if (!predicate.matches(ctx)) return false;
            if (rolloutPercent >= 100.0) return true;
            if (rolloutPercent <= 0.0) return false;
            long bucket = BucketAssigner.bucketOf(salt, ctx.unitId(), 10_000L);
            return bucket < Math.round(rolloutPercent / 100.0 * 10_000L);
        }

        public static Builder builder(String id) { return new Builder(id); }

        public static final class Builder {
            private final String id;
            private TargetingRule.Rule predicate;
            private Value value;
            private double rolloutPercent = 100.0;
            private String salt;

            private Builder(String id) { this.id = id; }

            public Builder when(TargetingRule.Rule p) { this.predicate = p; return this; }
            public Builder value(Value v) { this.value = v; return this; }
            public Builder rolloutPercent(double p) { this.rolloutPercent = p; return this; }
            public Builder salt(String s) { this.salt = s; return this; }
            public Rule build() { return new Rule(this); }
        }
    }

    public final String key;
    public final Type type;
    public final Value defaultValue;
    public final String description;
    public final List<Rule> rules;
    public final Instant updatedAt;

    private FeatureFlag(Builder b) {
        this.key = Objects.requireNonNull(b.key, "key");
        this.type = Objects.requireNonNull(b.type, "type");
        this.defaultValue = Objects.requireNonNull(b.defaultValue, "defaultValue");
        if (defaultValue.type != type) {
            throw new IllegalArgumentException("defaultValue type mismatch with flag type");
        }
        this.description = b.description != null ? b.description : "";
        List<Rule> r = new ArrayList<>(b.rules);
        this.rules = Collections.unmodifiableList(r);
        this.updatedAt = b.updatedAt != null ? b.updatedAt : Instant.now();
    }

    public Value evaluate(DiversionContext ctx) {
        for (Rule r : rules) {
            if (r.applies(ctx)) return r.value;
        }
        return defaultValue;
    }

    public boolean boolValue(DiversionContext ctx, boolean fallback) {
        Value v = evaluate(ctx);
        return v.type == Type.BOOLEAN ? v.boolValue : fallback;
    }

    public long intValue(DiversionContext ctx, long fallback) {
        Value v = evaluate(ctx);
        return v.type == Type.INTEGER ? v.intValue : fallback;
    }

    public double doubleValue(DiversionContext ctx, double fallback) {
        Value v = evaluate(ctx);
        return v.type == Type.DOUBLE ? v.doubleValue : fallback;
    }

    public String stringValue(DiversionContext ctx, String fallback) {
        Value v = evaluate(ctx);
        return v.type == Type.STRING ? v.stringValue : fallback;
    }

    public static Builder builder(String key) { return new Builder(key); }

    public static final class Builder {
        private final String key;
        private Type type;
        private Value defaultValue;
        private String description;
        private final List<Rule> rules = new ArrayList<>();
        private Instant updatedAt;

        private Builder(String key) { this.key = key; }

        public Builder type(Type t) { this.type = t; return this; }
        public Builder defaultValue(Value v) { this.defaultValue = v; return this; }
        public Builder description(String d) { this.description = d; return this; }
        public Builder addRule(Rule r) { this.rules.add(r); return this; }
        public Builder updatedAt(Instant t) { this.updatedAt = t; return this; }
        public FeatureFlag build() { return new FeatureFlag(this); }
    }
}
