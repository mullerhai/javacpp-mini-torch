/*
 * Force-assignment / override — operator-driven or whitelist-driven variant
 * assignment that bypasses the hashed bucket logic.
 *
 * Industry practice:
 *   - Meta XP: "forced variant" / "override" for QA, dogfood, canary QA pool,
 *     and CEO-tier employees.
 *   - ByteDance: "白名单" (whitelist) — designers / PMs / partner team uids
 *     pinned to a target variant before any hash check.
 *   - Alibaba / Tencent: "插桩名单" — debug or partner-tester uids are
 *     pre-assigned to a treatment to allow manual reproduction.
 *   - Google Ads: "manually assigned arm" for very high-priority users.
 *
 * Use cases:
 *   - QA reproducible testing of treatment path.
 *   - Dogfooding: internal employees always see the new variant.
 *   - Emergency escalation: force a region to control after incident.
 *   - Performance debugging: pin a uid to a specific variant to grab trace.
 *
 * Force rules are evaluated BEFORE bucket assignment. They never affect
 * holdout / guardrail metrics for general traffic (they tag the exposure
 * record with {@code forced=true}).
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of force-assignment rules with priority ordering and time window.
 */
public final class ForceAssignmentRegistry {

    /** A single force rule. */
    public static final class ForceRule {
        public enum Scope {
            /** Pin one unit to a fixed variant of one experiment. */
            UNIT,
            /** Pin all units of one region to control. */
            REGION,
            /** Pin all units matching an attribute to a variant. */
            ATTRIBUTE
        }

        public final String id;
        public final Scope scope;
        public final String experimentId;
        public final String variantId;
        public final String unitId;
        public final String attributeKey;
        public final String attributeValue;
        public final String reason;
        public final int priority;
        public final Instant validFrom;
        public final Instant validUntil;
        public final boolean enabled;

        private ForceRule(Builder b) {
            this.id = Objects.requireNonNull(b.id, "id");
            this.scope = Objects.requireNonNull(b.scope, "scope");
            this.experimentId = Objects.requireNonNull(b.experimentId, "experimentId");
            this.variantId = Objects.requireNonNull(b.variantId, "variantId");
            this.unitId = b.unitId;
            this.attributeKey = b.attributeKey;
            this.attributeValue = b.attributeValue;
            this.reason = b.reason != null ? b.reason : "";
            this.priority = b.priority;
            this.validFrom = b.validFrom;
            this.validUntil = b.validUntil;
            this.enabled = b.enabled;
            validate(this);
        }

        public boolean isActiveAt(Instant now) {
            if (!enabled) return false;
            if (validFrom != null && now.isBefore(validFrom)) return false;
            if (validUntil != null && !now.isBefore(validUntil)) return false;
            return true;
        }

        private static void validate(ForceRule r) {
            switch (r.scope) {
                case UNIT:
                    if (r.unitId == null || r.unitId.isEmpty()) {
                        throw new IllegalArgumentException("UNIT scope needs unitId");
                    }
                    break;
                case REGION:
                    // region encoded as attribute key="country" or "region"
                    break;
                case ATTRIBUTE:
                    if (r.attributeKey == null || r.attributeValue == null) {
                        throw new IllegalArgumentException("ATTRIBUTE scope needs key/value");
                    }
                    break;
            }
        }

        @Override
        public String toString() {
            return "ForceRule{id=" + id + ", scope=" + scope + ", exp=" + experimentId
                    + ", variant=" + variantId + ", prio=" + priority + "}";
        }

        public static Builder builder(String id) { return new Builder(id); }

        public static final class Builder {
            private final String id;
            private Scope scope;
            private String experimentId;
            private String variantId;
            private String unitId;
            private String attributeKey;
            private String attributeValue;
            private String reason;
            private int priority = 0;
            private Instant validFrom;
            private Instant validUntil;
            private boolean enabled = true;

            private Builder(String id) { this.id = id; }

            public Builder scope(Scope s) { this.scope = s; return this; }
            public Builder experimentId(String e) { this.experimentId = e; return this; }
            public Builder variantId(String v) { this.variantId = v; return this; }
            public Builder unitId(String u) { this.unitId = u; return this; }
            public Builder attribute(String key, String value) {
                this.attributeKey = key; this.attributeValue = value; return this;
            }
            public Builder reason(String r) { this.reason = r; return this; }
            public Builder priority(int p) { this.priority = p; return this; }
            public Builder validFrom(Instant t) { this.validFrom = t; return this; }
            public Builder validUntil(Instant t) { this.validUntil = t; return this; }
            public Builder enabled(boolean e) { this.enabled = e; return this; }
            public ForceRule build() { return new ForceRule(this); }
        }
    }

    private final ConcurrentHashMap<String, ForceRule> byId = new ConcurrentHashMap<>();

    public void add(ForceRule rule) {
        byId.put(rule.id, Objects.requireNonNull(rule));
    }

    public boolean remove(String ruleId) {
        return byId.remove(ruleId) != null;
    }

    public ForceRule get(String ruleId) {
        return byId.get(ruleId);
    }

    public List<ForceRule> list() {
        List<ForceRule> out = new ArrayList<>(byId.values());
        out.sort((a, b) -> Integer.compare(b.priority, a.priority));
        return Collections.unmodifiableList(out);
    }

    /**
     * Find the highest-priority active force rule for the given context.
     *
     * @return matching rule, or {@code null} if none applies
     */
    public ForceRule match(String experimentId, DiversionContext ctx) {
        Objects.requireNonNull(experimentId, "experimentId");
        Objects.requireNonNull(ctx, "ctx");
        Instant now = ctx.timestamp();
        ForceRule best = null;
        for (ForceRule r : byId.values()) {
            if (!r.experimentId.equals(experimentId)) continue;
            if (!r.isActiveAt(now)) continue;
            if (!scopeMatches(r, ctx)) continue;
            if (best == null || r.priority > best.priority) {
                best = r;
            }
        }
        return best;
    }

    private static boolean scopeMatches(ForceRule r, DiversionContext ctx) {
        switch (r.scope) {
            case UNIT:
                return r.unitId.equals(ctx.unitId());
            case REGION:
                return r.attributeValue != null
                        && r.attributeValue.equalsIgnoreCase(ctx.attribute(
                                r.attributeKey != null ? r.attributeKey : "country", ""));
            case ATTRIBUTE:
                return r.attributeValue.equalsIgnoreCase(
                        ctx.attribute(r.attributeKey, ""));
            default:
                return false;
        }
    }
}
