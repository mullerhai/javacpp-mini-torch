/*
 * Diversion context — request-time signals used to decide which units are
 * eligible for which experiments.
 *
 * Industry pattern:
 *   - Meta XP / Google: TargetingExpr with country, locale, age, platform,
 *     user_attribute, etc.
 *   - ByteDance Libra: experiment "hit conditions" on device/OS/app version
 *   - Alibaba / Taobao: crowd + scene + time window
 *   - Tencent: tag-based targeting with bitmask cohort ids
 *
 * This module provides a small, pure-Java expression language that is
 * composable with the existing layered experiment manager. We intentionally
 * keep the language simple but explicit so it's easy to serialize, replay
 * and audit (industry trend toward declarative targeting DSL).
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
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Immutable request-time context used to evaluate {@link TargetingRule}s.
 *
 * <p>Mirrors the "request scope" passed to upstream gates in Meta / Google.
 */
public final class DiversionContext {

    private final String unitId;
    private final DiversionUnit diversionUnit;
    private final String platform;
    private final String appVersion;
    private final String country;
    private final String locale;
    private final String userSegment;
    private final Map<String, String> attributes;
    private final Map<String, Boolean> cohortFlags;
    private final Instant timestamp;

    private DiversionContext(Builder b) {
        this.unitId = Objects.requireNonNull(b.unitId, "unitId");
        this.diversionUnit = b.diversionUnit != null ? b.diversionUnit : DiversionUnit.USER_ID;
        this.platform = b.platform != null ? b.platform : "";
        this.appVersion = b.appVersion != null ? b.appVersion : "";
        this.country = b.country != null ? b.country : "";
        this.locale = b.locale != null ? b.locale : "";
        this.userSegment = b.userSegment != null ? b.userSegment : "";
        // Surface canonical fields via the attribute bag so targeting rules can
        // reference them via attrIn("country", ...) etc. We populate the bag
        // BEFORE wrapping with unmodifiableMap so putIfAbsent is allowed.
        Map<String, String> bag = new LinkedHashMap<>(b.attributes);
        if (!this.platform.isEmpty()) bag.putIfAbsent("platform", this.platform);
        if (!this.appVersion.isEmpty()) bag.putIfAbsent("app_version", this.appVersion);
        if (!this.country.isEmpty()) bag.putIfAbsent("country", this.country);
        if (!this.locale.isEmpty()) bag.putIfAbsent("locale", this.locale);
        if (!this.userSegment.isEmpty()) bag.putIfAbsent("user_segment", this.userSegment);
        this.attributes = Collections.unmodifiableMap(bag);
        this.cohortFlags = Collections.unmodifiableMap(new LinkedHashMap<>(b.cohortFlags));
        this.timestamp = b.timestamp != null ? b.timestamp : Instant.now();
    }

    public static Builder builder(String unitId) {
        return new Builder(unitId);
    }

    public String unitId() { return unitId; }
    public DiversionUnit diversionUnit() { return diversionUnit; }
    public String platform() { return platform; }
    public String appVersion() { return appVersion; }
    public String country() { return country; }
    public String locale() { return locale; }
    public String userSegment() { return userSegment; }
    public Map<String, String> attributes() { return attributes; }
    public Map<String, Boolean> cohortFlags() { return cohortFlags; }
    public Instant timestamp() { return timestamp; }

    public String attribute(String key, String defaultValue) {
        String v = attributes.get(key);
        return v != null ? v : defaultValue;
    }

    public boolean hasCohort(String cohortId) {
        Boolean b = cohortFlags.get(cohortId);
        return b != null && b;
    }

    public boolean inAnyCohort(Set<String> cohortIds) {
        if (cohortIds == null) return false;
        for (String c : cohortIds) {
            Boolean v = cohortFlags.get(c);
            if (v != null && v) return true;
        }
        return false;
    }

    @Override
    public String toString() {
        return "DiversionContext{unit=" + unitId + ", platform=" + platform
                + ", country=" + country + ", segment=" + userSegment + "}";
    }

    public static final class Builder {
        private final String unitId;
        private DiversionUnit diversionUnit = DiversionUnit.USER_ID;
        private String platform;
        private String appVersion;
        private String country;
        private String locale;
        private String userSegment;
        private final Map<String, String> attributes = new LinkedHashMap<>();
        private final Map<String, Boolean> cohortFlags = new LinkedHashMap<>();
        private Instant timestamp;

        private Builder(String unitId) {
            this.unitId = unitId;
        }

        public Builder diversionUnit(DiversionUnit u) { this.diversionUnit = u; return this; }
        public Builder platform(String p) { this.platform = p; return this; }
        public Builder appVersion(String v) { this.appVersion = v; return this; }
        public Builder country(String c) { this.country = c; return this; }
        public Builder locale(String l) { this.locale = l; return this; }
        public Builder userSegment(String s) { this.userSegment = s; return this; }
        public Builder attribute(String k, String v) { this.attributes.put(k, v); return this; }
        public Builder attributes(Map<String, String> m) {
            if (m != null) this.attributes.putAll(m); return this;
        }
        public Builder cohortFlag(String cohortId, boolean inCohort) {
            this.cohortFlags.put(cohortId, inCohort); return this;
        }
        public Builder cohortFlags(Map<String, Boolean> m) {
            if (m != null) this.cohortFlags.putAll(m); return this;
        }
        public Builder timestamp(Instant t) { this.timestamp = t; return this; }
        public DiversionContext build() { return new DiversionContext(this); }
    }
}
