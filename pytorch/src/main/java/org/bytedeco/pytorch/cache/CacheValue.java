/*
 * CacheValue — typed value wrapper with TTL, event-time guard, version, and
 * lineage metadata.
 *
 * <p>Two time bases are tracked:
 * <ul>
 *   <li><b>eventTimestamp</b> — when the source data was actually produced
 *       (used by FeatureCrossingGuard to reject future-dated features).</li>
 *   <li><b>writtenAt</b> — when this entry was cached; used for stale-while-revalidate.</li>
 * </ul>
 *
 * <p>Three TTL modes are supported:
 * <ul>
 *   <li>{@link TtlMode#ABSOLUTE} — fixed expireAtMs</li>
 *   <li>{@link TtlMode#SLIDING} — expire on write, extended on read access</li>
 *   <li>{@link TtlMode#EVENT_TIME} — expire based on event time + maxAgeMs</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public final class CacheValue<V> {

    public static final Object NEGATIVE_SENTINEL = new Object() {
        @Override public String toString() { return "NEGATIVE_SENTINEL"; }
    };

    public enum TtlMode { ABSOLUTE, SLIDING, EVENT_TIME }

    private final V value;
    private final long eventTimestampMs;
    private final long writtenAtMs;
    private final long lastAccessedAtMs;
    private final long expireAtMs;
    private final long maxAgeMs;
    private final TtlMode ttlMode;
    private final long version;
    private final String sourceTag;     // origin pipeline (e.g. "flink:user_ctr_1h")
    private final Map<String, String> tags;

    private CacheValue(Builder<V> b) {
        this.value = b.value;
        this.eventTimestampMs = b.eventTimestampMs;
        this.writtenAtMs = b.writtenAtMs > 0 ? b.writtenAtMs : System.currentTimeMillis();
        this.lastAccessedAtMs = writtenAtMs;
        this.expireAtMs = b.expireAtMs;
        this.maxAgeMs = b.maxAgeMs;
        this.ttlMode = b.ttlMode == null ? TtlMode.ABSOLUTE : b.ttlMode;
        this.version = b.version;
        this.sourceTag = b.sourceTag;
        this.tags = b.tags == null ? Collections.emptyMap()
                                   : Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
    }

    public V value()                       { return value; }
    public long eventTimestampMs()         { return eventTimestampMs; }
    public long writtenAtMs()              { return writtenAtMs; }
    public long lastAccessedAtMs()         { return lastAccessedAtMs; }
    public long expireAtMs()               { return expireAtMs; }
    public long maxAgeMs()                 { return maxAgeMs; }
    public TtlMode ttlMode()               { return ttlMode; }
    public long version()                  { return version; }
    public String sourceTag()              { return sourceTag; }
    public Map<String, String> tags()      { return tags; }
    public String tag(String k)            { return tags.get(k); }

    public boolean isExpired(long nowMs) {
        switch (ttlMode) {
            case ABSOLUTE:
                return expireAtMs > 0 && nowMs >= expireAtMs;
            case EVENT_TIME:
                return maxAgeMs > 0 && eventTimestampMs > 0 && nowMs - eventTimestampMs > maxAgeMs;
            case SLIDING:
                return expireAtMs > 0 && nowMs >= expireAtMs;
            default:
                return false;
        }
    }

    public boolean isFreshlyStale(long nowMs, long staleWindowMs) {
        return expireAtMs > 0 && nowMs >= expireAtMs
                && nowMs - expireAtMs <= staleWindowMs;
    }

    public CacheValue<V> withTtlRefreshed(long nowMs, long newTtlMs) {
        if (ttlMode != TtlMode.SLIDING) return this;
        Builder<V> b = toBuilder().expireAtMs(nowMs + newTtlMs);
        return b.build();
    }

    public Builder<V> toBuilder() {
        return new Builder<V>()
                .value(value)
                .eventTimestampMs(eventTimestampMs)
                .writtenAtMs(writtenAtMs)
                .expireAtMs(expireAtMs)
                .maxAgeMs(maxAgeMs)
                .ttlMode(ttlMode)
                .version(version)
                .sourceTag(sourceTag)
                .tags(tags);
    }

    public static <V> Builder<V> of(V value) { return new Builder<V>().value(value); }

    public static final class Builder<V> {
        private V value;
        private long eventTimestampMs;
        private long writtenAtMs;
        private long expireAtMs;
        private long maxAgeMs;
        private TtlMode ttlMode = TtlMode.ABSOLUTE;
        private long version;
        private String sourceTag;
        private Map<String, String> tags;

        public Builder<V> value(V v)                    { this.value = v; return this; }
        public Builder<V> eventTimestampMs(long ts)     { this.eventTimestampMs = ts; return this; }
        public Builder<V> writtenAtMs(long ts)          { this.writtenAtMs = ts; return this; }
        public Builder<V> expireAtMs(long ts)           { this.expireAtMs = ts; return this; }
        public Builder<V> ttlFromNow(long ttlMs)        { this.expireAtMs = System.currentTimeMillis() + ttlMs; return this; }
        public Builder<V> maxAgeMs(long ms)             { this.maxAgeMs = ms; return this; }
        public Builder<V> ttlMode(TtlMode m)            { this.ttlMode = m; return this; }
        public Builder<V> version(long v)               { this.version = v; return this; }
        public Builder<V> sourceTag(String s)           { this.sourceTag = s; return this; }
        public Builder<V> tags(Map<String, String> m)   { if (m != null) { this.tags = m == null ? null : new LinkedHashMap<>(m); } return this; }
        public Builder<V> tag(String k, String v)       { if (this.tags == null) this.tags = new LinkedHashMap<>(); this.tags.put(k, v); return this; }

        public CacheValue<V> build() {
            Objects.requireNonNull(value, "value");
            return new CacheValue<>(this);
        }
    }
}
