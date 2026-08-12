/*
 * MetadataTagRedactionPolicy -- redact entries whose key tags contain a
 * declared sensitive tag (e.g. `pii=true`, `secret=high`).
 *
 * <p>Lightweight alternative to path-based redaction when the producer
 * annotates the cache key with sensitivity metadata.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.LinkedHashMap;
import java.util.Map;

public final class MetadataTagRedactionPolicy implements RedactionPolicy {

    public static final String REDACTED = "[REDACTED]";

    private final Map<String, String> sensitiveTags;

    public MetadataTagRedactionPolicy(Map<String, String> sensitiveTags) {
        this.sensitiveTags = sensitiveTags == null ? new LinkedHashMap<>() : new LinkedHashMap<>(sensitiveTags);
    }

    public static MetadataTagRedactionPolicy pii() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("pii", "true");
        return new MetadataTagRedactionPolicy(m);
    }

    public static MetadataTagRedactionPolicy secrets() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("sensitivity", "high");
        return new MetadataTagRedactionPolicy(m);
    }

    @Override public String name() { return "metadata-tag"; }

    @Override
    public CacheValue<Object> apply(CacheKey key, CacheValue<Object> value) {
        if (value == null) return value;
        boolean hit = false;
        for (Map.Entry<String, String> e : sensitiveTags.entrySet()) {
            String v = value.tag(e.getKey());
            if (v != null && v.equals(e.getValue())) { hit = true; break; }
        }
        if (hit) {
            return value.toBuilder().value(REDACTED).tag("redacted", "true").build();
        }
        return value;
    }
}
