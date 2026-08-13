/*
 * CacheKey — typed, hierarchical, versioned cache key.
 *
 * Designed for multi-tenant, multi-region, multi-view feature/score caches.
 * Inspired by ByteDance Abase, Alibaba Tair, Google Vertex AI Feature Store key
 * conventions: {env}:{tenant}:{scene}:{view}:{tag}:{entity}:{version}
 *
 * The components are intentionally separated to enable sharding by view / tenant
 * without exposing raw entity keys across clusters.
 */
package org.bytedeco.pytorch.cache;
import org.bytedeco.pytorch.distributed.*;

import java.util.Objects;

public final class CacheKey {

    private final String tenant;        // tenant / project, e.g. "feed", "rank", "default"
    private final String scene;         // business scene, e.g. "homepage", "recommend", "search"
    private final String view;          // logical view, e.g. "user_features_v3"
    private final String tag;           // optional sub-bucket, e.g. "embedding", "score", "raw"
    private final String entityKey;     // entity id, e.g. "u_12345" / "i_67890"
    private final long version;         // monotonic version for hot-loading / canary
    private final int hash;             // cached hashCode for shard routing

    private CacheKey(Builder b) {
        this.tenant = nullToDefault(b.tenant, "default");
        this.scene = nullToDefault(b.scene, "default");
        this.view = Objects.requireNonNull(b.view, "view");
        this.tag = nullToDefault(b.tag, "");
        this.entityKey = Objects.requireNonNull(b.entityKey, "entityKey");
        this.version = Math.max(0, b.version);
        this.hash = Objects.hash(tenant, scene, view, tag, entityKey, version);
    }

    public String tenant()    { return tenant; }
    public String scene()     { return scene; }
    public String view()      { return view; }
    public String tag()       { return tag; }
    public String entityKey() { return entityKey; }
    public long version()     { return version; }

    /** Raw string form for Redis / serialisation backends. */
    public String toStorageKey() {
        StringBuilder sb = new StringBuilder(96);
        sb.append(tenant).append(':').append(scene).append(':').append(view);
        if (!tag.isEmpty()) sb.append(':').append(tag);
        sb.append(':').append(entityKey);
        if (version > 0) sb.append(":v").append(version);
        return sb.toString();
    }

    /** Routing prefix (used by ConsistentHashRing for shard selection). */
    public String routingKey() {
        return tenant + ":" + scene + ":" + view;
    }

    @Override
    public String toString() { return toStorageKey(); }

    @Override
    public int hashCode() { return hash; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CacheKey)) return false;
        CacheKey k = (CacheKey) o;
        return version == k.version
                && tenant.equals(k.tenant)
                && scene.equals(k.scene)
                && view.equals(k.view)
                && tag.equals(k.tag)
                && entityKey.equals(k.entityKey);
    }

    /** Re-parse a storage key produced by {@link #toStorageKey()}. */
    public static CacheKey fromStorageKey(String s) {
        Objects.requireNonNull(s, "storage key");
        int vIdx = s.lastIndexOf(":v");
        long ver = 0;
        String body = s;
        if (vIdx > 0) {
            try {
                ver = Long.parseLong(s.substring(vIdx + 2));
                body = s.substring(0, vIdx);
            } catch (NumberFormatException ignore) { /* not a versioned key */ }
        }
        String[] parts = body.split(":");
        if (parts.length < 3) throw new IllegalArgumentException("bad key: " + s);
        Builder b = new Builder(parts[2], parts[parts.length - 1])
                .tenant(parts[0]).scene(parts[1]);
        if (parts.length == 4) b.tag(parts[3]);
        else if (parts.length > 4) {
            // join middle segments into tag
            StringBuilder tag = new StringBuilder();
            for (int i = 3; i < parts.length - 1; i++) {
                if (i > 3) tag.append(':');
                tag.append(parts[i]);
            }
            b.tag(tag.toString());
        }
        return b.version(ver).build();
    }

    public static Builder builder(String view, String entityKey) {
        return new Builder(view, entityKey);
    }

    private static String nullToDefault(String s, String def) {
        return s == null || s.isEmpty() ? def : s;
    }

    public static final class Builder {
        private String tenant;
        private String scene;
        private final String view;
        private String tag;
        private final String entityKey;
        private long version;

        private Builder(String view, String entityKey) {
            this.view = view;
            this.entityKey = entityKey;
        }

        public Builder tenant(String tenant)       { this.tenant = tenant; return this; }
        public Builder scene(String scene)         { this.scene = scene; return this; }
        public Builder tag(String tag)             { this.tag = tag; return this; }
        public Builder version(long version)       { this.version = version; return this; }

        public CacheKey build() { return new CacheKey(this); }
    }
}
