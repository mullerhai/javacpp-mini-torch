/*
 * RedisCacheBackend — L2 distributed tier backed by the project's pure-Java Redis client.
 *
 * <p>Used for online feature serving. Stores each entry as a JSON-encoded
 * {@link CacheValue} with optional TTL. Multi-get / multi-set piggy-back the
 * Redis mget / pipeline APIs to amortise round-trips.
 *
 * <p>Supports graceful degradation: if Redis is unreachable, reads return
 * Optional.empty() and writes are reported as failures (caller falls back to
 * source). Metrics surface the failure separately.
 */
package org.bytedeco.pytorch.cache;

import org.bytedeco.pytorch.cache.serialization.CacheCodec;
import org.bytedeco.pytorch.cache.serialization.JsonCacheCodec;
import org.bytedeco.pytorch.dataframe.redis.Redis;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

public final class RedisCacheBackend implements CacheBackend {

    private final Redis redis;
    private final String keyPrefix;
    private final CacheCodec codec;
    private final AtomicLong errors = new AtomicLong();

    public RedisCacheBackend(Redis redis, String keyPrefix) {
        this(redis, keyPrefix, JsonCacheCodec.INSTANCE);
    }

    public RedisCacheBackend(Redis redis, String keyPrefix, CacheCodec codec) {
        this.redis = redis;
        this.keyPrefix = keyPrefix == null ? "cache:" : keyPrefix;
        this.codec = codec;
    }

    @Override public String name() { return "redis-L2"; }
    @Override public int tier()    { return 2; }

    @Override
    public Optional<CacheValue<Object>> get(CacheKey key) {
        try {
            String storageKey = keyPrefix + key.toStorageKey();
            String raw = redis.get(storageKey);
            if (raw == null) return Optional.empty();
            return Optional.of(decode(key, raw));
        } catch (Exception e) {
            errors.incrementAndGet();
            return Optional.empty();
        }
    }

    @Override
    public Map<CacheKey, CacheValue<Object>> getBatch(java.util.Collection<CacheKey> keys) {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null || keys.isEmpty()) return out;
        try {
            String[] storageKeys = new String[keys.size()];
            int i = 0;
            for (CacheKey k : keys) storageKeys[i++] = keyPrefix + k.toStorageKey();
            Map<String, String> raws = new java.util.HashMap<>();
            java.util.List<String> vals = redis.mget(storageKeys);
            for (int j = 0; j < storageKeys.length; j++) {
                String v = j < vals.size() ? vals.get(j) : null;
                if (v != null) raws.put(storageKeys[j], v);
            }
            int idx = 0;
            for (CacheKey k : keys) {
                String raw = raws.get(storageKeys[idx++]);
                if (raw != null) out.put(k, decode(k, raw));
            }
        } catch (Exception e) {
            errors.incrementAndGet();
        }
        return out;
    }

    @Override
    public void put(CacheKey key, CacheValue<Object> value) {
        try {
            String storageKey = keyPrefix + key.toStorageKey();
            String encoded = encode(value);
            long ttlMs = ttlOf(value);
            if (ttlMs > 0) redis.setex(storageKey, java.time.Duration.ofMillis(ttlMs), encoded);
            else redis.set(storageKey, encoded);
        } catch (Exception e) {
            errors.incrementAndGet();
        }
    }

    @Override
    public void putBatch(Map<CacheKey, CacheValue<Object>> entries) {
        if (entries == null || entries.isEmpty()) return;
        try {
            Map<String, String> mset = new LinkedHashMap<>();
            for (Map.Entry<CacheKey, CacheValue<Object>> e : entries.entrySet()) {
                mset.put(keyPrefix + e.getKey().toStorageKey(), encode(e.getValue()));
            }
            redis.mset(mset);
            for (Map.Entry<CacheKey, CacheValue<Object>> e : entries.entrySet()) {
                long ttl = ttlOf(e.getValue());
                if (ttl > 0) redis.expire(keyPrefix + e.getKey().toStorageKey(),
                        java.time.Duration.ofMillis(ttl));
            }
        } catch (Exception ex) {
            errors.incrementAndGet();
        }
    }

    @Override
    public void delete(CacheKey key) {
        try {
            redis.del(keyPrefix + key.toStorageKey());
        } catch (Exception e) {
            errors.incrementAndGet();
        }
    }

    @Override
    public void deleteBatch(java.util.Collection<CacheKey> keys) {
        if (keys == null || keys.isEmpty()) return;
        try {
            String[] sk = new String[keys.size()];
            int i = 0;
            for (CacheKey k : keys) sk[i++] = keyPrefix + k.toStorageKey();
            redis.del(sk);
        } catch (Exception e) {
            errors.incrementAndGet();
        }
    }

    @Override
    public long size() { return -1; }

    @Override
    public boolean ping() {
        try { return "PONG".equalsIgnoreCase(redis.ping()); }
        catch (Exception e) { return false; }
    }

    @Override
    public void close() { try { redis.close(); } catch (Exception ignore) {} }

    public long errorCount() { return errors.get(); }

    private String encode(CacheValue<Object> v) {
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("value", v.value());
        envelope.put("eventTs", v.eventTimestampMs());
        envelope.put("writtenAt", v.writtenAtMs());
        envelope.put("expireAt", v.expireAtMs());
        envelope.put("maxAge", v.maxAgeMs());
        envelope.put("ttlMode", v.ttlMode().name());
        envelope.put("version", v.version());
        envelope.put("source", v.sourceTag());
        if (!v.tags().isEmpty()) envelope.put("tags", v.tags());
        return new String(codec.encode(envelope), java.nio.charset.StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    private CacheValue<Object> decode(CacheKey key, String raw) {
        Map<String, Object> env;
        try {
            Object parsed = org.bytedeco.pytorch.utils.json.Json.decode(raw);
            env = parsed instanceof Map
                    ? (Map<String, Object>) parsed
                    : java.util.Collections.emptyMap();
        } catch (java.io.IOException io) {
            return CacheValue.<Object>of(raw).sourceTag("decode-error").build();
        }
        Object v = env.get("value");
        CacheValue.Builder<Object> b = CacheValue.of(v)
                .eventTimestampMs(asLong(env.get("eventTs")))
                .writtenAtMs(asLong(env.get("writtenAt")))
                .expireAtMs(asLong(env.get("expireAt")))
                .maxAgeMs(asLong(env.get("maxAge")))
                .version(asLong(env.get("version")))
                .sourceTag(asString(env.get("source")));
        String mode = asString(env.get("ttlMode"));
        if (mode != null) {
            try { b.ttlMode(CacheValue.TtlMode.valueOf(mode)); } catch (Exception ignore) {}
        }
        Object tags = env.get("tags");
        if (tags instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) tags).entrySet()) {
                b.tag(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
            }
        }
        return b.build();
    }

    private long ttlOf(CacheValue<Object> v) {
        long now = System.currentTimeMillis();
        switch (v.ttlMode()) {
            case ABSOLUTE: return v.expireAtMs() > 0 ? Math.max(0, v.expireAtMs() - now) : 0;
            case EVENT_TIME: return v.maxAgeMs();
            case SLIDING: return v.expireAtMs() > 0 ? Math.max(0, v.expireAtMs() - now) : 0;
            default: return 0;
        }
    }

    private static long asLong(Object o) {
        if (o == null) return 0;
        if (o instanceof Number) return ((Number) o).longValue();
        try { return Long.parseLong(o.toString()); } catch (Exception e) { return 0; }
    }

    private static String asString(Object o) {
        return o == null ? null : o.toString();
    }
}
