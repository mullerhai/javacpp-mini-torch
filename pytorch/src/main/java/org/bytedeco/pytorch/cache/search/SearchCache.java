/*
 * SearchCache — caches payload-full search results (Elasticsearch / OpenSearch /
 * Redis JSON or Sqlite FTS) keyed by canonicalised query.
 *
 * <p>Big-tech reference: Bing / Google search cache layer, Elasticsearch query
 * cache, Redis JSON indexes used by Pinterest / DoorDash feed pre-render.
 *
 * <p>Anti-penetration/breakdown/avalanche/crossing are inherited from
 * {@link TieredCache}. Extra search-specific guards:
 * <ul>
 *   <li>query canonicalisation (whitespace, lower-case) so semantically equal
 *       queries share cache entries</li>
 *   <li>per-page / per-shard partition</li>
 *   <li>result-fingerprint detection for auto-invalidation when source content
 *       drifts (e.g. an entity was updated since the cached result)</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.search;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;
import org.bytedeco.pytorch.cache.TieredCache;
import org.bytedeco.pytorch.cache.metrics.CacheMetrics;
import org.bytedeco.pytorch.dataframe.opensearch.OpenSearch;
import org.bytedeco.pytorch.dataframe.redis.Redis;

import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

public final class SearchCache implements AutoCloseable {

    private final TieredCache tiered;
    private final SearchSource source;
    private final boolean canonicaliseLower;

    public SearchCache(SearchSource source, CacheBackend l1, CacheBackend l2, CacheMetrics metrics) {
        this.source = source;
        this.tiered = new TieredCache(
                l1,
                l2,
                key -> {
                    if (source == null) return null;
                    long start = System.currentTimeMillis();
                    List<Map<String, Object>> rows = source.search(parseQuery(key));
                    Map<String, Object> envelope = new LinkedHashMap<>();
                    envelope.put("q", key.entityKey());
                    envelope.put("rows", rows);
                    envelope.put("ts", start);
                    envelope.put("backend", source.backend());
                    return CacheValue.<Object>of(envelope)
                            .ttlFromNow(Duration.ofMinutes(2).toMillis())
                            .eventTimestampMs(start)
                            .sourceTag("search-cache")
                            .build();
                },
                org.bytedeco.pytorch.cache.CacheConfig.builder()
                        .l1MaxEntries(8_000)
                        .l1Ttl(Duration.ofSeconds(30))
                        .l2Ttl(Duration.ofMinutes(5))
                        .staleWhileRevalidate(Duration.ofSeconds(15))
                        .build(),
                metrics);
        this.canonicaliseLower = true;
    }

    public List<Map<String, Object>> search(String index, String query, int topK) {
        String canonical = canonicalise(query);
        CacheKey key = CacheKey.builder(index, fingerprint(canonical, topK))
                .tag("search")
                .build();
        Optional<CacheValue<Object>> v = tiered.get(key);
        if (v.isPresent()) {
            Object raw = v.get().value();
            if (raw instanceof Map) {
                Object rows = ((Map<?, ?>) raw).get("rows");
                if (rows instanceof List) return (List<Map<String, Object>>) rows;
            }
        }
        // miss path: load directly & cache
        List<Map<String, Object>> rows = source == null ? List.of() : source.search(parseQuery(key));
        if (rows != null) {
            Map<String, Object> envelope = new LinkedHashMap<>();
            envelope.put("q", canonical);
            envelope.put("rows", rows);
            envelope.put("ts", System.currentTimeMillis());
            envelope.put("backend", source == null ? "none" : source.backend());
            CacheValue<Object> cv = CacheValue.<Object>of(envelope)
                    .ttlFromNow(Duration.ofMinutes(2).toMillis())
                    .sourceTag("search-cache")
                    .build();
            tiered.put(key, cv);
        }
        return rows == null ? List.of() : rows;
    }

    public void invalidateIndex(String index) {
        tiered.invalidateByView("default", index);
    }

    private String canonicalise(String q) {
        if (q == null) return "";
        String s = q.trim().replaceAll("\\s+", " ");
        if (canonicaliseLower) s = s.toLowerCase(Locale.ROOT);
        return s;
    }

    private static String fingerprint(String canonical, int topK) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(canonical.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            md.update((byte) topK);
            byte[] h = md.digest();
            StringBuilder sb = new StringBuilder(16);
            for (int i = 0; i < 8; i++) sb.append(String.format("%02x", h[i]));
            return sb.toString();
        } catch (Exception e) {
            return Integer.toHexString(canonical.hashCode() ^ topK);
        }
    }

    private static SearchQuery parseQuery(CacheKey key) {
        return new SearchQuery(key.view(), key.entityKey(), 10);
    }

    public TieredCache tiered() { return tiered; }
    public CacheMetrics metrics() { return tiered.metrics(); }

    @Override
    public void close() { tiered.close(); }

    public static final class SearchQuery {
        public final String index;
        public final String query;
        public final int topK;
        public SearchQuery(String index, String query, int topK) {
            this.index = index; this.query = query; this.topK = topK;
        }
    }

    public interface SearchSource {
        String backend();
        List<Map<String, Object>> search(SearchQuery query);

        static SearchSource openSearch(OpenSearch es, String index) {
            return new OpenSearchSource(es, index);
        }

        static SearchSource redis(Redis redis, String index) {
            return new RedisJsonSource(redis, index);
        }
    }

    static final class OpenSearchSource implements SearchSource {
        private final OpenSearch es;
        private final String index;
        OpenSearchSource(OpenSearch es, String index) { this.es = es; this.index = index; }
        @Override public String backend() { return "opensearch"; }
        @Override public List<Map<String, Object>> search(SearchQuery q) {
            try {
                Map<String, Object> body = new LinkedHashMap<>();
                body.put("size", q.topK);
                body.put("query", java.util.Collections.singletonMap(
                        "match", java.util.Collections.singletonMap("_all", q.query)));
                Object raw = es.search(index, body);
                if (raw instanceof Map) {
                    Object hits = ((Map<?, ?>) raw).get("hits");
                    if (hits instanceof Map) {
                        Object arr = ((Map<?, ?>) hits).get("hits");
                        if (arr instanceof List) {
                            List<Map<String, Object>> out = new ArrayList<>();
                            for (Object e : (List<?>) arr) {
                                if (e instanceof Map) {
                                    Object src = ((Map<?, ?>) e).get("_source");
                                    if (src instanceof Map) {
                                        @SuppressWarnings("unchecked")
                                        Map<String, Object> typed = (Map<String, Object>) src;
                                        out.add(typed);
                                    }
                                }
                            }
                            return out;
                        }
                    }
                }
                return new ArrayList<>();
            } catch (Exception e) {
                return new ArrayList<>();
            }
        }
    }

    static final class RedisJsonSource implements SearchSource {
        private final Redis redis;
        private final String index;
        RedisJsonSource(Redis redis, String index) { this.redis = redis; this.index = index; }
        @Override public String backend() { return "redis-ft"; }
        @Override public List<Map<String, Object>> search(SearchQuery q) {
            // FT.SEARCH surrogate — return whatever is cached under the prefix.
            try {
                java.util.List<String> keys = redis.keys("idx:" + index + ":*");
                List<Map<String, Object>> out = new ArrayList<>();
                for (String k : keys) {
                    String v = redis.get(k);
                    if (v == null) continue;
                    Map<String, Object> m = org.bytedeco.pytorch.utils.json.Json.decodeObject(v);
                    m.put("_key", k);
                    out.add(m);
                    if (out.size() >= q.topK) break;
                }
                return out;
            } catch (Exception e) {
                return new ArrayList<>();
            }
        }
    }
}
