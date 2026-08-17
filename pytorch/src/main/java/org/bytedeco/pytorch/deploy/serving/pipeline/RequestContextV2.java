/*
 * Enhanced Request Context — enterprise-grade request context with advanced features.
 *
 * Key enhancements:
 *   1. User and session tracking
 *   2. Multi-tenant support
 *   3. Request tracing with traceId/spanId
 *   4. Feature store integration
 *   5. Context propagation
 *   6. Typed experiment parameters
 *   7. Deadline and budget tracking
 *   8. Request metadata
 *
 * Production patterns (Meta, Google, ByteDance, Alibaba):
 *   - Distributed tracing (Jaeger, Zipkin, OpenTelemetry)
 *   - Feature store lookups (Feast, Tecton)
 *   - Multi-tenancy for different business units
 *   - Request-level priority and SLAs
 */
package org.bytedeco.pytorch.deploy.serving.pipeline;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Enterprise-grade request context with comprehensive metadata.
 */
public final class RequestContextV2 {

    // ---- Core identifiers ----
    private final String requestId;
    private final String traceId;
    private final String spanId;
    private final String userId;
    private final String deviceId;
    private final String sessionId;
    private final String tenantId;

    // ---- Scene and context ----
    private final String scene;
    private final String page;
    private final String position;
    private final String platform;
    private final String clientVersion;

    // ---- Timing ----
    private final long startTimeMs;
    private final long deadlineMs;
    private final long timeoutMs;

    // ---- Data stores ----
    private final Map<String, String> experimentParams;
    private final Map<String, String> features;
    private final Map<String, Object> attributes;
    private final Map<String, double[]> embeddings;
    private final Map<String, List<String>> userHistory;

    // ---- Request metadata ----
    private final Map<String, String> metadata;
    private final String ipAddress;
    private final String userAgent;
    private final Map<String, String> headers;

    // ---- Debug and tracing ----
    private final boolean debug;
    private final boolean tracingEnabled;
    private final List<String> tags;

    private RequestContextV2(Builder b) {
        this.requestId = Objects.requireNonNull(b.requestId, "requestId");
        this.traceId = b.traceId != null ? b.traceId : generateTraceId();
        this.spanId = b.spanId != null ? b.spanId : generateSpanId();
        this.userId = b.userId != null ? b.userId : "";
        this.deviceId = b.deviceId != null ? b.deviceId : "";
        this.sessionId = b.sessionId != null ? b.sessionId : "";
        this.tenantId = b.tenantId != null ? b.tenantId : "default";

        this.scene = b.scene != null ? b.scene : "default";
        this.page = b.page != null ? b.page : "";
        this.position = b.position != null ? b.position : "";
        this.platform = b.platform != null ? b.platform : "unknown";
        this.clientVersion = b.clientVersion != null ? b.clientVersion : "";

        this.startTimeMs = b.startTimeMs > 0 ? b.startTimeMs : System.currentTimeMillis();
        this.deadlineMs = b.deadlineMs > 0 ? b.deadlineMs : this.startTimeMs + b.timeoutMs;
        this.timeoutMs = b.timeoutMs;

        this.experimentParams = Collections.unmodifiableMap(new LinkedHashMap<>(b.experimentParams));
        this.features = Collections.unmodifiableMap(new LinkedHashMap<>(b.features));
        this.attributes = new ConcurrentHashMap<>(b.attributes);
        this.embeddings = new ConcurrentHashMap<>(b.embeddings);
        this.userHistory = new ConcurrentHashMap<>(b.userHistory);

        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(b.metadata));
        this.ipAddress = b.ipAddress != null ? b.ipAddress : "";
        this.userAgent = b.userAgent != null ? b.userAgent : "";
        this.headers = Collections.unmodifiableMap(new LinkedHashMap<>(b.headers));

        this.debug = b.debug;
        this.tracingEnabled = b.tracingEnabled;
        this.tags = new CopyOnWriteArrayList<>(b.tags);
    }

    public static Builder builder(String requestId) {
        return new Builder(requestId);
    }

    // ---- Getters ----

    public String requestId() { return requestId; }
    public String traceId() { return traceId; }
    public String spanId() { return spanId; }
    public String userId() { return userId; }
    public String deviceId() { return deviceId; }
    public String sessionId() { return sessionId; }
    public String tenantId() { return tenantId; }

    public String scene() { return scene; }
    public String page() { return page; }
    public String position() { return position; }
    public String platform() { return platform; }
    public String clientVersion() { return clientVersion; }

    public long startTimeMs() { return startTimeMs; }
    public long deadlineMs() { return deadlineMs; }
    public long timeoutMs() { return timeoutMs; }

    public Map<String, String> experimentParams() { return experimentParams; }
    public Map<String, String> features() { return features; }
    public Map<String, Object> attributes() { return attributes; }
    public Map<String, double[]> embeddings() { return embeddings; }
    public Map<String, List<String>> userHistory() { return userHistory; }

    public Map<String, String> metadata() { return metadata; }
    public String ipAddress() { return ipAddress; }
    public String userAgent() { return userAgent; }
    public Map<String, String> headers() { return headers; }

    public boolean debug() { return debug; }
    public boolean tracingEnabled() { return tracingEnabled; }
    public List<String> tags() { return Collections.unmodifiableList(tags); }

    // ---- Computed properties ----

    public String diversionKey() {
        if (userId != null && !userId.isEmpty()) return userId;
        if (deviceId != null && !deviceId.isEmpty()) return deviceId;
        return requestId;
    }

    public long remainingBudgetMs() {
        return Math.max(0, deadlineMs - System.currentTimeMillis());
    }

    public boolean deadlineExceeded() {
        return System.currentTimeMillis() >= deadlineMs;
    }

    public long elapsedMs() {
        return System.currentTimeMillis() - startTimeMs;
    }

    public double deadlineProgress() {
        long total = deadlineMs - startTimeMs;
        if (total <= 0) return 1.0;
        return Math.min(1.0, (double) elapsedMs() / total);
    }

    public boolean isNewUser() {
        return userHistory.isEmpty() || userHistory.get("click_items") == null ||
               userHistory.get("click_items").isEmpty();
    }

    public boolean isHighPriority() {
        return "P0".equals(metadata.get("priority")) ||
               scene.contains("checkout") ||
               scene.contains("purchase");
    }

    // ---- Experiment parameters ----

    public String expParam(String key, String defaultValue) {
        return experimentParams.getOrDefault(key, defaultValue);
    }

    public int expParamInt(String key, int defaultValue) {
        String v = experimentParams.get(key);
        if (v == null) return defaultValue;
        try { return Integer.parseInt(v); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    public long expParamLong(String key, long defaultValue) {
        String v = experimentParams.get(key);
        if (v == null) return defaultValue;
        try { return Long.parseLong(v); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    public double expParamDouble(String key, double defaultValue) {
        String v = experimentParams.get(key);
        if (v == null) return defaultValue;
        try { return Double.parseDouble(v); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    public boolean expParamBool(String key, boolean defaultValue) {
        String v = experimentParams.get(key);
        if (v == null) return defaultValue;
        return "true".equalsIgnoreCase(v) || "1".equals(v) || "yes".equalsIgnoreCase(v);
    }

    // ---- Features ----

    public String feature(String key) {
        return features.get(key);
    }

    public String feature(String key, String defaultValue) {
        return features.getOrDefault(key, defaultValue);
    }

    public int featureInt(String key, int defaultValue) {
        String v = features.get(key);
        if (v == null) return defaultValue;
        try { return Integer.parseInt(v); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    public double featureDouble(String key, double defaultValue) {
        String v = features.get(key);
        if (v == null) return defaultValue;
        try { return Double.parseDouble(v); }
        catch (NumberFormatException e) { return defaultValue; }
    }

    // ---- Attributes (mutable) ----

    public void setAttribute(String key, Object value) {
        if (value == null) {
            attributes.remove(key);
        } else {
            attributes.put(key, value);
        }
    }

    @SuppressWarnings("unchecked")
    public <T> T getAttribute(String key) {
        return (T) attributes.get(key);
    }

    @SuppressWarnings("unchecked")
    public <T> T getAttribute(String key, T defaultValue) {
        return (T) attributes.getOrDefault(key, defaultValue);
    }

    public boolean hasAttribute(String key) {
        return attributes.containsKey(key);
    }

    // ---- Embeddings ----

    public void setEmbedding(String name, double[] embedding) {
        embeddings.put(name, embedding);
    }

    public double[] getEmbedding(String name) {
        return embeddings.get(name);
    }

    public boolean hasEmbedding(String name) {
        return embeddings.containsKey(name);
    }

    // ---- User history ----

    public void addToHistory(String listName, String itemId) {
        userHistory.computeIfAbsent(listName, k -> new CopyOnWriteArrayList<>()).add(itemId);
    }

    public List<String> getHistory(String listName) {
        return userHistory.getOrDefault(listName, List.of());
    }

    public boolean hasInHistory(String listName, String itemId) {
        List<String> history = userHistory.get(listName);
        return history != null && history.contains(itemId);
    }

    public int historySize(String listName) {
        List<String> history = userHistory.get(listName);
        return history != null ? history.size() : 0;
    }

    // ---- Tags ----

    public void addTag(String tag) {
        if (tag != null && !tags.contains(tag)) {
            tags.add(tag);
        }
    }

    public void removeTag(String tag) {
        tags.remove(tag);
    }

    public boolean hasTag(String tag) {
        return tags.contains(tag);
    }

    // ---- Conversion ----

    public RequestContext toRequestContext() {
        return RequestContext.builder(requestId)
                .userId(userId)
                .deviceId(deviceId)
                .scene(scene)
                .startEpochMs(startTimeMs)
                .deadlineEpochMs(deadlineMs)
                .experimentParams(experimentParams)
                .features(features)
                .debug(debug)
                .build();
    }

    // ---- Tracing ----

    public RequestContextV2 withChildSpan(String childSpanId) {
        return builder(requestId + "-child")
                .traceId(traceId)
                .spanId(childSpanId)
                .userId(userId)
                .deviceId(deviceId)
                .sessionId(sessionId)
                .tenantId(tenantId)
                .scene(scene)
                .page(page)
                .position(position)
                .platform(platform)
                .clientVersion(clientVersion)
                .startTimeMs(startTimeMs)
                .deadlineMs(deadlineMs)
                .timeoutMs(timeoutMs)
                .experimentParams(experimentParams)
                .features(features)
                .attributes(new HashMap<>(attributes))
                .debug(debug)
                .build();
    }

    public Map<String, String> toTracingHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("X-Trace-Id", traceId);
        headers.put("X-Span-Id", spanId);
        headers.put("X-Request-Id", requestId);
        headers.put("X-Tenant-Id", tenantId);
        return headers;
    }

    // ---- Helpers ----

    private static String generateTraceId() {
        return String.format("trace-%d-%04x", System.currentTimeMillis(),
                (int) (Math.random() * 0xFFFF));
    }

    private static String generateSpanId() {
        return String.format("span-%04x", (int) (Math.random() * 0xFFFF));
    }

    @Override
    public String toString() {
        return String.format("RequestContextV2{req=%s trace=%s user=%s scene=%s tenant=%s deadline=%.1fms}",
                requestId, traceId, userId, scene, tenantId, (double) remainingBudgetMs());
    }

    // ---- Builder ----

    public static final class Builder {
        private final String requestId;
        private String traceId;
        private String spanId;
        private String userId;
        private String deviceId;
        private String sessionId;
        private String tenantId;
        private String scene;
        private String page;
        private String position;
        private String platform;
        private String clientVersion;
        private long startTimeMs;
        private long deadlineMs;
        private long timeoutMs = 200L;
        private final Map<String, String> experimentParams = new LinkedHashMap<>();
        private final Map<String, String> features = new LinkedHashMap<>();
        private final Map<String, Object> attributes = new LinkedHashMap<>();
        private final Map<String, double[]> embeddings = new LinkedHashMap<>();
        private final Map<String, List<String>> userHistory = new LinkedHashMap<>();
        private final Map<String, String> metadata = new LinkedHashMap<>();
        private String ipAddress;
        private String userAgent;
        private final Map<String, String> headers = new LinkedHashMap<>();
        private boolean debug;
        private boolean tracingEnabled = true;
        private final List<String> tags = new ArrayList<>();

        private Builder(String requestId) {
            this.requestId = requestId;
        }

        // Setters
        public Builder traceId(String traceId) { this.traceId = traceId; return this; }
        public Builder spanId(String spanId) { this.spanId = spanId; return this; }
        public Builder userId(String userId) { this.userId = userId; return this; }
        public Builder deviceId(String deviceId) { this.deviceId = deviceId; return this; }
        public Builder sessionId(String sessionId) { this.sessionId = sessionId; return this; }
        public Builder tenantId(String tenantId) { this.tenantId = tenantId; return this; }
        public Builder scene(String scene) { this.scene = scene; return this; }
        public Builder page(String page) { this.page = page; return this; }
        public Builder position(String position) { this.position = position; return this; }
        public Builder platform(String platform) { this.platform = platform; return this; }
        public Builder clientVersion(String version) { this.clientVersion = version; return this; }
        public Builder startTimeMs(long startTimeMs) { this.startTimeMs = startTimeMs; return this; }
        public Builder deadlineMs(long deadlineMs) { this.deadlineMs = deadlineMs; return this; }
        public Builder timeoutMs(long timeoutMs) { this.timeoutMs = timeoutMs; return this; }
        public Builder experimentParam(String key, String value) { this.experimentParams.put(key, value); return this; }
        public Builder experimentParams(Map<String, String> params) { this.experimentParams.putAll(params); return this; }
        public Builder feature(String key, String value) { this.features.put(key, value); return this; }
        public Builder features(Map<String, String> features) { this.features.putAll(features); return this; }
        public Builder attribute(String key, Object value) { this.attributes.put(key, value); return this; }
        public Builder attributes(Map<String, Object> attributes) { this.attributes.putAll(attributes); return this; }
        public Builder embedding(String name, double[] emb) { this.embeddings.put(name, emb); return this; }
        public Builder userHistory(String listName, List<String> history) { this.userHistory.put(listName, history); return this; }
        public Builder metadata(String key, String value) { this.metadata.put(key, value); return this; }
        public Builder ipAddress(String ip) { this.ipAddress = ip; return this; }
        public Builder userAgent(String ua) { this.userAgent = ua; return this; }
        public Builder header(String key, String value) { this.headers.put(key, value); return this; }
        public Builder debug(boolean debug) { this.debug = debug; return this; }
        public Builder tracingEnabled(boolean enabled) { this.tracingEnabled = enabled; return this; }
        public Builder tag(String tag) { this.tags.add(tag); return this; }

        public RequestContextV2 build() {
            return new RequestContextV2(this);
        }
    }
}
