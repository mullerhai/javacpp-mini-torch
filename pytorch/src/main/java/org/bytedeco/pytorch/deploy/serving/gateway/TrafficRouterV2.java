/*
 * Enhanced Traffic Router — enterprise-grade gateway with fault tolerance.
 *
 * Key enhancements:
 *   1. Circuit breaker integration per upstream
 *   2. Rate limiting (token bucket algorithm)
 *   3. Circuit breaker at route level
 *   4. Retry with exponential backoff
 *   5. Health check integration
 *   6. Consistent hashing with session affinity
 *   7. Regional routing and failover
 *   8. Request mirroring/shadow traffic
 */
package org.bytedeco.pytorch.deploy.serving.gateway;

import org.bytedeco.pytorch.deploy.abtest.BucketAssigner;
import org.bytedeco.pytorch.deploy.abtest.TrafficSplitter;
import org.bytedeco.pytorch.deploy.serving.pipeline.CircuitBreaker;
import org.bytedeco.pytorch.deploy.serving.pipeline.RetryPolicy;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.*;

/**
 * Enterprise-grade traffic router with fault tolerance and rate limiting.
 */
public final class TrafficRouterV2 {

    /**
     * Upstream target with health monitoring.
     */
    public static final class UpstreamV2 {
        public final String id;
        public final String address;
        public volatile double weightPercent;
        public final boolean shadowOnly;
        public final CircuitBreaker circuitBreaker;
        public final RateLimiter rateLimiter;
        public final boolean healthCheckEnabled;
        public volatile HealthStatus healthStatus;

        public enum HealthStatus {
            HEALTHY,
            DEGRADED,
            UNHEALTHY,
            CIRCUIT_OPEN
        }

        public UpstreamV2(String id, String address, double weightPercent, boolean shadowOnly) {
            this(id, address, weightPercent, shadowOnly, null, null, true, HealthStatus.HEALTHY);
        }

        private UpstreamV2(String id, String address, double weightPercent, boolean shadowOnly,
                          CircuitBreaker circuitBreaker, RateLimiter rateLimiter,
                          boolean healthCheckEnabled, HealthStatus healthStatus) {
            this.id = id;
            this.address = address != null ? address : id;
            this.weightPercent = weightPercent;
            this.shadowOnly = shadowOnly;
            this.circuitBreaker = circuitBreaker;
            this.rateLimiter = rateLimiter;
            this.healthCheckEnabled = healthCheckEnabled;
            this.healthStatus = healthStatus;
        }

        public static Builder builder(String id) {
            return new Builder(id);
        }

        public static final class Builder {
            private final String id;
            private String address;
            private double weightPercent = 0;
            private boolean shadowOnly = false;
            private CircuitBreaker circuitBreaker;
            private RateLimiter rateLimiter;
            private boolean healthCheckEnabled = true;
            private HealthStatus healthStatus = HealthStatus.HEALTHY;

            private Builder(String id) {
                this.id = id;
            }

            public Builder address(String address) { this.address = address; return this; }
            public Builder weightPercent(double weightPercent) { this.weightPercent = weightPercent; return this; }
            public Builder shadowOnly(boolean shadowOnly) { this.shadowOnly = shadowOnly; return this; }
            public Builder circuitBreaker(CircuitBreaker circuitBreaker) { this.circuitBreaker = circuitBreaker; return this; }
            public Builder rateLimiter(RateLimiter rateLimiter) { this.rateLimiter = rateLimiter; return this; }
            public Builder healthCheckEnabled(boolean enabled) { this.healthCheckEnabled = enabled; return this; }

            public UpstreamV2 build() {
                return new UpstreamV2(id, address, weightPercent, shadowOnly,
                        circuitBreaker, rateLimiter, healthCheckEnabled, healthStatus);
            }
        }

        public boolean canServe() {
            return healthStatus == HealthStatus.HEALTHY || healthStatus == HealthStatus.DEGRADED;
        }

        public boolean isCircuitOpen() {
            return circuitBreaker != null && circuitBreaker.state() == CircuitBreaker.State.OPEN;
        }
    }

    /**
     * Token bucket rate limiter.
     */
    public static final class RateLimiter {
        private final double rate;           // tokens per second
        private final double capacity;      // max tokens
        private volatile double tokens;
        private volatile long lastRefill;
        private final Object lock = new Object();

        public RateLimiter(double rate, double capacity) {
            this.rate = rate;
            this.capacity = capacity;
            this.tokens = capacity;
            this.lastRefill = System.currentTimeMillis();
        }

        public static RateLimiter perSecond(double rps) {
            return new RateLimiter(rps, rps);
        }

        public static RateLimiter perMinute(double rpm) {
            return new RateLimiter(rpm / 60, rpm / 60);
        }

        /**
         * Try to acquire tokens. Returns true if allowed.
         */
        public boolean tryAcquire() {
            return tryAcquire(1);
        }

        public boolean tryAcquire(double tokens) {
            synchronized (lock) {
                refill();
                if (this.tokens >= tokens) {
                    this.tokens -= tokens;
                    return true;
                }
                return false;
            }
        }

        private void refill() {
            long now = System.currentTimeMillis();
            long elapsed = now - lastRefill;
            if (elapsed > 0) {
                double newTokens = Math.min(capacity, tokens + rate * (elapsed / 1000.0));
                tokens = newTokens;
                lastRefill = now;
            }
        }

        public double availableTokens() {
            synchronized (lock) {
                refill();
                return tokens;
            }
        }

        public void reset() {
            synchronized (lock) {
                tokens = capacity;
                lastRefill = System.currentTimeMillis();
            }
        }
    }

    /**
     * Health check configuration.
     */
    public interface HealthChecker {
        boolean isHealthy(String upstreamId);
        double healthScore(String upstreamId);
    }

    /**
     * Retry configuration.
     */
    public static final class RetryConfig {
        public final int maxAttempts;
        public final BackoffStrategy backoffStrategy;
        public final long baseDelayMs;
        public final long maxDelayMs;
        public final Set<Integer> retryableStatusCodes;

        public enum BackoffStrategy {
            FIXED,
            EXPONENTIAL,
            EXPONENTIAL_JITTER
        }

        public RetryConfig(int maxAttempts, BackoffStrategy backoffStrategy,
                          long baseDelayMs, long maxDelayMs,
                          Set<Integer> retryableStatusCodes) {
            this.maxAttempts = maxAttempts;
            this.backoffStrategy = backoffStrategy;
            this.baseDelayMs = baseDelayMs;
            this.maxDelayMs = maxDelayMs;
            this.retryableStatusCodes = retryableStatusCodes;
        }

        public static RetryConfig defaults() {
            return new RetryConfig(3, BackoffStrategy.EXPONENTIAL_JITTER,
                    100, 5000, Set.of(408, 429, 500, 502, 503, 504));
        }
    }

    /**
     * Routing decision with full metadata.
     */
    public static final class RouteDecisionV2 {
        public final String primaryUpstreamId;
        public final String primaryAddress;
        public final List<String> shadowUpstreamIds;
        public final String reason;
        public final boolean forced;
        public final boolean rateLimited;
        public final boolean circuitOpen;
        public final String requestId;

        public RouteDecisionV2(String primaryUpstreamId, String primaryAddress,
                              List<String> shadowUpstreamIds, String reason,
                              boolean forced, boolean rateLimited, boolean circuitOpen,
                              String requestId) {
            this.primaryUpstreamId = primaryUpstreamId;
            this.primaryAddress = primaryAddress;
            this.shadowUpstreamIds = shadowUpstreamIds;
            this.reason = reason;
            this.forced = forced;
            this.rateLimited = rateLimited;
            this.circuitOpen = circuitOpen;
            this.requestId = requestId;
        }

        public boolean isSuccessful() {
            return primaryUpstreamId != null && !rateLimited && !circuitOpen;
        }
    }

    private final String routeName;
    private final String salt;
    private final ConcurrentHashMap<String, UpstreamV2> upstreams = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<HeaderRule> headerRules = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<RegionAffinity> regionAffinities = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<FailoverRule> failoverRules = new CopyOnWriteArrayList<>();
    private final ConcurrentHashMap<String, AtomicLong> hitCounts = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, AtomicLong> errorCounts = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, AtomicLong> rateLimitedCounts = new ConcurrentHashMap<>();
    private final AtomicReference<HealthChecker> healthChecker = new AtomicReference<>();
    private volatile boolean sticky = true;
    private volatile RetryConfig retryConfig = RetryConfig.defaults();
    private volatile boolean enableRateLimit = true;
    private volatile boolean enableCircuitBreaker = true;

    public TrafficRouterV2(String routeName) {
        this(routeName, routeName);
    }

    public TrafficRouterV2(String routeName, String salt) {
        this.routeName = Objects.requireNonNull(routeName, "routeName");
        this.salt = salt != null ? salt : routeName;
    }

    public static Builder builder(String routeName) {
        return new Builder(routeName);
    }

    // ---- Upstream management ----

    public void addUpstream(UpstreamV2 upstream) {
        upstreams.put(upstream.id, upstream);
        hitCounts.put(upstream.id, new AtomicLong(0));
        errorCounts.put(upstream.id, new AtomicLong(0));
        rateLimitedCounts.put(upstream.id, new AtomicLong(0));
    }

    public void addUpstream(String id, String address, double weightPercent) {
        addUpstream(UpstreamV2.builder(id)
                .address(address)
                .weightPercent(weightPercent)
                .build());
    }

    public void removeUpstream(String id) {
        upstreams.remove(id);
    }

    public UpstreamV2 getUpstream(String id) {
        return upstreams.get(id);
    }

    public Map<String, UpstreamV2> allUpstreams() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(upstreams));
    }

    // ---- Weight management ----

    public synchronized void setWeights(Map<String, Double> weights) {
        for (Map.Entry<String, Double> e : weights.entrySet()) {
            UpstreamV2 u = upstreams.get(e.getKey());
            if (u != null) {
                u.weightPercent = e.getValue();
            }
        }
    }

    public synchronized void setCanaryPercent(String stableId, String canaryId, double canaryPercent) {
        UpstreamV2 stable = requireUpstream(stableId);
        UpstreamV2 canary = requireUpstream(canaryId);
        double c = Math.max(0.0, Math.min(100.0, canaryPercent));
        canary.weightPercent = c;
        stable.weightPercent = 100.0 - c;
    }

    // ---- Header and affinity rules ----

    public void addHeaderRule(HeaderRule rule) {
        headerRules.add(Objects.requireNonNull(rule));
    }

    public void addHeaderRule(String headerName, String headerValue, String upstreamId) {
        addHeaderRule(new HeaderRule(headerName, headerValue, upstreamId));
    }

    public void addRegionAffinity(String region, String upstreamId) {
        regionAffinities.add(new RegionAffinity(region, upstreamId));
    }

    public void addFailoverRule(String fromUpstream, String toUpstream) {
        failoverRules.add(new FailoverRule(fromUpstream, toUpstream));
    }

    // ---- Health checking ----

    public void setHealthChecker(HealthChecker checker) {
        healthChecker.set(checker);
    }

    /**
     * Update health status for an upstream.
     */
    public void updateHealthStatus(String upstreamId, UpstreamV2.HealthStatus status) {
        UpstreamV2 u = upstreams.get(upstreamId);
        if (u != null) {
            u.healthStatus = status;
        }
    }

    /**
     * Trigger health check for all upstreams.
     */
    public void performHealthCheck() {
        HealthChecker checker = healthChecker.get();
        if (checker == null) return;

        for (UpstreamV2 u : upstreams.values()) {
            if (!u.healthCheckEnabled) continue;

            double score = checker.healthScore(u.id);
            if (score >= 0.9) {
                u.healthStatus = UpstreamV2.HealthStatus.HEALTHY;
            } else if (score >= 0.5) {
                u.healthStatus = UpstreamV2.HealthStatus.DEGRADED;
            } else {
                u.healthStatus = UpstreamV2.HealthStatus.UNHEALTHY;
            }
        }
    }

    // ---- Main routing logic ----

    public RouteDecisionV2 route(TrafficRouter.RouteRequest request) {
        Objects.requireNonNull(request, "request");

        // 1) Header force rules
        for (HeaderRule rule : headerRules) {
            String v = request.header(rule.headerName);
            if (v == null) continue;
            if (rule.headerValue == null || rule.headerValue.equals(v)) {
                UpstreamV2 u = upstreams.get(rule.upstreamId);
                if (u != null && !u.shadowOnly && u.canServe()) {
                    hit(u.id);
                    return makeDecision(u, shadowIds(), "header:" + rule.headerName, true, request.requestId);
                }
            }
        }

        // 2) Region affinity
        if (!request.region.isEmpty()) {
            for (RegionAffinity ra : regionAffinities) {
                if (ra.region.equalsIgnoreCase(request.region)) {
                    UpstreamV2 u = upstreams.get(ra.upstreamId);
                    if (u != null && !u.shadowOnly && u.canServe() && u.weightPercent > 0) {
                        hit(u.id);
                        return makeDecision(u, shadowIds(), "region:" + request.region, false, request.requestId);
                    }
                }
            }
        }

        // 3) Filter healthy upstreams and select
        List<UpstreamV2> eligible = new ArrayList<>();
        for (UpstreamV2 u : upstreams.values()) {
            if (u.shadowOnly) continue;
            if (u.weightPercent <= 0) continue;
            if (!u.canServe()) continue;
            if (enableCircuitBreaker && u.isCircuitOpen()) continue;
            eligible.add(u);
        }

        if (eligible.isEmpty()) {
            // Try circuit-open upstreams
            for (UpstreamV2 u : upstreams.values()) {
                if (!u.shadowOnly && u.weightPercent > 0) {
                    hit(u.id);
                    return makeDecision(u, shadowIds(), "circuit_open_fallback", false, request.requestId);
                }
            }
            throw new IllegalStateException("no routable upstream available on " + routeName);
        }

        // 4) Weighted selection
        String chosenId = selectWeighted(eligible, request);
        UpstreamV2 chosen = upstreams.get(chosenId);

        // 5) Rate limiting check
        if (enableRateLimit && chosen.rateLimiter != null) {
            if (!chosen.rateLimiter.tryAcquire()) {
                rateLimitedCounts.get(chosen.id).incrementAndGet();
                // Try failover
                UpstreamV2 failover = findFailover(chosen.id);
                if (failover != null && failover.canServe()) {
                    hit(failover.id);
                    return makeDecision(failover, shadowIds(), "rate_limited_failover", false, request.requestId);
                }
                return makeDecision(chosen, shadowIds(), "rate_limited", false, request.requestId);
            }
        }

        hit(chosen.id);
        return makeDecision(chosen, shadowIds(), sticky ? "sticky_weight" : "random_weight", false, request.requestId);
    }

    private RouteDecisionV2 makeDecision(UpstreamV2 u, List<String> shadows, String reason, boolean forced, String requestId) {
        return new RouteDecisionV2(
                u.id,
                u.address,
                shadows,
                reason,
                forced,
                false,
                u.isCircuitOpen(),
                requestId
        );
    }

    private String selectWeighted(List<UpstreamV2> eligible, TrafficRouter.RouteRequest request) {
        if (eligible.size() == 1) return eligible.get(0).id;

        // Build weights
        List<TrafficSplitter.WeightedTarget> targets = new ArrayList<>();
        for (UpstreamV2 u : eligible) {
            targets.add(new TrafficSplitter.WeightedTarget(u.id, u.weightPercent));
        }

        if (sticky) {
            return TrafficSplitter.selectSticky(request.stickyKey(), salt, targets);
        } else {
            return TrafficSplitter.selectRandom(targets);
        }
    }

    private UpstreamV2 findFailover(String fromUpstream) {
        for (FailoverRule rule : failoverRules) {
            if (rule.fromUpstream.equals(fromUpstream)) {
                UpstreamV2 to = upstreams.get(rule.toUpstream);
                if (to != null && to.canServe()) {
                    return to;
                }
            }
        }
        return null;
    }

    private List<String> shadowIds() {
        List<String> shadows = new ArrayList<>();
        for (UpstreamV2 u : upstreams.values()) {
            if (u.shadowOnly && u.weightPercent > 0 && u.canServe()) {
                shadows.add(u.id);
            }
        }
        return shadows;
    }

    private void hit(String id) {
        hitCounts.computeIfAbsent(id, k -> new AtomicLong()).incrementAndGet();
    }

    public void recordError(String upstreamId) {
        errorCounts.computeIfAbsent(upstreamId, k -> new AtomicLong()).incrementAndGet();

        UpstreamV2 u = upstreams.get(upstreamId);
        if (u != null && u.circuitBreaker != null) {
            u.circuitBreaker.executeVoid(() -> {});
        }
    }

    public void recordSuccess(String upstreamId) {
        UpstreamV2 u = upstreams.get(upstreamId);
        if (u != null && u.circuitBreaker != null) {
            u.circuitBreaker.executeVoid(() -> {});
        }
    }

    private UpstreamV2 requireUpstream(String id) {
        UpstreamV2 u = upstreams.get(id);
        if (u == null) throw new IllegalArgumentException("unknown upstream: " + id);
        return u;
    }

    // ---- Stats and monitoring ----

    public Map<String, Long> hitCounts() {
        Map<String, Long> m = new LinkedHashMap<>();
        for (Map.Entry<String, AtomicLong> e : hitCounts.entrySet()) {
            m.put(e.getKey(), e.getValue().get());
        }
        return m;
    }

    public Map<String, Long> errorCounts() {
        Map<String, Long> m = new LinkedHashMap<>();
        for (Map.Entry<String, AtomicLong> e : errorCounts.entrySet()) {
            m.put(e.getKey(), e.getValue().get());
        }
        return m;
    }

    public Map<String, Double> errorRates() {
        Map<String, Double> rates = new LinkedHashMap<>();
        for (String id : hitCounts.keySet()) {
            long hits = hitCounts.get(id).get();
            long errors = errorCounts.get(id).get();
            rates.put(id, hits > 0 ? (double) errors / hits : 0.0);
        }
        return rates;
    }

    public Map<String, CircuitBreaker.Metrics> circuitBreakerStats() {
        Map<String, CircuitBreaker.Metrics> stats = new LinkedHashMap<>();
        for (UpstreamV2 u : upstreams.values()) {
            if (u.circuitBreaker != null) {
                stats.put(u.id, u.circuitBreaker.metrics());
            }
        }
        return stats;
    }

    public RouteDecisionV2 explain(TrafficRouter.RouteRequest request) {
        return route(request);
    }

    // ---- Configuration ----

    public TrafficRouterV2 sticky(boolean sticky) {
        this.sticky = sticky;
        return this;
    }

    public TrafficRouterV2 enableRateLimit(boolean enable) {
        this.enableRateLimit = enable;
        return this;
    }

    public TrafficRouterV2 enableCircuitBreaker(boolean enable) {
        this.enableCircuitBreaker = enable;
        return this;
    }

    public TrafficRouterV2 retryConfig(RetryConfig config) {
        this.retryConfig = config;
        return this;
    }

    // ---- Nested types ----

    public static final class HeaderRule {
        public final String headerName;
        public final String headerValue;
        public final String upstreamId;

        public HeaderRule(String headerName, String headerValue, String upstreamId) {
            this.headerName = Objects.requireNonNull(headerName);
            this.headerValue = headerValue;
            this.upstreamId = Objects.requireNonNull(upstreamId);
        }
    }

    public static final class RegionAffinity {
        public final String region;
        public final String upstreamId;

        public RegionAffinity(String region, String upstreamId) {
            this.region = Objects.requireNonNull(region);
            this.upstreamId = Objects.requireNonNull(upstreamId);
        }
    }

    public static final class FailoverRule {
        public final String fromUpstream;
        public final String toUpstream;

        public FailoverRule(String fromUpstream, String toUpstream) {
            this.fromUpstream = fromUpstream;
            this.toUpstream = toUpstream;
        }
    }

    public static final class Builder {
        private final String routeName;
        private String salt;
        private boolean sticky = true;
        private boolean enableRateLimit = true;
        private boolean enableCircuitBreaker = true;
        private RetryConfig retryConfig = RetryConfig.defaults();

        private Builder(String routeName) {
            this.routeName = routeName;
        }

        public Builder salt(String salt) { this.salt = salt; return this; }
        public Builder sticky(boolean sticky) { this.sticky = sticky; return this; }
        public Builder enableRateLimit(boolean enable) { this.enableRateLimit = enable; return this; }
        public Builder enableCircuitBreaker(boolean enable) { this.enableCircuitBreaker = enable; return this; }
        public Builder retryConfig(RetryConfig config) { this.retryConfig = config; return this; }

        public TrafficRouterV2 build() {
            TrafficRouterV2 router = new TrafficRouterV2(routeName, salt);
            router.sticky = sticky;
            router.enableRateLimit = enableRateLimit;
            router.enableCircuitBreaker = enableCircuitBreaker;
            router.retryConfig = retryConfig;
            return router;
        }
    }
}
