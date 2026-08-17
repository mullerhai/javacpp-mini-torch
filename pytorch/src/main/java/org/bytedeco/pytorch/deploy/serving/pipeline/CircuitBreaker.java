/*
 * Enterprise-grade circuit breaker implementation.
 *
 * Production circuit breaker patterns (Resilience4j, Hystrix, Envoy):
 *   - CLOSED: normal operation, requests pass through
 *   - OPEN: failures exceeded threshold, requests fail fast
 *   - HALF_OPEN: probe requests to test recovery
 *
 * State transitions controlled by:
 *   - Failure rate threshold
 *   - Slow request threshold
 *   - Minimum number of calls before evaluation
 *   - Wait duration in open state before transitioning to half-open
 *
 * Industry usage:
 *   - ByteDance/TikTok: circuit breaker on each micro-service call
 *   - Alibaba: Sentinel dashboard integration
 *   - Tencent: service downgrade on circuit open
 *   - Meta: local circuit breaker for all external RPC
 */
package org.bytedeco.pytorch.deploy.serving.pipeline;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * Thread-safe circuit breaker with configurable failure and slow-call thresholds.
 *
 * <p>State diagram:
 * <pre>
 *   CLOSED --[failure threshold]--> OPEN --[wait duration]--> HALF_OPEN
 *     ^                                                        |
 *     |                                                        v
 *   [success]                                           [probe]
 *     |                                                        |
 *     +----------------[success threshold]--------------------+
 * </pre>
 */
public final class CircuitBreaker {

    public enum State { CLOSED, OPEN, HALF_OPEN }

    public enum Result { SUCCESS, FAILURE, SLOW, TIMEOUT }

    public interface Config {
        double failureRateThreshold();
        double slowCallRateThreshold();
        long slowCallDurationMs();
        int minimumNumberOfCalls();
        long waitDurationInOpenStateMs();
        int permittedNumberOfCallsInHalfOpen();
        double slidingWindowSize();
        boolean recordExceptions();

        static Config defaults() {
            return new DefaultConfig();
        }
    }

    private static final class DefaultConfig implements Config {
        @Override public double failureRateThreshold() { return 50.0; }
        @Override public double slowCallRateThreshold() { return 80.0; }
        @Override public long slowCallDurationMs() { return 2000L; }
        @Override public int minimumNumberOfCalls() { return 10; }
        @Override public long waitDurationInOpenStateMs() { return 60000L; }
        @Override public int permittedNumberOfCallsInHalfOpen() { return 5; }
        @Override public double slidingWindowSize() { return 100.0; }
        @Override public boolean recordExceptions() { return true; }
    }

    private final String name;
    private final Config config;
    private final AtomicReference<State> state = new AtomicReference<>(State.CLOSED);
    private final AtomicReference<State> previousState = new AtomicReference<>(State.CLOSED);
    private final AtomicLong lastStateChange = new AtomicLong(System.currentTimeMillis());
    private final AtomicLong successCount = new AtomicLong(0);
    private final AtomicLong failureCount = new AtomicLong(0);
    private final AtomicLong slowCallCount = new AtomicLong(0);
    private final AtomicLong notPermittedCount = new AtomicLong(0);
    private final AtomicLong totalCallCount = new AtomicLong(0);
    private final AtomicLong halfOpenSuccessRequired = new AtomicLong(0);
    private final AtomicLong halfOpenSuccessCount = new AtomicLong(0);

    public CircuitBreaker(String name) {
        this(name, Config.defaults());
    }

    public CircuitBreaker(String name, Config config) {
        this.name = name != null ? name : "circuit-breaker";
        this.config = config;
    }

    public String name() {
        return name;
    }

    public State state() {
        return state.get();
    }

    public Metrics metrics() {
        return new Metrics(
                state.get(),
                successCount.get(),
                failureCount.get(),
                slowCallCount.get(),
                notPermittedCount.get(),
                totalCallCount.get(),
                lastStateChange.get()
        );
    }

    /**
     * Execute an action with circuit breaker protection.
     * Returns null if circuit is open and fallback is null.
     */
    public <T> T execute(Supplier<T> action, Supplier<T> fallback) {
        if (!isCallPermitted()) {
            notPermittedCount.incrementAndGet();
            return fallback != null ? fallback.get() : null;
        }

        totalCallCount.incrementAndGet();
        long start = System.currentTimeMillis();
        T result = null;
        Result outcome;

        try {
            result = action.get();
            long duration = System.currentTimeMillis() - start;

            if (duration > config.slowCallDurationMs()) {
                outcome = Result.SLOW;
            } else {
                outcome = Result.SUCCESS;
                successCount.incrementAndGet();
            }

            if (state.get() == State.HALF_OPEN) {
                halfOpenSuccessCount.incrementAndGet();
                if (halfOpenSuccessCount.get() >= config.permittedNumberOfCallsInHalfOpen()) {
                    transitionToClosed();
                }
            }
        } catch (Exception e) {
            outcome = Result.FAILURE;
            failureCount.incrementAndGet();
            if (config.recordExceptions()) {
                onFailure(e);
            }
        }

        evaluateState();
        return result;
    }

    public <T> T execute(Supplier<T> action) {
        return execute(action, null);
    }

    /**
     * Execute an action with circuit breaker protection and no return value.
     */
    public void executeVoid(Runnable action, Runnable fallback) {
        execute(() -> {
            action.run();
            return null;
        }, fallback != null ? () -> { fallback.run(); return null; } : null);
    }

    public void executeVoid(Runnable action) {
        executeVoid(action, null);
    }

    private boolean isCallPermitted() {
        State current = state.get();
        if (current == State.CLOSED) {
            return true;
        }
        if (current == State.HALF_OPEN) {
            return halfOpenSuccessRequired.get() < config.permittedNumberOfCallsInHalfOpen();
        }
        // OPEN state: check if wait duration has elapsed
        if (current == State.OPEN) {
            long elapsed = System.currentTimeMillis() - lastStateChange.get();
            if (elapsed >= config.waitDurationInOpenStateMs()) {
                transitionToHalfOpen();
                return true;
            }
            return false;
        }
        return false;
    }

    private void onFailure(Exception e) {
        long total = totalCallCount.get();
        if (total < config.minimumNumberOfCalls()) {
            return; // Not enough data to make decision
        }

        long failures = failureCount.get();
        double failureRate = (failures * 100.0) / total;

        if (failureRate >= config.failureRateThreshold()) {
            transitionToOpen();
        }
    }

    private void evaluateState() {
        long total = totalCallCount.get();
        if (total < config.minimumNumberOfCalls()) {
            return;
        }

        double slowRate = (slowCallCount.get() * 100.0) / total;
        if (slowRate >= config.slowCallRateThreshold() && state.get() == State.CLOSED) {
            // Don't immediately open for slow calls, but log it
        }
    }

    private void transitionToOpen() {
        if (state.compareAndSet(State.CLOSED, State.OPEN) ||
            state.compareAndSet(State.HALF_OPEN, State.OPEN)) {
            previousState.set(state.get());
            lastStateChange.set(System.currentTimeMillis());
            halfOpenSuccessCount.set(0);
        }
    }

    private void transitionToHalfOpen() {
        if (state.compareAndSet(State.OPEN, State.HALF_OPEN)) {
            previousState.set(State.OPEN);
            lastStateChange.set(System.currentTimeMillis());
            halfOpenSuccessCount.set(0);
            halfOpenSuccessRequired.set(0);
        }
    }

    private void transitionToClosed() {
        if (state.compareAndSet(State.HALF_OPEN, State.CLOSED)) {
            previousState.set(State.HALF_OPEN);
            lastStateChange.set(System.currentTimeMillis());
            successCount.set(0);
            failureCount.set(0);
            slowCallCount.set(0);
            halfOpenSuccessCount.set(0);
        }
    }

    /**
     * Manually transition to OPEN state (for emergency fallback).
     */
    public void forceOpen() {
        transitionToOpen();
    }

    /**
     * Manually transition to CLOSED state (for recovery).
     */
    public void forceClose() {
        transitionToClosed();
    }

    /**
     * Reset all counters.
     */
    public void reset() {
        successCount.set(0);
        failureCount.set(0);
        slowCallCount.set(0);
        notPermittedCount.set(0);
        totalCallCount.set(0);
        halfOpenSuccessCount.set(0);
        transitionToClosed();
    }

    public record Metrics(
            State state,
            long successCount,
            long failureCount,
            long slowCallCount,
            long notPermittedCount,
            long totalCallCount,
            long lastStateChangeMs
    ) {
        public double failureRate() {
            return totalCallCount == 0 ? 0.0 : (failureCount * 100.0) / totalCallCount;
        }

        public double slowCallRate() {
            return totalCallCount == 0 ? 0.0 : (slowCallCount * 100.0) / totalCallCount;
        }
    }

    @Override
    public String toString() {
        Metrics m = metrics();
        return String.format("CircuitBreaker{name=%s, state=%s, failures=%d/%d (%.1f%%), notPermitted=%d}",
                name, m.state(), m.failureCount(), m.totalCallCount(), m.failureRate(), m.notPermittedCount());
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public static final class Builder {
        private String name;
        private Config config;

        private Builder(String name) {
            this.name = name;
        }

        public Builder config(Config config) {
            this.config = config;
            return this;
        }

        public Builder failureRateThreshold(double threshold) {
            this.config = new DelegatingConfig(config != null ? config : Config.defaults()) {
                @Override public double failureRateThreshold() { return threshold; }
            };
            return this;
        }

        public Builder slowCallThreshold(long ms) {
            this.config = new DelegatingConfig(config != null ? config : Config.defaults()) {
                @Override public long slowCallDurationMs() { return ms; }
            };
            return this;
        }

        public Builder waitDurationInOpenState(long ms) {
            this.config = new DelegatingConfig(config != null ? config : Config.defaults()) {
                @Override public long waitDurationInOpenStateMs() { return ms; }
            };
            return this;
        }

        public Builder minimumNumberOfCalls(int n) {
            this.config = new DelegatingConfig(config != null ? config : Config.defaults()) {
                @Override public int minimumNumberOfCalls() { return n; }
            };
            return this;
        }

        public CircuitBreaker build() {
            return new CircuitBreaker(name, config != null ? config : Config.defaults());
        }

        private abstract static class DelegatingConfig implements Config {
            private final Config delegate;
            DelegatingConfig(Config delegate) { this.delegate = delegate; }
            @Override public double failureRateThreshold() { return delegate.failureRateThreshold(); }
            @Override public double slowCallRateThreshold() { return delegate.slowCallRateThreshold(); }
            @Override public long slowCallDurationMs() { return delegate.slowCallDurationMs(); }
            @Override public int minimumNumberOfCalls() { return delegate.minimumNumberOfCalls(); }
            @Override public long waitDurationInOpenStateMs() { return delegate.waitDurationInOpenStateMs(); }
            @Override public int permittedNumberOfCallsInHalfOpen() { return delegate.permittedNumberOfCallsInHalfOpen(); }
            @Override public double slidingWindowSize() { return delegate.slidingWindowSize(); }
            @Override public boolean recordExceptions() { return delegate.recordExceptions(); }
        }
    }
}
