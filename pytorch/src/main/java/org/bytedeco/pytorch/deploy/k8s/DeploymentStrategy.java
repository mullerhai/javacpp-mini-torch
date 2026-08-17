/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.deploy.k8s;

import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Enterprise-grade Kubernetes deployment strategies supporting:
 * <ul>
 *   <li>Rolling Update - Gradual replacement with zero-downtime</li>
 *   <li>Blue-Green - Instant switch with environment duplication</li>
 *   <li>Canary - Progressive traffic shifting with monitoring</li>
 *   <li>In-Place Update - Zero-replacement configuration updates</li>
 * </ul>
 *
 * <p>Reference: Inspired by zio-k8s patterns, Argo Rollouts, Flagger
 */
public final class DeploymentStrategy {

    private DeploymentStrategy() {}

    // ========== Strategy Type Enum ==========

    /**
     * Deployment strategy types.
     */
    public enum StrategyType {
        /** Gradual replacement with configurable surge/unavailable */
        ROLLING_UPDATE,
        /** Full environment duplication, instant switch */
        BLUE_GREEN,
        /** Progressive traffic shifting with automated analysis */
        CANARY,
        /** Configuration-only updates without pod replacement */
        IN_PLACE
    }

    // ========== Strategy Configuration ==========

    /**
     * Base configuration for all deployment strategies.
     */
    public static class StrategyConfig {
        private final StrategyType type;
        private final Duration timeout;
        private final Duration pollingInterval;
        private final int maxRetries;
        private final boolean autoRollbackOnFailure;
        private final Map<String, String> annotations;

        protected StrategyConfig(Builder<?> b) {
            this.type = b.type;
            this.timeout = b.timeout;
            this.pollingInterval = b.pollingInterval;
            this.maxRetries = b.maxRetries;
            this.autoRollbackOnFailure = b.autoRollbackOnFailure;
            this.annotations = Map.copyOf(b.annotations);
        }

        public StrategyType type() { return type; }
        public Duration timeout() { return timeout; }
        public Duration pollingInterval() { return pollingInterval; }
        public int maxRetries() { return maxRetries; }
        public boolean autoRollbackOnFailure() { return autoRollbackOnFailure; }
        public Map<String, String> annotations() { return annotations; }

        public abstract static class Builder<T extends Builder<T>> {
            protected StrategyType type = StrategyType.ROLLING_UPDATE;
            protected Duration timeout = Duration.ofMinutes(10);
            protected Duration pollingInterval = Duration.ofSeconds(5);
            protected int maxRetries = 3;
            protected boolean autoRollbackOnFailure = true;
            protected Map<String, String> annotations = new ConcurrentHashMap<>();

            @SuppressWarnings("unchecked")
            protected T self() { return (T) this; }

            public T type(StrategyType type) { this.type = type; return self(); }
            public T timeout(Duration timeout) { this.timeout = timeout; return self(); }
            public T pollingInterval(Duration interval) { this.pollingInterval = interval; return self(); }
            public T maxRetries(int retries) { this.maxRetries = retries; return self(); }
            public T autoRollbackOnFailure(boolean auto) { this.autoRollbackOnFailure = auto; return self(); }
            public T annotation(String key, String value) { this.annotations.put(key, value); return self(); }
        }
    }

    /**
     * Configuration for Rolling Update strategy.
     */
    public static final class RollingUpdateConfig extends StrategyConfig {
        private final int maxSurge;
        private final int maxUnavailable;
        private final boolean waitForReady;

        private RollingUpdateConfig(Builder b) {
            super(b);
            this.maxSurge = b.maxSurge;
            this.maxUnavailable = b.maxUnavailable;
            this.waitForReady = b.waitForReady;
        }

        public int maxSurge() { return maxSurge; }
        public int maxUnavailable() { return maxUnavailable; }
        public boolean waitForReady() { return waitForReady; }

        public static Builder builder() { return new Builder(); }

        public static final class Builder extends StrategyConfig.Builder<Builder> {
            private int maxSurge = 1;
            private int maxUnavailable = 0;
            private boolean waitForReady = true;

            public Builder() { type(StrategyType.ROLLING_UPDATE); }

            public Builder maxSurge(int surge) { this.maxSurge = surge; return this; }
            public Builder maxUnavailable(int unavailable) { this.maxUnavailable = unavailable; return this; }
            public Builder waitForReady(boolean wait) { this.waitForReady = wait; return this; }

            public RollingUpdateConfig build() {
                return new RollingUpdateConfig(this);
            }
        }
    }

    /**
     * Configuration for Blue-Green deployment strategy.
     */
    public static final class BlueGreenConfig extends StrategyConfig {
        private final String activeColor;
        private final int prePromotionChecks;
        private final Duration prePromotionDelay;
        private final boolean autoPromote;
        private final String trafficPolicy;
        private final int maxSurge;

        private BlueGreenConfig(Builder b) {
            super(b);
            this.activeColor = b.activeColor;
            this.prePromotionChecks = b.prePromotionChecks;
            this.prePromotionDelay = b.prePromotionDelay;
            this.autoPromote = b.autoPromote;
            this.trafficPolicy = b.trafficPolicy;
            this.maxSurge = b.maxSurge;
        }

        public String activeColor() { return activeColor; }
        public int prePromotionChecks() { return prePromotionChecks; }
        public Duration prePromotionDelay() { return prePromotionDelay; }
        public boolean autoPromote() { return autoPromote; }
        public String trafficPolicy() { return trafficPolicy; }
        public int maxSurge() { return maxSurge; }

        public static Builder builder() { return new Builder(); }

        public static final class Builder extends StrategyConfig.Builder<Builder> {
            private String activeColor = "blue";
            private int prePromotionChecks = 3;
            private Duration prePromotionDelay = Duration.ofSeconds(30);
            private boolean autoPromote = true;
            private String trafficPolicy = "service-selector"; // or "istio", "nginx"
            private int maxSurge = 1;

            public Builder() { type(StrategyType.BLUE_GREEN); }

            public Builder activeColor(String color) { this.activeColor = color; return this; }
            public Builder prePromotionChecks(int checks) { this.prePromotionChecks = checks; return this; }
            public Builder prePromotionDelay(Duration delay) { this.prePromotionDelay = delay; return this; }
            public Builder autoPromote(boolean auto) { this.autoPromote = auto; return this; }
            public Builder trafficPolicy(String policy) { this.trafficPolicy = policy; return this; }
            public Builder maxSurge(int surge) { this.maxSurge = surge; return this; }

            public BlueGreenConfig build() {
                return new BlueGreenConfig(this);
            }
        }
    }

    /**
     * Configuration for Canary deployment strategy.
     */
    public static final class CanaryConfig extends StrategyConfig {
        private final int analysisIntervalSeconds;
        private final float initialWeight;
        private final float stepWeight;
        private final float maxWeight;
        private final Duration analysisDuration;
        private final Map<String, String> matchLabels;
        private final boolean pauseOnPromotion;
        private final CanaryMetric[] metrics;

        private CanaryConfig(Builder b) {
            super(b);
            this.analysisIntervalSeconds = b.analysisIntervalSeconds;
            this.initialWeight = b.initialWeight;
            this.stepWeight = b.stepWeight;
            this.maxWeight = b.maxWeight;
            this.analysisDuration = b.analysisDuration;
            this.matchLabels = Map.copyOf(b.matchLabels);
            this.pauseOnPromotion = b.pauseOnPromotion;
            this.metrics = b.metrics.toArray(new CanaryMetric[0]);
        }

        public int analysisIntervalSeconds() { return analysisIntervalSeconds; }
        public float initialWeight() { return initialWeight; }
        public float stepWeight() { return stepWeight; }
        public float maxWeight() { return maxWeight; }
        public Duration analysisDuration() { return analysisDuration; }
        public Map<String, String> matchLabels() { return matchLabels; }
        public boolean pauseOnPromotion() { return pauseOnPromotion; }
        public CanaryMetric[] metrics() { return metrics; }

        public static Builder builder() { return new Builder(); }

        public static final class Builder extends StrategyConfig.Builder<Builder> {
            private int analysisIntervalSeconds = 60;
            private float initialWeight = 5.0f;
            private float stepWeight = 10.0f;
            private float maxWeight = 100.0f;
            private Duration analysisDuration = Duration.ofMinutes(10);
            private final Map<String, String> matchLabels = new java.util.LinkedHashMap<>();
            private boolean pauseOnPromotion = false;
            private final java.util.List<CanaryMetric> metrics = new java.util.ArrayList<>();

            public Builder() { type(StrategyType.CANARY); }

            public Builder analysisInterval(int seconds) { this.analysisIntervalSeconds = seconds; return this; }
            public Builder initialWeight(float weight) { this.initialWeight = weight; return this; }
            public Builder stepWeight(float step) { this.stepWeight = step; return this; }
            public Builder maxWeight(float max) { this.maxWeight = max; return this; }
            public Builder analysisDuration(Duration duration) { this.analysisDuration = duration; return this; }
            public Builder matchLabel(String key, String value) { this.matchLabels.put(key, value); return this; }
            public Builder pauseOnPromotion(boolean pause) { this.pauseOnPromotion = pause; return this; }
            public Builder metric(CanaryMetric metric) { this.metrics.add(metric); return this; }

            public CanaryConfig build() {
                return new CanaryConfig(this);
            }
        }
    }

    /**
     * Canary analysis metric specification.
     */
    public static final class CanaryMetric {
        private final String name;
        private final String query;
        private final float threshold;
        private final MetricDirection direction;
        private final int interval;

        public enum MetricDirection {
            INCREASE, DECREASE, EITHER
        }

        private CanaryMetric(String name, String query, float threshold, MetricDirection direction, int interval) {
            this.name = Objects.requireNonNull(name, "name");
            this.query = Objects.requireNonNull(query, "query");
            this.threshold = threshold;
            this.direction = direction;
            this.interval = interval;
        }

        public static CanaryMetric of(String name, String query, float threshold) {
            return new CanaryMetric(name, query, threshold, MetricDirection.EITHER, 60);
        }

        public static CanaryMetric errorRate(float threshold) {
            return new CanaryMetric("error-rate", "rate(http_requests_total{status=~\"5..\"}[5m])", threshold, MetricDirection.DECREASE, 60);
        }

        public static CanaryMetric latency(String name, String query, float threshold) {
            return new CanaryMetric(name, query, threshold, MetricDirection.DECREASE, 60);
        }

        public String name() { return name; }
        public String query() { return query; }
        public float threshold() { return threshold; }
        public MetricDirection direction() { return direction; }
        public int interval() { return interval; }
    }

    // ========== Factory Methods ==========

    /**
     * Create a default rolling update configuration.
     */
    public static RollingUpdateConfig rollingUpdate() {
        return RollingUpdateConfig.builder().build();
    }

    /**
     * Create a blue-green deployment configuration.
     */
    public static BlueGreenConfig blueGreen() {
        return BlueGreenConfig.builder().build();
    }

    /**
     * Create a canary deployment configuration.
     */
    public static CanaryConfig canary() {
        return CanaryConfig.builder().build();
    }
}
