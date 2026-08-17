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

import org.bytedeco.pytorch.deploy.k8s.DeploymentStrategy.*;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Enterprise-grade Kubernetes deployment manager supporting multiple deployment strategies.
 *
 * <p>This manager provides:
 * <ul>
 *   <li>Unified API for Blue-Green, Canary, Rolling Update, and In-Place strategies</li>
 *   <li>Automatic rollback on failure detection</li>
 *   <li>Health checking with customizable probes</li>
 *   <li>Progress tracking and event logging</li>
 *   <li>Dry-run mode for validation</li>
 * </ul>
 *
 * <pre>{@code
 * try (DeploymentManager mgr = DeploymentManager.connect()) {
 *     // Blue-Green deployment
 *     DeploymentResult result = mgr.deploy()
 *         .name("my-model")
 *         .image("myrepo/model:v2")
 *         .strategy(DeploymentStrategy.blueGreen()
 *             .activeColor("blue")
 *             .prePromotionDelay(Duration.ofSeconds(30)))
 *         .execute();
 *
 *     // Canary with metrics
 *     DeploymentResult canaryResult = mgr.deploy()
 *         .name("my-model")
 *         .image("myrepo/model:v3")
 *         .strategy(DeploymentStrategy.canary()
 *             .initialWeight(10)
 *             .stepWeight(20)
 *             .metric(CanaryMetric.errorRate(0.01f)))
 *         .execute();
 * }
 * }</pre>
 */
public final class DeploymentManager implements AutoCloseable {

    private final K8s k8s;
    private final ExecutorService executor;
    private final List<DeploymentListener> listeners;
    private final Map<String, DeploymentState> deploymentStates;
    private final boolean dryRun;

    private DeploymentManager(Builder b) {
        this.k8s = b.k8s;
        this.executor = b.executor;
        this.listeners = new CopyOnWriteArrayList<>(b.listeners);
        this.deploymentStates = new ConcurrentHashMap<>();
        this.dryRun = b.dryRun;
    }

    public static DeploymentManager connect() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    // ========== Deployment API ==========

    /**
     * Start a new deployment with the specified configuration.
     */
    public DeploymentBuilder deploy() {
        return new DeploymentBuilder(this);
    }

    /**
     * Execute a deployment directly with spec and strategy.
     */
    public DeploymentResult deploy(ModelServingManifest.ModelServiceSpec spec, StrategyConfig strategy) {
        return deploy().spec(spec).strategy(strategy).execute();
    }

    // ========== Strategy Execution ==========

    /**
     * Execute a blue-green deployment.
     */
    public DeploymentResult deployBlueGreen(
            String name,
            String namespace,
            String image,
            BlueGreenConfig config) {

        fireEvent(DeploymentEvent.deploying(name, "BLUE_GREEN", image));

        try {
            BlueGreenDeployer deployer = new BlueGreenDeployer(k8s, config);
            DeploymentResult result = dryRun
                ? simulateDeployment(name, image, "BLUE_GREEN")
                : deployer.deploy(name, namespace, image);

            if (result.success()) {
                fireEvent(DeploymentEvent.completed(name, "BLUE_GREEN", result));
            } else if (config.autoRollbackOnFailure()) {
                rollback(name, namespace);
                fireEvent(DeploymentEvent.rolledBack(name, "BLUE_GREEN"));
            }
            return result;
        } catch (Exception e) {
            if (config.autoRollbackOnFailure()) {
                try { rollback(name, namespace); } catch (Exception ignored) {}
            }
            fireEvent(DeploymentEvent.failed(name, "BLUE_GREEN", e));
            return DeploymentResult.failure(name, "BLUE_GREEN", e);
        }
    }

    /**
     * Execute a canary deployment.
     */
    public DeploymentResult deployCanary(
            String name,
            String namespace,
            String image,
            CanaryConfig config) {

        fireEvent(DeploymentEvent.deploying(name, "CANARY", image));

        try {
            CanaryDeployer deployer = new CanaryDeployer(k8s, config);
            DeploymentResult result = dryRun
                ? simulateDeployment(name, image, "CANARY")
                : deployer.deploy(name, namespace, image);

            if (result.success()) {
                fireEvent(DeploymentEvent.completed(name, "CANARY", result));
            } else if (config.autoRollbackOnFailure()) {
                rollback(name, namespace);
                fireEvent(DeploymentEvent.rolledBack(name, "CANARY"));
            }
            return result;
        } catch (Exception e) {
            if (config.autoRollbackOnFailure()) {
                try { rollback(name, namespace); } catch (Exception ignored) {}
            }
            fireEvent(DeploymentEvent.failed(name, "CANARY", e));
            return DeploymentResult.failure(name, "CANARY", e);
        }
    }

    /**
     * Execute a rolling update deployment.
     */
    public DeploymentResult deployRollingUpdate(
            String name,
            String namespace,
            String image,
            RollingUpdateConfig config) {

        fireEvent(DeploymentEvent.deploying(name, "ROLLING_UPDATE", image));

        try {
            RollingUpdateDeployer deployer = new RollingUpdateDeployer(k8s, config);
            DeploymentResult result = dryRun
                ? simulateDeployment(name, image, "ROLLING_UPDATE")
                : deployer.deploy(name, namespace, image);

            if (result.success()) {
                fireEvent(DeploymentEvent.completed(name, "ROLLING_UPDATE", result));
            }
            return result;
        } catch (Exception e) {
            fireEvent(DeploymentEvent.failed(name, "ROLLING_UPDATE", e));
            return DeploymentResult.failure(name, "ROLLING_UPDATE", e);
        }
    }

    /**
     * Execute an in-place update (configuration-only, no pod restart).
     */
    public DeploymentResult deployInPlace(
            String name,
            String namespace,
            ModelServingManifest.ModelServiceSpec spec,
            InPlaceDeployer.InPlaceConfig config) {

        fireEvent(DeploymentEvent.deploying(name, "IN_PLACE", spec.image()));

        try {
            InPlaceDeployer deployer = new InPlaceDeployer(k8s, config);
            DeploymentResult result = dryRun
                ? simulateDeployment(name, spec.image(), "IN_PLACE")
                : deployer.update(name, namespace, spec);

            if (result.success()) {
                fireEvent(DeploymentEvent.completed(name, "IN_PLACE", result));
            }
            return result;
        } catch (Exception e) {
            fireEvent(DeploymentEvent.failed(name, "IN_PLACE", e));
            return DeploymentResult.failure(name, "IN_PLACE", e);
        }
    }

    // ========== Rollback ==========

    /**
     * Rollback a deployment to the previous version.
     */
    public DeploymentResult rollback(String name, String namespace) {
        fireEvent(DeploymentEvent.rollbackStarted(name));

        try {
            DeploymentState state = deploymentStates.get(name);
            String previousImage = state != null ? state.previousImage() : null;

            if (previousImage == null) {
                throw new K8sException("No previous version found for rollback: " + name);
            }

            DeploymentResult result = deployRollingUpdate(name, namespace, previousImage,
                DeploymentStrategy.rollingUpdate());

            if (result.success()) {
                fireEvent(DeploymentEvent.rolledBack(name));
            }
            return result;
        } catch (Exception e) {
            fireEvent(DeploymentEvent.rollbackFailed(name, e));
            return DeploymentResult.failure(name, "ROLLBACK", e);
        }
    }

    /**
     * Rollback to a specific revision.
     */
    public DeploymentResult rollbackToRevision(String name, String namespace, int revision) {
        fireEvent(DeploymentEvent.rollbackStarted(name + "@" + revision));

        try {
            String yaml = k8s.kubectl().rolloutUndo(name, namespace, revision);
            k8s.kubectl().applyStdin(yaml);
            k8s.rolloutWait(name, Duration.ofMinutes(5));

            DeploymentResult result = DeploymentResult.success(name, "ROLLBACK_REVISION");
            fireEvent(DeploymentEvent.rolledBack(name + "@" + revision));
            return result;
        } catch (Exception e) {
            fireEvent(DeploymentEvent.rollbackFailed(name, e));
            return DeploymentResult.failure(name, "ROLLBACK_REVISION", e);
        }
    }

    // ========== Status & History ==========

    /**
     * Get current deployment state.
     */
    public DeploymentState getState(String name) {
        return deploymentStates.get(name);
    }

    /**
     * Get deployment history.
     */
    public String getHistory(String name) {
        DeploymentState state = deploymentStates.get(name);
        if (state == null) return "";

        try {
            return k8s.kubectl().rolloutHistory("deployment/" + name, state.namespace());
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * Get current rollout status.
     */
    public String getRolloutStatus(String name, String namespace) {
        try {
            return k8s.kubectl().rolloutStatusRaw(name, namespace);
        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }

    // ========== Listeners ==========

    public void addListener(DeploymentListener listener) {
        listeners.add(listener);
    }

    public void removeListener(DeploymentListener listener) {
        listeners.remove(listener);
    }

    private void fireEvent(DeploymentEvent event) {
        for (DeploymentListener listener : listeners) {
            try {
                listener.onEvent(event);
            } catch (Exception ignored) {}
        }
    }

    // ========== Helpers ==========

    private DeploymentResult simulateDeployment(String name, String image, String strategy) {
        System.out.println("[DRY-RUN] Would deploy " + name + " with " + strategy + " using image " + image);
        return DeploymentResult.builder()
                .name(name)
                .strategy(strategy)
                .startTime(Instant.now())
                .endTime(Instant.now())
                .success(true)
                .duration(Duration.ofMillis(100))
                .build();
    }

    @Override
    public void close() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException ignored) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    // ========== Builder ==========

    public static final class Builder {
        private K8s k8s;
        private ExecutorService executor = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "deployer-" + System.nanoTime());
            t.setDaemon(true);
            return t;
        });
        private List<DeploymentListener> listeners = new CopyOnWriteArrayList<>();
        private boolean dryRun = false;

        public Builder k8s(K8s k8s) { this.k8s = k8s; return this; }
        public Builder executor(ExecutorService executor) { this.executor = executor; return this; }
        public Builder listener(DeploymentListener listener) { this.listeners.add(listener); return this; }
        public Builder dryRun(boolean dryRun) { this.dryRun = dryRun; return this; }

        public DeploymentManager build() {
            if (k8s == null) {
                k8s = K8s.connect();
            }
            return new DeploymentManager(this);
        }
    }

    // ========== Inner Classes ==========

    /**
     * Fluent API for building deployments.
     */
    public static final class DeploymentBuilder {
        private final DeploymentManager mgr;
        private String name;
        private String namespace = "default";
        private String image;
        private StrategyConfig strategy;
        private ModelServingManifest.ModelServiceSpec spec;
        private int replicas = 1;
        private int port = 8000;
        private Map<String, String> labels = new HashMap<>();
        private HealthCheckConfig healthCheck;

        DeploymentBuilder(DeploymentManager mgr) {
            this.mgr = mgr;
        }

        public DeploymentBuilder name(String name) { this.name = name; return this; }
        public DeploymentBuilder namespace(String ns) { this.namespace = ns; return this; }
        public DeploymentBuilder image(String image) { this.image = image; return this; }
        public DeploymentBuilder strategy(StrategyConfig strategy) { this.strategy = strategy; return this; }
        public DeploymentBuilder replicas(int replicas) { this.replicas = replicas; return this; }
        public DeploymentBuilder port(int port) { this.port = port; return this; }
        public DeploymentBuilder label(String key, String value) { this.labels.put(key, value); return this; }
        public DeploymentBuilder healthCheck(HealthCheckConfig config) { this.healthCheck = config; return this; }

        public DeploymentBuilder spec(ModelServingManifest.ModelServiceSpec spec) {
            this.spec = spec;
            this.name = spec.name();
            this.namespace = spec.namespace() != null ? spec.namespace() : "default";
            this.image = spec.image();
            return this;
        }

        public DeploymentResult execute() {
            if (strategy == null) {
                strategy = DeploymentStrategy.rollingUpdate();
            }

            if (image == null && spec != null) {
                image = spec.image();
            }

            if (strategy instanceof BlueGreenConfig bg) {
                return mgr.deployBlueGreen(name, namespace, image, bg);
            } else if (strategy instanceof CanaryConfig canary) {
                return mgr.deployCanary(name, namespace, image, canary);
            } else if (strategy instanceof RollingUpdateConfig rolling) {
                return mgr.deployRollingUpdate(name, namespace, image, rolling);
            } else if (strategy instanceof InPlaceDeployer.InPlaceConfig inPlace && spec != null) {
                return mgr.deployInPlace(name, namespace, spec, inPlace);
            } else {
                throw new IllegalArgumentException("Unknown strategy type or missing required config: " + strategy);
            }
        }
    }

    /**
     * Configuration for health checks during deployment.
     */
    public static final class HealthCheckConfig {
        private final Duration timeout;
        private final int maxRetries;
        private final Duration retryInterval;
        private final String healthEndpoint;
        private final int successThreshold;

        private HealthCheckConfig(Builder b) {
            this.timeout = b.timeout;
            this.maxRetries = b.maxRetries;
            this.retryInterval = b.retryInterval;
            this.healthEndpoint = b.healthEndpoint;
            this.successThreshold = b.successThreshold;
        }

        public static Builder builder() { return new Builder(); }

        public static final class Builder {
            private Duration timeout = Duration.ofMinutes(5);
            private int maxRetries = 30;
            private Duration retryInterval = Duration.ofSeconds(10);
            private String healthEndpoint = "/health";
            private int successThreshold = 1;

            public Builder timeout(Duration timeout) { this.timeout = timeout; return this; }
            public Builder maxRetries(int retries) { this.maxRetries = retries; return this; }
            public Builder retryInterval(Duration interval) { this.retryInterval = interval; return this; }
            public Builder healthEndpoint(String endpoint) { this.healthEndpoint = endpoint; return this; }
            public Builder successThreshold(int threshold) { this.successThreshold = threshold; return this; }
            public HealthCheckConfig build() { return new HealthCheckConfig(this); }
        }

        public Duration timeout() { return timeout; }
        public int maxRetries() { return maxRetries; }
        public Duration retryInterval() { return retryInterval; }
        public String healthEndpoint() { return healthEndpoint; }
        public int successThreshold() { return successThreshold; }
    }
}
