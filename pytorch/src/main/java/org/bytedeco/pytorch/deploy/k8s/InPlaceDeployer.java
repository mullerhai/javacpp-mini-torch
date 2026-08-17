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
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * In-place update deployer that modifies configuration without pod replacement.
 *
 * <p>This deployer is useful for:
 * <ul>
 *   <li>Updating environment variables</li>
 *   <li>Changing resource limits</li>
 *   <li>Updating ConfigMaps/Secrets</li>
 *   <li>Adjusting scaling parameters</li>
 * </ul>
 */
public final class InPlaceDeployer {

    private final K8s k8s;
    private final InPlaceConfig config;

    public InPlaceDeployer(K8s k8s, InPlaceConfig config) {
        this.k8s = Objects.requireNonNull(k8s, "k8s");
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Update deployment configuration in-place.
     */
    public DeploymentResult update(String name, String namespace, ModelServingManifest.ModelServiceSpec newSpec) throws Exception {
        Instant start = Instant.now();
        System.out.println("[InPlace] Starting update: name=" + name);

        // Apply configuration changes via patch
        Map<String, Object> patch = buildPatch(newSpec);
        String patchYaml = Resources.toYaml(patch);

        k8s.kubectl().patch("deployment", name, namespace, "merge", patchYaml);

        // Wait for rollout to complete if needed
        if (config.waitForRollout()) {
            k8s.rolloutWait(name, config.timeout());
        }

        Duration duration = Duration.between(start, Instant.now());
        System.out.println("[InPlace] Update completed: name=" + name + ", duration=" + duration);

        return DeploymentResult.builder()
                .name(name)
                .strategy("IN_PLACE")
                .startTime(start)
                .endTime(Instant.now())
                .success(true)
                .duration(duration)
                .metadata("type", "configuration")
                .build();
    }

    /**
     * Scale deployment in-place.
     */
    public DeploymentResult scale(String name, String namespace, int replicas) throws Exception {
        Instant start = Instant.now();
        System.out.println("[InPlace] Scaling: name=" + name + ", replicas=" + replicas);

        k8s.scale(name, replicas);

        Duration duration = Duration.between(start, Instant.now());
        return DeploymentResult.builder()
                .name(name)
                .strategy("IN_PLACE")
                .startTime(start)
                .endTime(Instant.now())
                .success(true)
                .duration(duration)
                .metadata("action", "scale")
                .metadata("replicas", String.valueOf(replicas))
                .build();
    }

    /**
     * Update environment variables.
     */
    public void updateEnv(String name, String namespace, Map<String, String> envVars) throws Exception {
        Map<String, Object> patch = Map.of(
                "spec", Map.of(
                        "template", Map.of(
                                "spec", Map.of(
                                        "containers", List.of(Map.of(
                                                "name", name,
                                                "env", envVars.entrySet().stream()
                                                        .map(e -> Map.of("name", e.getKey(), "value", e.getValue()))
                                                        .toList()
                                        ))
                                )
                        )
                )
        );

        k8s.kubectl().patch("deployment", name, namespace, "merge", Resources.toYaml(patch));
    }

    /**
     * Update resources (CPU/memory limits).
     */
    public void updateResources(String name, String namespace,
                               String cpuRequest, String memoryRequest,
                               String cpuLimit, String memoryLimit) throws Exception {
        Map<String, Object> patch = Map.of(
                "spec", Map.of(
                        "template", Map.of(
                                "spec", Map.of(
                                        "containers", List.of(Map.of(
                                                "name", name,
                                                "resources", Map.of(
                                                        "requests", Map.of(
                                                                "cpu", cpuRequest != null ? cpuRequest : "100m",
                                                                "memory", memoryRequest != null ? memoryRequest : "128Mi"
                                                        ),
                                                        "limits", Map.of(
                                                                "cpu", cpuLimit != null ? cpuLimit : "500m",
                                                                "memory", memoryLimit != null ? memoryLimit : "512Mi"
                                                        )
                                                )
                                        ))
                                )
                        )
                )
        );

        k8s.kubectl().patch("deployment", name, namespace, "merge", Resources.toYaml(patch));
    }

    private Map<String, Object> buildPatch(ModelServingManifest.ModelServiceSpec spec) {
        return Map.of(
                "metadata", Map.of("annotations", Map.of(
                        "deploy.jnitorch.io/in-place", "true",
                        "deploy.jnitorch.io/timestamp", String.valueOf(System.currentTimeMillis())
                )),
                "spec", Map.of(
                        "replicas", spec.replicas(),
                        "template", Map.of(
                                "metadata", Map.of("labels", spec.labels()),
                                "spec", Map.of(
                                        "containers", List.of(Map.of(
                                                "name", spec.name(),
                                                "image", spec.image(),
                                                "env", spec.env().entrySet().stream()
                                                        .map(e -> Map.of("name", e.getKey(), "value", e.getValue()))
                                                        .toList()
                                        ))
                                )
                        )
                )
        );
    }

    /**
     * In-place update configuration.
     */
    public static final class InPlaceConfig extends DeploymentStrategy.StrategyConfig {
        private final boolean waitForRollout;
        private final boolean restartPods;

        private InPlaceConfig(Builder b) {
            super(b);
            this.waitForRollout = b.waitForRollout;
            this.restartPods = b.restartPods;
        }

        public boolean waitForRollout() { return waitForRollout; }
        public boolean restartPods() { return restartPods; }

        public static Builder builder() { return new Builder(); }

        public static final class Builder extends DeploymentStrategy.StrategyConfig.Builder<Builder> {
            private boolean waitForRollout = false;
            private boolean restartPods = false;

            public Builder() { type(DeploymentStrategy.StrategyType.IN_PLACE); }

            public Builder waitForRollout(boolean wait) { this.waitForRollout = wait; return this; }
            public Builder restartPods(boolean restart) { this.restartPods = restart; return this; }

            public InPlaceConfig build() { return new InPlaceConfig(this); }
        }
    }
}
