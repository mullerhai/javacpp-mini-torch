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

import org.bytedeco.pytorch.deploy.k8s.DeploymentStrategy.BlueGreenConfig;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;

/**
 * Blue-Green deployment implementation for enterprise-grade zero-downtime releases.
 *
 * <p>This deployer creates a full duplicate environment (blue/green) and performs
 * instant traffic switching after validation.
 *
 * <ul>
 *   <li>Creates parallel deployment with color-coded labels</li>
 *   <li>Waits for all replicas to be ready</li>
 *   <li>Validates with pre-promotion checks</li>
 *   <li>Switches service selector atomically</li>
 *   <li>Optionally auto-promotes after delay</li>
 * </ul>
 */
public final class BlueGreenDeployer {

    private final K8s k8s;
    private final BlueGreenConfig config;

    public BlueGreenDeployer(K8s k8s, BlueGreenConfig config) {
        this.k8s = Objects.requireNonNull(k8s, "k8s");
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Execute blue-green deployment.
     */
    public DeploymentResult deploy(String name, String namespace, String image) throws Exception {
        Instant start = Instant.now();
        String targetColor = getInactiveColor(name, namespace);
        String activeColor = getActiveColor(name, namespace);

        System.out.println("[BlueGreen] Starting deployment: name=" + name + ", target=" + targetColor + ", active=" + activeColor);

        // Step 1: Deploy the inactive version
        String inactiveName = name + "-" + targetColor;
        deployVersion(inactiveName, namespace, image, targetColor);

        // Step 2: Wait for readiness
        waitForReady(inactiveName, namespace);

        // Step 3: Pre-promotion checks
        if (!runPrePromotionChecks(name + "-" + targetColor, namespace)) {
            throw new K8sException("Pre-promotion checks failed for " + inactiveName);
        }

        // Step 4: Switch traffic
        switchTraffic(name, namespace, targetColor);

        // Step 5: Scale down old version (if not auto-promoting immediately)
        if (!config.autoPromote() && activeColor != null) {
            String activeName = name + "-" + activeColor;
            k8s.scale(activeName, 0);
        }

        // Step 6: Wait for promotion delay if configured
        if (config.autoPromote() && !config.prePromotionDelay().isZero()) {
            Thread.sleep(config.prePromotionDelay().toMillis());
        }

        Duration duration = Duration.between(start, Instant.now());
        System.out.println("[BlueGreen] Deployment completed: name=" + name + ", color=" + targetColor + ", duration=" + duration);

        return DeploymentResult.builder()
                .name(name)
                .strategy("BLUE_GREEN")
                .startTime(start)
                .endTime(Instant.now())
                .success(true)
                .duration(duration)
                .metadata("targetColor", targetColor)
                .metadata("activeColor", activeColor != null ? activeColor : "")
                .metadata("image", image)
                .build();
    }

    /**
     * Switch traffic to the specified color.
     */
    public void switchTraffic(String name, String namespace, String color) throws Exception {
        System.out.println("[BlueGreen] Switching traffic to " + color);

        switch (config.trafficPolicy()) {
            case "service-selector" -> switchServiceSelector(name, namespace, color);
            case "istio" -> switchIstio(name, namespace, color);
            case "nginx" -> switchNginxIngress(name, namespace, color);
            default -> switchServiceSelector(name, namespace, color);
        }

        System.out.println("[BlueGreen] Traffic switched to " + color);
    }

    private void switchServiceSelector(String name, String namespace, String color) throws Exception {
        // Update the service selector to point to the new deployment
        Map<String, Object> servicePatch = Map.of(
                "spec", Map.of("selector", Map.of(
                        "app", name,
                        "color", color
                ))
        );

        String patchYaml = Resources.toYaml(servicePatch);
        k8s.kubectl().patch("service", name, namespace, "merge", patchYaml);
    }

    private void switchIstio(String name, String namespace, String color) throws Exception {
        // Create/update VirtualService to route 100% traffic to new version
        Map<String, Object> vs = Map.of(
                "apiVersion", "networking.istio.io/v1",
                "kind", "VirtualService",
                "metadata", Map.of("name", name, "namespace", namespace),
                "spec", Map.of(
                        "hosts", List.of(name),
                        "http", List.of(Map.of(
                                "route", List.of(Map.of(
                                        "destination", Map.of(
                                                "host", name + "-" + color,
                                                "port", Map.of("number", 8000)
                                        ),
                                        "weight", 100
                                ))
                        ))
                )
        );

        k8s.applyManifest(Manifest.of(vs));
    }

    private void switchNginxIngress(String name, String namespace, String color) throws Exception {
        // Annotate the service or use nginx ingress annotation
        Map<String, Object> annotationPatch = Map.of(
                "metadata", Map.of("annotations", Map.of(
                        "nginx.ingress.kubernetes.io/canary-weight", "100"
                ))
        );

        String patchYaml = Resources.toYaml(annotationPatch);
        k8s.kubectl().patch("service", name, namespace, "merge", patchYaml);
    }

    /**
     * Get the currently active color (blue/green).
     */
    public String getActiveColor(String name, String namespace) {
        try {
            return k8s.kubectl().cmd("get", "service", name)
                    .ns(namespace)
                    .output("jsonpath={.spec.selector.color}")
                    .runOk();
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Get the inactive color (the one to deploy to).
     */
    public String getInactiveColor(String name, String namespace) {
        String active = getActiveColor(name, namespace);
        if ("blue".equals(active)) return "green";
        return "blue";
    }

    /**
     * Abort the current blue-green deployment.
     */
    public void abort(String name, String namespace) throws Exception {
        String inactiveColor = getInactiveColor(name, namespace);
        String inactiveName = name + "-" + inactiveColor;

        System.out.println("[BlueGreen] Aborting deployment: " + inactiveName);

        // Scale down the inactive deployment
        k8s.scale(inactiveName, 0);

        // Clean up if needed
        // k8s.kubectl().delete("deployment", inactiveName, namespace);
    }

    private void deployVersion(String deploymentName, String namespace, String image, String color) throws Exception {
        ModelServingManifest.ModelServiceSpec spec = ModelServingManifest.ModelServiceSpec.builder(deploymentName, image)
                .namespace(namespace)
                .label("app", deploymentName)
                .label("color", color)
                .label("deploy.jnitorch.io/strategy", "blue-green")
                .replicas(config.maxSurge() > 0 ? config.maxSurge() : 1)
                .build();

        k8s.deployModelService(spec);
        System.out.println("[BlueGreen] Deployed: " + deploymentName + " with image " + image);
    }

    private void waitForReady(String name, String namespace) throws Exception {
        Duration timeout = config.timeout();
        Instant deadline = Instant.now().plus(timeout);

        while (Instant.now().isBefore(deadline)) {
            String status = k8s.kubectl().rolloutStatusRaw("deployment/" + name, namespace);

            if (status.contains("successfully rolled out")) {
                System.out.println("[BlueGreen] Ready: " + name);
                return;
            }

            if (status.contains("error") || status.contains("failed")) {
                throw new K8sException("Deployment failed: " + status);
            }

            Thread.sleep(config.pollingInterval().toMillis());
        }

        throw new K8sException("Timeout waiting for deployment: " + name);
    }

    private boolean runPrePromotionChecks(String name, String namespace) throws Exception {
        System.out.println("[BlueGreen] Running pre-promotion checks (" + config.prePromotionChecks() + " checks)");

        for (int i = 0; i < config.prePromotionChecks(); i++) {
            if (!checkPodHealth(name, namespace)) {
                System.out.println("[BlueGreen] Check " + (i + 1) + " failed");
                return false;
            }
            System.out.println("[BlueGreen] Check " + (i + 1) + " passed");

            if (i < config.prePromotionChecks() - 1) {
                Thread.sleep(config.pollingInterval().toMillis());
            }
        }

        return true;
    }

    private boolean checkPodHealth(String name, String namespace) {
        try {
            // Check deployment status
            String output = k8s.kubectl().cmd("get", "deployment", name)
                    .ns(namespace)
                    .output("jsonpath={.status.readyReplicas}")
                    .runOk();
            int readyReplicas = output.isBlank() ? 0 : Integer.parseInt(output.trim());

            output = k8s.kubectl().cmd("get", "deployment", name)
                    .ns(namespace)
                    .output("jsonpath={.spec.replicas}")
                    .runOk();
            int desiredReplicas = output.isBlank() ? 1 : Integer.parseInt(output.trim());

            return readyReplicas >= desiredReplicas;
        } catch (Exception e) {
            System.out.println("[BlueGreen] Health check error: " + e.getMessage());
            return false;
        }
    }
}
