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
import java.util.*;




/**
 * Deployment event for listeners.
 */
public final class DeploymentEvent {
    private final Type type;
    private final String deploymentName;
    private final String strategy;
    private final Instant timestamp;
    private final Object payload;
    private final Throwable error;

    public enum Type {
        DEPLOYING, COMPLETED, FAILED, ROLLED_BACK, ROLLBACK_STARTED, ROLLBACK_FAILED,
        PAUSED, RESUMED, ABORTED
    }

    public DeploymentEvent(Type type, String deploymentName, String strategy,
                          Object payload, Throwable error) {
        this.type = type;
        this.deploymentName = deploymentName;
        this.strategy = strategy;
        this.timestamp = Instant.now();
        this.payload = payload;
        this.error = error;
    }

    public static DeploymentEvent deploying(String name, String strategy, String image) {
        return new DeploymentEvent(Type.DEPLOYING, name, strategy, image, null);
    }

    public static DeploymentEvent completed(String name, String strategy, DeploymentResult result) {
        return new DeploymentEvent(Type.COMPLETED, name, strategy, result, null);
    }

    public static DeploymentEvent failed(String name, String strategy, Throwable error) {
        return new DeploymentEvent(Type.FAILED, name, strategy, null, error);
    }

    public static DeploymentEvent rolledBack(String name, String strategy) {
        return new DeploymentEvent(Type.ROLLED_BACK, name, strategy, null, null);
    }

    public static DeploymentEvent rolledBack(String name) {
        return new DeploymentEvent(Type.ROLLED_BACK, name, null, null, null);
    }

    public static DeploymentEvent rollbackStarted(String name) {
        return new DeploymentEvent(Type.ROLLBACK_STARTED, name, null, null, null);
    }

    public static DeploymentEvent rollbackFailed(String name, Throwable error) {
        return new DeploymentEvent(Type.ROLLBACK_FAILED, name, null, null, error);
    }

    public Type type() { return type; }
    public String deploymentName() { return deploymentName; }
    public String strategy() { return strategy; }
    public Instant timestamp() { return timestamp; }
    public Object payload() { return payload; }
    public Throwable error() { return error; }

    public String toString() {
        return String.format("DeploymentEvent{type=%s, name='%s', strategy=%s, time=%s}",
                type, deploymentName, strategy, timestamp);
    }
}


