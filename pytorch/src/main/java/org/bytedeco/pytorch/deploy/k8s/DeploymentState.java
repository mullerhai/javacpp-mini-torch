package org.bytedeco.pytorch.deploy.k8s;

import java.time.Instant;

/**
 * Current state of a deployment.
 */
public final class DeploymentState {
    private final String name;
    private final String namespace;
    private final String currentImage;
    private final String previousImage;
    private final String strategy;
    private final DeploymentStatus status;
    private final Instant lastUpdate;

    public enum DeploymentStatus {
        PENDING, IN_PROGRESS, COMPLETED, FAILED, ROLLED_BACK
    }

    public DeploymentState(String name, String namespace, String currentImage,
                           String previousImage, String strategy,
                           DeploymentStatus status, Instant lastUpdate) {
        this.name = name;
        this.namespace = namespace;
        this.currentImage = currentImage;
        this.previousImage = previousImage;
        this.strategy = strategy;
        this.status = status;
        this.lastUpdate = lastUpdate;
    }

    public String name() { return name; }
    public String namespace() { return namespace; }
    public String currentImage() { return currentImage; }
    public String previousImage() { return previousImage; }
    public String strategy() { return strategy; }
    public DeploymentStatus status() { return status; }
    public Instant lastUpdate() { return lastUpdate; }
}
