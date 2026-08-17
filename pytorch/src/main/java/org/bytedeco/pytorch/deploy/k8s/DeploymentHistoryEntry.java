package org.bytedeco.pytorch.deploy.k8s;

import java.time.Instant;

/**
 * Deployment history entry.
 */
public final class DeploymentHistoryEntry {
    private final int revision;
    private final String changeCause;
    private final Instant createdAt;
    private final String image;

    public DeploymentHistoryEntry(int revision, String changeCause, Instant createdAt, String image) {
        this.revision = revision;
        this.changeCause = changeCause;
        this.createdAt = createdAt;
        this.image = image;
    }

    public int revision() { return revision; }
    public String changeCause() { return changeCause; }
    public Instant createdAt() { return createdAt; }
    public String image() { return image; }
}