package org.bytedeco.pytorch.deploy.k8s;

/**
 * Listener interface for deployment events.
 */
public interface DeploymentListener {
    void onEvent(DeploymentEvent event);
}