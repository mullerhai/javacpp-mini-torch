/*
 * Canonical metric point / series / parameter / artifact models.
 *
 * These wrap the wire-format encoding for ClearML / MLflow / Kubeflow.
 */
package org.bytedeco.pytorch.deploy.integrations;

import java.time.Instant;
import java.util.Objects;

/**
 * Scalar metric point: (name, value, step, timestamp).
 */
public final class MetricPoint {
    public final String name;
    public final double value;
    public final long step;
    public final Instant timestamp;

    public MetricPoint(String name, double value, long step) {
        this(name, value, step, Instant.now());
    }

    public MetricPoint(String name, double value, long step, Instant timestamp) {
        this.name = Objects.requireNonNull(name, "name");
        this.value = value;
        this.step = step;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.ROOT, "MetricPoint{%s=%g @step=%d}", name, value, step);
    }
}
