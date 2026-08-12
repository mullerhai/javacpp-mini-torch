/*
 * Time-series metric (ordered list of MetricPoint).
 */
package org.bytedeco.pytorch.deploy.integrations;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Time-series metric: (name, ordered history).
 */
public final class MetricSeries {
    public final String name;
    public final List<MetricPoint> points;

    public MetricSeries(String name, List<MetricPoint> points) {
        this.name = Objects.requireNonNull(name, "name");
        this.points = Collections.unmodifiableList(new ArrayList<>(points));
    }

    public MetricSeries append(MetricPoint p) {
        if (!p.name.equals(name)) {
            throw new IllegalArgumentException("name mismatch: " + name + " vs " + p.name);
        }
        List<MetricPoint> next = new ArrayList<>(points);
        next.add(p);
        return new MetricSeries(name, next);
    }
}
