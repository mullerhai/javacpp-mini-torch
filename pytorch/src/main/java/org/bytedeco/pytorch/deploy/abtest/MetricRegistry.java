/*
 * Metric registry — typed catalog of metrics an organization tracks.
 *
 * Mirrors:
 *   - Meta XP / Microsoft ExP "metric store"
 *   - ByteDance "指标库"
 *   - Google internal experiment metric catalog
 *
 * The registry acts as the single source of truth for metric type, units
 * and direction so that analyzer / dashboard / guardrail modules all
 * agree on what "ctr" or "latency_p99" means.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe in-memory metric registry.
 */
public final class MetricRegistry {

    private final ConcurrentHashMap<String, MetricDefinition> byKey = new ConcurrentHashMap<>();

    public MetricRegistry register(MetricDefinition def) {
        Objects.requireNonNull(def, "def");
        MetricDefinition prev = byKey.putIfAbsent(def.key, def);
        if (prev != null && prev != def) {
            throw new IllegalStateException("duplicate metric key: " + def.key);
        }
        return this;
    }

    public MetricDefinition get(String key) {
        return byKey.get(key);
    }

    public boolean contains(String key) {
        return byKey.containsKey(key);
    }

    public List<MetricDefinition> list() {
        return List.copyOf(byKey.values());
    }

    public List<MetricDefinition> primaryMetrics() {
        return byKey.values().stream().filter(d -> d.primary).toList();
    }

    public List<MetricDefinition> guardrailMetrics() {
        return byKey.values().stream().filter(d -> d.guardrail).toList();
    }

    public void clear() { byKey.clear(); }
}
