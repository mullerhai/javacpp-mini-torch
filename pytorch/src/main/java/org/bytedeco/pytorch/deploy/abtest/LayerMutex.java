/*
 * Layer mutex / conflict detector — enforces that experiments in the same
 * layer don't double-expose units AND that experiments across layers have
 * valid combinations (e.g. two experiments trying to override the same
 * parameter to conflicting values).
 *
 * Industry rules:
 *   1. Within a layer: only one experiment may assign a unit.
 *      (enforced by reserved-range allocator in {@link ExperimentLayer}).
 *   2. Across layers: experiments are independent by hash. BUT they may
 *      specify parameter overlays that conflict. Industry platforms
 *      expose a "conflict detection" tool that scans parameters
 *      declared by experiments and warns about collisions.
 *   3. Mutual exclusion flags: certain experiments (e.g. "kill old model")
 *      must never run with others ("new model rollout").
 *
 * Mirrors:
 *   - Meta XP "holdback overlap" detection
 *   - ByteDance Libra "层互斥校验"
 *   - Google Ads "mutually exclusive experiment" framework
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Detects inter-layer / cross-experiment parameter collisions and
 * declares mutex relationships.
 */
public final class LayerMutex {

    private LayerMutex() {}

    /**
     * Detect conflicting parameter overlays across two experiments
     * whose variant parameter maps intersect on the same key with
     * different values.
     *
     * <p>This is a static check on experiment specs — does not require
     * running the manager.
     */
    public static List<Conflict> detectParameterConflicts(List<Experiment> experiments) {
        Objects.requireNonNull(experiments, "experiments");
        List<Conflict> out = new ArrayList<>();
        for (int i = 0; i < experiments.size(); i++) {
            for (int j = i + 1; j < experiments.size(); j++) {
                Experiment a = experiments.get(i);
                Experiment b = experiments.get(j);
                out.addAll(detectPairConflicts(a, b));
            }
        }
        return out;
    }

    private static List<Conflict> detectPairConflicts(Experiment a, Experiment b) {
        Map<String, Set<String>> keysA = collectKeys(a);
        Map<String, Set<String>> keysB = collectKeys(b);
        List<Conflict> out = new ArrayList<>();
        for (Map.Entry<String, Set<String>> entry : keysA.entrySet()) {
            String key = entry.getKey();
            Set<String> vsB = keysB.get(key);
            if (vsB == null) continue;
            Set<String> intersection = new LinkedHashSet<>(entry.getValue());
            intersection.retainAll(vsB);
            if (intersection.isEmpty()) {
                // Same key, different values across variants.
                out.add(new Conflict(a.id(), b.id(), key,
                        new ArrayList<>(entry.getValue()),
                        new ArrayList<>(vsB),
                        ConflictKind.PARAMETER_VALUE_CONFLICT));
            }
        }
        return out;
    }

    private static Map<String, Set<String>> collectKeys(Experiment e) {
        Map<String, Set<String>> keys = new LinkedHashMap<>();
        for (Variant v : e.variants()) {
            for (String k : v.parameters().keySet()) {
                keys.computeIfAbsent(k, x -> new LinkedHashSet<>()).add(v.parameters().get(k));
            }
        }
        return keys;
    }

    /**
     * Detect capacity violations when the requested total traffic percent
     * exceeds 100% within any single layer.
     */
    public static List<CapacityViolation> detectCapacity(List<ExperimentLayer> layers) {
        Objects.requireNonNull(layers, "layers");
        List<CapacityViolation> out = new ArrayList<>();
        for (ExperimentLayer layer : layers) {
            double used = layer.usedTrafficPercent();
            if (used > 100.0 + 1e-9) {
                out.add(new CapacityViolation(layer.id(), used, layer.experiments()));
            }
        }
        return out;
    }

    /** Two experiments overlap on the same diversion unit if their bucket
     *  windows intersect in the same layer. Caller is responsible for
     *  checking whether the (layerId, unitId) tuple was already assigned. */
    public static boolean overlapsBucketWindow(Experiment a, Experiment b, String unitId) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        if (!a.layerId().equals(b.layerId())) return false;
        long bucketA = BucketAssigner.bucketOf(a.salt(), unitId, a.bucketCount());
        long bucketB = BucketAssigner.bucketOf(b.salt(), unitId, b.bucketCount());
        return bucketA == bucketB; // same bucket -> both could enter if sum(percent)>bucket_pos
    }

    /** Conflict record. */
    public static final class Conflict {
        public final String experimentA;
        public final String experimentB;
        public final String parameterKey;
        public final List<String> valuesA;
        public final List<String> valuesB;
        public final ConflictKind kind;

        public Conflict(String experimentA, String experimentB, String parameterKey,
                        List<String> valuesA, List<String> valuesB, ConflictKind kind) {
            this.experimentA = experimentA;
            this.experimentB = experimentB;
            this.parameterKey = parameterKey;
            this.valuesA = Collections.unmodifiableList(valuesA);
            this.valuesB = Collections.unmodifiableList(valuesB);
            this.kind = kind;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "Conflict{exp=[%s,%s] key=%s valuesA=%s valuesB=%s}",
                    experimentA, experimentB, parameterKey, valuesA, valuesB);
        }
    }

    public enum ConflictKind {
        /** Same parameter key, different values. */
        PARAMETER_VALUE_CONFLICT
    }

    public static final class CapacityViolation {
        public final String layerId;
        public final double usedPercent;
        public final List<Experiment> experiments;

        public CapacityViolation(String layerId, double usedPercent, List<Experiment> experiments) {
            this.layerId = layerId;
            this.usedPercent = usedPercent;
            this.experiments = Collections.unmodifiableList(experiments);
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "CapacityViolation{layer=%s used=%.2f%%}",
                    layerId, usedPercent);
        }
    }
}