/*
 * Unified domain model for MLOps platform integration.
 *
 * Bridges the gap between BYTETensor's in-process abstractions
 * (Experiments, Trials, Metrics, Models, Pipelines) and the three
 * external platforms:
 *   - ClearML (Allegro AI)         : server + auto-agent + dashboard
 *   - MLflow                       : Tracking / Models / Registry
 *   - Kubeflow (Pipelines + Katib) : k8s-native orchestration
 *
 * The goal is to map to a *canonical* model that is a superset of the
 * three platforms — every field exposed by any one of them can be
 * represented canonically.
 *
 * Convention: all factories return immutable, side-effect-free objects
 * that can be serialized to the platform-specific wire format.
 */
package org.bytedeco.pytorch.deploy.integrations;

import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Canonical experiment / run representation.
 *
 * <p>Maps to:
 *   - ClearML: Task (project, name, type=TRAINING/TESTING/INFERENCE)
 *   - MLflow:  Run (experiment_id, run_name, run_id)
 *   - Kubeflow: Experiment + Trial (KF API)
 */
public final class CanonicalExperiment {

    public enum Type {
        TRAINING, TESTING, INFERENCE, OPTIMIZATION, A_B_TEST, DATA_PREP, OTHER
    }

    public final String id;
    public final String project;
    public final String name;
    public final Type type;
    public final String owner;
    public final String description;
    public final Instant createdAt;
    public final Instant startedAt;
    public final Instant finishedAt;
    public final String status;     // "queued|running|completed|failed|killed"
    public final Map<String, String> tags;
    public final Map<String, String> systemic;
    public final List<String> inputArtifacts;
    public final List<String> outputArtifacts;

    private CanonicalExperiment(Builder b) {
        this.id = Objects.requireNonNull(b.id, "id");
        this.project = b.project != null ? b.project : "default";
        this.name = b.name != null ? b.name : b.id;
        this.type = b.type != null ? b.type : Type.OTHER;
        this.owner = b.owner != null ? b.owner : "";
        this.description = b.description != null ? b.description : "";
        this.createdAt = b.createdAt != null ? b.createdAt : Instant.now();
        this.startedAt = b.startedAt;
        this.finishedAt = b.finishedAt;
        this.status = b.status != null ? b.status : "queued";
        this.tags = Collections.unmodifiableMap(new LinkedHashMap<>(b.tags));
        this.systemic = Collections.unmodifiableMap(new LinkedHashMap<>(b.systemic));
        this.inputArtifacts = List.copyOf(b.inputArtifacts);
        this.outputArtifacts = List.copyOf(b.outputArtifacts);
    }

    public static Builder builder(String id) { return new Builder(id); }

    public boolean isTerminal() {
        return "completed".equals(status) || "failed".equals(status) || "killed".equals(status);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "CanonicalExperiment{id=%s project=%s name=%s status=%s}",
                id, project, name, status);
    }

    public static final class Builder {
        private final String id;
        private String project;
        private String name;
        private Type type;
        private String owner;
        private String description;
        private Instant createdAt;
        private Instant startedAt;
        private Instant finishedAt;
        private String status;
        private final Map<String, String> tags = new LinkedHashMap<>();
        private final Map<String, String> systemic = new LinkedHashMap<>();
        private final java.util.List<String> inputArtifacts = new java.util.ArrayList<>();
        private final java.util.List<String> outputArtifacts = new java.util.ArrayList<>();

        private Builder(String id) { this.id = id; }

        public Builder project(String p) { this.project = p; return this; }
        public Builder name(String n) { this.name = n; return this; }
        public Builder type(Type t) { this.type = t; return this; }
        public Builder owner(String o) { this.owner = o; return this; }
        public Builder description(String d) { this.description = d; return this; }
        public Builder createdAt(Instant t) { this.createdAt = t; return this; }
        public Builder startedAt(Instant t) { this.startedAt = t; return this; }
        public Builder finishedAt(Instant t) { this.finishedAt = t; return this; }
        public Builder status(String s) { this.status = s; return this; }
        public Builder tag(String k, String v) { this.tags.put(k, v); return this; }
        public Builder systemic(String k, String v) { this.systemic.put(k, v); return this; }
        public Builder inputArtifact(String uri) { this.inputArtifacts.add(uri); return this; }
        public Builder outputArtifact(String uri) { this.outputArtifacts.add(uri); return this; }

        public CanonicalExperiment build() { return new CanonicalExperiment(this); }
    }
}
