/*
 * MLflow integration.
 *
 * MLflow Tracking REST API:
 *   - POST  /api/2.0/mlflow/runs/create
 *   - POST  /api/2.0/mlflow/runs/log-metric
 *   - POST  /api/2.0/mlflow/runs/log-parameter
 *   - POST  /api/2.0/mlflow/runs/log-artifact
 *   - POST  /api/2.0/mlflow/runs/update
 *   - POST  /api/2.0/mlflow/runs/set-terminated
 *
 * MLflow Model Registry REST API:
 *   - POST  /api/2.0/mlflow/model-versions/create
 *
 * Auth: OAuth bearer token sent as `Authorization: Bearer ...` (MLflow
 *       supports this via the AUTH configuration).
 *
 * Reference: https://mlflow.org/docs/latest/rest-api.html
 */
package org.bytedeco.pytorch.deploy.integrations.mlflow;

import org.bytedeco.pytorch.deploy.integrations.Artifact;
import org.bytedeco.pytorch.deploy.integrations.CanonicalExperiment;
import org.bytedeco.pytorch.deploy.integrations.MetricPoint;
import org.bytedeco.pytorch.deploy.integrations.MlopsSink;
import org.bytedeco.pytorch.deploy.integrations.ExperimentParameter;
import org.bytedeco.pytorch.deploy.integrations.SinkConfig;
import org.bytedeco.pytorch.deploy.integrations.clearml.HttpJsonClient;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * MLflow sink.
 */
public final class MlflowSink implements MlopsSink {

    private SinkConfig config;
    private final HttpJsonClient http;
    private final Map<String, String> experimentIdMapping = new ConcurrentHashMap<>();
    private final Map<String, String> runNameToId = new ConcurrentHashMap<>();
    private String defaultExperimentId;

    public MlflowSink() { this(new HttpJsonClient()); }
    public MlflowSink(HttpJsonClient http) { this.http = Objects.requireNonNull(http); }

    @Override
    public String platformName() { return "mlflow"; }

    @Override
    public void configure(SinkConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        http.setAuthToken(config.authToken);
        // Resolve experiment id (creating it if missing).
        defaultExperimentId = ensureExperiment(config.projectNamespace);
    }

    private String ensureExperiment(String name) {
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("name", name);
        Map<String, Object> resp = http.postJson(endpoint("/api/2.0/mlflow/experiments/create"), req);
        Object id = resp.get("experiment_id");
        if (id == null && resp.get("error_code") != null) {
            // Already exists; look it up via search.
            req.put("view_type", "ACTIVE_ONLY");
            resp = http.postJson(endpoint("/api/2.0/mlflow/experiments/search"), req);
            @SuppressWarnings("unchecked")
            List<Object> exps = (List<Object>) resp.get("experiments");
            if (exps != null && !exps.isEmpty()) {
                @SuppressWarnings("unchecked")
                Map<String, Object> e = (Map<String, Object>) exps.get(0);
                id = e.get("experiment_id");
            }
        }
        if (id == null) {
            throw new IllegalStateException("MLflow experiment lookup failed: " + resp);
        }
        return id.toString();
    }

    private String endpoint(String path) {
        String base = config.endpointUrl != null ? config.endpointUrl : "http://localhost:5000";
        if (base.endsWith("/")) base = base.substring(0, base.length() - 1);
        return base + path;
    }

    @Override
    public String startExperiment(CanonicalExperiment exp) {
        requireConfig();
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("experiment_id", defaultExperimentId);
        req.put("run_name", exp.name);
        req.put("start_time", exp.createdAt != null ? exp.createdAt.toEpochMilli() : Instant.now().toEpochMilli());
        req.put("run_id", exp.id != null ? exp.id : UUID.randomUUID().toString());
        req.put("tags", mlTags(exp));

        Map<String, Object> resp = http.postJson(endpoint("/api/2.0/mlflow/runs/create"), req);
        Map<String, Object> run = (Map<String, Object>) resp.get("run");
        if (run == null) throw new IllegalStateException("MLflow did not return run: " + resp);
        String runId = (String) run.get("info.run_id");
        if (runId == null) runId = (String) run.get("run_id");
        if (runId == null) throw new IllegalStateException("MLflow did not return run_id");
        experimentIdMapping.put(exp.id, runId);
        runNameToId.put(exp.name, runId);
        return runId;
    }

    private static List<Map<String, Object>> mlTags(CanonicalExperiment exp) {
        List<Map<String, Object>> tags = new ArrayList<>();
        for (Map.Entry<String, String> e : exp.tags.entrySet()) {
            tags.add(mlTag(e.getKey(), e.getValue()));
        }
        tags.add(mlTag("project", exp.project));
        tags.add(mlTag("owner", exp.owner));
        tags.add(mlTag("type", exp.type.name()));
        tags.add(mlTag("mlflow.user", exp.owner));
        return tags;
    }

    private static Map<String, Object> mlTag(String key, String value) {
        Map<String, Object> t = new LinkedHashMap<>();
        t.put("key", key);
        t.put("value", value != null ? value : "");
        return t;
    }

    @Override
    public void updateExperiment(String experimentId, CanonicalExperiment update) {
        requireConfig();
        String runId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("run_id", runId);
        req.put("status", toMlflowStatus(update.status));
        req.put("end_time", update.finishedAt != null ? update.finishedAt.toEpochMilli() : Instant.now().toEpochMilli());
        req.put("run_name", update.name);
        http.postJson(endpoint("/api/2.0/mlflow/runs/update"), req);
    }

    @Override
    public void logMetrics(String experimentId, List<MetricPoint> points) {
        requireConfig();
        String runId = resolve(experimentId);
        // MLflow supports batch via /runs/log-metric with multiple keys.
        for (MetricPoint p : points) {
            Map<String, Object> req = new LinkedHashMap<>();
            req.put("run_id", runId);
            req.put("key", p.name);
            req.put("value", p.value);
            req.put("timestamp", p.timestamp.toEpochMilli());
            req.put("step", p.step);
            http.postJson(endpoint("/api/2.0/mlflow/runs/log-metric"), req);
        }
    }

    @Override
    public void logParameters(String experimentId, List<ExperimentParameter> params) {
        requireConfig();
        String runId = resolve(experimentId);
        for (ExperimentParameter p : params) {
            Map<String, Object> req = new LinkedHashMap<>();
            req.put("run_id", runId);
            req.put("key", p.key);
            req.put("value", p.value);
            http.postJson(endpoint("/api/2.0/mlflow/runs/log-parameter"), req);
        }
    }

    @Override
    public void logArtifact(String experimentId, Artifact artifact) {
        requireConfig();
        String runId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("run_id", runId);
        req.put("path", artifact.name);
        req.put("uri", artifact.uri);
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("kind", artifact.kind.name());
        meta.put("size", artifact.sizeBytes);
        meta.put("content_type", artifact.contentType);
        req.put("metadata", meta);
        http.postJson(endpoint("/api/2.0/mlflow/runs/log-artifact"), req);
        // If it's a model, also register a model version.
        if (artifact.kind == Artifact.ArtifactKind.MODEL) {
            registerModelVersion(runId, artifact);
        }
    }

    private void registerModelVersion(String runId, Artifact artifact) {
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("name", artifact.name);
        req.put("source", artifact.uri);
        req.put("run_id", runId);
        try {
            http.postJson(endpoint("/api/2.0/mlflow/model-versions/create"), req);
        } catch (RuntimeException e) {
            // Best-effort: model registry may not be configured.
        }
    }

    @Override
    public void completeExperiment(String experimentId) {
        requireConfig();
        String runId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("run_id", runId);
        req.put("status", "FINISHED");
        http.postJson(endpoint("/api/2.0/mlflow/runs/set-terminated"), req);
    }

    @Override
    public void failExperiment(String experimentId, String reason) {
        requireConfig();
        String runId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("run_id", runId);
        req.put("status", "FAILED");
        req.put("end_time", Instant.now().toEpochMilli());
        http.postJson(endpoint("/api/2.0/mlflow/runs/set-terminated"), req);
    }

    @Override
    public String resolveExperimentId(CanonicalExperiment exp) {
        return experimentIdMapping.getOrDefault(exp.id, exp.id);
    }

    private String resolve(String experimentId) {
        return experimentIdMapping.getOrDefault(experimentId, experimentId);
    }

    private void requireConfig() {
        if (config == null) throw new IllegalStateException("configure() not called");
    }

    private static String toMlflowStatus(String s) {
        switch (s) {
            case "running": return "RUNNING";
            case "completed": return "FINISHED";
            case "failed": return "FAILED";
            case "killed": return "KILLED";
            case "queued": return "SCHEDULED";
            default: return "RUNNING";
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "MlflowSink{endpoint=%s experiment=%s}",
                config != null ? config.endpointUrl : "?", defaultExperimentId);
    }
}