/*
 * ClearML (Allegro AI) integration.
 *
 * ClearML exposes a REST/HTTP API documented as the "ClearML Server" API
 * with three core services:
 *   - fileserver (default port 8081) : artifact upload/download
 *   - api       (default port 8008) : tasks, projects, metrics, params
 *   - webserver (default port 8080) : dashboard
 *
 * Auto-log / RemoteExecution uses the clearml-agent drop-in; we instead
 * implement pure-Java REST so the deployment has no Python dependency.
 *
 * Reference docs:
 *   https://clear.ml/docs/latest/docs/references/api_ref
 *
 * Authentication: bearer token (X-ClearML-Authorization) configured via
 * {@link SinkConfig#authToken}; the token is the user's "api credentials".
 *
 * Wire format used:
 *   POST {api}/tasks.create        -> {"task": "<id>", ...}
 *   POST {api}/tasks.edit           -> {}
 *   POST {api}/tasks.set_status     -> {}
 *   POST {api}/events.add_batch     -> {"added": true}
 *   POST {api}/tasks.add_or_update_parameters -> {}
 *   POST {files}/upload             -> multipart
 *
 * All HTTP calls go through {@link HttpJsonClient} (pure Java) so this
 * works without external dependencies.
 */
package org.bytedeco.pytorch.deploy.integrations.clearml;

import org.bytedeco.pytorch.deploy.integrations.Artifact;
import org.bytedeco.pytorch.deploy.integrations.CanonicalExperiment;
import org.bytedeco.pytorch.deploy.integrations.MetricPoint;
import org.bytedeco.pytorch.deploy.integrations.MlopsSink;
import org.bytedeco.pytorch.deploy.integrations.ExperimentParameter;
import org.bytedeco.pytorch.deploy.integrations.SinkConfig;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ClearML sink — REST-based, no Python required.
 */
public final class ClearMLSink implements MlopsSink {

    private SinkConfig config;
    private final HttpJsonClient http;
    private final Map<String, String> experimentIdMapping = new ConcurrentHashMap<>();

    public ClearMLSink() {
        this(new HttpJsonClient());
    }

    public ClearMLSink(HttpJsonClient http) {
        this.http = Objects.requireNonNull(http);
    }

    @Override
    public String platformName() { return "clearml"; }

    @Override
    public void configure(SinkConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        http.setAuthToken(config.authToken);
    }

    /** Two API endpoints: api (8008) for tasks / events, files (8081) for artifacts. */
    private String apiEndpoint() {
        return config.endpointUrl != null ? trimSlash(config.endpointUrl) : "http://localhost:8008";
    }

    private String filesEndpoint() {
        // Convention: same host, port+1 for fileserver.
        return apiEndpoint().replaceAll(":\\d+$", ":8081");
    }

    private static String trimSlash(String s) {
        return s.endsWith("/") ? s.substring(0, s.length() - 1) : s;
    }

    @Override
    public String startExperiment(CanonicalExperiment exp) {
        requireConfig();
        Map<String, Object> req = new LinkedHashMap<>();
        Map<String, Object> task = new LinkedHashMap<>();
        task.put("name", exp.name);
        task.put("project", config.projectNamespace + "/" + exp.project);
        task.put("type", "training");
        task.put("status", "in_progress");
        task.put("created", exp.createdAt != null ? exp.createdAt.toString() : Instant.now().toString());
        Map<String, Object> system = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : exp.systemic.entrySet()) {
            system.put(e.getKey(), e.getValue());
        }
        task.put("system_tags", system);
        task.put("tags", new ArrayList<>(exp.tags.keySet()));
        req.put("task", task);
        task.put("comment", exp.description);
        req.put("publish_description", "false");
        req.put("return_only_id", "true");

        Map<String, Object> resp = http.postJson(apiEndpoint() + "/tasks.create", req);
        Object id = resp.get("task_id");
        if (id == null) id = resp.get("id");
        if (id == null) throw new IllegalStateException("ClearML did not return a task id");
        String taskId = id.toString();
        experimentIdMapping.put(exp.id, taskId);
        return taskId;
    }

    @Override
    public void updateExperiment(String experimentId, CanonicalExperiment update) {
        requireConfig();
        String taskId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        Map<String, Object> task = new LinkedHashMap<>();
        task.put("id", taskId);
        if (update.status != null) task.put("status", toClearMLStatus(update.status));
        if (update.finishedAt != null) task.put("completed", update.finishedAt.toString());
        task.put("tags", new ArrayList<>(update.tags.keySet()));
        req.put("task", task);
        http.postJson(apiEndpoint() + "/tasks.edit", req);
    }

    @Override
    public void logMetrics(String experimentId, List<MetricPoint> points) {
        requireConfig();
        String taskId = resolve(experimentId);
        List<Map<String, Object>> events = new ArrayList<>();
        for (MetricPoint p : points) {
            Map<String, Object> ev = new LinkedHashMap<>();
            ev.put("task", taskId);
            ev.put("type", "scalar");
            ev.put("metric", p.name);
            ev.put("value", p.value);
            ev.put("iter", p.step);
            ev.put("timestamp", p.timestamp.toEpochMilli());
            events.add(ev);
        }
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("events", events);
        req.put("also_handle_iter", false);
        http.postJson(apiEndpoint() + "/events.add_batch", req);
    }

    @Override
    public void logParameters(String experimentId, List<ExperimentParameter> params) {
        requireConfig();
        String taskId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("task", taskId);
        Map<String, Object> pmap = new LinkedHashMap<>();
        int idx = 0;
        for (ExperimentParameter p : params) {
            Map<String, Object> sec = new LinkedHashMap<>();
            sec.put("section", "General");
            sec.put("name", p.key);
            sec.put("value", p.value);
            sec.put("type", toClearMLParamType(p.type));
            pmap.put("p" + idx++, sec);
        }
        req.put("parameters", pmap);
        req.put("replace", "true");
        http.postJson(apiEndpoint() + "/tasks.add_or_update_parameters", req);
    }

    @Override
    public void logArtifact(String experimentId, Artifact artifact) {
        requireConfig();
        String taskId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("task", taskId);
        req.put("uri", artifact.uri);
        req.put("name", artifact.name);
        req.put("type", toClearMLArtifactType(artifact.kind));
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("size", artifact.sizeBytes);
        meta.put("content_type", artifact.contentType);
        for (Map.Entry<String, String> e : artifact.metadata.entrySet()) {
            meta.put(e.getKey(), e.getValue());
        }
        req.put("metadata", meta);
        http.postJson(apiEndpoint() + "/tasks.add_or_update_artifacts", req);
    }

    @Override
    public void completeExperiment(String experimentId) {
        requireConfig();
        String taskId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("task", taskId);
        req.put("status", "completed");
        req.put("publish_reason", "Auto-completed by abtest/integrations");
        http.postJson(apiEndpoint() + "/tasks.set_status", req);
    }

    @Override
    public void failExperiment(String experimentId, String reason) {
        requireConfig();
        String taskId = resolve(experimentId);
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("task", taskId);
        req.put("status", "failed");
        req.put("reason", reason);
        http.postJson(apiEndpoint() + "/tasks.set_status", req);
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

    private static String toClearMLStatus(String s) {
        switch (s) {
            case "running": return "in_progress";
            case "completed": return "completed";
            case "failed": return "failed";
            case "killed": return "stopped";
            case "queued": return "created";
            default: return "in_progress";
        }
    }

    private static String toClearMLParamType(ExperimentParameter.ParameterType t) {
        switch (t) {
            case DOUBLE: return "float";
            case LONG: return "int";
            case BOOLEAN: return "bool";
            default: return "string";
        }
    }

    private static String toClearMLArtifactType(Artifact.ArtifactKind k) {
        switch (k) {
            case MODEL: return "model";
            case CHECKPOINT: return "checkpoint";
            case DATASET: return "dataset";
            case FEATURE_STORE: return "feature-store";
            case PLOT: return "plot";
            case REPORT: return "report";
            case LOG: return "log";
            default: return "file";
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "ClearMLSink{endpoint=%s}", apiEndpoint());
    }
}