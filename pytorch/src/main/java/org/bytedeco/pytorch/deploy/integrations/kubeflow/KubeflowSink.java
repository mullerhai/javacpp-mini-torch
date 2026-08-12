/*
 * Kubeflow integration.
 *
 * Kubeflow is a Kubernetes-native MLOps platform. Two sub-systems are
 * relevant for an experiment tracking integration:
 *
 *   1. Kubeflow Pipelines (KFP) - REST API on port 8888 (default)
 *      - POST /apis/v1beta1/runs                -> start a run
 *      - GET  /apis/v1beta1/runs/<id>           -> read run
 *      - GET  /apis/v1beta1/runs/<id>/nodes     -> list nodes
 *      - POST /apis/v1beta1/experiments         -> create experiment
 *
 *   2. Kubeflow Katib - hyperparameter tuning
 *      - POST /api/v1/namespaces/<ns>/experiments
 *      - GET  /api/v1/namespaces/<ns>/experiments/<name>
 *      - GET  /api/v1/namespaces/<ns>/trials
 *      - POST /api/v1/namespaces/<ns>/trials    -> (verbs on experiments)
 *
 *   3. ML Metadata (MLMD) - model registry
 *      - POST /api/v1alpha1/artifact_types, context_types, etc.
 *
 * We expose a single KubeflowSink that covers all three:
 *   - Logs metrics via Katib (since Katib collects metrics via pod logs)
 *   - Logs artifacts via MLMD
 *   - Tracks experiments via KFP
 *
 * Reference: https://www.kubeflow.org/docs/components/
 */
package org.bytedeco.pytorch.deploy.integrations.kubeflow;

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
import java.util.concurrent.ConcurrentHashMap;

/**
 * Kubeflow sink covering KFP + Katib + MLMD.
 */
public final class KubeflowSink implements MlopsSink {

    private SinkConfig config;
    private final HttpJsonClient http;
    private final Map<String, String> expIdMapping = new ConcurrentHashMap<>();
    private final Map<String, String> trialIdMapping = new ConcurrentHashMap<>();
    private final Map<String, String> artifactIdMapping = new ConcurrentHashMap<>();

    public KubeflowSink() { this(new HttpJsonClient()); }
    public KubeflowSink(HttpJsonClient http) { this.http = Objects.requireNonNull(http); }

    @Override public String platformName() { return "kubeflow"; }

    @Override
    public void configure(SinkConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        http.setAuthToken(config.authToken);
    }

    private String endpoint(String path) {
        String base = config.endpointUrl != null ? config.endpointUrl : "http://localhost:8888";
        if (base.endsWith("/")) base = base.substring(0, base.length() - 1);
        return base + path;
    }

    private String namespace() {
        return config.projectNamespace != null ? config.projectNamespace : "default";
    }

    @Override
    public String startExperiment(CanonicalExperiment exp) {
        requireConfig();
        // Create or look up KFP experiment.
        String kfpExpId = ensureKfpExperiment(exp.project);
        // Create a Katib Experiment (representing the trial).
        Map<String, Object> katibReq = new LinkedHashMap<>();
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("name", exp.id);
        meta.put("namespace", namespace());
        katibReq.put("metadata", meta);
        Map<String, Object> spec = new LinkedHashMap<>();
        Map<String, Object> objective = new LinkedHashMap<>();
        objective.put("type", "maximize");
        objective.put("goal", 0.0);
        objective.put("objectiveMetricName", "metric");
        spec.put("objective", objective);
        Map<String, Object> algorithm = new LinkedHashMap<>();
        algorithm.put("algorithmName", "random");
        spec.put("algorithm", algorithm);
        Map<String, Object> nas = new LinkedHashMap<>();
        nas.put("objective", objective);
        spec.put("nasConfig", nas);
        spec.put("maxTrialCount", 1);
        spec.put("parallelTrialCount", 1);
        Map<String, Object> params = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : exp.tags.entrySet()) {
            params.put(e.getKey(), e.getValue());
        }
        spec.put("parameters", params);
        Map<String, Object> template = new LinkedHashMap<>();
        template.put("primaryContainerName", "training");
        spec.put("trialTemplate", template);
        katibReq.put("spec", spec);

        try {
            Map<String, Object> resp = http.postJson(
                    endpoint("/api/v1/namespaces/" + namespace() + "/experiments"), katibReq);
            Object trialId = resp.get("name");
            if (trialId != null) trialIdMapping.put(exp.id, trialId.toString());
        } catch (RuntimeException e) {
            // Best-effort: Katib may not be installed.
        }

        // Create a KFP run for this canonical experiment.
        Map<String, Object> runReq = new LinkedHashMap<>();
        runReq.put("name", exp.name);
        runReq.put("experiment_id", kfpExpId);
        Map<String, Object> pipelineSpec = new LinkedHashMap<>();
        pipelineSpec.put("pipeline_id", exp.project);
        pipelineSpec.put("parameters", new LinkedHashMap<>());
        runReq.put("pipeline_spec", pipelineSpec);
        Map<String, Object> resp = http.postJson(endpoint("/apis/v1beta1/runs"), runReq);
        Map<String, Object> run = (Map<String, Object>) resp.get("run");
        String runId = run != null ? (String) run.get("id") : exp.id;
        expIdMapping.put(exp.id, runId);
        return runId;
    }

    private String ensureKfpExperiment(String project) {
        Map<String, Object> req = new LinkedHashMap<>();
        req.put("name", project);
        req.put("description", "Auto-created by abtest/integrations");
        try {
            Map<String, Object> resp = http.postJson(endpoint("/apis/v1beta1/experiments"), req);
            Map<String, Object> created = (Map<String, Object>) resp.get("experiment");
            return created != null ? (String) created.get("id") : project;
        } catch (RuntimeException e) {
            return project; // fall back to name
        }
    }

    @Override
    public void updateExperiment(String experimentId, CanonicalExperiment update) {
        requireConfig();
        String runId = resolve(experimentId);
        Map<String, Object> patch = new LinkedHashMap<>();
        patch.put("status", toKfpStatus(update.status));
        if (update.finishedAt != null) {
            patch.put("finished_at", update.finishedAt.toString());
        }
        try {
            // KFP has a "terminated" patch op.
            http.postJson(endpoint("/apis/v1beta1/runs/" + runId), patch);
        } catch (RuntimeException ignored) {}
    }

    @Override
    public void logMetrics(String experimentId, List<MetricPoint> points) {
        requireConfig();
        String trialName = trialIdMapping.getOrDefault(experimentId, experimentId);
        for (MetricPoint p : points) {
            Map<String, Object> req = new LinkedHashMap<>();
            req.put("name", trialName);
            req.put("namespace", namespace());
            req.put("metric_name", p.name);
            req.put("value", p.value);
            req.put("timestamp", p.timestamp.toString());
            try {
                http.postJson(endpoint("/api/v1/namespaces/" + namespace() + "/trials") + "/" + trialName + "/metrics", req);
            } catch (RuntimeException e) {
                // Best-effort: metrics may not be supported in this KF version.
            }
        }
    }

    @Override
    public void logParameters(String experimentId, List<ExperimentParameter> params) {
        requireConfig();
        String trialName = trialIdMapping.getOrDefault(experimentId, experimentId);
        for (ExperimentParameter p : params) {
            Map<String, Object> req = new LinkedHashMap<>();
            req.put("name", trialName);
            req.put("namespace", namespace());
            req.put("parameter_name", p.getKey());
            req.put("value", p.getValue());
            try {
                http.postJson(endpoint("/api/v1/namespaces/" + namespace() + "/trials") + "/" + trialName + "/parameters", req);
            } catch (RuntimeException ignored) {}
        }
    }

    @Override
    public void logArtifact(String experimentId, Artifact artifact) {
        requireConfig();
        // Use MLMD to record artifact.
        Map<String, Object> req = new LinkedHashMap<>();
        Map<String, Object> type = new LinkedHashMap<>();
        type.put("name", artifact.kind.name());
        req.put("artifact_type", type);
        Map<String, Object> uri = new LinkedHashMap<>();
        uri.put("uri", artifact.uri);
        req.put("uri", uri);
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("name", artifact.name);
        props.put("experiment_id", experimentId);
        props.put("size", artifact.sizeBytes);
        req.put("properties", props);
        try {
            http.postJson(endpoint("/api/v1alpha1/artifact_types"), req);
        } catch (RuntimeException ignored) {}
        artifactIdMapping.put(artifact.name, artifact.uri);
    }

    @Override
    public void completeExperiment(String experimentId) {
        updateExperiment(experimentId, CanonicalExperiment.builder(experimentId).status("completed").build());
    }

    @Override
    public void failExperiment(String experimentId, String reason) {
        updateExperiment(experimentId, CanonicalExperiment.builder(experimentId).status("failed").build());
    }

    @Override
    public String resolveExperimentId(CanonicalExperiment exp) {
        return expIdMapping.getOrDefault(exp.id, exp.id);
    }

    private String resolve(String experimentId) {
        return expIdMapping.getOrDefault(experimentId, experimentId);
    }

    private void requireConfig() {
        if (config == null) throw new IllegalStateException("configure() not called");
    }

    private static String toKfpStatus(String s) {
        switch (s) {
            case "running": return "Running";
            case "completed": return "Succeeded";
            case "failed": return "Failed";
            case "killed": return "Skipped";
            default: return "Pending";
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "KubeflowSink{endpoint=%s namespace=%s}",
                config != null ? config.endpointUrl : "?", namespace());
    }
}