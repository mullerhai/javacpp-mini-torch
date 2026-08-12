/*
 * MLflow <-> ClearML <-> Kubeflow import shims.
 *
 * Operational reality: many teams have MLflow for tracking, ClearML for
 * pipelines, and Kubeflow on k8s. Supporting all three with one schema
 * is the value of this integration layer.
 *
 * The {@link MlopsCompatibility} class detects which platform is
 * configured (from environment variables) and exposes a uniform
 * {@link MlopsClient} that delegates accordingly.
 *
 * Detection order:
 *   1. MLOPS_ENDPOINT  - explicit platform URL prefix
 *   2. MLFLOW_TRACKING_URI - MLflow
 *   3. CLEARML_API_HOST   - ClearML
 *   4. KUBEFLOW_ENDPOINT  - Kubeflow
 *   5. None              - no-op sink
 */
package org.bytedeco.pytorch.deploy.integrations;

import org.bytedeco.pytorch.deploy.integrations.clearml.ClearMLSink;
import org.bytedeco.pytorch.deploy.integrations.kubeflow.KubeflowSink;
import org.bytedeco.pytorch.deploy.integrations.mlflow.MlflowSink;

import java.util.Locale;
import java.util.Objects;

/**
 * Auto-detects MLOps backend and exposes a uniform client.
 */
public final class MlopsCompatibility {

    private MlopsCompatibility() {}

    /** Backend enumeration. */
    public enum Backend {
        MLFLOW, CLEARML, KUBEFLOW, BROADCAST, NOOP
    }

    /**
     * Returns the detected backend.
     */
    public static Backend detect() {
        if (env("MLOPS_BROADCAST", null) != null) return Backend.BROADCAST;
        if (env("MLFLOW_TRACKING_URI", null) != null) return Backend.MLFLOW;
        if (env("CLEARML_API_HOST", null) != null) return Backend.CLEARML;
        if (env("KUBEFLOW_ENDPOINT", null) != null) return Backend.KUBEFLOW;
        if (env("MLOPS_ENDPOINT", null) != null) {
            String u = env("MLOPS_ENDPOINT", "").toLowerCase(Locale.ROOT);
            if (u.contains("mlflow")) return Backend.MLFLOW;
            if (u.contains("clear") || u.contains("allegro")) return Backend.CLEARML;
            if (u.contains("kubeflow") || u.contains("kfp")) return Backend.KUBEFLOW;
        }
        return Backend.NOOP;
    }

    /** Build a sink based on env detection. */
    public static MlopsSink autoSink() {
        Backend b = detect();
        switch (b) {
            case MLFLOW: {
                MlflowSink s = new MlflowSink();
                s.configure(SinkConfig.builder()
                        .endpointUrl(env("MLFLOW_TRACKING_URI", "http://localhost:5000"))
                        .authToken(env("MLFLOW_TRACKING_TOKEN", null))
                        .projectNamespace(env("MLFLOW_EXPERIMENT_NAME", "default"))
                        .build());
                return s;
            }
            case CLEARML: {
                ClearMLSink s = new ClearMLSink();
                s.configure(SinkConfig.builder()
                        .endpointUrl(env("CLEARML_API_HOST", "http://localhost:8008"))
                        .authToken(env("CLEARML_AUTH_TOKEN", null))
                        .projectNamespace(env("CLEARML_PROJECT", "default"))
                        .build());
                return s;
            }
            case KUBEFLOW: {
                KubeflowSink s = new KubeflowSink();
                s.configure(SinkConfig.builder()
                        .endpointUrl(env("KUBEFLOW_ENDPOINT", "http://localhost:8888"))
                        .authToken(env("KUBEFLOW_BEARER_TOKEN", null))
                        .projectNamespace(env("KUBEFLOW_NAMESPACE", "kubeflow"))
                        .build());
                return s;
            }
            case BROADCAST: {
                MlopsSinkRegistry.registerIfMissing(Backend.MLFLOW);
                MlopsSinkRegistry.registerIfMissing(Backend.CLEARML);
                MlopsSinkRegistry.registerIfMissing(Backend.KUBEFLOW);
                return null;
            }
            default:
                return new NoopMlopsSink();
        }
    }

    /** Build a client based on env detection. */
    public static MlopsClient autoClient() {
        Backend b = detect();
        if (b == Backend.BROADCAST) return MlopsClient.broadcast();
        MlopsSink s = autoSink();
        return s == null ? MlopsClient.broadcast() : MlopsClient.single(s);
    }

    private static String env(String key, String fallback) {
        String v = System.getenv(key);
        return v != null ? v : fallback;
    }

    /** Inner accessor — extend MlopsSinkRegistry with auto-register helpers. */
    public static final class Static {
        private Static() {}
    }
}

/**
 * No-op sink for environments with no MLOps backend.
 */
final class NoopMlopsSink implements MlopsSink {
    @Override public void configure(SinkConfig config) {}
    @Override public String startExperiment(CanonicalExperiment exp) { return exp.id; }
    @Override public void updateExperiment(String experimentId, CanonicalExperiment update) {}
    @Override public void logMetrics(String experimentId, java.util.List<MetricPoint> points) {}
    @Override public void logParameters(String experimentId, java.util.List<ExperimentParameter> params) {}
    @Override public void logArtifact(String experimentId, Artifact artifact) {}
    @Override public void completeExperiment(String experimentId) {}
    @Override public void failExperiment(String experimentId, String reason) {}
    @Override public String platformName() { return "noop"; }
}