/*
 * Platform-agnostic MLOps sink interface.
 *
 * All three platforms (ClearML, MLflow, Kubeflow) expose a similar API
 * (create experiment, log params/metrics/artifacts, finalize). This
 * interface normalizes them.
 */
package org.bytedeco.pytorch.deploy.integrations;

/**
 * Common MLOps sink interface.
 *
 * <p>Implementations register themselves in {@link MlopsSinkRegistry}.
 * Service code uses the high-level {@link MlopsClient} which dispatches
 * to a single configured sink (or "broadcast" to all if configured).
 */
public interface MlopsSink {

    /** Configure the sink (idempotent). */
    void configure(SinkConfig config);

    /** Create / start a new experiment. Returns the platform-assigned id. */
    String startExperiment(CanonicalExperiment exp);

    /** Update experiment metadata (status, finished_at, etc.). */
    void updateExperiment(String experimentId, CanonicalExperiment update);

    /** Log scalar metrics (batch). */
    void logMetrics(String experimentId, java.util.List<MetricPoint> points);

    /** Log parameters (batch). */
    void logParameters(String experimentId, java.util.List<ExperimentParameter> params);

    /** Upload an artifact (typically model file). */
    void logArtifact(String experimentId, Artifact artifact);

    /** Mark the experiment as completed. */
    void completeExperiment(String experimentId);

    /** Mark the experiment as failed. */
    void failExperiment(String experimentId, String reason);

    /** Read back the registered id for a canonical experiment. */
    default String resolveExperimentId(CanonicalExperiment exp) {
        return exp.id;
    }

    /** Friendly string for logs. */
    default String platformName() {
        return getClass().getSimpleName();
    }
}
