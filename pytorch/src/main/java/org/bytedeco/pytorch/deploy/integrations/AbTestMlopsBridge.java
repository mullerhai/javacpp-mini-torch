/*
 * Bridge between the abtest infrastructure and the MLOps sinks.
 *
 * Auto-emits:
 *   - Experiment starts/transitions  -> startExperiment / updateExperiment
 *   - Guardrail evaluations          -> logMetrics (metric: guardrail_fires, decision)
 *   - Exposure records              -> logMetrics (exposure_count, per_variant)
 *   - Metric observations           -> logMetrics (raw metric value)
 *   - Variant parameter changes     -> logParameters (param.*)
 *   - Model artifacts (LR/MODEL)    -> logArtifact
 *
 * History shown to user via /experiment/{id} on ClearML / MLflow / KFP UI.
 */
package org.bytedeco.pytorch.deploy.integrations;

import org.bytedeco.pytorch.deploy.abtest.AbTestClient;
import org.bytedeco.pytorch.deploy.abtest.Experiment;
import org.bytedeco.pytorch.deploy.abtest.ExperimentStatus;
import org.bytedeco.pytorch.deploy.abtest.LayeredExperimentManager;
import org.bytedeco.pytorch.deploy.abtest.OnlineMetricsCollector;
import org.bytedeco.pytorch.deploy.abtest.StatisticalTest;
import org.bytedeco.pytorch.deploy.abtest.Variant;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

import org.bytedeco.pytorch.deploy.integrations.ExperimentParameter;

/**
 * Pushes abtest events to a single MLOps platform.
 */
public final class AbTestMlopsBridge {

    private final MlopsClient client;
    private final LayeredExperimentManager manager;
    private final AbTestClient abClient;
    private final Map<String, String> canonicalIds = new ConcurrentHashMap<>();
    private final Map<String, Long> stepCounters = new ConcurrentHashMap<>();

    public AbTestMlopsBridge(MlopsClient client, LayeredExperimentManager manager, AbTestClient abClient) {
        this.client = Objects.requireNonNull(client);
        this.manager = Objects.requireNonNull(manager);
        this.abClient = abClient;
    }

    /** Wire into the abtest manager (status / traffic / exposure events). */
    public void install() {
        manager.addListener(this::onAuditEvent);
        if (abClient != null) {
            abClient.addExposureListener(this::onExposure);
        }
    }

    private void onAuditEvent(LayeredExperimentManager.AuditEvent event) {
        if (event.experimentId() == null) return;
        Experiment exp = manager.getExperiment(event.experimentId());
        if (exp == null) return;
        CanonicalExperiment canonical = toCanonical(exp);
        switch (event.type()) {
            case EXPERIMENT_REGISTERED:
                client.startExperiment(canonical);
                canonicalIds.put(exp.id(), exp.id());
                break;
            case STATUS_CHANGED:
                ExperimentStatus st = parseStatusFromDetail(event.detail());
                if (st != null) {
                    client.updateExperiment(exp.id(), CanonicalExperiment.builder(exp.id())
                            .status(toMlopsStatus(st))
                            .finishedAt(st.isTerminal() ? Instant.now() : null)
                            .build());
                    if (st.isTerminal()) {
                        client.completeExperiment(exp.id());
                    }
                }
                break;
            case TRAFFIC_CHANGED:
                // Log a "traffic" parameter and a metric snapshot.
                client.logParameters(exp.id(), List.of(new ExperimentParameter("traffic_percent", exp.trafficPercent())));
                client.logMetrics(exp.id(), List.of(new MetricPoint("traffic_percent", exp.trafficPercent(), step(exp.id()))));
                break;
            default:
                break;
        }
    }

    private void onExposure(AbTestClient.ExposureRecord rec) {
        // Log a marker metric and exposure count.
        long step = step(rec.experimentId);
        List<MetricPoint> pts = new ArrayList<>();
        pts.add(new MetricPoint("exposure_" + rec.variantId, 1.0, step));
        client.logMetrics(rec.experimentId, pts);
    }

    /**
     * Push a full guardrail evaluation snapshot (used by {@link org.bytedeco.pytorch.deploy.abtest.ExperimentAnalyzer}).
     */
    public void pushGuardrailSnapshot(Experiment exp, String controlVariant, String treatmentVariant,
                                       OnlineMetricsCollector collector) {
        long step = step(exp.id());
        List<MetricPoint> pts = new ArrayList<>();
        for (String metric : exp.primaryMetrics()) {
            OnlineMetricsCollector.StatsSnapshot c = collector.stats(exp.id(), controlVariant, metric);
            OnlineMetricsCollector.StatsSnapshot t = collector.stats(exp.id(), treatmentVariant, metric);
            pts.add(new MetricPoint("control_mean_" + metric, c.mean, step));
            pts.add(new MetricPoint("treatment_mean_" + metric, t.mean, step));
            pts.add(new MetricPoint("delta_" + metric, t.mean - c.mean, step));
        }
        try {
            StatisticalTest.SrmResult srm = collector.srm(exp, 0.001);
            pts.add(new MetricPoint("srm_pvalue", srm.pValue, step));
        } catch (RuntimeException ignored) {}
        client.logMetrics(exp.id(), pts);
    }

    /**
     * Push a model artifact (e.g. trained pytorch model file).
     */
    public void pushModelArtifact(Experiment exp, String variantId, String localPath, String remoteUri) {
        Artifact a = new Artifact(
                "model_" + variantId,
                remoteUri,
                Artifact.ArtifactKind.MODEL,
                -1L,
                "application/octet-stream",
                Map.of("local_path", localPath, "variant", variantId));
        client.logArtifact(exp.id(), a);
    }

    private static CanonicalExperiment toCanonical(Experiment exp) {
        CanonicalExperiment.Builder b = CanonicalExperiment.builder(exp.id())
                .project(exp.layerId())
                .name(exp.name())
                .type(CanonicalExperiment.Type.A_B_TEST)
                .owner(exp.owner())
                .description(exp.description())
                .status(toMlopsStatus(exp.status()));
        b.tag("layer", exp.layerId());
        b.tag("diversion_unit", exp.diversionUnit().name());
        for (Map.Entry<String, String> e : exp.tags().entrySet()) b.tag(e.getKey(), e.getValue());
        for (String m : exp.primaryMetrics()) b.tag("primary_metric", m);
        for (String m : exp.guardrailMetrics()) b.tag("guardrail_metric", m);
        for (Variant v : exp.variants()) {
            b.tag("variant_" + v.id(), "weight=" + v.trafficWeight());
        }
        return b.build();
    }

    private static String toMlopsStatus(ExperimentStatus s) {
        switch (s) {
            case DRAFT: return "queued";
            case REVIEW: return "queued";
            case AA_RUNNING: return "running";
            case RUNNING: return "running";
            case PAUSED: return "running";
            case COMPLETED: return "completed";
            case KILLED: return "killed";
            case ROLLED_BACK: return "completed";
            default: return "running";
        }
    }

    private static ExperimentStatus parseStatusFromDetail(String detail) {
        if (detail == null) return null;
        // detail format: "DRAFT->REVIEW"
        int arrow = detail.indexOf("->");
        if (arrow < 0) return null;
        try { return ExperimentStatus.valueOf(detail.substring(arrow + 2).trim()); }
        catch (Exception e) { return null; }
    }

    private long step(String experimentId) {
        return stepCounters.merge(experimentId, 1L, Long::sum);
    }
}