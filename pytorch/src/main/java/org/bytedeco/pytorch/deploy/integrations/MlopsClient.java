/*
 * High-level MLOps client facade.
 *
 * Routes calls to either a single configured sink or broadcasts to all
 * registered sinks (for teams that maintain multiple platforms in parallel).
 */
package org.bytedeco.pytorch.deploy.integrations;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * High-level facade used by service code.
 *
 * <p>Routes calls to one configured sink (default) or broadcasts to all
 * registered sinks if {@link Mode#BROADCAST}.
 */
public final class MlopsClient {

    public enum Mode { SINGLE, BROADCAST }

    private final Mode mode;
    private final MlopsSink primary;
    private final List<Consumer<String>> logListeners = new CopyOnWriteArrayList<>();

    public MlopsClient(Mode mode, MlopsSink primary) {
        this.mode = mode != null ? mode : Mode.SINGLE;
        this.primary = primary;
    }

    public static MlopsClient single(MlopsSink s) { return new MlopsClient(Mode.SINGLE, s); }

    public static MlopsClient broadcast() {
        return new MlopsClient(Mode.BROADCAST, null);
    }

    public Mode mode() { return mode; }

    public MlopsSink primarySink() { return primary; }

    public void addLogListener(Consumer<String> l) { logListeners.add(l); }

    public String startExperiment(CanonicalExperiment exp) {
        log("start experiment: " + exp);
        if (mode == Mode.SINGLE) return requireSink().startExperiment(exp);
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.startExperiment(exp); } catch (RuntimeException e) { log("sink " + s + ": " + e.getMessage()); }
        }
        return exp.id;
    }

    public void logMetrics(String experimentId, List<MetricPoint> points) {
        if (points == null || points.isEmpty()) return;
        if (mode == Mode.SINGLE) { requireSink().logMetrics(experimentId, points); return; }
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.logMetrics(experimentId, points); } catch (RuntimeException e) { log("sink " + s + ": " + e.getMessage()); }
        }
    }

    public void logParameters(String experimentId, java.util.List<ExperimentParameter> params) {
        if (params == null || params.isEmpty()) return;
        if (mode == Mode.SINGLE) { requireSink().logParameters(experimentId, params); return; }
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.logParameters(experimentId, params); } catch (RuntimeException e) { log("sink " + s + ": " + e.getMessage()); }
        }
    }

    public void updateExperiment(String experimentId, CanonicalExperiment update) {
        if (mode == Mode.SINGLE) { requireSink().updateExperiment(experimentId, update); return; }
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.updateExperiment(experimentId, update); } catch (RuntimeException e) { log("sink " + s + ": " + e.getMessage()); }
        }
    }

    public void logArtifact(String experimentId, Artifact artifact) {
        log("log artifact: " + artifact.name + " -> " + artifact.uri);
        if (mode == Mode.SINGLE) { requireSink().logArtifact(experimentId, artifact); return; }
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.logArtifact(experimentId, artifact); } catch (RuntimeException e) { log("sink " + s + ": " + e.getMessage()); }
        }
    }

    public void completeExperiment(String experimentId) {
        log("complete experiment: " + experimentId);
        if (mode == Mode.SINGLE) { requireSink().completeExperiment(experimentId); return; }
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.completeExperiment(experimentId); } catch (RuntimeException ignored) {}
        }
    }

    public void failExperiment(String experimentId, String reason) {
        log("fail experiment: " + experimentId + " reason=" + reason);
        if (mode == Mode.SINGLE) { requireSink().failExperiment(experimentId, reason); return; }
        for (MlopsSink s : MlopsSinkRegistry.all()) {
            try { s.failExperiment(experimentId, reason); } catch (RuntimeException ignored) {}
        }
    }

    private MlopsSink requireSink() {
        if (primary == null) throw new IllegalStateException("no primary sink configured");
        return primary;
    }

    private void log(String msg) {
        for (Consumer<String> l : logListeners) {
            try { l.accept(String.format(Locale.ROOT, "[MlopsClient] %s", msg)); }
            catch (RuntimeException ignored) {}
        }
    }
}
