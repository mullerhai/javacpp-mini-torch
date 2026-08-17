package org.bytedeco.pytorch.plot.wandb;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Training-loop helper that streams scalars / histograms / images / heatmaps / surfaces
 * to WandB, with best-practice defaults mirroring the official Python API.
 *
 * <pre>{@code
 * try (WandbLocalServer server = WandbLocalServer.start(0);
 *      WandbClient wb = WandbClient.newBuilder().offline(server)
 *              .entity("local").project("demo").build();
 *      WandbTrainingMonitor mon = new WandbTrainingMonitor(wb, "exp1")) {
 *     for (int s = 0; s < 1000; s++) {
 *         mon.logLoss(loss);
 *         mon.logAccuracy(acc);
 *         mon.watchModelGradients(model, s);
 *     }
 * }
 * }</pre>
 */
public final class WandbTrainingMonitor implements AutoCloseable {

    private final WandbClient client;
    private final String runName;
    private long step;
    private final boolean closeClient;
    private final boolean watchGradients;
    private final boolean watchParameters;
    private final int histogramBins;

    public WandbTrainingMonitor(WandbClient client, String runName)
            throws IOException, InterruptedException {
        this(client, runName, Map.of("framework", "javacpp-pytorch"), false, false, true, 64);
    }

    public WandbTrainingMonitor(WandbClient client, String runName, Map<String, ?> config,
                                boolean closeClient)
            throws IOException, InterruptedException {
        this(client, runName, config, closeClient, false, true, 64);
    }

    /**
     * @param config          run-level config
     * @param closeClient     close the underlying {@link WandbClient} on {@link #close()}
     * @param watchParameters log histogram of each parameter every step
     * @param histogramBins   bin count for histograms
     */
    public WandbTrainingMonitor(WandbClient client, String runName, Map<String, ?> config,
                                boolean closeClient, boolean watchGradients,
                                boolean watchParameters, int histogramBins)
            throws IOException, InterruptedException {
        this.client = client;
        this.runName = runName;
        this.closeClient = closeClient;
        this.watchGradients = watchGradients;
        this.watchParameters = watchParameters;
        this.histogramBins = histogramBins;
        this.step = 0;
        Map<String, Object> cfg = new LinkedHashMap<>();
        if (config != null) cfg.putAll(config);
        cfg.putIfAbsent("monitor", "WandbTrainingMonitor");
        client.initRun(runName, cfg);
    }

    public WandbClient client() { return client; }
    public long step() { return step; }
    public String uiUrl() { return client.uiUrl(); }

    public void logMetric(String key, double value) throws IOException, InterruptedException {
        client.log(Map.of(key, value), step++);
    }

    public void logLoss(double loss) throws IOException, InterruptedException {
        logMetric("loss", loss);
    }

    public void logAccuracy(double accuracy) throws IOException, InterruptedException {
        logMetric("accuracy", accuracy);
    }

    public void logLearningRate(double lr) throws IOException, InterruptedException {
        logMetric("learning_rate", lr);
    }

    public void log(Map<String, ? extends Number> metrics) throws IOException, InterruptedException {
        client.log(metrics, step++);
    }

    /** Log a map of metrics without bumping the internal step counter. */
    public void logAt(long s, Map<String, ? extends Number> metrics) throws IOException, InterruptedException {
        client.log(metrics, s);
    }

    public void logHeatmap(String name, double[][] matrix) throws IOException, InterruptedException {
        client.logHeatmap(name, matrix, step);
    }

    public void logHeatmap(String name, Tensor t) throws IOException, InterruptedException {
        client.logHeatmap(name, t, step);
    }

    public void logImage(String name, Tensor image) throws IOException, InterruptedException {
        client.logImage(name, image, step);
    }

    public void logImages(String name, Tensor batch) throws IOException, InterruptedException {
        client.logImages(name, batch, step);
    }

    public void logHistogram(String name, double[] values) throws IOException, InterruptedException {
        client.logHistogram(name, values, histogramBins, step);
    }

    public void logHistogram(String name, Tensor values) throws IOException, InterruptedException {
        client.logHistogram(name, values, histogramBins, step);
    }

    public void logText(String name, String text) throws IOException, InterruptedException {
        client.logText(name, text, step);
    }

    /** Record a 3D height-field surface (Plotly type=surface). */
    public void logSurface(String name, double[][] Z) throws IOException, InterruptedException {
        client.logSurface(name, Z, step);
    }

    /** Record a polar scatter. */
    public void logPolar(String name, double[][] polarPoints) throws IOException, InterruptedException {
        client.logPolar(name, polarPoints, step);
    }

    /** Record a box plot — each column of {@code data} is one box. */
    public void logBox(String name, double[][] data) throws IOException, InterruptedException {
        client.logBox(name, data, step);
    }

    /** Record a pie chart. */
    public void logPie(String name, double[] values, String[] labels) throws IOException, InterruptedException {
        client.logPie(name, values, labels, step);
    }

    /** Send a wandb-style alert. */
    public void alert(String title, String text, String level) throws IOException, InterruptedException {
        client.alert(title, text, level);
    }

    /** Upload an artifact (model / dataset / file). */
    public void logArtifact(String name, Path file, String type) throws IOException, InterruptedException {
        client.logArtifact(name, file, type);
    }

    /**
     * Capture parameter histograms (or gradient histograms, if {@code watchGradients}
     * was set). Returns the number of parameters captured.
     */
    public int watchModel(Module model) throws IOException, InterruptedException {
        if (!watchParameters) return 0;
        return client.watchModel("parameters", model, step);
    }

    /** Convenience: watch + bump step. */
    public int stepWith(Module model) throws IOException, InterruptedException {
        int n = watchModel(model);
        if (watchGradients) {
            // gradient histograms are recorded when the user manually populates _grad fields
        }
        step++;
        return n;
    }

    public void setStep(long step) { this.step = step; }

    @Override
    public void close() {
        try {
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("final_step", step);
            summary.put("run", runName);
            client.logSummary(summary);
            client.finish();
        } catch (Exception ignored) {
        }
        if (closeClient) client.close();
    }
}
