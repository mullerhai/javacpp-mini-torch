package org.bytedeco.pytorch.plot.swanlab;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Training-loop helper for SwanLab (mirrors {@code swanlab.log} usage with extra
 * surface / polar / box / artifact helpers and a best-practice API).
 */
public final class SwanLabTrainingMonitor implements AutoCloseable {

    private final SwanLabClient client;
    private final String experiment;
    private long step;
    private final boolean closeClient;
    private final boolean watchParameters;
    private final int histogramBins;

    public SwanLabTrainingMonitor(SwanLabClient client)
            throws IOException, InterruptedException {
        this(client, Map.of("framework", "javacpp-pytorch"), false, true, 64);
    }

    public SwanLabTrainingMonitor(SwanLabClient client, Map<String, ?> config, boolean closeClient)
            throws IOException, InterruptedException {
        this(client, config, closeClient, true, 64);
    }

    public SwanLabTrainingMonitor(SwanLabClient client, Map<String, ?> config,
                                  boolean closeClient, boolean watchParameters, int histogramBins)
            throws IOException, InterruptedException {
        this.client = client;
        this.experiment = client.experiment();
        this.closeClient = closeClient;
        this.watchParameters = watchParameters;
        this.histogramBins = histogramBins;
        this.step = 0;
        Map<String, Object> cfg = new LinkedHashMap<>();
        if (config != null) cfg.putAll(config);
        cfg.putIfAbsent("monitor", "SwanLabTrainingMonitor");
        client.init(cfg);
    }

    public SwanLabClient client() { return client; }
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

    public void logSurface(String name, double[][] Z) throws IOException, InterruptedException {
        client.logSurface(name, Z, step);
    }

    public void logPolar(String name, double[][] polarPoints) throws IOException, InterruptedException {
        client.logPolar(name, polarPoints, step);
    }

    public void logBox(String name, double[][] data) throws IOException, InterruptedException {
        client.logBox(name, data, step);
    }

    public void logArtifact(String name, Path file, String type) throws IOException, InterruptedException {
        client.logArtifact(name, file, type);
    }

    /** Capture parameter histograms once and bump step. Returns parameters recorded. */
    public int stepWith(Module model) throws IOException, InterruptedException {
        int n = 0;
        if (watchParameters) n = client.watchModel("parameters", model, step);
        step++;
        return n;
    }

    public void setStep(long step) { this.step = step; }

    @Override
    public void close() {
        try {
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("final_step", step);
            summary.put("experiment", experiment);
            client.logSummary(summary);
            client.finish();
        } catch (Exception ignored) {
        }
        if (closeClient) client.close();
    }
}
