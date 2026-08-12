/*
 * TrainingBackend adapter for SummaryWriter (TensorBoard) (Phase 5).
 */
package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.plot.tensorboard.SummaryWriter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public final class TensorBoardBackend implements TrainingBackend {

    private final SummaryWriter writer;
    private final boolean ownsWriter;
    private String runName;

    private TensorBoardBackend(SummaryWriter writer, boolean ownsWriter) {
        this.writer = writer;
        this.ownsWriter = ownsWriter;
    }

    public static TensorBoardBackend of(String runName, String logDir) throws IOException {
        java.io.File dir = new java.io.File(logDir);
        SummaryWriter w = new SummaryWriter(dir.getPath());
        TensorBoardBackend b = new TensorBoardBackend(w, true);
        b.runName = runName;
        return b;
    }

    /** Wrap an externally-managed SummaryWriter. */
    public static TensorBoardBackend wrap(SummaryWriter writer) {
        return new TensorBoardBackend(writer, false);
    }

    @Override public String name() { return "tensorboard"; }

    @Override public boolean isReady() { return writer != null; }

    @Override
    public String init(String runName, Map<String, ?> config) throws IOException {
        this.runName = runName;
        if (config != null) {
            try {
                Map<String, Object> hparams = new java.util.LinkedHashMap<>();
                hparams.put("run_name", runName);
                for (Map.Entry<String, ?> e : config.entrySet()) {
                    hparams.put(e.getKey(), e.getValue());
                }
                // hparams → log as scalars (best effort)
                long s = 0L;
                for (Map.Entry<String, ?> e : config.entrySet()) {
                    if (e.getValue() instanceof Number) {
                        writer.add_scalar("hparam/" + e.getKey(),
                                ((Number) e.getValue()).doubleValue(), s++);
                    }
                }
            } catch (Exception ignored) {}
        }
        return runName;
    }

    @Override
    public void finish() throws IOException {
        if (ownsWriter) writer.close();
    }

    @Override
    public void log(Map<String, ? extends Number> metrics, long step) throws IOException {
        if (metrics == null) return;
        for (Map.Entry<String, ? extends Number> e : metrics.entrySet()) {
            writer.add_scalar(e.getKey(), e.getValue().doubleValue(), step);
        }
    }

    @Override
    public void logSummary(Map<String, ?> summary) throws IOException {
        if (summary == null) return;
        long s = Long.MAX_VALUE - 1L;
        for (Map.Entry<String, ?> e : summary.entrySet()) {
            if (e.getValue() instanceof Number) {
                writer.add_scalar("summary/" + e.getKey(),
                        ((Number) e.getValue()).doubleValue(), s);
            } else {
                writer.add_text("summary/" + e.getKey(),
                        String.valueOf(e.getValue()), s);
            }
        }
    }

    @Override
    public void logImage(String name, byte[] png, long step) throws IOException {
        // TensorBoard needs a Tensor; skip if not provided
        // (caller should use SummaryWriter directly for image-heavy workloads)
    }

    @Override
    public void logArtifact(String name, Path file, String type) throws IOException {
        if (file == null || !Files.exists(file)) return;
        logSummary(Map.of(
                "artifact/" + name + "/path", file.toString(),
                "artifact/" + name + "/type", type == null ? "model" : type,
                "artifact/" + name + "/bytes", Files.size(file)));
    }

    public SummaryWriter writer() { return writer; }

    @Override
    public void close() {
        try { writer.flush(); } catch (IOException ignored) {}
        if (ownsWriter) {
            try { writer.close(); } catch (Exception ignored) {}
        }
    }
}