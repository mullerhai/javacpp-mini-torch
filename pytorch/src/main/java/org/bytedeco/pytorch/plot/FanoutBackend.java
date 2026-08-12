/*
 * Multi-backend fanout writer (Phase 5).
 *
 * 同时写入多个 TrainingBackend, 用于"训练主循环 0 等待"场景.
 * 也可用于自动 fallback: 主 backend 失败 → 切到 LocalServer.
 */
package org.bytedeco.pytorch.plot;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Fan-out writer that forwards every call to N child {@link TrainingBackend}s.
 * <p>Usage:
 * <pre>{@code
 *   TrainingBackend wb = WandbBackend.of("myrun", Map.of("lr", 1e-3));
 *   TrainingBackend tb = TensorBoardBackend.of("runs/exp");
 *   try (FanoutBackend fan = FanoutBackend.of(wb, tb)) {
 *       for (int step = 0; step < 1000; step++) {
 *           fan.log(Map.of("loss", loss), step);
 *       }
 *   }
 * }</pre>
 */
public final class FanoutBackend implements TrainingBackend {

    private final String primaryName;
    private final CopyOnWriteArrayList<TrainingBackend> children = new CopyOnWriteArrayList<>();
    private final java.util.concurrent.atomic.AtomicReference<String> activeRunId =
            new java.util.concurrent.atomic.AtomicReference<>();
    private boolean closed = false;

    private FanoutBackend(TrainingBackend... backends) {
        if (backends.length == 0) {
            throw new IllegalArgumentException("FanoutBackend requires at least one backend");
        }
        this.primaryName = backends[0].name();
        for (TrainingBackend b : backends) children.add(b);
    }

    public static FanoutBackend of(TrainingBackend... backends) {
        return new FanoutBackend(backends);
    }

    /** Number of registered children. */
    public int size() { return children.size(); }

    public List<TrainingBackend> children() { return new ArrayList<>(children); }

    @Override public String name() { return primaryName; }

    @Override public boolean isReady() {
        for (TrainingBackend b : children) {
            if (b.isReady()) return true;
        }
        return false;
    }

    @Override
    public String init(String runName, Map<String, ?> config) throws IOException {
        String firstId = null;
        IOException lastErr = null;
        for (TrainingBackend b : children) {
            try {
                String id = b.init(runName, config);
                if (firstId == null) firstId = id;
            } catch (IOException e) {
                lastErr = e;
                // fall through to next backend — fan-out is best-effort
            }
        }
        if (firstId == null && lastErr != null) throw lastErr;
        activeRunId.set(firstId);
        return firstId;
    }

    @Override
    public void finish() throws IOException {
        IOException lastErr = null;
        for (TrainingBackend b : children) {
            try { b.finish(); }
            catch (IOException e) { lastErr = e; }
        }
        if (lastErr != null) throw lastErr;
    }

    @Override
    public void log(Map<String, ? extends Number> metrics, long step) throws IOException {
        IOException lastErr = null;
        int ok = 0;
        for (TrainingBackend b : children) {
            try { b.log(metrics, step); ok++; }
            catch (IOException e) { lastErr = e; }
        }
        if (ok == 0 && lastErr != null) throw lastErr;
    }

    @Override
    public void logSummary(Map<String, ?> summary) throws IOException {
        for (TrainingBackend b : children) {
            try { b.logSummary(summary); } catch (IOException ignored) {}
        }
    }

    @Override
    public void logImage(String name, byte[] png, long step) throws IOException {
        for (TrainingBackend b : children) {
            try { b.logImage(name, png, step); } catch (IOException ignored) {}
        }
    }

    @Override
    public void logArtifact(String name, Path file, String type) throws IOException {
        for (TrainingBackend b : children) {
            try { b.logArtifact(name, file, type); } catch (IOException ignored) {}
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (TrainingBackend b : children) {
            try { b.close(); } catch (Exception ignored) {}
        }
    }
}