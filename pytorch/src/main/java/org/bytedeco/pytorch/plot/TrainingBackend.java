/*
 * Training-monitor unified abstraction (Phase 5).
 *
 * 多个 backend (wandb / swanlab / tensorboard / visdom) 共享同一接口,
 * 用户可以 fan-out 到多个 backend 或自动 fallback 到 LocalServer.
 *
 * Design:
 *   - Backend: 抽象一个训练监控 backend (initRun/log/finish/upload/...)
 *   - BackendRegistry: 全局注册表
 *   - Fanout: 同时发到多个 backend
 *   - AutoDetect: 检测网络可达性, 自动 fallback 到 LocalServer
 */
package org.bytedeco.pytorch.plot;

/** Common backend abstraction for any training monitor. */
public interface TrainingBackend extends AutoCloseable {
    /** Display name ("wandb" / "swanlab" / "tensorboard" / "visdom"). */
    String name();

    /** True if this backend is ready to accept log() calls. */
    boolean isReady();

    /** Init a run with optional config map. Returns runId. */
    String init(String runName, java.util.Map<String, ?> config) throws java.io.IOException;

    /** Finish the run. */
    void finish() throws java.io.IOException;

    /** Log metrics at given step. */
    void log(java.util.Map<String, ? extends Number> metrics, long step)
            throws java.io.IOException;

    /** Log summary at end of run. */
    void logSummary(java.util.Map<String, ?> summary) throws java.io.IOException;

    /** Optional: log an image. Backends that don't support can no-op. */
    default void logImage(String name, byte[] png, long step) throws java.io.IOException {}

    /** Optional: log an artifact (model ckpt, dataset, etc.). */
    default void logArtifact(String name, java.nio.file.Path file, String type) throws java.io.IOException {}

    /** Default no-op close. */
    @Override
    default void close() {}
}