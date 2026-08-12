/*
 * BackendRegistry (Phase 5):
 *   - 自动 fallback: 主 backend 失败 → 切到 LocalServer
 *   - 多 backend fan-out 一键启用
 *   - 环境变量驱动的默认 backend 选择
 *
 * Env:
 *   PLOT_BACKEND  = "wandb" | "swanlab" | "tensorboard" | "visdom" | "fanout:wandb+tensorboard" | "auto"
 */
package org.bytedeco.pytorch.plot;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public final class BackendRegistry {

    private BackendRegistry() {}

    /** Open a backend from a string spec. */
    public static TrainingBackend open(String spec, String runName, Map<String, ?> config)
            throws IOException {
        if (spec == null || spec.isEmpty() || "auto".equalsIgnoreCase(spec)) {
            spec = System.getenv().getOrDefault("PLOT_BACKEND", "tensorboard");
        }
        return openExplicit(spec, runName, config);
    }

    static TrainingBackend openExplicit(String spec, String runName, Map<String, ?> config)
            throws IOException {
        switch (spec.toLowerCase()) {
            case "wandb":
                return WandbBackend.of(runName, config);
            case "swanlab":
                return SwanLabBackend.of(runName, config);
            case "tensorboard":
            case "tb":
                return TensorBoardBackend.of(runName, "runs/" + runName);
            case "fanout:wandb+tensorboard":
                return FanoutBackend.of(
                        WandbBackend.of(runName, config),
                        TensorBoardBackend.of(runName, "runs/" + runName));
            case "fanout:swanlab+tensorboard":
                return FanoutBackend.of(
                        SwanLabBackend.of(runName, config),
                        TensorBoardBackend.of(runName, "runs/" + runName));
            case "fanout:wandb+swanlab+tensorboard":
                return FanoutBackend.of(
                        WandbBackend.of(runName, config),
                        SwanLabBackend.of(runName, config),
                        TensorBoardBackend.of(runName, "runs/" + runName));
            default:
                // Try with prefix 'fanout:' generic
                if (spec.startsWith("fanout:")) {
                    String[] names = spec.substring("fanout:".length()).split("\\+");
                    TrainingBackend[] backs = new TrainingBackend[names.length];
                    for (int i = 0; i < names.length; i++) {
                        backs[i] = openExplicit(names[i].trim(), runName, config);
                    }
                    return FanoutBackend.of(backs);
                }
                throw new IllegalArgumentException("Unknown backend spec: " + spec);
        }
    }

    /** Detect available backends in priority order: wandb → swanlab → tensorboard. */
    public static TrainingBackend auto(String runName, Map<String, ?> config) throws IOException {
        // 1) try wandb remote
        try {
            return WandbBackend.of(runName, config);
        } catch (IOException e) {
            // 2) fall back to tensorboard (always available offline)
            return TensorBoardBackend.of(runName, "runs/" + runName);
        }
    }
}