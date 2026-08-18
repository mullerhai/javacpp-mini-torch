/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.mlflow;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tiny MLflow-like logger. Writes metrics / params / artifacts to a local directory. Compatible
 * with offline MLflow tracking clients (the schema matches).
 */
public final class MLflowClient {

    private final String trackingDir;
    private final String experimentName;
    private final String runId;

    public MLflowClient(String experimentName, String trackingDir) {
        this.experimentName = experimentName;
        this.trackingDir = trackingDir;
        this.runId = Long.toHexString(System.currentTimeMillis());
        try {
            Path root = Paths.get(trackingDir);
            Files.createDirectories(root.resolve(experimentName).resolve(runId));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public MLflowClient logParam(String key, String value) {
        append("params", key, value);
        return this;
    }

    public MLflowClient logMetric(String key, double value) {
        append("metrics", key, String.valueOf(value));
        return this;
    }

    public MLflowClient logMetric(String key, double value, long step) {
        append("metrics", key, String.valueOf(value) + "," + step);
        return this;
    }

    public MLflowClient logArtifact(String sourcePath) {
        try {
            Path p = Paths.get(sourcePath);
            Path target = Paths.get(trackingDir, experimentName, runId, "artifacts", p.getFileName().toString());
            Files.createDirectories(target.getParent());
            Files.copy(p, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return this;
    }

    public MLflowClient logTextArtifact(String name, String content) {
        try {
            Path target = Paths.get(trackingDir, experimentName, runId, "artifacts", name);
            Files.createDirectories(target.getParent());
            Files.write(target, content.getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return this;
    }

    public MLflowClient end() {
        return this;
    }

    public String runId() { return runId; }
    public String experimentName() { return experimentName; }

    private void append(String kind, String key, String value) {
        try {
            Path p = Paths.get(trackingDir, experimentName, runId, kind + ".tsv");
            Files.write(p, (key + "\t" + value + "\n").getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
