/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Configuration for the Web Demo HTTP server. Plain DTO — no validation, defaults only.
 * Mutable so the CLI can populate it.
 */
public final class WebDemoConfig {

    public String bindHost = "0.0.0.0";
    public int bindPort = 7860;
    public Path staticDir = Paths.get("pytorch/samples/llm/web/static");
    public int maxConcurrentTurns = 8;
    public long idleSessionTtlMs = 30L * 60L * 1000L;
    public int maxSessions = 200;
    public boolean corsAny = true;
    public String title = "JavaCPP-PyTorch LLM Web Demo";
    public String subtitle = "Pure-Java Gradio-style chat — zero Python dependencies";
    public String version = "1.0";
    public int executorThreads = 4;

    public WebDemoConfig() {}
}