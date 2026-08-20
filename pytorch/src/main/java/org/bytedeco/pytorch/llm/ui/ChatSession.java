/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
/**
 * Per-browser-session state. Each HTTP session (cookie {@code JSESSION_ID}) has exactly
 * one {@code ChatSession} managed by {@link SessionRegistry}.
 *
 * <p>Holds:
 * <ul>
 *   <li>Conversation history (bounded deque, oldest dropped first).</li>
 *   <li>Current LoRA {@code tool} name (one of the {@code Engine.tools()} names).</li>
 *   <li>Current {@code GenerationConfig} baseline (overridable per-turn).</li>
 *   <li>{@code StreamingListener} — the active SSE writer (if any) for the in-flight turn.</li>
 *   <li>Concurrency tokens ({@code busy}, {@code cancelRequested}, {@code seqNum}).</li>
 * </ul>
 *
 * <p>This class is <b>not</b> internally synchronized — callers (the engine) must hold
 * the {@link SessionRegistry} lock when mutating it, or use the concurrent primitives
 * exposed ({@link #cancelRequested} / {@link #seqNum}).
 */
public final class ChatSession {

    public static final int DEFAULT_MAX_HISTORY = 50;

    public final String sessionId;
    public final long createdAtMillis;
    public volatile long lastAccessMillis;
    /** Currently selected tool name (matches a key in {@code Engine.tools()}). */
    public volatile String currentTool = "base";
    /** Currently selected template name (matches {@code Engine.templates()}). */
    public volatile String currentTemplate = "balanced";
    /** Per-session baseline {@code GenerationConfig} (defaults to the engine default). */
    public volatile GenerationConfig effectiveGenerationConfig;
    /** Bounded turn history. */
    public final Deque<ChatTurn> history;
    /** {@code true} while a turn is being generated. */
    public volatile boolean busy = false;
    /** Set to {@code true} to interrupt the current turn as soon as possible. */
    public volatile boolean cancelRequested = false;
    /** Monotonic sequence number — increments every turn. */
    public final AtomicInteger turnCounter = new AtomicInteger(0);
    /** Monotonic SSE event sequence number — used by the client to detect lost events. */
    public final AtomicLong seqNum = new AtomicLong(0L);
    /** Active stream listener; {@code null} between turns. */
    public volatile ChunkListener activeListener;
    /** Active {@code TextIteratorStreamer} for the in-flight turn. */
    public volatile org.bytedeco.pytorch.llm.generation.TextIteratorStreamer activeStreamer;

    public ChatSession(String sessionId, int maxHistory) {
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId");
        this.createdAtMillis = System.currentTimeMillis();
        this.lastAccessMillis = this.createdAtMillis;
        this.history = new ArrayDeque<>(Math.max(2, maxHistory));
    }

    public ChatSession(String sessionId) {
        this(sessionId, DEFAULT_MAX_HISTORY);
    }

    public int maxHistory() { return DEFAULT_MAX_HISTORY; }

    public void touch() {
        lastAccessMillis = System.currentTimeMillis();
    }

    public void appendTurn(ChatTurn turn) {
        Objects.requireNonNull(turn, "turn");
        if (history.size() >= DEFAULT_MAX_HISTORY) {
            history.pollFirst();
        }
        history.offerLast(turn);
        touch();
    }

    public void clearHistory() {
        history.clear();
        touch();
    }

    public List<ChatTurn> snapshotHistory() {
        return new java.util.ArrayList<>(history);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("sessionId", sessionId);
        m.put("createdAtMillis", createdAtMillis);
        m.put("lastAccessMillis", lastAccessMillis);
        m.put("currentTool", currentTool);
        m.put("currentTemplate", currentTemplate);
        m.put("busy", busy);
        m.put("turnCounter", turnCounter.get());
        m.put("historySize", history.size());
        return m;
    }
}